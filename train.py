import sys
from models import ModelFactory
import json
import tempfile
import time
import copy
import os
import torchinfo
# from datasets import DatasetFactory
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import HTRDataset
import torchvision
import dataclasses
from typing import Any, Dict
import torch
from torch import nn
import mlflow
from mlflow.tracking.client import MlflowClient
import wandb
import numpy as np
import experiments
from autoaugment import CIFAR10Policy
from utils import metrics
import cv2
from sklearn.cluster import KMeans
from joblib import parallel_backend
from torch.ao.quantization import QuantStub, DeQuantStub


# WANDB_API_KEY = "e3dd99069f49577edaf1c80982c52f12fcdebb36"
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WANDB_ENTITY = "emascandela"

wandb.login(key=WANDB_API_KEY)

import itertools


class FixedHeightResize(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, image, **kwargs):
        if image.shape[0] != self.height:
            size = (self._calc_new_width(image), self.height)
            image = cv2.resize(image, size)
        return {"image": image, **kwargs}

    def _calc_new_width(self, img):
        old_height, old_width = img.shape[:2]
        aspect_ratio = old_width / old_height
        return round(self.height * aspect_ratio)


def collate_fn(batch_list):
    widths = torch.tensor([b[0][0].shape[2] for b in batch_list])
    width = torch.max(widths)

    batch_images = []
    batch_masks = []
    batch_labels = []
    batch_label_sizes = []

    for ((image, mask), (label, label_size)) in batch_list:
        pad = width - image.shape[2]
        batch_images.append(torch.nn.functional.pad(image, (0, pad)))
        batch_masks.append(torch.nn.functional.pad(mask, (0, pad), ))
        batch_labels.append(label)
        batch_label_sizes.append(label_size)

    batch_images = torch.stack(batch_images, 0)
    batch_masks = torch.stack(batch_masks, 0)
    batch_labels = torch.cat(batch_labels, 0)
    batch_label_sizes = torch.stack(batch_label_sizes, 0)

    return (batch_images, batch_masks), (batch_labels, batch_label_sizes)

def get_quantized_model(model, dataloader):
    model_fp32 = nn.Sequential(
        QuantStub(),
        model,
        DeQuantStub(),
    )

    model_fp32.cpu()
    model_fp32.eval()

    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    input_data = next(iter(dataloader))
    (input_data, _), _ = input_data

    model_fp32_prepared(input_data)

    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    return model_int8


def grid_search(dataset_names, base_params, share_params, train_params):
    grid = itertools.product(
        *(
            [dataset_names]
            + list(base_params.values())
            + list(share_params.values())
            + list(train_params.values())
        )
    )

    # dataset_names_keys = list(base_params.keys())
    base_params_keys = list(base_params.keys())
    share_params_keys = list(share_params.keys())
    train_params_keys = list(train_params.keys())


    for i, params in enumerate(grid):
        _dataset_name = params[0]
        _base_params = dict(zip(base_params_keys, params[1 : len(base_params_keys)+1]))
        _share_params = dict(
            zip(
                share_params_keys,
                params[
                    len(base_params_keys)+1 : len(base_params_keys)
                    + len(share_params_keys) + 1
                ],
            )
        )
        _train_params = dict(zip(train_params_keys, params[-len(train_params_keys) :]))

        yield _dataset_name, _base_params, _share_params, _train_params

def run_grid_search_experiment(base_params, share_params, train_params, model_name, dataset_names, experiment_name):
    for _dataset_name, _base_params, _share_params, _train_params in grid_search(
        dataset_names, base_params, share_params, train_params
    ):
        _experiment_name = f"{experiment_name} - {_dataset_name}"
        res = train_model(
            model_name=model_name,
            dataset_name=_dataset_name,
            base_params=_base_params,
            share_params=_share_params,
            train_params=_train_params,
            experiment_name=_experiment_name,
        )

        if res:
            return False

        # if model is not None:
        #     evaluate_model(
        #         model=model,
        #         model_name=model_name,
        #         dataset_name=_dataset_name,
        #         base_params=_base_params,
        #         share_params=_share_params,
        #         train_params=_train_params,
        #         experiment_name=_experiment_name,
        #     )
        #     return
    return True

import gc


def get_run_mlflow(experiment, md5: str):
    run = MlflowClient().search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string=f"tags.md5 = '{md5}'",
        max_results=1,
        # order_by=["metrics.accuracy DESC"],
    )
    if len(run) > 0:
        return run[0]
    return None

def get_run(experiment_name, md5: str):
    api = wandb.Api(api_key=WANDB_API_KEY)
    run = api.runs(path=f"{WANDB_ENTITY}/{experiment_name}", filters={"config.md5": md5})

    if len(run) > 0:
        return run[0]
    return None


def evaluate(model, dataloader, device):
    model.eval()
    cer = []
    model.to(device)

    for val_step, data in enumerate(dataloader):
        (inputs, mask), (flat_labels, label_lens) = data
        with torch.no_grad():
            # inputs = transform(inputs.to(device))
            inputs = inputs.to(device)

            outputs = model(inputs)

        mask = torch.squeeze(
            torchvision.transforms.functional.resize(torch.unsqueeze(mask, 0), (outputs.size()[0], outputs.size()[1]), interpolation=torchvision.transforms.InterpolationMode.NEAREST), 0)
        input_lengths = torch.tensor([mask.shape[1]] * mask.shape[0]) #mask.sum(1).to(torch.int32)

        # print("mask", mask.sum(1))

        # print("prev out")
        # print(outputs[:2])
        outputs = torch.argmax(outputs, 2)
        # print("mid out")

        # print(outputs[:2])
        outputs *= mask.to(torch.long).to(device)
        # print("post out")
        # print(outputs[:2])

        labels = []
        i = 0
        for lab_len in label_lens:
            labels.append(flat_labels[i:i+lab_len])
            i += lab_len


        # print(labels[:5])

        outputs = outputs.cpu()
        non_eps_mask = outputs[:, :-1] != outputs[:, 1:]
        non_eps_mask = torch.cat([torch.ones([outputs.shape[0], 1]), non_eps_mask], axis=1)
        outputs = outputs * non_eps_mask

        preds = []
        for out in outputs:
            preds.append(out[torch.where(out != 0)[0]])

        # correct.extend((outputs == labels.to(device)).cpu().numpy().tolist())
        for pred, label in zip(preds, labels):
            cer.append(metrics.cer(pred, label))

    cer = np.mean(cer)
    return cer


@dataclasses.dataclass
class TrainParams:
    batch_size: int = 16
    val_size: float = 0.2
    optimizer: str = "sgd"
    optimizer_params: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"lr": 1e-3}
    )
    epochs: int = 30
    min_epochs: int = 30
    scheduler: str = None
    scheduler_params: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"milestones": [15, 20]}
    )
    patience: int = 20
    augment: str = None
    lr_warmup: int = None
    fold: int = 0


def train_model(
    model_name, dataset_name, base_params, share_params, train_params, experiment_name
):
    params = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "base_params": base_params,
        "share_params": share_params,
        "train_params": train_params,
        "experiment_name": experiment_name,
    }
    if not ModelFactory.is_legal(model_name, base_params, share_params):
        print("Illegal config, skipping")
        return

    device = "cuda"
    from functools import partial

    train_transform = A.Compose(
        [
            FixedHeightResize(base_params['input_size']),
            A.augmentations.geometric.transforms.Affine(scale=(0.85, 1.15), p=0.5),
            A.augmentations.geometric.transforms.Affine(translate_percent=(-0.03, 0.03), p=0.5),
            A.augmentations.geometric.transforms.Affine(rotate=(-5, 5), p=0.5),
            A.augmentations.geometric.transforms.Affine(shear=(-5, 5), p=0.5),
            A.ToGray(),
            A.ChannelShuffle(p=1.0),
            # Rotate
            # VCrops
            # Shift

            A.Normalize(0.5, 0.5),
            ToTensorV2()
        ]
    )
    val_transform = A.Compose(
        [
            FixedHeightResize(base_params['input_size']),
            A.Normalize(0.5, 0.5),
            ToTensorV2()
        ]
    )

    train_params = TrainParams(**train_params)
    if train_params.fold == 0:
        train_split = [0, 1, 2, 3, 4]
        val_split = [5]
        test_split = [6, 7]
    elif train_params.fold == 1:
        train_split = [0, 1, 2, 6, 7]
        val_split = [3]
        test_split = [4, 5]
    elif train_params.fold == 2:
        train_split = [0, 4, 5, 6, 7]
        val_split = [1]
        test_split = [2, 3]
    elif train_params.fold == 3:
        train_split = [2, 3, 4, 5, 6]
        val_split = [7]
        test_split = [0, 1]

    # if dataset_name == "IAM_S":
    #     print("Running with subset of train split IAM")
    #     train_dataset = HTRDataset(f"data/{dataset_name}", splits=[2, 3], transform=train_transform)
    # if dataset_name == "IAM_M":
    #     print("Running with subset of train split IAM")
    #     train_dataset = HTRDataset(f"data/{dataset_name}", splits=[2, 3, 4, 5], transform=train_transform)

    # else:
    train_dataset = HTRDataset(f"data/{dataset_name}", splits=train_split, transform=train_transform)
    val_dataset = HTRDataset(f"data/{dataset_name}", splits=val_split, transform=val_transform)
    test_dataset = HTRDataset(f"data/{dataset_name}", splits=test_split, transform=val_transform)

    # num_characters = len(train_dataset.get_chars())
    # base_params = base_params.copy()
    # base_params["num_outputs"] = num_characters + 1


    model = ModelFactory.get_model(
        model_name,
        dataset_name=dataset_name,
        base_params=base_params,
        share_params=share_params,
        train_params=dataclasses.asdict(train_params),
    )
    base_model = model.get_base_model()

    # experiment = mlflow.set_experiment(experiment_name) #!!
    run = get_run(experiment_name, model.md5())

    if run is not None:
        print("Model already trained, skipping.")
        return False

    # with mlflow.start_run(run_id=run_id):
    #     mlflow.set_tag("md5", model.md5())
    #     mlflow.set_tag("base_md5", base_model.md5())
    #     mlflow.set_tag("eval", False)
    #     mlflow.set_tag("mlflow.runName", model.get_name())
    #     mlflow.log_params(model.as_dict(flatten=True))
    if True:
        print(f"Training model {model.get_name()} in experiment {experiment_name}")
        wandb.init(project=experiment_name, entity=WANDB_ENTITY, name=model.get_name())
        wandb.config.update(model.as_dict(flatten=True)  )
        wandb.config.update({"md5": model.md5()})
        wandb.run.summary["repr_params"] = params

        torchinfo.summary(model, (1, 3, base_params["input_size"], 300))


        # transform = torchvision.transforms.Compose(
        #     [
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225]),
        #     ]
        # )
        # train_transform = transform

        # if train_params.augment is not None:
        #     if train_params.augment == "resnet_simple":
        #         train_transform = torchvision.transforms.Compose([
        #             torchvision.transforms.RandomHorizontalFlip(),
        #             torchvision.transforms.RandomCrop(32, 4),
        #             train_transform
        #         ])
        #     elif train_params.augment == "autoaugment_cifar10":
        #         train_transform = torchvision.transforms.Compose([
        #             torchvision.transforms.RandomHorizontalFlip(),
        #             torchvision.transforms.RandomCrop(32, 4),
        #             CIFAR10Policy(),
        #             train_transform
        #         ])
        #     else:
        #         raise Exception(f"Unknown augmentation method {train_params.augment}")


        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_params.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=train_params.batch_size, shuffle=False, num_workers=4, 
            pin_memory=True,
            collate_fn=collate_fn,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train_params.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
        )

        loss_fn = torch.nn.CTCLoss(blank=0)

        if train_params.optimizer == "adam":
            optimizer_class = torch.optim.Adam
        elif train_params.optimizer == "sgd":
            optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(model.parameters(), **train_params.optimizer_params)

        if train_params.scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, **train_params.scheduler_params
            )
            if train_params.lr_warmup is not None:
                lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
                    torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=train_params.lr_warmup),
                    lr_scheduler,
                ], milestones=[train_params.lr_warmup])
        elif train_params.scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **train_params.scheduler_params
            )
            if train_params.lr_warmup is not None:
                lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=train_params.lr_warmup),
                    lr_scheduler,
                ], milestones=[train_params.lr_warmup])
            
        if model.share_params.share_cnn:
            if model.share_params.cnn_share_mode == "all":
                cnn_cluster_axis = []
            elif model.share_params.cnn_share_mode == "channel":
                cnn_cluster_axis = [0, 1]
            elif model.share_params.cnn_share_mode == "unit":
                cnn_cluster_axis = [0, 1, 2, 3]
            
            model.set_clusterable_param(nn.Conv2d, "weight", cnn_cluster_axis, avoid_params=[f"block{i+1}.0" for i in range(5)])

        if model.share_params.share_lstm:
            if model.share_params.lstm_share_mode == "all":
                lstm_cluster_axis = []
            elif model.share_params.lstm_share_mode == "channel":
                lstm_cluster_axis = [0]
            elif model.share_params.lstm_share_mode == "unit":
                lstm_cluster_axis = [0, 1]

            model.set_clusterable_param(nn.LSTM, "weight_hh_l0", lstm_cluster_axis)
            model.set_clusterable_param(nn.LSTM, "weight_hh_l0_reverse", lstm_cluster_axis)
            model.set_clusterable_param(nn.LSTM, "weight_ih_l0", lstm_cluster_axis)
            model.set_clusterable_param(nn.LSTM, "weight_ih_l0_reverse", lstm_cluster_axis)
            

        model.to(device)

        global_step = 0
        best_cer = np.inf
        last_best_epoch = 0
        losses = []
        for epoch in range(train_params.epochs):
            start_time = time.time()
            if (epoch - last_best_epoch) > train_params.patience and (epoch > (model.share_params.warmup_steps + train_params.min_epochs)) and (best_cer < 0.4):
                break

            # mlflow.log_metric("epoch", epoch, step=global_step)
            # mlflow.log_metric("lr", lr_scheduler.get_last_lr()[0], step=global_step)
            wandb.log({"epoch": epoch, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
            model.train()

            for i, data in enumerate(train_dataloader, 0):
                # print(f"It {i} {time.time() - start_time}", device)
                # inputs, labels = data
                (inputs, mask), (labels, label_lengths) = data

                optimizer.zero_grad()
                # inputs = resize(inputs.to(device))

                with torch.autocast(device_type="cuda"):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    # print(f"Inf {i} {time.time() - start_time}" )
                    # outputs = outputs.log_softmax(2)

                    with torch.no_grad():
                        mask = torch.squeeze(
                            torchvision.transforms.functional.resize(torch.unsqueeze(mask, 0), (outputs.size()[0], outputs.size()[1]), interpolation=torchvision.transforms.InterpolationMode.NEAREST), 0)
                        input_lengths = torch.tensor([mask.shape[1]] * mask.shape[0]) #mask.sum(1).to(torch.int32)
                    
                    # print(f"Resize {i} {time.time() - start_time}")

                    loss = loss_fn(outputs.permute(1, 0, 2).log_softmax(2), labels.to(device), input_lengths.to(device), label_lengths.to(device))
                    # print(f"Loss {i} {time.time() - start_time}")
                loss.backward()
                # print(f"Back {i} {time.time() - start_time}")
                optimizer.step()
                # print(f"Optim step {i} {time.time() - start_time}")

                # print statistics
                losses.append(loss.item())

                if (global_step + 1) % 50 == 0:  # print every 2000 mini-batches
                    print(
                        f"[{epoch + 1}, {global_step + 1:5d}] loss: {np.mean(losses):.3f}"
                    )

                    # mlflow.log_metric("loss", np.mean(losses), step=global_step)
                    wandb.log({"loss": np.mean(losses)}, step=global_step)
                    losses = []

                global_step += 1

            lr_scheduler.step()


            # print(f"Train time: {time.time() - start_time} s")
            val_start = time.time()
            val_cer = evaluate(model, val_dataloader, device)
            # print(f"Validation time: {time.time() - val_start} s")
            # mlflow.log_metric("val_cer", val_cer, step=global_step)
            wandb.log({"val_cer": val_cer}, step=global_step)
            ## print(f"Val cer: {val_cer:.4f}")

            if val_cer < best_cer:
                last_best_epoch = epoch
                best_cer = val_cer
                print("Decreased best cer")
                # model.save()
                model.save()
                cer = evaluate(model, test_dataloader, device)
                # mlflow.log_metric("cer", cer)
                # mlflow.log_metric("params", model.get_params())
                wandb.run.summary["cer"] = cer
                wandb.run.summary["params"] = model.get_params()

            if epoch >= model.share_params.warmup_steps and epoch < (model.share_params.warmup_steps + model.share_params.cluster_steps):
                # print("Clustering weights")
                model.make_clusterable()
                # clusterable_weights = model.get_clusterable_weights()
                # cluster_weights(clusterable_weights, model.share_params.cluster_ratio)
                clust_metrics = model.cluster(model.share_params.cluster_ratio)
                # mlflow.log_metric("cluster_metrics", clust_metrics)
                new_cer = evaluate(model, val_dataloader, device)
                print(f"After cluster cer: {new_cer:.4f}")
            elif epoch == (model.share_params.warmup_steps + model.share_params.cluster_steps):
                # print("Clustering weights and setting new params")
                params_b = model.get_params()
                model.make_clusterable()
                # clusterable_weights = model.get_clusterable_weights()
                # cluster_weights(clusterable_weights, model.share_params.cluster_ratio, replace_params=True)
                clust_metrics = model.cluster(model.share_params.cluster_ratio, replace_params=True)
                # Revisar si se necesita
                # with open("cluster_metrics.json", "w") as f:
                #     f.write(json.dumps(clust_metrics))
                #     mlflow.log_artifact("cluster_metrics.json", "cluster_metrics.json")
                params_a = model.get_params()
                optimizer.param_groups.clear()
                optimizer.add_param_group({"params": list(model.parameters())})
                print(f"Params before/after: {params_b} {params_a}")

            # print(f"Total epoch time: {time.time() - start_time} s")

        # model.save()
    wandb.finish()

    if model.share_params.precision == "FP":
        model = torch.load(model.get_model_path())
        model.share_params.precision = "Q8"

        wandb.init(project=experiment_name, entity=WANDB_ENTITY, name=model.get_name())
        wandb.config.update(model.as_dict(flatten=True))
        wandb.config.update({"md5": model.md5()})
        wandb.run.summary["repr_params"] = params


        model_q8 = get_quantized_model(model, train_dataloader)
        cer = evaluate(model_q8, test_dataloader, "cpu")
        wandb.run.summary["cer"] = cer
        # evaluate_q8(model, train_dataloader, test_dataloader)
    # del model
    # model.load()

    return True


def evaluate_model(
    model, model_name, dataset_name, base_params, share_params, train_params, experiment_name
):
    print(model_name, dataset_name, base_params, share_params, train_params, sep="\n")
    if not ModelFactory.is_legal(model_name, base_params, share_params):
        # print("Illegal config, skipping")
        return

    device = "cuda"


    train_params = TrainParams(**train_params)
    # model = ModelFactory.get_model(
    #     model_name,
    #     dataset_name=dataset_name,
    #     base_params=base_params,
    #     share_params=share_params,
    #     train_params=dataclasses.asdict(train_params),
    # )
    base_model = model.get_base_model()

    print(model.md5())

    experiment = mlflow.set_experiment(experiment_name)
    run = get_run(experiment, model.md5())
    base_run = get_run(experiment, base_model.md5())

    if run is None:
        print("Model not trained, skipping.")
        return
    # if run.data.tags["eval"] == "True":
    #     print("Model already evaluated, skipping.")
    #     return
    # with mlflow.start_run(run_id=run.info.run_id):
    #     mlflow.set_tag("mlflow.runName", model.get_name())
    #     return

    transform = A.Compose(
        [
            FixedHeightResize(base_params['input_size']),
            A.Normalize(0.5, 0.5),
            ToTensorV2()
        ]
    )
    test_dataset = HTRDataset(f"data/{dataset_name}", splits=[1], transform=transform)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_params.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    model.eval()

    with mlflow.start_run(run_id=run.info.run_id):
        # model.load()
        model.to(device)

        # mlflow.log_metric("params", model.get_params())

        cer = evaluate(model, test_dataloader, device)
        mlflow.log_metric("cer", cer)

        # if model.md5() != base_model.md5():
        #     mlflow.log_metric("params-baseline", base_model.get_params())
        #     mlflow.log_metric(
        #         "params-rel_diff", model.get_params() / base_model.get_params()
        #     )

        #     baseline_cer = base_run.data.metrics["cer"]
        #     mlflow.log_metric("cer-baseline", baseline_cer)
        #     mlflow.log_metric("cer-rel_diff", cer / baseline_cer)
        #     mlflow.log_metric("cer-abs_diff", cer - baseline_cer)

        # mlflow.set_tag("eval", True)


if __name__ == "__main__":
    # for units in [64, 256]:
    #     for num_layers in [4, 6, 8, 10]:
    #         for use_bn in [True, False]:
    #             for share_bn in [False, True]:
    #                 for weight_scale in [False, True]:
    #                     for share_layers in [False, True]:
    #                         base_params = {
    #                             "units": units,
    #                             "num_layers": num_layers,
    #                             "num_outputs": 10,
    #                             "input_size": (28, 28),
    #                             "input_channels": 1,
    #                             "use_batch_norm": use_bn,
    #                         }
    #                         share_params = {
    #                             "share_layers": share_layers,
    #                             "share_batch_norm": share_bn,
    #                             "weight_scale": weight_scale
    #                         }
    #                         train_params = {

    #                         }

    #                         # train_model(model_name="mlp", dataset_name="mnist", base_params=base_params, share_params=share_params, train_params=train_params, experiment_name="WeightSharing")
    #                         evaluate_model(model_name="mlp", dataset_name="mnist", base_params=base_params, share_params=share_params, train_params=train_params, experiment_name="WeightSharing")

    # run_experiment(*experiments.cfresnet())
    # run_experiment(*experiments.crnn())
    # run_experiment(*experiments.find_best_model())

    finished = run_grid_search_experiment(*experiments.baseline_reduction())
    if not finished:
        sys.exit(0)

    finished = run_grid_search_experiment(*experiments.wclus_reduction())
    if not finished:
        sys.exit(0)

    finished = run_grid_search_experiment(*experiments.baseline_recursion())
    if not finished:
        sys.exit(0)

    sys.exit(2)
