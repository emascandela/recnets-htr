def cfresnet():
    base_params = {
        "variant": ["r20", "r32", "r44"],
        "input_size": [64],
        "lstm_layers": [1, 2],
        "lstm_hidden_size": [64, 128, 256],
        "first_layer_stride": [2],
        "num_outputs": [163],
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": [1, 2, 4, 6],
        "share_rnn_stride": [1, 2],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False, True],
    }
    train_params = {
        "batch_size": [32],
        "val_size": [0.1],
        "optimizer": ["adam"],
        "epochs": [20],
        "optimizer_params": [
            {
                "lr": 0.001,
            }
        ],
        "scheduler_params": [{"milestones": [5, 10, 18]}],
        "augment": ["resnet_simple"],
    }

    model_name = "cf_resnet"
    dataset_names = ["IAM", "PARZIVAL", "SAINT_GALL", "WASHINGTON"]
    experiment_name = "CFResNet"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )


def crnn():
    base_params = {
        "input_size": [64],
        "num_filters": [32, 64, 128],
        "block_size": [1, 2, 3],
        "lstm_layers": [1, 2],
        "lstm_hidden_size": [64, 128, 256],
        "num_outputs": [163],
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": [1, 2, 3],
        "share_rnn_stride": [1, 2],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False, True],
    }
    train_params = {
        "batch_size": [32],
        "val_size": [0.1],
        "optimizer": ["adam"],
        "epochs": [20],
        "optimizer_params": [
            {
                "lr": 0.001,
            }
        ],
        "scheduler_params": [{"milestones": [5, 10, 18]}],
        "augment": ["resnet_simple"],
    }

    model_name = "crnn"
    dataset_names = ["IAM", "PARZIVAL", "SAINT_GALL", "WASHINGTON"]
    experiment_name = "CRNN"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )


def baseline():
    base_params = {
        "input_size": [64],
        "num_filters": [16],
        "block_size": reversed([1, 2, 3]),
        "lstm_layers": reversed([2, 3]),
        "lstm_hidden_size": [64, 256],
        "num_outputs": [163],
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": [1, 2, 3],
        "share_rnn_stride": [1, 2, 3],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False, True],
    }
    train_params = {
        # "batch_size": [32],
        "batch_size": [64],
        "optimizer": ["adam"],
        "epochs": [1000],
        "min_epochs": [80],
        "patience": [20],
        "optimizer_params": [
            {
                "lr": 0.0003,
            }
        ],
        "scheduler_params": [{"milestones": [1000000]}],
        "augment": ["resnet_simple"],
        "model_id": [0],
    }

    model_name = "crnn"
    # dataset_names = ["IAM", "WASHINGTON", "PARZIVAL", "SAINT_GALL"]
    dataset_names = ["IAM_S"]
    experiment_name = "CRNN"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )
