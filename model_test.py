from models.cf_resnet import CFResNet

import torchinfo
import torch
import numpy as np
import torchvision.transforms.functional as F
import datasets

class FixedHeightResize(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        size = (self.height, self._calc_new_width(img))
        return F.resize(img, size)

    def _calc_new_width(self, img):
        old_width, old_height = img.size
        aspect_ratio = old_width / old_height
        return round(self.height * aspect_ratio)

if __name__ == "__main__":
    base_params = {
        "input_size": 64,
        "num_outputs": 32,
        "lstm_hidden_size": 256,
        "lstm_layers": 2,
        "variant": "r56",
    }

    cnn = CFResNet(
        dataset_name="foobar",
        base_params=base_params,
        share_params={"share_layer_stride": 1, "share_layer_batch_norm": False},
        train_params={}
    )

    print(cnn.get_base_model().get_params())
    print(cnn.get_params())

    torchinfo.summary(cnn, (1, 3, 64, 1024))
    quit()

    import torchvision

    transform = torchvision.transforms.Compose(
        [
            FixedHeightResize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.HTRDataset("data/*/", splits=list(range(10)), transform=transform)
    print(len(dataset.get_chars()))
    (image, mask), (label, label_sizes) = dataset[0]
    print(image.shape)
    print(mask.shape)
    print(label.shape)

    from torch.utils.data import DataLoader

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
            batch_masks.append(torch.nn.functional.pad(mask, (0, pad)))
            batch_labels.append(label)
            batch_label_sizes.append(label_size)

        batch_images = torch.stack(batch_images, 0)
        batch_masks = torch.stack(batch_masks, 0)
        batch_labels = torch.cat(batch_labels, 0)
        batch_label_sizes = torch.stack(batch_label_sizes, 0)

        return (batch_images, batch_masks), (batch_labels, batch_label_sizes)

    dl = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    (im, msk), (lb, lleb) = next(iter(dl))

    print(im.shape)
    print(msk.shape)
    print(lb.shape)
    print(lleb.shape)
    print(torch.sum(lleb))


    quit()


    class A():
        _last_v = None

        def __init__(self, use_last_v = False):
            if use_last_v:
                self.v = self._last_v
            else:
                self.v = (A._last_v + 1) if self._last_v else 1
                A._last_v = self.v
            print(self.v)
    
    A()
    A()
    A(use_last_v=True)
    A(use_last_v=True)
    A()
    A(use_last_v=True)
    A(use_last_v=True)
