import torchvision
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from typing import Type, List, Union, Optional
import os
import glob

class DatasetFactory():
    datasets = {}

    @classmethod
    def register(cls, name: str, dataset_class: Type):
        if name in cls.datasets:
            raise Exception(f"{name} is already registered")
        
        cls.datasets[name] = dataset_class
    
    @classmethod
    def get_dataset(cls, dataset_name: str, **kwargs):
        if dataset_name not in cls.datasets.keys():
            raise Exception(f"Unknown dataset {dataset_name}")
        
        dataset_class = cls.datasets[dataset_name]
        return dataset_class(**kwargs)


class HTRDataset(Dataset):
    def __init__(self, path: str, splits: List[int], transform=None):
        self.path = path
        self.gt_paths = glob.glob(os.path.join(path, "all_gt", f"[{','.join(map(str, splits))}]", "*.txt"))
        png_image_paths = [p.replace( "all_gt", "all_images").replace(".txt", ".png") for p in self.gt_paths]
        jpg_image_paths = [p.replace( "all_gt", "all_images").replace(".txt", ".jpg") for p in self.gt_paths]

        if os.path.isfile(jpg_image_paths[0]):
            self.image_paths = jpg_image_paths
        else:
            self.image_paths = png_image_paths

        print("Loading images")
        self.images = [Image.open(image_path).convert("RGB") for image_path in self.image_paths]
        print("Images loaded")

        self.transform = transform

        self.chars = self.get_chars(splits=list(range(10)))
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        gt_path = self.gt_paths[idx]
        # image_path = self.image_paths[idx]
        # image = Image.open(image_path).convert("RGB")
        image = self.images[idx]

        with open(gt_path, "r") as f:
            label = f.read().splitlines()
        label = self.encode(label)

        image = self.transform(image=np.asarray(image))["image"]
        mask = torch.ones(image.size()[2])
        mask.to(torch.int32)

        return (image, mask), (torch.tensor(label), torch.tensor(len(label)))
    
    def encode(self, label):
        return [self.chars.index(l) + 1 for l in label]

    def decode(self, label):
        return [self.chars[l - 1] for l in label if l != 0]

    def get_chars(self, splits: Optional[List[Union[int, str]]]):
        gt_paths = glob.glob(os.path.join(self.path, "all_gt", f"[{','.join(map(str, splits))}]", "*.txt"))

        characters = set()
        for gt_path in gt_paths:
            with open(gt_path, "r") as f:
                label = f.read().splitlines()
                characters.update(label)
        return sorted(list(characters))


if __name__ == "__main__":
    dataset = HTRDataset("data/IAM/", splits=[0, 1, 2])
    image, label = dataset[0]
    print(image.shape)
    print(label)
    print(np.unique(label))

    chars = get_dataset_chars("data/*/")
    print(chars)
    print(len(chars))
# DatasetFactory.register("mnist", torchvision.datasets.MNIST)
# DatasetFactory.register("cifar10", torchvision.datasets.CIFAR10)
# DatasetFactory.register("cifar100", torchvision.datasets.CIFAR100)