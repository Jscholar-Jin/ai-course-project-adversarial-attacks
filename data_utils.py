import random
from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_limit_dataset(dataset, max_samples: Optional[int]):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, range(max_samples))


def build_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    workers: int,
    download: bool,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_tf = transforms.ToTensor()

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_tf,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_tf,
    )

    train_set = maybe_limit_dataset(train_set, train_samples)
    test_set = maybe_limit_dataset(test_set, test_samples)

    pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
