import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models import get_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def data_loaders(data_dir, batch_size, workers, download):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
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


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        size = labels.size(0)
        loss_sum += loss.item() * size
        correct += logits.argmax(1).eq(labels).sum().item()
        total += size

    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def test(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="test", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        size = labels.size(0)
        loss_sum += loss.item() * size
        correct += logits.argmax(1).eq(labels).sum().item()
        total += size

    return loss_sum / total, 100.0 * correct / total


def fit(name, train_loader, test_loader, args, device):
    model = get_model(name).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = []
    ckpt_path = os.path.join(args.checkpoint_dir, f"{name}.pt")
    log_path = os.path.join(args.log_dir, f"{name}.csv")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        test_loss, test_acc = test(model, test_loader, loss_fn, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"{name} epoch {epoch}/{args.epochs} "
            f"train={train_acc:.2f}% test={test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "name": name,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "acc": test_acc,
                },
                ckpt_path,
            )

    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"{name} best={best_acc:.2f}% checkpoint={ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--model", default="both", choices=["resnet", "cnn", "both"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = data_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.download,
    )

    names = ["resnet", "cnn"] if args.model == "both" else [args.model]
    for name in names:
        fit(name, train_loader, test_loader, args, device)


if __name__ == "__main__":
    main()
