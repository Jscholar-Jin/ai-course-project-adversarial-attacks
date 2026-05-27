# train_baseline.py

import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import build_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(data_dir, batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{total_loss / total:.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    avg_loss = total_loss / total
    acc = 100.0 * correct / total

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Eval", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total

    return avg_loss, acc


def train_model(
    model_tag,
    model_name,
    train_loader,
    test_loader,
    device,
    epochs,
    lr,
    weight_decay,
    save_dir,
    log_dir
):
    print("=" * 80)
    print(f"开始训练 {model_tag}: {model_name}")
    print("=" * 80)

    model = build_model(model_name).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_acc = 0.0
    history = []

    best_path = os.path.join(save_dir, f"{model_tag}_{model_name}_best.pth")
    last_path = os.path.join(save_dir, f"{model_tag}_{model_name}_last.pth")

    for epoch in range(1, epochs + 1):
        print(f"\n[{model_tag}-{model_name}] Epoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": current_lr
        })

        if test_acc > best_acc:
            best_acc = test_acc

            torch.save({
                "model_tag": model_tag,
                "model_name": model_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_acc": test_acc,
            }, best_path)

            print(f"保存最佳模型: {best_path}")
            print(f"当前最佳测试准确率: {best_acc:.2f}%")

        torch.save({
            "model_tag": model_tag,
            "model_name": model_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "test_acc": test_acc,
        }, last_path)

    log_path = os.path.join(log_dir, f"{model_tag}_{model_name}_train_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False, encoding="utf-8-sig")

    print("\n训练完成")
    print(f"模型: {model_tag}-{model_name}")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print(f"最佳模型路径: {best_path}")
    print(f"训练日志路径: {log_path}")

    return best_acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--target",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="A=ResNet18 源模型, B=SimpleCNN 目标模型, both=两个都训练"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("CIFAR-10 对抗攻击大作业：基础模型训练")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    results = {}

    if args.target in ["A", "both"]:
        acc_A = train_model(
            model_tag="model_A",
            model_name="resnet18",
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
        results["model_A_resnet18"] = acc_A

    if args.target in ["B", "both"]:
        acc_B = train_model(
            model_tag="model_B",
            model_name="simplecnn",
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
        results["model_B_simplecnn"] = acc_B

    print("\n" + "=" * 80)
    print("基础模型训练结果汇总")
    print("=" * 80)

    for name, acc in results.items():
        print(f"{name}: Best Test Acc = {acc:.2f}%")

    print("\n生成文件：")
    print("checkpoints/model_A_resnet18_best.pth")
    print("checkpoints/model_B_simplecnn_best.pth")
    print("logs/model_A_resnet18_train_log.csv")
    print("logs/model_B_simplecnn_train_log.csv")


if __name__ == "__main__":
    main()