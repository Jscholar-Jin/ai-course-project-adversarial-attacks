# train_pgd_defense.py
# 功能：训练 PGD 对抗训练防御模型 Model B-PGD-AT

import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import build_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clamp(x, min_value=0.0, max_value=1.0):
    return torch.clamp(x, min_value, max_value)


def pgd_for_training(
    model,
    images,
    labels,
    eps=8 / 255,
    alpha=2 / 255,
    steps=5,
    random_start=True
):
    """
    用当前模型生成 PGD 对抗样本。
    """
    was_training = model.training
    model.eval()

    ori_images = images.clone().detach()
    labels = labels.clone().detach()

    if random_start:
        adv_images = ori_images + torch.empty_like(ori_images).uniform_(-eps, eps)
        adv_images = clamp(adv_images, 0.0, 1.0)
    else:
        adv_images = ori_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_sign = adv_images.grad.sign()

        adv_images = adv_images + alpha * grad_sign

        delta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = clamp(ori_images + delta, 0.0, 1.0).detach()

    model.zero_grad(set_to_none=True)

    if was_training:
        model.train()

    return adv_images


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


def load_checkpoint_if_needed(model, checkpoint_path, device):
    if checkpoint_path is None:
        print("未加载预训练模型，将从随机初始化开始训练。")
        return model

    if not os.path.exists(checkpoint_path):
        print(f"未找到预训练模型: {checkpoint_path}")
        print("将从随机初始化开始训练。")
        return model

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"已加载预训练模型: {checkpoint_path}")

    if "test_acc" in checkpoint:
        print(f"原模型测试准确率: {checkpoint['test_acc']:.2f}%")

    return model


def train_one_epoch_pgd_at(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    eps,
    alpha,
    pgd_steps,
    clean_weight,
    adv_weight
):
    model.train()

    total_loss = 0.0
    total_clean_loss = 0.0
    total_adv_loss = 0.0

    clean_correct = 0
    adv_correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="PGD Adv Train", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        adv_images = pgd_for_training(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=pgd_steps,
            random_start=True
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        clean_outputs = model(images)
        adv_outputs = model(adv_images)

        clean_loss = criterion(clean_outputs, labels)
        adv_loss = criterion(adv_outputs, labels)

        loss = clean_weight * clean_loss + adv_weight * adv_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_clean_loss += clean_loss.item() * batch_size
        total_adv_loss += adv_loss.item() * batch_size

        clean_preds = clean_outputs.argmax(dim=1)
        adv_preds = adv_outputs.argmax(dim=1)

        clean_correct += clean_preds.eq(labels).sum().item()
        adv_correct += adv_preds.eq(labels).sum().item()
        total += batch_size

        pbar.set_postfix({
            "loss": f"{total_loss / total:.4f}",
            "clean": f"{100.0 * clean_correct / total:.2f}%",
            "pgd": f"{100.0 * adv_correct / total:.2f}%"
        })

    return {
        "train_loss": total_loss / total,
        "train_clean_loss": total_clean_loss / total,
        "train_adv_loss": total_adv_loss / total,
        "train_clean_acc": 100.0 * clean_correct / total,
        "train_pgd_acc": 100.0 * adv_correct / total,
    }


def evaluate_clean_and_pgd(
    model,
    test_loader,
    criterion,
    device,
    eps,
    alpha,
    pgd_steps
):
    model.eval()

    total_clean_loss = 0.0
    total_adv_loss = 0.0

    clean_correct = 0
    adv_correct = 0
    total = 0

    pbar = tqdm(test_loader, desc="Eval Clean+PGD", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        with torch.no_grad():
            clean_outputs = model(images)
            clean_loss = criterion(clean_outputs, labels)
            clean_preds = clean_outputs.argmax(dim=1)

        adv_images = pgd_for_training(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=pgd_steps,
            random_start=True
        )

        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_loss = criterion(adv_outputs, labels)
            adv_preds = adv_outputs.argmax(dim=1)

        total_clean_loss += clean_loss.item() * batch_size
        total_adv_loss += adv_loss.item() * batch_size

        clean_correct += clean_preds.eq(labels).sum().item()
        adv_correct += adv_preds.eq(labels).sum().item()
        total += batch_size

        pbar.set_postfix({
            "clean": f"{100.0 * clean_correct / total:.2f}%",
            "pgd": f"{100.0 * adv_correct / total:.2f}%"
        })

    return {
        "test_clean_loss": total_clean_loss / total,
        "test_pgd_loss": total_adv_loss / total,
        "test_clean_acc": 100.0 * clean_correct / total,
        "test_pgd_acc": 100.0 * adv_correct / total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="./checkpoints/model_B_simplecnn_best.pth"
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd_steps", type=int, default=5)

    parser.add_argument("--clean_weight", type=float, default=0.3)
    parser.add_argument("--adv_weight", type=float, default=0.7)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("PGD 对抗训练防御：Model B-PGD-AT")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Init checkpoint: {args.init_checkpoint}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"eps: {args.eps * 255:.1f}/255")
    print(f"alpha: {args.alpha * 255:.1f}/255")
    print(f"PGD steps: {args.pgd_steps}")
    print(f"Clean weight: {args.clean_weight}")
    print(f"Adv weight: {args.adv_weight}")

    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = build_model("simplecnn").to(device)

    model = load_checkpoint_if_needed(
        model=model,
        checkpoint_path=args.init_checkpoint,
        device=device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_score = -1.0

    best_path = os.path.join(args.save_dir, "model_B_simplecnn_pgd_at_best.pth")
    last_path = os.path.join(args.save_dir, "model_B_simplecnn_pgd_at_last.pth")

    history = []

    for epoch in range(1, args.epochs + 1):
        print("\n" + "-" * 100)
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 100)

        train_metrics = train_one_epoch_pgd_at(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            eps=args.eps,
            alpha=args.alpha,
            pgd_steps=args.pgd_steps,
            clean_weight=args.clean_weight,
            adv_weight=args.adv_weight
        )

        test_metrics = evaluate_clean_and_pgd(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            eps=args.eps,
            alpha=args.alpha,
            pgd_steps=args.pgd_steps
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        robust_score = 0.5 * test_metrics["test_clean_acc"] + 0.5 * test_metrics["test_pgd_acc"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            **train_metrics,
            **test_metrics,
            "robust_score": robust_score
        }

        history.append(row)

        print(
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Train Clean Acc: {train_metrics['train_clean_acc']:.2f}% | "
            f"Train PGD Acc: {train_metrics['train_pgd_acc']:.2f}%"
        )

        print(
            f"Test Clean Acc: {test_metrics['test_clean_acc']:.2f}% | "
            f"Test PGD Acc: {test_metrics['test_pgd_acc']:.2f}% | "
            f"Robust Score: {robust_score:.2f} | "
            f"LR: {current_lr:.6f}"
        )

        checkpoint = {
            "model_name": "simplecnn",
            "defense": "pgd_adversarial_training",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "test_clean_acc": test_metrics["test_clean_acc"],
            "test_pgd_acc": test_metrics["test_pgd_acc"],
            "robust_score": robust_score,
            "eps": args.eps,
            "alpha": args.alpha,
            "pgd_steps": args.pgd_steps,
        }

        torch.save(checkpoint, last_path)

        if robust_score > best_score:
            best_score = robust_score
            torch.save(checkpoint, best_path)
            print(f"保存最佳 PGD 对抗训练模型: {best_path}")
            print(f"当前最佳 Robust Score: {best_score:.2f}")

    log_path = os.path.join(args.log_dir, "model_B_simplecnn_pgd_at_train_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("PGD 对抗训练完成")
    print("=" * 100)
    print(f"最佳模型路径: {best_path}")
    print(f"最后模型路径: {last_path}")
    print(f"训练日志路径: {log_path}")


if __name__ == "__main__":
    main()