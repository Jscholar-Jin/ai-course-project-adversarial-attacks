# train_fgsm_adv.py
# 功能：
# 使用 FGSM 对抗训练方法训练防御后的模型
#
# 输入：
# 原始模型权重：./checkpoints/cnn.pt
#
# 输出：
# 1. checkpoints/cnn_fgsm_adv_train.pt
# 2. logs/fgsm_adv_training_log.csv
#
# 训练思想：
# 对每个 batch：
# 1. 先用当前模型生成 FGSM 对抗样本；
# 2. 再同时使用 clean image 和 adv image 训练；
# 3. 使模型既保持正常分类能力，又提升对抗鲁棒性。

import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from data_utils import build_cifar10_loaders
from models import load_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clamp_unit(x):
    return torch.clamp(x, 0.0, 1.0)


def generate_fgsm_adv(
    model,
    images,
    labels,
    eps=8 / 255,
):
    """
    生成 FGSM 对抗样本。

    公式：
    x_adv = x + eps * sign(grad_x loss)

    注意：
    这里用于对抗训练，所以对抗样本根据当前模型动态生成。
    """

    model.eval()

    images = images.clone().detach()
    labels = labels.clone().detach()

    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_sign = images.grad.detach().sign()

    adv_images = images + eps * grad_sign
    adv_images = clamp_unit(adv_images)

    return adv_images.detach()


def evaluate_clean(model, test_loader, device):
    """
    测试干净样本准确率。
    """

    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100.0

    return acc


def evaluate_fgsm(model, test_loader, device, eps=8 / 255):
    """
    测试 FGSM 对抗样本准确率。

    这里统计的是：
    模型在 FGSM 对抗样本上的分类准确率。
    """

    model.eval()

    total = 0
    correct = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        adv_images = generate_fgsm_adv(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
        )

        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(dim=1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100.0

    return acc


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    eps,
    adv_weight,
    clean_weight,
):
    """
    训练一轮 FGSM 对抗训练。

    loss = clean_weight * CE(clean) + adv_weight * CE(adv)
    """

    model.train()

    total = 0
    total_loss = 0.0

    clean_correct = 0
    adv_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. 先根据当前模型生成 FGSM 对抗样本
        adv_images = generate_fgsm_adv(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
        )

        # 2. 切回训练模式
        model.train()

        optimizer.zero_grad(set_to_none=True)

        clean_outputs = model(images)
        adv_outputs = model(adv_images)

        clean_loss = F.cross_entropy(clean_outputs, labels)
        adv_loss = F.cross_entropy(adv_outputs, labels)

        loss = clean_weight * clean_loss + adv_weight * adv_loss

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total += batch_size

        total_loss += loss.item() * batch_size

        clean_preds = clean_outputs.argmax(dim=1)
        adv_preds = adv_outputs.argmax(dim=1)

        clean_correct += clean_preds.eq(labels).sum().item()
        adv_correct += adv_preds.eq(labels).sum().item()

    avg_loss = total_loss / total
    clean_acc = clean_correct / total * 100.0
    adv_acc = adv_correct / total * 100.0

    return avg_loss, clean_acc, adv_acc


def main():
    parser = argparse.ArgumentParser(
        description="FGSM 对抗训练代码"
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument(
        "--clean-checkpoint",
        type=str,
        default="./checkpoints/cnn.pt",
        help="原始 clean 模型权重路径"
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="./checkpoints/cnn_fgsm_adv_train.pt",
        help="对抗训练后模型保存路径"
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="./logs/fgsm_adv_training_log.csv",
        help="训练日志保存路径"
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        help="FGSM 扰动大小，默认 8/255"
    )

    parser.add_argument(
        "--clean-weight",
        type=float,
        default=0.5,
        help="clean loss 权重"
    )

    parser.add_argument(
        "--adv-weight",
        type=float,
        default=0.5,
        help="adv loss 权重"
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    ensure_dir(os.path.dirname(args.save_path))
    ensure_dir(os.path.dirname(args.log_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("FGSM 对抗训练")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Clean checkpoint: {args.clean_checkpoint}")
    print(f"Save path: {args.save_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"eps = {args.eps * 255:.1f}/255")
    print(f"clean_weight = {args.clean_weight}")
    print(f"adv_weight = {args.adv_weight}")

    print("\n加载 CIFAR-10 数据集...")
    train_loader, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.download,
        test_samples=None,
    )
    print("数据集加载完成")

    print("\n加载原始 clean 模型...")
    model = load_model(
        args.model,
        args.clean_checkpoint,
        device
    )
    print("模型加载完成")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    logs = []

    best_score = -1.0
    best_epoch = -1

    print("\n开始 FGSM 对抗训练...")

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        train_loss, train_clean_acc, train_fgsm_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            eps=args.eps,
            clean_weight=args.clean_weight,
            adv_weight=args.adv_weight,
        )

        test_clean_acc = evaluate_clean(
            model=model,
            test_loader=test_loader,
            device=device,
        )

        test_fgsm_acc = evaluate_fgsm(
            model=model,
            test_loader=test_loader,
            device=device,
            eps=args.eps,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # 综合分数：
        # 同时考虑 clean acc 和 FGSM robust acc
        robust_score = 0.5 * test_clean_acc + 0.5 * test_fgsm_acc

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_clean_acc": train_clean_acc,
            "train_fgsm_acc": train_fgsm_acc,
            "test_clean_acc": test_clean_acc,
            "test_fgsm_acc": test_fgsm_acc,
            "robust_score": robust_score,
            "lr": current_lr,
            "eps_255": args.eps * 255,
            "clean_weight": args.clean_weight,
            "adv_weight": args.adv_weight,
        }

        logs.append(log_row)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Clean Acc: {train_clean_acc:.2f}%")
        print(f"Train FGSM Acc: {train_fgsm_acc:.2f}%")
        print(f"Test Clean Acc: {test_clean_acc:.2f}%")
        print(f"Test FGSM Acc: {test_fgsm_acc:.2f}%")
        print(f"Robust Score: {robust_score:.2f}")
        print(f"LR: {current_lr:.6f}")

        if robust_score > best_score:
            best_score = robust_score
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "test_clean_acc": test_clean_acc,
                    "test_fgsm_acc": test_fgsm_acc,
                    "eps": args.eps,
                    "model": args.model,
                },
                args.save_path,
            )

            print(f"保存当前最优模型：{args.save_path}")

        df = pd.DataFrame(logs)
        df.to_csv(args.log_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("FGSM 对抗训练完成")
    print("=" * 80)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Robust Score: {best_score:.2f}")
    print(f"模型已保存：{args.save_path}")
    print(f"日志已保存：{args.log_path}")


if __name__ == "__main__":
    main()