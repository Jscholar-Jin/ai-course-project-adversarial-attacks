# train_pgd_adv_final.py
# 功能：
# PGD 对抗训练最终版
#
# 目标：
# 训练一个比 FGSM 对抗训练更强的鲁棒模型，用于抵抗 PGD / FGSM / DeepFool / 迁移攻击。
#
# 输出：
# 1. checkpoints/cnn_pgd_adv_final.pt
# 2. logs/pgd_adv_training_log.csv
#
# 默认设置：
# eps = 8/255
# alpha = 2/255
# PGD train steps = 7
# PGD eval steps = 10
# epochs = 20

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
    if path:
        os.makedirs(path, exist_ok=True)


def clamp_unit(x):
    return torch.clamp(x, 0.0, 1.0)


def pgd_attack_for_training(
    model,
    images,
    labels,
    eps=8 / 255,
    alpha=2 / 255,
    steps=7,
    random_start=True,
):
    """
    PGD 对抗样本生成。

    用于对抗训练阶段。

    公式：
    x_{t+1} = Proj_{B_eps(x)}(x_t + alpha * sign(grad_x loss))

    注意：
    这里只生成对抗样本，不更新模型参数。
    """

    model.eval()

    ori_images = images.detach()

    if random_start:
        adv_images = ori_images + torch.empty_like(ori_images).uniform_(-eps, eps)
        adv_images = clamp_unit(adv_images)
    else:
        adv_images = ori_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True

        logits = model(adv_images)
        loss = F.cross_entropy(logits, labels)

        model.zero_grad(set_to_none=True)

        if adv_images.grad is not None:
            adv_images.grad.zero_()

        loss.backward()

        grad = adv_images.grad.detach()

        adv_images = adv_images.detach() + alpha * grad.sign()

        delta = torch.clamp(
            adv_images - ori_images,
            min=-eps,
            max=eps
        )

        adv_images = clamp_unit(ori_images + delta).detach()

    return adv_images


def evaluate_clean(model, test_loader, device, max_samples=None):
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

            batch_size = images.size(0)

            if max_samples is not None and total >= max_samples:
                break

            if max_samples is not None and total + batch_size > max_samples:
                remain = max_samples - total
                images = images[:remain]
                labels = labels[:remain]
                batch_size = images.size(0)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += preds.eq(labels).sum().item()
            total += batch_size

    return correct / total * 100.0


def evaluate_pgd(
    model,
    test_loader,
    device,
    eps=8 / 255,
    alpha=2 / 255,
    steps=10,
    max_samples=None,
):
    """
    测试 PGD 对抗准确率。
    """

    model.eval()

    total = 0
    clean_correct = 0
    adv_correct = 0
    clean_correct_for_asr = 0
    attack_success = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        if max_samples is not None and total >= max_samples:
            break

        if max_samples is not None and total + batch_size > max_samples:
            remain = max_samples - total
            images = images[:remain]
            labels = labels[:remain]
            batch_size = images.size(0)

        with torch.no_grad():
            clean_logits = model(images)
            clean_pred = clean_logits.argmax(dim=1)

        clean_mask = clean_pred.eq(labels)

        clean_correct += clean_mask.sum().item()
        clean_correct_for_asr += clean_mask.sum().item()

        adv_images = pgd_attack_for_training(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=True,
        )

        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_pred = adv_logits.argmax(dim=1)

        adv_mask = adv_pred.eq(labels)

        adv_correct += adv_mask.sum().item()

        success_mask = clean_mask & adv_pred.ne(labels)
        attack_success += success_mask.sum().item()

        total += batch_size

    clean_acc = clean_correct / total * 100.0
    adv_acc = adv_correct / total * 100.0

    if clean_correct_for_asr == 0:
        asr = 0.0
    else:
        asr = attack_success / clean_correct_for_asr * 100.0

    return clean_acc, adv_acc, asr


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    eps,
    alpha,
    pgd_steps,
    clean_weight,
    adv_weight,
):
    """
    PGD 对抗训练一轮。

    loss = clean_weight * CE(clean) + adv_weight * CE(adv)
    """

    model.train()

    total = 0
    total_loss = 0.0

    clean_correct = 0
    adv_correct = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 1. 生成 PGD 对抗样本
        adv_images = pgd_attack_for_training(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=pgd_steps,
            random_start=True,
        )

        # 2. 用 clean + adv 一起训练
        model.train()

        optimizer.zero_grad(set_to_none=True)

        clean_logits = model(images)
        adv_logits = model(adv_images)

        clean_loss = F.cross_entropy(clean_logits, labels)
        adv_loss = F.cross_entropy(adv_logits, labels)

        loss = clean_weight * clean_loss + adv_weight * adv_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total += batch_size

        total_loss += loss.item() * batch_size

        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)

        clean_correct += clean_pred.eq(labels).sum().item()
        adv_correct += adv_pred.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / total
    train_clean_acc = clean_correct / total * 100.0
    train_pgd_acc = adv_correct / total * 100.0

    return avg_loss, train_clean_acc, train_pgd_acc


def save_checkpoint(
    model,
    save_path,
    epoch,
    best_score,
    test_clean_acc,
    test_pgd_acc,
    test_pgd_asr,
    args,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "test_clean_acc": test_clean_acc,
            "test_pgd_acc": test_pgd_acc,
            "test_pgd_asr": test_pgd_asr,
            "model": args.model,
            "eps": args.eps,
            "alpha": args.alpha,
            "train_pgd_steps": args.train_pgd_steps,
            "eval_pgd_steps": args.eval_pgd_steps,
        },
        save_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="PGD 对抗训练最终版"
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--model", type=str, default="cnn")

    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="./checkpoints/cnn.pt",
        help="原始 clean 模型权重路径"
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="./checkpoints/cnn_pgd_adv_final.pt",
        help="PGD 对抗训练后模型保存路径"
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="./logs/pgd_adv_training_log.csv",
        help="训练日志保存路径"
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        help="扰动上限，默认 8/255"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 255,
        help="PGD 每步步长，默认 2/255"
    )

    parser.add_argument(
        "--train-pgd-steps",
        type=int,
        default=7,
        help="训练阶段 PGD 迭代次数"
    )

    parser.add_argument(
        "--eval-pgd-steps",
        type=int,
        default=10,
        help="评估阶段 PGD 迭代次数"
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

    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=2000,
        help="每轮评估最多使用多少测试样本。设为 None 会完整测试集，但更慢"
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    ensure_dir(os.path.dirname(args.save_path))
    ensure_dir(os.path.dirname(args.log_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PGD 对抗训练最终版")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Init checkpoint: {args.init_checkpoint}")
    print(f"Save path: {args.save_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"eps = {args.eps * 255:.1f}/255")
    print(f"alpha = {args.alpha * 255:.1f}/255")
    print(f"train_pgd_steps = {args.train_pgd_steps}")
    print(f"eval_pgd_steps = {args.eval_pgd_steps}")
    print(f"clean_weight = {args.clean_weight}")
    print(f"adv_weight = {args.adv_weight}")
    print(f"eval_max_samples = {args.eval_max_samples}")

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
        args.init_checkpoint,
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

    print("\n开始 PGD 对抗训练...")

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        train_loss, train_clean_acc, train_pgd_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            eps=args.eps,
            alpha=args.alpha,
            pgd_steps=args.train_pgd_steps,
            clean_weight=args.clean_weight,
            adv_weight=args.adv_weight,
        )

        test_clean_acc = evaluate_clean(
            model=model,
            test_loader=test_loader,
            device=device,
            max_samples=args.eval_max_samples,
        )

        _, test_pgd_acc, test_pgd_asr = evaluate_pgd(
            model=model,
            test_loader=test_loader,
            device=device,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.eval_pgd_steps,
            max_samples=args.eval_max_samples,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # 最终模型选择：
        # 更看重 PGD 鲁棒性，同时保留一定 clean accuracy
        robust_score = 0.4 * test_clean_acc + 0.6 * test_pgd_acc

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_clean_acc": train_clean_acc,
            "train_pgd_acc": train_pgd_acc,
            "test_clean_acc": test_clean_acc,
            "test_pgd_acc": test_pgd_acc,
            "test_pgd_asr": test_pgd_asr,
            "robust_score": robust_score,
            "lr": current_lr,
            "eps_255": args.eps * 255,
            "alpha_255": args.alpha * 255,
            "train_pgd_steps": args.train_pgd_steps,
            "eval_pgd_steps": args.eval_pgd_steps,
        }

        logs.append(log_row)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Clean Acc: {train_clean_acc:.2f}%")
        print(f"Train PGD Acc: {train_pgd_acc:.2f}%")
        print(f"Test Clean Acc: {test_clean_acc:.2f}%")
        print(f"Test PGD Acc: {test_pgd_acc:.2f}%")
        print(f"Test PGD ASR: {test_pgd_asr:.2f}%")
        print(f"Robust Score: {robust_score:.2f}")
        print(f"LR: {current_lr:.6f}")

        df = pd.DataFrame(logs)
        df.to_csv(args.log_path, index=False, encoding="utf-8-sig")

        if robust_score > best_score:
            best_score = robust_score
            best_epoch = epoch

            save_checkpoint(
                model=model,
                save_path=args.save_path,
                epoch=epoch,
                best_score=best_score,
                test_clean_acc=test_clean_acc,
                test_pgd_acc=test_pgd_acc,
                test_pgd_asr=test_pgd_asr,
                args=args,
            )

            print(f"保存当前最优模型：{args.save_path}")

    print("\n" + "=" * 80)
    print("PGD 对抗训练完成")
    print("=" * 80)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Robust Score: {best_score:.2f}")
    print(f"模型已保存：{args.save_path}")
    print(f"日志已保存：{args.log_path}")


if __name__ == "__main__":
    main()