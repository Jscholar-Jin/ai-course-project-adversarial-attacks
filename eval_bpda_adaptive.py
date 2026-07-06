# eval_bpda_adaptive.py
# 功能：
# 使用 PGD + BPDA 自适应攻击评估输入预处理防御
#
# 防御方法：
# 1. JPEG 压缩
# 2. 位深压缩 squeeze
#
# 输出：
# 1. results/bpda_adaptive_results.csv
# 2. figures/bpda_adaptive_adv_acc_bar.png
# 3. figures/bpda_adaptive_asr_bar.png
#
# 说明：
# 普通预处理防御可能因为不可导操作造成“梯度遮蔽”。
# BPDA 的作用是：
# 前向传播使用真实预处理；
# 反向传播近似为恒等映射；
# 从而更公平地评估预处理防御是否真的有效。

import os
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from data_utils import build_cifar10_loaders
from models import load_model

from defenses import (
    apply_preprocess,
    logits_with_preprocess,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clamp_unit(x):
    return torch.clamp(x, 0.0, 1.0)


def parse_defense_list(defense_list_str):
    """
    输入：
    --defenses jpeg,squeeze

    返回：
    ["jpeg", "squeeze"]
    """
    defenses = []

    for item in defense_list_str.split(","):
        item = item.strip().lower()
        if item:
            defenses.append(item)

    return defenses


def get_defense_kwargs(defense, args):
    """
    不同防御方法对应不同参数。
    """
    if defense == "jpeg":
        return {
            "jpeg_quality": args.jpeg_quality
        }

    if defense == "squeeze":
        return {
            "bit_depth": args.bit_depth
        }

    if defense == "none":
        return {}

    raise ValueError(f"未知防御方法: {defense}")


def defense_display_name(defense):
    if defense == "jpeg":
        return "JPEG Compression"

    if defense == "squeeze":
        return "Feature Squeezing"

    if defense == "none":
        return "None"

    return defense


def pgd_bpda_attack(
    model,
    images,
    labels,
    defense,
    eps=8 / 255,
    alpha=2 / 255,
    steps=10,
    random_start=True,
    **defense_kwargs,
):
    """
    PGD + BPDA 自适应攻击。

    目标：
    攻击带输入预处理防御的模型。

    普通前向：
        x_processed = preprocess(x)
        logits = model(x_processed)

    BPDA 前向/反向：
        前向仍然使用 preprocess(x)
        反向近似使用 identity gradient

    在 defense.py 中对应：
        logits_with_preprocess(model, x, defense, bpda=True)

    攻击公式：
        x_adv = Proj_{B_eps(x)}(
                    x_adv + alpha * sign(grad_x loss)
                )
    """

    model.eval()

    ori_images = images.clone().detach()
    labels = labels.clone().detach()

    if random_start:
        adv_images = ori_images + torch.empty_like(ori_images).uniform_(-eps, eps)
        adv_images = clamp_unit(adv_images)
    else:
        adv_images = ori_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True

        logits = logits_with_preprocess(
            model=model,
            x=adv_images,
            defense=defense,
            bpda=True,
            **defense_kwargs,
        )

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
            max=eps,
        )

        adv_images = clamp_unit(ori_images + delta).detach()

    return adv_images


def evaluate_bpda_attack(
    model,
    test_loader,
    device,
    defense,
    eps,
    alpha,
    steps,
    max_samples=None,
    **defense_kwargs,
):
    """
    评估某个防御方法在 PGD + BPDA 下的鲁棒性。

    Clean Acc：
        对干净样本先做预处理，再输入模型。

    Adv Acc：
        对 PGD + BPDA 生成的对抗样本先做预处理，再输入模型。

    ASR：
        只在原本 clean 预测正确的样本上统计攻击成功率。
    """

    model.eval()

    total = 0

    clean_correct = 0
    adv_correct = 0

    clean_correct_for_asr = 0
    attack_success = 0

    for batch_idx, (images, labels) in enumerate(test_loader):
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

        # 1. 干净样本经过预处理后的准确率
        with torch.no_grad():
            clean_logits = logits_with_preprocess(
                model=model,
                x=images,
                defense=defense,
                bpda=False,
                **defense_kwargs,
            )
            clean_pred = clean_logits.argmax(dim=1)

        clean_mask = clean_pred.eq(labels)

        clean_correct += clean_mask.sum().item()
        clean_correct_for_asr += clean_mask.sum().item()

        # 2. 用 PGD + BPDA 生成自适应对抗样本
        adv_images = pgd_bpda_attack(
            model=model,
            images=images,
            labels=labels,
            defense=defense,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=True,
            **defense_kwargs,
        )

        # 3. 对抗样本仍然要经过真实预处理再输入模型
        with torch.no_grad():
            adv_logits = logits_with_preprocess(
                model=model,
                x=adv_images,
                defense=defense,
                bpda=False,
                **defense_kwargs,
            )
            adv_pred = adv_logits.argmax(dim=1)

        adv_mask = adv_pred.eq(labels)

        adv_correct += adv_mask.sum().item()

        success_mask = clean_mask & adv_pred.ne(labels)
        attack_success += success_mask.sum().item()

        total += batch_size

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(test_loader)}] "
                f"processed samples = {total}"
            )

    clean_acc = clean_correct / total * 100.0
    adv_acc = adv_correct / total * 100.0

    if clean_correct_for_asr == 0:
        asr = 0.0
    else:
        asr = attack_success / clean_correct_for_asr * 100.0

    return {
        "total_samples": total,
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "asr": asr,
    }


def plot_bar(df, value_col, title, ylabel, save_path):
    plt.figure(figsize=(7.5, 5.0))

    labels = df["method_label"].tolist()
    values = df[value_col].tolist()

    bars = plt.bar(labels, values)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="PGD + BPDA 自适应攻击评估输入预处理防御"
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/cnn.pt",
        help="待评估模型权重路径，通常是原始 Model B"
    )

    parser.add_argument(
        "--defenses",
        type=str,
        default="jpeg,squeeze",
        help="防御方法列表，例如 jpeg,squeeze"
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        help="JPEG 压缩质量，默认 75"
    )

    parser.add_argument(
        "--bit-depth",
        type=int,
        default=5,
        help="位深压缩 bit depth，默认 5"
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多测试多少张样本。JPEG + BPDA 较慢，可以先设为 1000"
    )

    parser.add_argument("--result-dir", type=str, default="./results")
    parser.add_argument("--figure-dir", type=str, default="./figures")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    ensure_dir(args.result_dir)
    ensure_dir(args.figure_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    defenses = parse_defense_list(args.defenses)

    print("=" * 80)
    print("PGD + BPDA 自适应攻击评估")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Defenses: {defenses}")
    print(f"eps = {args.eps * 255:.1f}/255")
    print(f"alpha = {args.alpha * 255:.1f}/255")
    print(f"steps = {args.steps}")
    print(f"jpeg_quality = {args.jpeg_quality}")
    print(f"bit_depth = {args.bit_depth}")
    print(f"batch_size = {args.batch_size}")
    print(f"max_samples = {args.max_samples}")

    print("\n加载 CIFAR-10 测试集...")
    _, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.download,
        test_samples=args.max_samples,
    )
    print("测试集加载完成")

    print("\n加载模型...")
    model = load_model(
        args.model,
        args.checkpoint,
        device
    )
    print("模型加载完成")

    results = []

    for defense in defenses:
        print("\n" + "=" * 80)
        print(f"当前防御方法: {defense_display_name(defense)}")
        print("=" * 80)

        defense_kwargs = get_defense_kwargs(defense, args)

        metrics = evaluate_bpda_attack(
            model=model,
            test_loader=test_loader,
            device=device,
            defense=defense,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            max_samples=args.max_samples,
            **defense_kwargs,
        )

        row = {
            "defense": defense,
            "defense_label": defense_display_name(defense),
            "attack_name": "pgd_bpda",
            "method_label": f"{defense_display_name(defense)} + PGD-BPDA",
            "eps_255": args.eps * 255,
            "alpha_255": args.alpha * 255,
            "steps": args.steps,
            "jpeg_quality": args.jpeg_quality if defense == "jpeg" else "",
            "bit_depth": args.bit_depth if defense == "squeeze" else "",
            "total_samples": metrics["total_samples"],
            "clean_acc": metrics["clean_acc"],
            "adv_acc": metrics["adv_acc"],
            "asr": metrics["asr"],
        }

        results.append(row)

        print(f"Defense: {row['defense_label']}")
        print(f"Attack: PGD + BPDA")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Clean Acc: {metrics['clean_acc']:.2f}%")
        print(f"Adv Acc: {metrics['adv_acc']:.2f}%")
        print(f"ASR: {metrics['asr']:.2f}%")

    df = pd.DataFrame(results)

    csv_path = os.path.join(args.result_dir, "bpda_adaptive_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\nBPDA 自适应攻击结果：")
    print(df.to_string(index=False))
    print(f"\nCSV 已保存：{csv_path}")

    adv_acc_fig = os.path.join(args.figure_dir, "bpda_adaptive_adv_acc_bar.png")
    asr_fig = os.path.join(args.figure_dir, "bpda_adaptive_asr_bar.png")

    plot_bar(
        df=df,
        value_col="adv_acc",
        title="PGD + BPDA: Adversarial Accuracy",
        ylabel="Adv Acc (%)",
        save_path=adv_acc_fig,
    )

    plot_bar(
        df=df,
        value_col="asr",
        title="PGD + BPDA: Attack Success Rate",
        ylabel="ASR (%)",
        save_path=asr_fig,
    )

    print(f"Adv Acc 图已保存：{adv_acc_fig}")
    print(f"ASR 图已保存：{asr_fig}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()