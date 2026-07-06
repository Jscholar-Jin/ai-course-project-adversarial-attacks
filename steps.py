# steps.py
# 功能：
# 单独测试 Transfer DeepFool 在不同 steps 下的攻击效果
#
# 攻击路径：
# Model A -> Model B
#
# 输出：
# 1. results/transfer_deepfool_steps_sensitivity.csv
# 2. figures/transfer_deepfool_steps_asr.png
# 3. figures/transfer_deepfool_steps_adv_acc.png
#
# 默认测试：
# steps = 1, 3, 5, 10, 20
#
# 固定参数：
# eps = 8/255
#
# 说明：
# DeepFool 的 max_deepfool_steps 默认跟随当前 steps

import os
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from attacks import make_adv
from data_utils import build_cifar10_loaders
from models import load_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def evaluate_attack(
    attack_name,
    source_model,
    target_model,
    test_loader,
    device,
    eps,
    alpha,
    steps,
    deepfool_steps,
    max_samples=None,
):
    """
    评估某一种攻击在指定 steps 下的攻击效果。

    对于 Transfer DeepFool：

    source_model = Model A
        用于生成 DeepFool 对抗样本。

    target_model = Model B
        用于测试对抗样本是否可以迁移攻击成功。

    统计指标：
    clean_acc：目标模型在干净样本上的准确率
    adv_acc：目标模型在对抗样本上的准确率
    asr：攻击成功率

    ASR 统计方式：
    只在目标模型原本分类正确的样本上统计攻击成功率。
    """

    source_model.eval()
    target_model.eval()

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

        # 先用目标模型 Model B 测试干净样本
        with torch.no_grad():
            clean_logits = target_model(images)
            clean_pred = clean_logits.argmax(dim=1)

        clean_mask = clean_pred.eq(labels)

        clean_correct += clean_mask.sum().item()
        clean_correct_for_asr += clean_mask.sum().item()

        # 在源模型 Model A 上生成 DeepFool 对抗样本
        adv_images = make_adv(
            attack=attack_name,
            model=source_model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps,
            max_deepfool_steps=deepfool_steps,
        )

        # 在目标模型 Model B 上测试迁移攻击效果
        with torch.no_grad():
            adv_logits = target_model(adv_images)
            adv_pred = adv_logits.argmax(dim=1)

        adv_mask = adv_pred.eq(labels)
        adv_correct += adv_mask.sum().item()

        # 攻击成功：Model B 原本预测正确，但对抗样本预测错误
        success_mask = clean_mask & adv_pred.ne(labels)
        attack_success += success_mask.sum().item()

        total += batch_size

    clean_acc = clean_correct / total * 100.0
    adv_acc = adv_correct / total * 100.0

    if clean_correct_for_asr == 0:
        asr = 0.0
    else:
        asr = attack_success / clean_correct_for_asr * 100.0

    return clean_acc, adv_acc, asr, total


def parse_steps_list(steps_list_str):
    """
    输入格式：
    --steps-list 1,3,5,10,20

    返回：
    [1, 3, 5, 10, 20]
    """

    steps_values = []

    for item in steps_list_str.split(","):
        item = item.strip()

        if item == "":
            continue

        steps_values.append(int(item))

    return steps_values


def plot_steps_curve(df, value_col, title, ylabel, save_path):
    plt.figure(figsize=(8.0, 5.2))

    plt.plot(
        df["steps"],
        df[value_col],
        marker="o",
        linewidth=2,
        label="Transfer DeepFool",
    )

    for x, y in zip(df["steps"], df[value_col]):
        plt.text(
            x,
            y + 1,
            f"{y:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(df["steps"].unique()))
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="单独测试 Transfer DeepFool 在不同 steps 下的迁移攻击效果"
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)

    # Model A：迁移攻击源模型
    parser.add_argument("--model-a", type=str, default="resnet")
    parser.add_argument("--checkpoint-a", type=str, default="./checkpoints/resnet.pt")

    # Model B：目标模型
    parser.add_argument("--model-b", type=str, default="cnn")
    parser.add_argument("--checkpoint-b", type=str, default="./checkpoints/cnn.pt")

    parser.add_argument(
        "--steps-list",
        type=str,
        default="1,3,5,10,20",
        help="DeepFool 最大迭代次数列表，例如 1,3,5,10,20"
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        help="最大扰动范围，默认 8/255"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 255,
        help="这里为了统一接口保留 alpha，DeepFool 实际不使用该参数"
    )

    parser.add_argument(
        "--deepfool-steps",
        type=int,
        default=None,
        help="DeepFool 最大迭代次数。默认跟随当前 steps"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多测试多少张样本，默认完整测试集"
    )

    parser.add_argument("--result-dir", type=str, default="./results")
    parser.add_argument("--figure-dir", type=str, default="./figures")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    ensure_dir(args.result_dir)
    ensure_dir(args.figure_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps_values = parse_steps_list(args.steps_list)

    print("=" * 80)
    print("Transfer DeepFool steps 敏感性实验")
    print("=" * 80)
    print(f"Device: {device}")
    print("Attack: Transfer DeepFool")
    print("Path: Model A -> Model B")
    print(f"steps list: {steps_values}")
    print(f"eps = {args.eps * 255:.1f}/255")
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

    print("\n加载 Model A：迁移攻击源模型...")
    model_a = load_model(
        args.model_a,
        args.checkpoint_a,
        device
    )
    print("Model A 加载完成")

    print("\n加载 Model B：目标模型...")
    model_b = load_model(
        args.model_b,
        args.checkpoint_b,
        device
    )
    print("Model B 加载完成")

    experiments = [
        {
            "experiment": "transfer",
            "attack_name": "deepfool",
            "method_label": "Transfer DeepFool",
            "source_model": model_a,
            "target_model": model_b,
            "source_model_name": "Model A",
            "target_model_name": "Model B",
        },
    ]

    results = []

    print("\n开始测试 Transfer DeepFool 在不同 steps 下的攻击效果...")

    for steps in steps_values:
        print("\n" + "=" * 80)
        print(f"当前 DeepFool steps = {steps}")
        print("=" * 80)

        for exp in experiments:
            print("-" * 80)
            print(f"实验类型: {exp['experiment']}")
            print(f"攻击方法: {exp['method_label']}")
            print(f"攻击路径: {exp['source_model_name']} -> {exp['target_model_name']}")

            # DeepFool 的最大迭代次数默认跟随当前 steps
            if args.deepfool_steps is None:
                current_deepfool_steps = steps
            else:
                current_deepfool_steps = args.deepfool_steps

            current_steps = steps

            clean_acc, adv_acc, asr, total = evaluate_attack(
                attack_name=exp["attack_name"],
                source_model=exp["source_model"],
                target_model=exp["target_model"],
                test_loader=test_loader,
                device=device,
                eps=args.eps,
                alpha=args.alpha,
                steps=current_steps,
                deepfool_steps=current_deepfool_steps,
                max_samples=args.max_samples,
            )

            row = {
                "experiment": exp["experiment"],
                "attack_name": exp["attack_name"],
                "method_label": exp["method_label"],
                "source_model": exp["source_model_name"],
                "target_model": exp["target_model_name"],
                "eps_255": args.eps * 255,
                "alpha_255": args.alpha * 255,
                "steps": steps,
                "deepfool_steps": current_deepfool_steps,
                "total_samples": total,
                "clean_acc": clean_acc,
                "adv_acc": adv_acc,
                "asr": asr,
            }

            results.append(row)

            print(f"Total Samples: {total}")
            print(f"Clean Acc: {clean_acc:.2f}%")
            print(f"Adv Acc: {adv_acc:.2f}%")
            print(f"ASR: {asr:.2f}%")

    df = pd.DataFrame(results)

    csv_path = os.path.join(
        args.result_dir,
        "transfer_deepfool_steps_sensitivity.csv"
    )
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\nTransfer DeepFool steps 敏感性实验结果：")
    print(df.to_string(index=False))
    print(f"\nCSV 已保存：{csv_path}")

    asr_fig = os.path.join(
        args.figure_dir,
        "transfer_deepfool_steps_asr.png"
    )
    adv_acc_fig = os.path.join(
        args.figure_dir,
        "transfer_deepfool_steps_adv_acc.png"
    )

    plot_steps_curve(
        df=df,
        value_col="asr",
        title="Transfer DeepFool: Effect of Steps on ASR",
        ylabel="ASR (%)",
        save_path=asr_fig,
    )

    plot_steps_curve(
        df=df,
        value_col="adv_acc",
        title="Transfer DeepFool: Effect of Steps on Adv Acc",
        ylabel="Adv Acc (%)",
        save_path=adv_acc_fig,
    )

    print(f"ASR 曲线图已保存：{asr_fig}")
    print(f"Adv Acc 曲线图已保存：{adv_acc_fig}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()