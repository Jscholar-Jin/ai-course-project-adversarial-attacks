# eval_eps_sensitivity.py
# 功能：
# 测试扰动大小 eps 对攻击效果的影响
#
# 输出：
# 1. results/eps_sensitivity.csv
# 2. figures/eps_sensitivity_asr.png
# 3. figures/eps_sensitivity_adv_acc.png
#
# 默认测试：
# eps = 2/255, 4/255, 8/255, 12/255, 16/255
#
# 攻击方法：
# White-box FGSM
# White-box PGD
# White-box DeepFool
# Transfer FGSM
# Transfer PGD
# Transfer MI-FGSM

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

        with torch.no_grad():
            clean_logits = target_model(images)
            clean_pred = clean_logits.argmax(dim=1)

        clean_mask = clean_pred.eq(labels)

        clean_correct += clean_mask.sum().item()
        clean_correct_for_asr += clean_mask.sum().item()

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

        with torch.no_grad():
            adv_logits = target_model(adv_images)
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

    return clean_acc, adv_acc, asr, total


def plot_eps_curve(df, value_col, title, ylabel, save_path):
    plt.figure(figsize=(9.5, 5.5))

    for method in df["method_label"].unique():
        sub = df[df["method_label"] == method]
        plt.plot(
            sub["eps_255"],
            sub[value_col],
            marker="o",
            linewidth=2,
            label=method,
        )

    plt.xlabel("eps / 255")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(df["eps_255"].unique()))
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def parse_eps_list(eps_list_str):
    """
    输入格式：
    --eps-list 2,4,8,12,16

    返回：
    [2/255, 4/255, 8/255, 12/255, 16/255]
    """
    eps_values = []

    for item in eps_list_str.split(","):
        item = item.strip()

        if item == "":
            continue

        eps_255 = float(item)
        eps_values.append(eps_255 / 255.0)

    return eps_values


def main():
    parser = argparse.ArgumentParser(
        description="扰动大小 eps 敏感性实验"
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--model-a", type=str, default="resnet")
    parser.add_argument("--checkpoint-a", type=str, default="./checkpoints/resnet.pt")

    parser.add_argument("--model-b", type=str, default="cnn")
    parser.add_argument("--checkpoint-b", type=str, default="./checkpoints/cnn.pt")

    parser.add_argument(
        "--eps-list",
        type=str,
        default="2,4,8,12,16",
        help="扰动大小列表，单位是 /255，例如 2,4,8,12,16"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 255,
        help="PGD 和 MI-FGSM 的步长，默认 2/255"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="PGD 和 MI-FGSM 迭代次数"
    )

    parser.add_argument(
        "--deepfool-steps",
        type=int,
        default=20,
        help="DeepFool 最大迭代次数"
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

    eps_values = parse_eps_list(args.eps_list)

    print("=" * 80)
    print("扰动大小 eps 敏感性实验")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"eps list: {[round(eps * 255, 1) for eps in eps_values]}/255")
    print(f"alpha = {args.alpha * 255:.1f}/255")
    print(f"steps = {args.steps}")
    print(f"deepfool_steps = {args.deepfool_steps}")
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
            "experiment": "white-box",
            "attack_name": "fgsm",
            "method_label": "White-box FGSM",
            "source_model": model_b,
            "target_model": model_b,
            "source_model_name": "Model B",
            "target_model_name": "Model B",
        },
        {
            "experiment": "white-box",
            "attack_name": "pgd",
            "method_label": "White-box PGD",
            "source_model": model_b,
            "target_model": model_b,
            "source_model_name": "Model B",
            "target_model_name": "Model B",
        },
        {
            "experiment": "white-box",
            "attack_name": "deepfool",
            "method_label": "White-box DeepFool",
            "source_model": model_b,
            "target_model": model_b,
            "source_model_name": "Model B",
            "target_model_name": "Model B",
        },
        {
            "experiment": "transfer",
            "attack_name": "fgsm",
            "method_label": "Transfer FGSM",
            "source_model": model_a,
            "target_model": model_b,
            "source_model_name": "Model A",
            "target_model_name": "Model B",
        },
        {
            "experiment": "transfer",
            "attack_name": "pgd",
            "method_label": "Transfer PGD",
            "source_model": model_a,
            "target_model": model_b,
            "source_model_name": "Model A",
            "target_model_name": "Model B",
        },
        {
            "experiment": "transfer",
            "attack_name": "mi-fgsm",
            "method_label": "Transfer MI-FGSM",
            "source_model": model_a,
            "target_model": model_b,
            "source_model_name": "Model A",
            "target_model_name": "Model B",
        },
    ]

    results = []

    print("\n开始测试不同 eps 下的攻击效果...")

    for eps in eps_values:
        print("\n" + "=" * 80)
        print(f"当前 eps = {eps * 255:.1f}/255")
        print("=" * 80)

        for exp in experiments:
            print("-" * 80)
            print(f"实验类型: {exp['experiment']}")
            print(f"攻击方法: {exp['method_label']}")
            print(f"攻击路径: {exp['source_model_name']} -> {exp['target_model_name']}")

            clean_acc, adv_acc, asr, total = evaluate_attack(
                attack_name=exp["attack_name"],
                source_model=exp["source_model"],
                target_model=exp["target_model"],
                test_loader=test_loader,
                device=device,
                eps=eps,
                alpha=args.alpha,
                steps=args.steps,
                deepfool_steps=args.deepfool_steps,
                max_samples=args.max_samples,
            )

            row = {
                "experiment": exp["experiment"],
                "attack_name": exp["attack_name"],
                "method_label": exp["method_label"],
                "source_model": exp["source_model_name"],
                "target_model": exp["target_model_name"],
                "eps_255": eps * 255,
                "alpha_255": args.alpha * 255,
                "steps": args.steps,
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

    csv_path = os.path.join(args.result_dir, "eps_sensitivity.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n扰动大小敏感性实验结果：")
    print(df.to_string(index=False))
    print(f"\nCSV 已保存：{csv_path}")

    asr_fig = os.path.join(args.figure_dir, "eps_sensitivity_asr.png")
    adv_acc_fig = os.path.join(args.figure_dir, "eps_sensitivity_adv_acc.png")

    plot_eps_curve(
        df=df,
        value_col="asr",
        title="Effect of eps on Attack Success Rate",
        ylabel="ASR (%)",
        save_path=asr_fig,
    )

    plot_eps_curve(
        df=df,
        value_col="adv_acc",
        title="Effect of eps on Adversarial Accuracy",
        ylabel="Adv Acc (%)",
        save_path=adv_acc_fig,
    )

    print(f"ASR 曲线图已保存：{asr_fig}")
    print(f"Adv Acc 曲线图已保存：{adv_acc_fig}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()