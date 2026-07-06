# eval_adv_training_result.py
# 功能：
# 评估 FGSM 对抗训练后的模型鲁棒性
#
# 输出：
# 1. results/adv_training_eval_results.csv
# 2. figures/adv_training_adv_acc_bar.png
# 3. figures/adv_training_asr_bar.png
#
# 评估内容：
# 白盒攻击：FGSM / PGD / DeepFool
# 黑盒攻击：SPSA
# 迁移攻击：FGSM / PGD / MI-FGSM / DeepFool
#
# 说明：
# Model A：迁移攻击源模型，例如 ResNet
# Model B_adv：对抗训练后的目标模型，例如 cnn_fgsm_adv_train.pt

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


def evaluate_one_attack(
    attack_name,
    source_model,
    target_model,
    test_loader,
    device,
    eps,
    alpha,
    steps,
    deepfool_steps,
    spsa_samples,
    max_samples=None,
):
    """
    评估一次攻击效果。

    source_model：
        用于生成对抗样本的模型。

    target_model：
        被攻击、被评估的目标模型。

    白盒攻击：
        source_model = target_model = 对抗训练后的模型

    迁移攻击：
        source_model = Model A
        target_model = 对抗训练后的模型

    黑盒 SPSA：
        source_model = target_model = 对抗训练后的模型
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

        # 干净样本准确率：用目标模型评估
        with torch.no_grad():
            clean_logits = target_model(images)
            clean_pred = clean_logits.argmax(dim=1)

        clean_mask = clean_pred.eq(labels)

        clean_correct += clean_mask.sum().item()
        clean_correct_for_asr += clean_mask.sum().item()

        # 在 source_model 上生成对抗样本
        adv_images = make_adv(
            attack=attack_name,
            model=source_model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps,
            max_deepfool_steps=deepfool_steps,
            spsa_samples=spsa_samples,
        )

        # 在 target_model 上评估对抗样本
        with torch.no_grad():
            adv_logits = target_model(adv_images)
            adv_pred = adv_logits.argmax(dim=1)

        adv_mask = adv_pred.eq(labels)
        adv_correct += adv_mask.sum().item()

        # ASR：只在原本分类正确的样本上统计攻击成功率
        success_mask = clean_mask & adv_pred.ne(labels)
        attack_success += success_mask.sum().item()

        total += batch_size

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
    plt.figure(figsize=(11.5, 5.5))

    x_labels = df["method_label"].tolist()
    values = df[value_col].tolist()

    bars = plt.bar(x_labels, values)

    plt.xticks(rotation=35, ha="right")
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
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="评估 FGSM 对抗训练后模型的防御效果"
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)

    # Model A：迁移攻击源模型
    parser.add_argument("--model-a", type=str, default="resnet")
    parser.add_argument(
        "--checkpoint-a",
        type=str,
        default="./checkpoints/resnet.pt",
        help="Model A 权重路径，用于迁移攻击源模型"
    )

    # 对抗训练后的目标模型
    parser.add_argument("--model-b", type=str, default="cnn")
    parser.add_argument(
        "--checkpoint-b-adv",
        type=str,
        default="./checkpoints/cnn_fgsm_adv_train.pt",
        help="对抗训练后的 Model B 权重路径"
    )

    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--deepfool-steps", type=int, default=20)

    # SPSA 比较慢，默认只测 1000 张
    parser.add_argument("--spsa-samples", type=int, default=16)
    parser.add_argument("--spsa-max-samples", type=int, default=1000)

    # 其他攻击默认完整测试集
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--result-dir", type=str, default="./results")
    parser.add_argument("--figure-dir", type=str, default="./figures")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    ensure_dir(args.result_dir)
    ensure_dir(args.figure_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("FGSM 对抗训练模型鲁棒性评估")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"eps = {args.eps * 255:.1f}/255")
    print(f"alpha = {args.alpha * 255:.1f}/255")
    print(f"steps = {args.steps}")
    print(f"deepfool_steps = {args.deepfool_steps}")
    print(f"spsa_samples = {args.spsa_samples}")

    print("\n加载 CIFAR-10 测试集...")
    _, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.download,
        test_samples=None,
    )
    print("测试集加载完成")

    print("\n加载 Model A：迁移攻击源模型...")
    model_a = load_model(
        args.model_a,
        args.checkpoint_a,
        device
    )
    print("Model A 加载完成")

    print("\n加载对抗训练后的 Model B...")
    model_b_adv = load_model(
        args.model_b,
        args.checkpoint_b_adv,
        device
    )
    print("对抗训练后的 Model B 加载完成")

    experiments = []

    # 白盒攻击：对抗训练后的 Model B -> 对抗训练后的 Model B
    experiments.extend([
        {
            "experiment": "white-box",
            "attack_name": "fgsm",
            "method_label": "White-box FGSM",
            "source_model_name": "Adv-trained Model B",
            "target_model_name": "Adv-trained Model B",
            "source_model": model_b_adv,
            "target_model": model_b_adv,
            "max_samples": args.max_samples,
        },
        {
            "experiment": "white-box",
            "attack_name": "pgd",
            "method_label": "White-box PGD",
            "source_model_name": "Adv-trained Model B",
            "target_model_name": "Adv-trained Model B",
            "source_model": model_b_adv,
            "target_model": model_b_adv,
            "max_samples": args.max_samples,
        },
        {
            "experiment": "white-box",
            "attack_name": "deepfool",
            "method_label": "White-box DeepFool",
            "source_model_name": "Adv-trained Model B",
            "target_model_name": "Adv-trained Model B",
            "source_model": model_b_adv,
            "target_model": model_b_adv,
            "max_samples": args.max_samples,
        },
    ])

    # 迁移攻击：Model A -> 对抗训练后的 Model B
    experiments.extend([
        {
            "experiment": "transfer",
            "attack_name": "fgsm",
            "method_label": "Transfer FGSM",
            "source_model_name": "Model A",
            "target_model_name": "Adv-trained Model B",
            "source_model": model_a,
            "target_model": model_b_adv,
            "max_samples": args.max_samples,
        },
        {
            "experiment": "transfer",
            "attack_name": "pgd",
            "method_label": "Transfer PGD",
            "source_model_name": "Model A",
            "target_model_name": "Adv-trained Model B",
            "source_model": model_a,
            "target_model": model_b_adv,
            "max_samples": args.max_samples,
        },
        {
            "experiment": "transfer",
            "attack_name": "mi-fgsm",
            "method_label": "Transfer MI-FGSM",
            "source_model_name": "Model A",
            "target_model_name": "Adv-trained Model B",
            "source_model": model_a,
            "target_model": model_b_adv,
            "max_samples": args.max_samples,
        },
        {
            "experiment": "transfer",
            "attack_name": "deepfool",
            "method_label": "Transfer DeepFool",
            "source_model_name": "Model A",
            "target_model_name": "Adv-trained Model B",
            "source_model": model_a,
            "target_model": model_b_adv,
            "max_samples": args.max_samples,
        },
    ])

    # 黑盒攻击：SPSA Query Adv-trained Model B -> Adv-trained Model B
    experiments.append(
        {
            "experiment": "black-box",
            "attack_name": "spsa",
            "method_label": "Black-box SPSA",
            "source_model_name": "Query Adv-trained Model B",
            "target_model_name": "Adv-trained Model B",
            "source_model": model_b_adv,
            "target_model": model_b_adv,
            "max_samples": args.spsa_max_samples,
        }
    )

    results = []

    print("\n开始评估对抗训练后的模型...")

    for exp in experiments:
        print("-" * 80)
        print(f"实验类型: {exp['experiment']}")
        print(f"攻击方法: {exp['attack_name']}")
        print(f"攻击路径: {exp['source_model_name']} -> {exp['target_model_name']}")

        metrics = evaluate_one_attack(
            attack_name=exp["attack_name"],
            source_model=exp["source_model"],
            target_model=exp["target_model"],
            test_loader=test_loader,
            device=device,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            deepfool_steps=args.deepfool_steps,
            spsa_samples=args.spsa_samples,
            max_samples=exp["max_samples"],
        )

        row = {
            "defense": "fgsm_adversarial_training",
            "experiment": exp["experiment"],
            "attack_name": exp["attack_name"],
            "method_label": exp["method_label"],
            "source_model": exp["source_model_name"],
            "target_model": exp["target_model_name"],
            "eps_255": args.eps * 255,
            "alpha_255": args.alpha * 255,
            "steps": args.steps,
            "total_samples": metrics["total_samples"],
            "clean_acc": metrics["clean_acc"],
            "adv_acc": metrics["adv_acc"],
            "asr": metrics["asr"],
        }

        results.append(row)

        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Clean Acc: {metrics['clean_acc']:.2f}%")
        print(f"Adv Acc: {metrics['adv_acc']:.2f}%")
        print(f"ASR: {metrics['asr']:.2f}%")

    df = pd.DataFrame(results)

    csv_path = os.path.join(args.result_dir, "adv_training_eval_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\nFGSM 对抗训练模型评估结果：")
    print(df.to_string(index=False))
    print(f"\nCSV 已保存：{csv_path}")

    adv_acc_fig = os.path.join(args.figure_dir, "adv_training_adv_acc_bar.png")
    asr_fig = os.path.join(args.figure_dir, "adv_training_asr_bar.png")

    plot_bar(
        df=df,
        value_col="adv_acc",
        title="Adversarial Accuracy after FGSM Adversarial Training",
        ylabel="Adv Acc (%)",
        save_path=adv_acc_fig,
    )

    plot_bar(
        df=df,
        value_col="asr",
        title="Attack Success Rate after FGSM Adversarial Training",
        ylabel="ASR (%)",
        save_path=asr_fig,
    )

    print(f"Adv Acc 柱状图已保存：{adv_acc_fig}")
    print(f"ASR 柱状图已保存：{asr_fig}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()