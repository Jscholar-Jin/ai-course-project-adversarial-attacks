# visual_tsne_decision_boundary.py
# 功能：
# 使用 t-SNE 可视化正常样本和对抗样本的特征分布差异
#
# 输出：
# 1. figures/tsne_clean_vs_attacks.png
#    Clean / FGSM / PGD / DeepFool 总体分布图
#
# 2. figures/tsne_clean_vs_each_attack.png
#    Clean vs FGSM、Clean vs PGD、Clean vs DeepFool 三个子图
#
# 3. results/tsne_features.csv
#    t-SNE 坐标和样本信息
#
# 默认：
# model = cnn
# checkpoint = ./checkpoints/cnn.pt
# attacks = fgsm,pgd,deepfool
# num_samples = 300

import os
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from attacks import make_adv
from data_utils import CIFAR10_CLASSES, build_cifar10_loaders
from models import load_model


ATTACK_LABELS = {
    "clean": "Clean",
    "fgsm": "FGSM",
    "pgd": "PGD",
    "deepfool": "DeepFool",
    "mi-fgsm": "MI-FGSM",
    "mifgsm": "MI-FGSM",
}

KIND_COLORS = {
    "clean": "#D62728",
    "fgsm": "#4C78A8",
    "pgd": "#F58518",
    "deepfool": "#54A24B",
    "mi-fgsm": "#B279A2",
    "mifgsm": "#B279A2",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "figure.titlesize": 18,
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def parse_attack_list(attack_list_str):
    attacks = []
    for item in attack_list_str.split(","):
        item = item.strip().lower()
        if item:
            attacks.append(item)
    return attacks


def attack_display_name(name):
    return ATTACK_LABELS.get(name.lower(), name.upper())


def extract_features(model, images):
    """
    从模型中提取特征。

    优先级：
    1. 如果模型有 forward_features()，使用它；
    2. 如果模型有 features 模块，使用它；
    3. 否则使用 logits 作为 fallback。

    输出：
    features: [B, D]
    """

    model.eval()

    base_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        if hasattr(base_model, "forward_features") and callable(base_model.forward_features):
            features = base_model.forward_features(images)

        elif hasattr(base_model, "features") and callable(base_model.features):
            features = base_model.features(images)

        else:
            # fallback：如果没有显式特征层，就用最后 logits 做可视化
            features = model(images)

    if isinstance(features, (tuple, list)):
        features = features[0]

    if features.dim() > 2:
        features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
        features = features.flatten(1)
    else:
        features = features.flatten(1)

    return features.detach().cpu()


def collect_features(
    model,
    test_loader,
    device,
    args,
):
    """
    收集 Clean 和多种攻击样本的特征。

    只选取目标模型在 clean 样本上预测正确的样本，
    这样 t-SNE 分布更适合解释“攻击导致样本偏离原本类别区域”。
    """

    model.eval()

    attacks = parse_attack_list(args.attacks)

    feature_list = []
    record_list = []

    collected = 0
    sample_id = 0

    print("\n开始收集 Clean 和 Adversarial 特征...")

    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_logits = model(images)
            clean_pred = clean_logits.argmax(dim=1)

        clean_mask = clean_pred.eq(labels)

        if clean_mask.sum().item() == 0:
            continue

        images = images[clean_mask]
        labels = labels[clean_mask]
        clean_pred = clean_pred[clean_mask]

        remaining = args.num_samples - collected
        if images.size(0) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
            clean_pred = clean_pred[:remaining]

        batch_size = images.size(0)

        # 1. Clean features
        clean_features = extract_features(model, images)
        feature_list.append(clean_features)

        for i in range(batch_size):
            label_id = labels[i].item()
            pred_id = clean_pred[i].item()

            record_list.append(
                {
                    "sample_id": sample_id + i,
                    "kind": "clean",
                    "attack": "clean",
                    "true_label": CIFAR10_CLASSES[label_id],
                    "pred_label": CIFAR10_CLASSES[pred_id],
                    "attack_success": False,
                }
            )

        # 2. Adversarial features
        for attack in attacks:
            print(
                f"Batch {batch_idx + 1}: generating {attack_display_name(attack)} "
                f"for {batch_size} samples..."
            )

            adv_images = make_adv(
                attack=attack,
                model=model,
                images=images,
                labels=labels,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                max_deepfool_steps=args.deepfool_steps,
            )

            with torch.no_grad():
                adv_logits = model(adv_images)
                adv_pred = adv_logits.argmax(dim=1)

            adv_features = extract_features(model, adv_images)
            feature_list.append(adv_features)

            for i in range(batch_size):
                label_id = labels[i].item()
                adv_pred_id = adv_pred[i].item()
                success = adv_pred_id != label_id

                record_list.append(
                    {
                        "sample_id": sample_id + i,
                        "kind": attack,
                        "attack": attack,
                        "true_label": CIFAR10_CLASSES[label_id],
                        "pred_label": CIFAR10_CLASSES[adv_pred_id],
                        "attack_success": success,
                    }
                )

        collected += batch_size
        sample_id += batch_size

        print(f"已收集 clean 样本数: {collected}/{args.num_samples}")

        if collected >= args.num_samples:
            break

    if len(feature_list) == 0:
        raise RuntimeError("没有收集到任何特征，请检查模型和数据集。")

    features = torch.cat(feature_list, dim=0).numpy()
    records = pd.DataFrame(record_list)

    return features, records


def run_tsne(features, args):
    """
    对特征做标准化后使用 t-SNE 降维到二维。
    """

    print("\n开始运行 t-SNE...")

    features = StandardScaler().fit_transform(features)

    n_samples = features.shape[0]

    perplexity = min(args.perplexity, max(5, (n_samples - 1) // 3))

    print(f"t-SNE samples: {n_samples}")
    print(f"t-SNE perplexity: {perplexity}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )

    embedding = tsne.fit_transform(features)

    return embedding


def save_combined_tsne_plot(df, output_path, args):
    """
    保存 Clean + 多种对抗样本的总体 t-SNE 分布图。
    """

    plt.figure(figsize=(8.5, 6.8))

    kinds = ["clean"] + parse_attack_list(args.attacks)

    for kind in kinds:
        sub = df[df["kind"] == kind]

        if sub.empty:
            continue

        color = KIND_COLORS.get(kind, None)

        plt.scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            s=18,
            alpha=0.65,
            label=attack_display_name(kind),
            c=color,
            edgecolors="none",
        )

    plt.title("t-SNE Feature Distribution: Clean vs Adversarial Samples")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"总览 t-SNE 图已保存：{output_path}")


def save_each_attack_tsne_plot(df, output_path, args):
    """
    保存 Clean vs 每一种攻击的对比图。

    每个子图中：
    - 灰色点：Clean
    - 彩色点：对应攻击样本
    - 淡线：连接同一个样本的 clean 位置和 adversarial 位置
    """

    attacks = parse_attack_list(args.attacks)

    n_cols = len(attacks)

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(5.2 * n_cols, 5.0),
        sharex=True,
        sharey=True,
    )

    if n_cols == 1:
        axes = [axes]

    clean_df = df[df["kind"] == "clean"]

    clean_lookup = clean_df.set_index("sample_id")[["tsne_x", "tsne_y"]]

    for ax, attack in zip(axes, attacks):
        adv_df = df[df["kind"] == attack]

        color = KIND_COLORS.get(attack, "#4C78A8")

        # 画连接线，表示同一个样本从 clean 到 adv 的特征偏移
        count_lines = 0
        for _, row in adv_df.iterrows():
            sid = row["sample_id"]

            if sid not in clean_lookup.index:
                continue

            clean_x = clean_lookup.loc[sid, "tsne_x"]
            clean_y = clean_lookup.loc[sid, "tsne_y"]

            ax.plot(
                [clean_x, row["tsne_x"]],
                [clean_y, row["tsne_y"]],
                color=color,
                alpha=0.18,
                linewidth=0.7,
            )

            count_lines += 1
            if count_lines >= args.max_lines:
                break

        ax.scatter(
            clean_df["tsne_x"],
            clean_df["tsne_y"],
            s=16,
            alpha=0.50,
            c="#222222",
            label="Clean",
            edgecolors="none",
        )

        ax.scatter(
            adv_df["tsne_x"],
            adv_df["tsne_y"],
            s=18,
            alpha=0.70,
            c=color,
            label=attack_display_name(attack),
            edgecolors="none",
        )

        success_rate = adv_df["attack_success"].mean() * 100.0

        ax.set_title(
            f"Clean vs {attack_display_name(attack)}\n"
            f"Attack Success in selected samples: {success_rate:.1f}%"
        )

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(frameon=True, fontsize=9)

    axes[0].set_ylabel("t-SNE Dimension 2")

    fig.suptitle(
        "Feature Shift from Clean Samples to Adversarial Samples",
        fontsize=18,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"分攻击 t-SNE 图已保存：{output_path}")


def save_attack_success_bar(df, output_path, args):
    """
    保存当前选中样本上的攻击成功率柱状图。
    """

    attacks = parse_attack_list(args.attacks)

    rows = []

    for attack in attacks:
        sub = df[df["kind"] == attack]
        if sub.empty:
            continue

        rows.append(
            {
                "attack": attack_display_name(attack),
                "success_rate": sub["attack_success"].mean() * 100.0,
            }
        )

    result = pd.DataFrame(rows)

    plt.figure(figsize=(6.5, 4.5))

    bars = plt.bar(
        result["attack"],
        result["success_rate"],
    )

    plt.ylim(0, 105)
    plt.ylabel("Attack Success Rate (%)")
    plt.title("Attack Success Rate on Selected Samples")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, result["success_rate"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"攻击成功率图已保存：{output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE 可视化正常样本和对抗样本的特征分布差异"
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/cnn.pt",
        help="模型权重路径"
    )

    parser.add_argument(
        "--attacks",
        type=str,
        default="fgsm,pgd,deepfool",
        help="攻击方法列表，例如 fgsm,pgd,deepfool"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=300,
        help="选取多少个 clean 分类正确样本做 t-SNE"
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--deepfool-steps", type=int, default=20)

    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity"
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=80,
        help="每个子图最多画多少条 clean -> adv 连接线"
    )

    parser.add_argument("--figure-dir", type=str, default="./figures")
    parser.add_argument("--result-dir", type=str, default="./results")
    parser.add_argument("--prefix", type=str, default="tsne")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    setup_matplotlib()

    ensure_dir(args.figure_dir)
    ensure_dir(args.result_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("t-SNE 决策边界 / 特征分布可视化")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Attacks: {parse_attack_list(args.attacks)}")
    print(f"num_samples = {args.num_samples}")
    print(f"eps = {args.eps * 255:.1f}/255")
    print(f"alpha = {args.alpha * 255:.1f}/255")
    print(f"steps = {args.steps}")
    print(f"deepfool_steps = {args.deepfool_steps}")

    print("\n加载 CIFAR-10 测试集...")
    _, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.download,
        test_samples=None,
    )
    print("测试集加载完成")

    print("\n加载模型...")
    model = load_model(
        args.model,
        args.checkpoint,
        device
    )
    print("模型加载完成")

    features, records = collect_features(
        model=model,
        test_loader=test_loader,
        device=device,
        args=args,
    )

    embedding = run_tsne(features, args)

    records["tsne_x"] = embedding[:, 0]
    records["tsne_y"] = embedding[:, 1]

    csv_path = os.path.join(
        args.result_dir,
        f"{args.prefix}_features.csv"
    )

    records.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nt-SNE 坐标 CSV 已保存：{csv_path}")

    combined_path = os.path.join(
        args.figure_dir,
        f"{args.prefix}_clean_vs_attacks.png"
    )

    each_path = os.path.join(
        args.figure_dir,
        f"{args.prefix}_clean_vs_each_attack.png"
    )

    success_path = os.path.join(
        args.figure_dir,
        f"{args.prefix}_attack_success_selected.png"
    )

    save_combined_tsne_plot(
        df=records,
        output_path=combined_path,
        args=args,
    )

    save_each_attack_tsne_plot(
        df=records,
        output_path=each_path,
        args=args,
    )

    save_attack_success_bar(
        df=records,
        output_path=success_path,
        args=args,
    )

    print("\n全部完成。输出文件：")
    print(combined_path)
    print(each_path)
    print(success_path)
    print(csv_path)


if __name__ == "__main__":
    main()