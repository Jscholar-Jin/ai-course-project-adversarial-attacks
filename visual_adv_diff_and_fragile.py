# visual_adv_diff_and_fragile.py
# 功能：
# 1. 可视化对抗样本与原图差异：Clean / Perturbation xN / Adversarial
# 2. 统计模型在哪些类别样本上最脆弱
# 3. 导出最脆弱样本表和类别统计表
#
# 默认攻击：
# FGSM / PGD / DeepFool
#
# 默认模型：
# cnn
# ./checkpoints/cnn.pt
#
# 输出：
# figures/adv_diff_examples.png
# figures/fragile_class_counts.png
# figures/fragile_class_asr.png
# results/fragile_samples.csv
# results/fragile_class_stats.csv

import os
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from attacks import make_adv
from data_utils import CIFAR10_CLASSES, build_cifar10_loaders
from models import load_model


ATTACK_LABELS = {
    "fgsm": "FGSM",
    "pgd": "PGD",
    "deepfool": "DeepFool",
    "mi-fgsm": "MI-FGSM",
    "mifgsm": "MI-FGSM",
    "spsa": "SPSA",
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
            "axes.titlesize": 12,
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


def attack_display_name(attack):
    return ATTACK_LABELS.get(attack.lower(), attack.upper())


def tensor_to_numpy_image(x):
    """
    Tensor [3, H, W] -> numpy [H, W, 3]
    """
    x = x.detach().cpu()
    x = x.permute(1, 2, 0).numpy()
    x = np.clip(x, 0.0, 1.0)
    return x


def make_delta_vis(clean_image, adv_image, magnify=12.0):
    """
    可视化扰动：

    delta = adv - clean

    因为 delta 有正有负，直接显示不明显。
    所以这里做：
        delta_vis = delta * magnify + 0.5

    这样灰色附近表示扰动小，
    偏彩色区域表示扰动更明显。
    """
    delta = adv_image - clean_image
    delta_vis = delta * magnify + 0.5
    delta_vis = torch.clamp(delta_vis, 0.0, 1.0)
    return delta_vis


def compute_confidence(logits):
    """
    返回 softmax 最大置信度。
    """
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return conf, pred


def collect_fragile_samples(
    model,
    test_loader,
    device,
    args,
):
    """
    收集攻击成功样本，并统计类别脆弱性。

    脆弱样本定义：
    1. clean 样本原本分类正确；
    2. 攻击后分类错误。

    排序依据：
    margin_drop = clean 时真实类别 logit - adv 时真实类别 logit
    margin_drop 越大，说明攻击对真实类别置信度打击越明显。
    """

    model.eval()

    attacks = parse_attack_list(args.attacks)

    fragile_rows = []
    visual_candidates = {attack: [] for attack in attacks}

    # 统计每个类别 clean 正确数量
    class_clean_correct = {
        class_name: 0 for class_name in CIFAR10_CLASSES
    }

    # 统计每个类别在不同攻击下被成功攻击次数
    class_attack_success = {
        class_name: {attack: 0 for attack in attacks}
        for class_name in CIFAR10_CLASSES
    }

    total_seen = 0

    print("\n开始收集对抗样本与脆弱类别统计...")

    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        if args.max_samples is not None and total_seen >= args.max_samples:
            break

        if args.max_samples is not None and total_seen + batch_size > args.max_samples:
            remain = args.max_samples - total_seen
            images = images[:remain]
            labels = labels[:remain]
            batch_size = images.size(0)

        with torch.no_grad():
            clean_logits = model(images)
            clean_conf, clean_pred = compute_confidence(clean_logits)

        clean_correct_mask = clean_pred.eq(labels)

        # 先统计 clean 正确类别数量
        for i in range(batch_size):
            if clean_correct_mask[i].item():
                class_name = CIFAR10_CLASSES[labels[i].item()]
                class_clean_correct[class_name] += 1

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
                spsa_samples=args.spsa_samples,
            )

            with torch.no_grad():
                adv_logits = model(adv_images)
                adv_conf, adv_pred = compute_confidence(adv_logits)

            true_clean_logit = clean_logits.gather(
                1,
                labels.unsqueeze(1)
            ).squeeze(1)

            true_adv_logit = adv_logits.gather(
                1,
                labels.unsqueeze(1)
            ).squeeze(1)

            margin_drop = true_clean_logit - true_adv_logit

            success_mask = clean_correct_mask & adv_pred.ne(labels)

            for i in range(batch_size):
                label_id = labels[i].item()
                clean_pred_id = clean_pred[i].item()
                adv_pred_id = adv_pred[i].item()

                true_name = CIFAR10_CLASSES[label_id]
                clean_pred_name = CIFAR10_CLASSES[clean_pred_id]
                adv_pred_name = CIFAR10_CLASSES[adv_pred_id]

                success = bool(success_mask[i].item())

                if success:
                    class_attack_success[true_name][attack] += 1

                    linf = (adv_images[i] - images[i]).abs().max().item()
                    l2 = (adv_images[i] - images[i]).view(-1).norm(p=2).item()

                    row = {
                        "sample_index": total_seen + i,
                        "attack": attack_display_name(attack),
                        "attack_key": attack,
                        "true_label": true_name,
                        "clean_pred": clean_pred_name,
                        "adv_pred": adv_pred_name,
                        "clean_conf": clean_conf[i].item(),
                        "adv_conf": adv_conf[i].item(),
                        "confidence_drop": clean_conf[i].item() - adv_conf[i].item(),
                        "margin_drop": margin_drop[i].item(),
                        "linf_255": linf * 255.0,
                        "l2": l2,
                        "attack_success": success,
                    }

                    fragile_rows.append(row)

                    visual_candidates[attack].append(
                        {
                            "sample_index": total_seen + i,
                            "attack": attack,
                            "image": images[i].detach().cpu(),
                            "adv_image": adv_images[i].detach().cpu(),
                            "true_label": true_name,
                            "clean_pred": clean_pred_name,
                            "adv_pred": adv_pred_name,
                            "clean_conf": clean_conf[i].item(),
                            "adv_conf": adv_conf[i].item(),
                            "margin_drop": margin_drop[i].item(),
                            "linf_255": linf * 255.0,
                        }
                    )

        total_seen += batch_size

        print(f"已处理样本数: {total_seen}")

        # 如果只是为了可视化，可以提前停止；
        # 但为了类别统计，建议完整跑完 max_samples。
        if args.max_samples is not None and total_seen >= args.max_samples:
            break

    fragile_df = pd.DataFrame(fragile_rows)

    # 每种攻击选 margin_drop 最大的若干样本用于可视化
    selected_visual = {}

    for attack in attacks:
        candidates = visual_candidates[attack]
        candidates = sorted(
            candidates,
            key=lambda item: item["margin_drop"],
            reverse=True,
        )

        selected_visual[attack] = candidates[:args.num_examples_per_attack]

    # 类别统计表
    class_rows = []

    for class_name in CIFAR10_CLASSES:
        clean_count = class_clean_correct[class_name]

        row = {
            "class": class_name,
            "clean_correct_count": clean_count,
        }

        total_success_all_attacks = 0

        for attack in attacks:
            success_count = class_attack_success[class_name][attack]
            total_success_all_attacks += success_count

            row[f"{attack_display_name(attack)}_success_count"] = success_count

            if clean_count > 0:
                row[f"{attack_display_name(attack)}_asr"] = success_count / clean_count * 100.0
            else:
                row[f"{attack_display_name(attack)}_asr"] = 0.0

        row["total_success_count"] = total_success_all_attacks

        if clean_count > 0 and len(attacks) > 0:
            row["mean_asr"] = total_success_all_attacks / (clean_count * len(attacks)) * 100.0
        else:
            row["mean_asr"] = 0.0

        class_rows.append(row)

    class_stats_df = pd.DataFrame(class_rows)

    class_stats_df = class_stats_df.sort_values(
        ["mean_asr", "total_success_count"],
        ascending=[False, False],
    )

    return fragile_df, class_stats_df, selected_visual


def save_adv_diff_examples(selected_visual, output_path, args):
    """
    保存对抗样本差异可视化图。

    布局：
    每种攻击占 num_examples_per_attack 行；
    每行 3 列：
        Clean / Perturbation xN / Adversarial
    """

    attacks = parse_attack_list(args.attacks)

    rows = sum(len(selected_visual.get(attack, [])) for attack in attacks)
    cols = 3

    if rows == 0:
        print("没有可视化样本，跳过 adv_diff_examples.png")
        return

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(11.2, 3.0 * rows)
    )

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        "Adversarial Samples and Magnified Perturbations",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    row_idx = 0

    for attack in attacks:
        examples = selected_visual.get(attack, [])

        for local_idx, item in enumerate(examples):
            clean_img = item["image"]
            adv_img = item["adv_image"]
            delta_vis = make_delta_vis(
                clean_image=clean_img,
                adv_image=adv_img,
                magnify=args.magnify,
            )

            clean_np = tensor_to_numpy_image(clean_img)
            adv_np = tensor_to_numpy_image(adv_img)
            delta_np = tensor_to_numpy_image(delta_vis)

            # Clean
            ax = axes[row_idx, 0]
            ax.imshow(clean_np)
            ax.set_title(
                f"Clean\n"
                f"True: {item['true_label']}\n"
                f"Pred: {item['clean_pred']}",
                fontsize=10,
            )

            # Delta
            ax = axes[row_idx, 1]
            ax.imshow(delta_np)
            ax.set_title(
                f"Perturbation ×{args.magnify:.0f}\n"
                f"L∞ = {item['linf_255']:.1f}/255",
                fontsize=10,
            )

            # Adv
            ax = axes[row_idx, 2]
            ax.imshow(adv_np)
            ax.set_title(
                f"Adversarial\n"
                f"Pred: {item['adv_pred']}\n"
                f"Attack: {attack_display_name(attack)}",
                fontsize=10,
            )

            axes[row_idx, 0].set_ylabel(
                f"{attack_display_name(attack)}\nSample {local_idx + 1}",
                fontsize=11,
                fontweight="bold",
            )

            for col in range(cols):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])

            row_idx += 1

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"对抗样本差异图已保存：{output_path}")


def save_fragile_class_counts(class_stats_df, output_path, args):
    """
    保存各类别被攻击成功次数堆叠柱状图。
    """

    attacks = parse_attack_list(args.attacks)

    df = class_stats_df.copy()

    # 为了图更直观，按 total_success_count 降序
    df = df.sort_values("total_success_count", ascending=False)

    labels = df["class"].tolist()
    x = np.arange(len(labels))

    bottom = np.zeros(len(labels))

    plt.figure(figsize=(10.5, 5.5))

    for attack in attacks:
        col = f"{attack_display_name(attack)}_success_count"

        if col not in df.columns:
            continue

        values = df[col].values

        plt.bar(
            x,
            values,
            bottom=bottom,
            label=attack_display_name(attack),
        )

        bottom += values

    plt.xticks(x, labels, rotation=25)
    plt.ylabel("Successful Attack Count")
    plt.title("Most Fragile Classes: Successful Attacks by Class")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=True)

    for i, total in enumerate(bottom):
        plt.text(
            i,
            total + max(bottom) * 0.01,
            f"{int(total)}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"类别攻击成功次数图已保存：{output_path}")


def save_fragile_class_asr(class_stats_df, output_path, args):
    """
    保存各类别平均 ASR 图。
    """

    df = class_stats_df.copy()
    df = df.sort_values("mean_asr", ascending=False)

    plt.figure(figsize=(9.5, 5.2))

    bars = plt.bar(
        df["class"],
        df["mean_asr"],
    )

    plt.xticks(rotation=25)
    plt.ylabel("Mean ASR (%)")
    plt.title("Most Fragile Classes: Mean Attack Success Rate")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, df["mean_asr"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"类别平均 ASR 图已保存：{output_path}")


def print_top_fragile_classes(class_stats_df, top_k=5):
    print("\n最脆弱类别 Top-K：")
    print("-" * 80)

    show_cols = [
        "class",
        "clean_correct_count",
        "total_success_count",
        "mean_asr",
    ]

    print(
        class_stats_df[show_cols]
        .head(top_k)
        .to_string(index=False)
    )


def print_top_fragile_samples(fragile_df, top_k=10):
    if fragile_df.empty:
        print("\n没有攻击成功样本。")
        return

    print("\n最脆弱样本 Top-K：")
    print("-" * 80)

    show_cols = [
        "sample_index",
        "attack",
        "true_label",
        "clean_pred",
        "adv_pred",
        "margin_drop",
        "linf_255",
    ]

    print(
        fragile_df.sort_values("margin_drop", ascending=False)
        [show_cols]
        .head(top_k)
        .to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="可视化对抗样本与原图差异，并分析模型最脆弱类别"
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

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--deepfool-steps", type=int, default=20)
    parser.add_argument("--spsa-samples", type=int, default=16)

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多分析多少张测试样本，默认完整测试集"
    )

    parser.add_argument(
        "--num-examples-per-attack",
        type=int,
        default=3,
        help="每种攻击可视化几个样本"
    )

    parser.add_argument(
        "--magnify",
        type=float,
        default=12.0,
        help="扰动图放大倍数"
    )

    parser.add_argument("--figure-dir", type=str, default="./figures")
    parser.add_argument("--result-dir", type=str, default="./results")
    parser.add_argument("--prefix", type=str, default="fragile")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    setup_matplotlib()

    ensure_dir(args.figure_dir)
    ensure_dir(args.result_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("可视化对抗样本与原图差异 + 脆弱类别分析")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Attacks: {parse_attack_list(args.attacks)}")
    print(f"eps = {args.eps * 255:.1f}/255")
    print(f"alpha = {args.alpha * 255:.1f}/255")
    print(f"steps = {args.steps}")
    print(f"deepfool_steps = {args.deepfool_steps}")
    print(f"max_samples = {args.max_samples}")
    print(f"num_examples_per_attack = {args.num_examples_per_attack}")
    print(f"magnify = {args.magnify}")

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

    fragile_df, class_stats_df, selected_visual = collect_fragile_samples(
        model=model,
        test_loader=test_loader,
        device=device,
        args=args,
    )

    # 保存 CSV
    fragile_csv = os.path.join(
        args.result_dir,
        f"{args.prefix}_samples.csv"
    )

    class_csv = os.path.join(
        args.result_dir,
        f"{args.prefix}_class_stats.csv"
    )

    fragile_df.to_csv(fragile_csv, index=False, encoding="utf-8-sig")
    class_stats_df.to_csv(class_csv, index=False, encoding="utf-8-sig")

    print(f"\n最脆弱样本 CSV 已保存：{fragile_csv}")
    print(f"类别脆弱性统计 CSV 已保存：{class_csv}")

    print_top_fragile_samples(fragile_df, top_k=10)
    print_top_fragile_classes(class_stats_df, top_k=5)

    # 保存图片
    adv_diff_path = os.path.join(
        args.figure_dir,
        "adv_diff_examples.png"
    )

    fragile_counts_path = os.path.join(
        args.figure_dir,
        "fragile_class_counts.png"
    )

    fragile_asr_path = os.path.join(
        args.figure_dir,
        "fragile_class_asr.png"
    )

    save_adv_diff_examples(
        selected_visual=selected_visual,
        output_path=adv_diff_path,
        args=args,
    )

    save_fragile_class_counts(
        class_stats_df=class_stats_df,
        output_path=fragile_counts_path,
        args=args,
    )

    save_fragile_class_asr(
        class_stats_df=class_stats_df,
        output_path=fragile_asr_path,
        args=args,
    )

    print("\n全部完成。输出文件：")
    print(adv_diff_path)
    print(fragile_counts_path)
    print(fragile_asr_path)
    print(fragile_csv)
    print(class_csv)


if __name__ == "__main__":
    main()