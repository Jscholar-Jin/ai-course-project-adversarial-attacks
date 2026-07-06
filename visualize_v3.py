# visual_4rows_attacks.py
# 功能：
# 生成 4 行攻击效果展示图：
# Clean / FGSM / PGD / DeepFool
#
# 特点：
# 1. 去掉 Delta 扰动图；
# 2. 只保留 4 行；
# 3. 不显示每张小图的小标题；
# 4. 每一列对应同一张原始图像；
# 5. 适合直接放进 PPT 展示。
#
# 输出：
# figures/vis_4rows.png

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from attacks import make_adv
from data_utils import build_cifar10_loaders
from models import load_model


ATTACKS = ("fgsm", "pgd", "deepfool")

ATTACK_LABELS = {
    "clean": "Clean",
    "fgsm": "FGSM",
    "pgd": "PGD",
    "deepfool": "DeepFool",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 12,
            "figure.titlesize": 18,
            "axes.unicode_minus": False,
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
        }
    )


def tensor_to_numpy_image(x):
    """
    Tensor [3, H, W] -> numpy [H, W, 3]
    """
    x = x.detach().cpu()
    x = x.permute(1, 2, 0).numpy()
    x = np.clip(x, 0.0, 1.0)
    return x


def collect_common_examples(model, loader, device, args):
    """
    收集 num_examples 个样本。

    每个样本都保存：
    1. Clean 原图；
    2. FGSM 对抗样本；
    3. PGD 对抗样本；
    4. DeepFool 对抗样本。

    默认条件：
    - 原始图像必须被模型分类正确；
    - 至少有一种攻击成功。

    如果开启 --require-all-success：
    - 三种攻击都必须成功。
    """

    model.eval()

    selected = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_logits = model(images)
            clean_pred = clean_logits.argmax(dim=1)

        clean_correct = clean_pred.eq(labels)

        # 一次性生成三种攻击结果
        adv_dict = {}
        adv_pred_dict = {}

        for attack in ATTACKS:
            adv_images = make_adv(
                attack,
                model,
                images,
                labels,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                max_deepfool_steps=args.deepfool_steps,
            )

            with torch.no_grad():
                adv_logits = model(adv_images)
                adv_pred = adv_logits.argmax(dim=1)

            adv_dict[attack] = adv_images.detach()
            adv_pred_dict[attack] = adv_pred.detach()

        for idx in range(images.size(0)):
            if len(selected) >= args.num_examples:
                break

            # 原图必须分类正确
            if not clean_correct[idx].item():
                continue

            label_id = labels[idx].item()

            attack_success = {}

            for attack in ATTACKS:
                attack_success[attack] = adv_pred_dict[attack][idx].item() != label_id

            if args.require_all_success:
                keep_sample = all(attack_success.values())
            else:
                keep_sample = any(attack_success.values())

            if not keep_sample:
                continue

            item = {
                "clean": images[idx].detach().cpu(),
                "fgsm": adv_dict["fgsm"][idx].detach().cpu(),
                "pgd": adv_dict["pgd"][idx].detach().cpu(),
                "deepfool": adv_dict["deepfool"][idx].detach().cpu(),
                "label": label_id,
                "clean_pred": clean_pred[idx].item(),
                "fgsm_pred": adv_pred_dict["fgsm"][idx].item(),
                "pgd_pred": adv_pred_dict["pgd"][idx].item(),
                "deepfool_pred": adv_pred_dict["deepfool"][idx].item(),
                "fgsm_success": attack_success["fgsm"],
                "pgd_success": attack_success["pgd"],
                "deepfool_success": attack_success["deepfool"],
            }

            selected.append(item)

        if len(selected) >= args.num_examples:
            break

    return selected


def save_4row_figure(samples, output_path, args):
    """
    保存 4 行图：

    第 1 行：Clean
    第 2 行：FGSM
    第 3 行：PGD
    第 4 行：DeepFool

    不显示每张小图标题。
    """

    if len(samples) == 0:
        print("没有找到可视化样本，无法保存图片。")
        return

    rows = 4
    cols = len(samples)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.6 * cols, 8.2)
    )

    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    row_keys = ["clean", "fgsm", "pgd", "deepfool"]

    for row_idx, key in enumerate(row_keys):
        for col_idx, sample in enumerate(samples):
            ax = axes[row_idx, col_idx]

            image_np = tensor_to_numpy_image(sample[key])
            ax.imshow(image_np)

            ax.set_xticks([])
            ax.set_yticks([])

            # 去掉每个小图的小标题
            ax.set_title("")

            # 只在每一行最左边显示行标签
            if col_idx == 0:
                ax.set_ylabel(
                    ATTACK_LABELS[key],
                    fontsize=15,
                    fontweight="bold",
                    rotation=90,
                    labelpad=18
                )

            # 去掉边框，让图更干净
            for spine in ax.spines.values():
                spine.set_visible(False)

    # 总标题也可以不要。
    # 如果你想要总标题，把下面三行取消注释即可。
    # fig.suptitle(
    #     "Clean Images and Adversarial Examples",
    #     fontsize=18,
    #     fontweight="bold",
    # )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.98,
        bottom=0.03,
        wspace=0.08,
        hspace=0.08
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"\n4 行攻击展示图已保存：{output_path}")


def print_sample_info(samples):
    print("\n样本预测信息：")

    for i, sample in enumerate(samples, start=1):
        print("-" * 60)
        print(f"样本 {i}")
        print(f"真实类别 ID: {sample['label']}")
        print(f"Clean 预测 ID: {sample['clean_pred']}")
        print(
            f"FGSM 预测 ID: {sample['fgsm_pred']} | "
            f"攻击成功: {sample['fgsm_success']}"
        )
        print(
            f"PGD 预测 ID: {sample['pgd_pred']} | "
            f"攻击成功: {sample['pgd_success']}"
        )
        print(
            f"DeepFool 预测 ID: {sample['deepfool_pred']} | "
            f"攻击成功: {sample['deepfool_success']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成 Clean / FGSM / PGD / DeepFool 四行展示图"
    )

    parser.add_argument(
        "--data-dir",
        default="./data",
        help="CIFAR-10 数据集路径"
    )

    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cnn.pt",
        help="模型权重路径"
    )

    parser.add_argument(
        "--model",
        default="cnn",
        choices=["cnn", "resnet"],
        help="模型名称，需要和 models.py 中 load_model 支持的名称一致"
    )

    parser.add_argument(
        "--figure-dir",
        default="./figures",
        help="输出图片文件夹"
    )

    parser.add_argument(
        "--output-name",
        default="vis_4rows.png",
        help="输出图片文件名"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
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
        help="PGD 每步步长，默认 2/255"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="PGD 迭代次数"
    )

    parser.add_argument(
        "--deepfool-steps",
        type=int,
        default=20,
        help="DeepFool 最大迭代次数"
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=4,
        help="展示几个样本，默认 4"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="最多搜索多少个测试样本"
    )

    parser.add_argument(
        "--require-all-success",
        action="store_true",
        help="要求 FGSM / PGD / DeepFool 三种攻击都成功"
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="如果本地没有 CIFAR-10，是否自动下载"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def main():
    args = parse_args()

    setup_style()
    set_seed(args.seed)

    os.makedirs(args.figure_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("生成 Clean / FGSM / PGD / DeepFool 四行展示图")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"eps = {args.eps * 255:.1f}/255")
    print(f"alpha = {args.alpha * 255:.1f}/255")
    print(f"steps = {args.steps}")
    print(f"deepfool_steps = {args.deepfool_steps}")
    print(f"展示样本数: {args.num_examples}")
    print(f"最多搜索样本数: {args.max_samples}")
    print(f"是否要求三种攻击都成功: {args.require_all_success}")

    print("\n加载模型...")
    model = load_model(
        args.model,
        args.checkpoint,
        device
    )
    print("模型加载完成")

    print("\n加载 CIFAR-10 测试集...")
    _, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.download,
        test_samples=args.max_samples,
    )
    print("测试集加载完成")

    print("\n搜索可视化样本...")
    samples = collect_common_examples(
        model=model,
        loader=test_loader,
        device=device,
        args=args
    )

    if len(samples) == 0:
        print("\n没有找到满足条件的样本。")
        print("可以尝试：")
        print("1. 去掉 --require-all-success")
        print("2. 增大 --max-samples")
        print("3. 增大 --eps，例如 --eps 0.062745")
        return

    print(f"\n成功收集 {len(samples)} 个样本。")
    print_sample_info(samples)

    output_path = os.path.join(args.figure_dir, args.output_name)

    print("\n保存图片...")
    save_4row_figure(
        samples=samples,
        output_path=output_path,
        args=args
    )

    print("\n全部完成。")


if __name__ == "__main__":
    main()