# eval_preprocess_defense.py
# 功能：
# 1. 评估输入预处理防御
# 2. 默认评估两个代表性攻击：
#    - white-box PGD
#    - transfer MI-FGSM
# 3. 输出 Clean Acc、Adv Acc、ASR
#
# 说明：
# 这里属于非自适应防御评估：
# 攻击样本仍由原始模型生成，然后在测试阶段加入输入预处理防御。

import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import load_model
from attacks import generate_adversarial_examples


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PreprocessDefense(nn.Module):
    """
    输入预处理防御包装器
    输入范围默认为 [0, 1]
    """

    def __init__(
        self,
        base_model,
        defense="bit_depth_smooth",
        bit_depth=4,
        noise_sigma=0.03
    ):
        super().__init__()
        self.base_model = base_model
        self.defense = defense
        self.bit_depth = bit_depth
        self.noise_sigma = noise_sigma

    def bit_depth_reduce(self, x):
        """
        位深压缩：
        bit_depth = 4 时，图像被量化到 16 个灰度等级
        """
        levels = 2 ** self.bit_depth
        x = torch.round(x * (levels - 1)) / (levels - 1)
        return torch.clamp(x, 0.0, 1.0)

    def avg_smooth(self, x):
        """
        3x3 平均平滑：
        用于削弱高频对抗扰动
        """
        return F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def gaussian_noise(self, x):
        """
        高斯噪声随机化：
        测试时给输入加入轻微噪声
        """
        noise = torch.randn_like(x) * self.noise_sigma
        return torch.clamp(x + noise, 0.0, 1.0)

    def preprocess(self, x):
        if self.defense == "none":
            return x

        elif self.defense == "bit_depth":
            x = self.bit_depth_reduce(x)
            return x

        elif self.defense == "avg_smooth":
            x = self.avg_smooth(x)
            return x

        elif self.defense == "gaussian_noise":
            x = self.gaussian_noise(x)
            return x

        elif self.defense == "bit_depth_smooth":
            x = self.bit_depth_reduce(x)
            x = self.avg_smooth(x)
            return x

        else:
            raise ValueError(f"未知防御方法: {self.defense}")

    def forward(self, x):
        x = self.preprocess(x)
        return self.base_model(x)


def get_test_loader(data_dir, batch_size, num_workers):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


def build_attack_experiments(attack_set, model_A, model_B):
    """
    attack_set:
    - strong: 只测试最有代表性的强攻击
    - all: 测试所有白盒和迁移攻击
    """

    if attack_set == "strong":
        experiments = [
            {
                "experiment": "white-box",
                "attack_name": "pgd",
                "source_model": model_B,
                "source_model_name": "Model B SimpleCNN",
                "target_model_name": "Model B SimpleCNN",
            },
            {
                "experiment": "transfer",
                "attack_name": "mi-fgsm",
                "source_model": model_A,
                "source_model_name": "Model A ResNet18",
                "target_model_name": "Model B SimpleCNN",
            },
        ]

    elif attack_set == "all":
        experiments = [
            {
                "experiment": "white-box",
                "attack_name": "fgsm",
                "source_model": model_B,
                "source_model_name": "Model B SimpleCNN",
                "target_model_name": "Model B SimpleCNN",
            },
            {
                "experiment": "white-box",
                "attack_name": "pgd",
                "source_model": model_B,
                "source_model_name": "Model B SimpleCNN",
                "target_model_name": "Model B SimpleCNN",
            },
            {
                "experiment": "transfer",
                "attack_name": "fgsm",
                "source_model": model_A,
                "source_model_name": "Model A ResNet18",
                "target_model_name": "Model B SimpleCNN",
            },
            {
                "experiment": "transfer",
                "attack_name": "pgd",
                "source_model": model_A,
                "source_model_name": "Model A ResNet18",
                "target_model_name": "Model B SimpleCNN",
            },
            {
                "experiment": "transfer",
                "attack_name": "mi-fgsm",
                "source_model": model_A,
                "source_model_name": "Model A ResNet18",
                "target_model_name": "Model B SimpleCNN",
            },
        ]

    else:
        raise ValueError(f"未知 attack_set: {attack_set}")

    return experiments


def evaluate_defense(
    defense_name,
    attack_info,
    defended_target_model,
    test_loader,
    device,
    eps,
    alpha,
    steps,
    max_samples=None
):
    attack_name = attack_info["attack_name"]
    source_model = attack_info["source_model"]

    source_model.eval()
    defended_target_model.eval()

    total = 0
    clean_correct = 0
    adv_correct = 0

    clean_correct_total = 0
    attack_success = 0

    desc = f"{defense_name} | {attack_info['experiment']} | {attack_name}"

    pbar = tqdm(test_loader, desc=desc, leave=True)

    for images, labels in pbar:
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

        # 1. 防御模型在干净样本上的结果
        with torch.no_grad():
            clean_outputs = defended_target_model(images)
            clean_preds = clean_outputs.argmax(dim=1)

        # 2. 用原始 source model 生成对抗样本
        adv_images = generate_adversarial_examples(
            attack_name=attack_name,
            source_model=source_model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps
        )

        # 3. 对抗样本经过输入预处理防御后，再送入目标模型
        with torch.no_grad():
            adv_outputs = defended_target_model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)

        clean_correct_mask = clean_preds.eq(labels)
        adv_correct_mask = adv_preds.eq(labels)

        clean_correct += clean_correct_mask.sum().item()
        adv_correct += adv_correct_mask.sum().item()

        clean_correct_total += clean_correct_mask.sum().item()
        attack_success += (clean_correct_mask & (~adv_correct_mask)).sum().item()

        total += batch_size

        clean_acc = 100.0 * clean_correct / total
        adv_acc = 100.0 * adv_correct / total
        asr = 100.0 * attack_success / max(clean_correct_total, 1)

        pbar.set_postfix({
            "CleanAcc": f"{clean_acc:.2f}%",
            "AdvAcc": f"{adv_acc:.2f}%",
            "ASR": f"{asr:.2f}%"
        })

    clean_acc = 100.0 * clean_correct / total
    adv_acc = 100.0 * adv_correct / total
    asr = 100.0 * attack_success / max(clean_correct_total, 1)

    result = {
        "defense": defense_name,
        "experiment": attack_info["experiment"],
        "attack_name": attack_name,
        "source_model": attack_info["source_model_name"],
        "target_model": attack_info["target_model_name"],
        "eps_255": eps * 255,
        "alpha_255": alpha * 255,
        "steps": steps,
        "total_samples": total,
        "clean_acc_defended": clean_acc,
        "adv_acc_defended": adv_acc,
        "asr_defended": asr,
        "clean_correct_total": clean_correct_total,
        "attack_success": attack_success,
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_A", type=str, default="./checkpoints/model_A_resnet18_best.pth")
    parser.add_argument("--checkpoint_B", type=str, default="./checkpoints/model_B_simplecnn_best.pth")
    parser.add_argument("--result_dir", type=str, default="./results")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)

    parser.add_argument(
        "--defenses",
        nargs="+",
        default=["bit_depth_smooth"],
        choices=["none", "bit_depth", "avg_smooth", "gaussian_noise", "bit_depth_smooth"],
        help="可以一次测试多个防御方法"
    )

    parser.add_argument(
        "--attack_set",
        type=str,
        default="strong",
        choices=["strong", "all"],
        help="strong=只测 PGD 白盒和 MI-FGSM 迁移；all=测全部白盒和迁移攻击"
    )

    parser.add_argument("--bit_depth", type=int, default=4)
    parser.add_argument("--noise_sigma", type=float, default=0.03)

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="快速测试可设置为 1000；正式实验不设置，默认全测试集"
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("输入预处理防御实验")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Defenses: {args.defenses}")
    print(f"Attack set: {args.attack_set}")
    print(f"eps = {args.eps * 255:.1f}/255")
    print(f"alpha = {args.alpha * 255:.1f}/255")
    print(f"steps = {args.steps}")
    print(f"bit_depth = {args.bit_depth}")
    print(f"noise_sigma = {args.noise_sigma}")
    print(f"max_samples = {args.max_samples}")

    test_loader = get_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print("\n加载模型...")
    model_A = load_model("resnet18", args.checkpoint_A, device)
    model_B = load_model("simplecnn", args.checkpoint_B, device)
    print("模型加载完成")

    attack_experiments = build_attack_experiments(
        attack_set=args.attack_set,
        model_A=model_A,
        model_B=model_B
    )

    all_results = []

    for defense_name in args.defenses:
        print("\n" + "=" * 100)
        print(f"当前防御方法: {defense_name}")
        print("=" * 100)

        defended_model_B = PreprocessDefense(
            base_model=model_B,
            defense=defense_name,
            bit_depth=args.bit_depth,
            noise_sigma=args.noise_sigma
        ).to(device)

        for attack_info in attack_experiments:
            result = evaluate_defense(
                defense_name=defense_name,
                attack_info=attack_info,
                defended_target_model=defended_model_B,
                test_loader=test_loader,
                device=device,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                max_samples=args.max_samples
            )

            all_results.append(result)

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 120)
    print("输入预处理防御结果汇总")
    print("=" * 120)

    show_cols = [
        "defense",
        "experiment",
        "attack_name",
        "source_model",
        "target_model",
        "total_samples",
        "clean_acc_defended",
        "adv_acc_defended",
        "asr_defended",
    ]

    print(df[show_cols].to_string(index=False))

    save_path = os.path.join(args.result_dir, "defense_preprocess_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n结果已保存到：")
    print(save_path)

    print("\n报告解释建议：")
    print("输入预处理防御通过位深压缩、平滑或随机化削弱输入中的对抗扰动。")
    print("若防御后 Adv Acc 上升、ASR 下降，说明该防御方法对非自适应对抗样本具有一定缓解作用。")


if __name__ == "__main__":
    main()