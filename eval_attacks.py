# eval_attacks.py
# 功能：
# 1. 评估白盒攻击：Model B -> Model B
# 2. 评估迁移攻击：Model A -> Model B
# 3. 输出 Clean Acc、Adv Acc、ASR，并保存 CSV 结果

import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

from models import load_model
from attacks import generate_adversarial_examples


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


@torch.no_grad()
def predict(model, images):
    outputs = model(images)
    preds = outputs.argmax(dim=1)
    return preds


def evaluate_attack(
    attack_name,
    source_model,
    target_model,
    test_loader,
    device,
    eps,
    alpha,
    steps,
    max_samples=None
):
    """
    source_model：生成对抗样本的模型
    target_model：被攻击、被评估的模型

    指标说明：
    Clean Acc：目标模型在干净样本上的准确率
    Adv Acc：目标模型在对抗样本上的准确率
    ASR：Attack Success Rate
         在原本被目标模型分类正确的样本中，被攻击后分类错误的比例
    """

    source_model.eval()
    target_model.eval()

    total = 0

    clean_correct = 0
    adv_correct = 0

    attack_success = 0
    clean_correct_total = 0

    pbar = tqdm(test_loader, desc=f"Evaluating {attack_name}", leave=True)

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

        # 1. 目标模型在干净样本上的预测
        with torch.no_grad():
            clean_outputs = target_model(images)
            clean_preds = clean_outputs.argmax(dim=1)

        # 2. 用 source_model 生成对抗样本
        adv_images = generate_adversarial_examples(
            attack_name=attack_name,
            source_model=source_model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps
        )

        # 3. 目标模型在对抗样本上的预测
        with torch.no_grad():
            adv_outputs = target_model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)

        clean_correct_mask = clean_preds.eq(labels)
        adv_correct_mask = adv_preds.eq(labels)

        clean_correct += clean_correct_mask.sum().item()
        adv_correct += adv_correct_mask.sum().item()

        # ASR 只统计原本分类正确的样本
        attack_success += (clean_correct_mask & (~adv_correct_mask)).sum().item()
        clean_correct_total += clean_correct_mask.sum().item()

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
        "attack_name": attack_name,
        "eps": eps,
        "eps_255": eps * 255,
        "alpha": alpha,
        "alpha_255": alpha * 255,
        "steps": steps,
        "total_samples": total,
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "asr": asr,
        "clean_correct_total": clean_correct_total,
        "attack_success": attack_success
    }

    return result


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


def print_result_table(results):
    print("\n" + "=" * 100)
    print("攻击实验结果汇总")
    print("=" * 100)

    df = pd.DataFrame(results)

    show_cols = [
        "experiment",
        "attack_name",
        "source_model",
        "target_model",
        "eps_255",
        "steps",
        "clean_acc",
        "adv_acc",
        "asr"
    ]

    print(df[show_cols].to_string(index=False))

    print("\n说明：")
    print("Clean Acc：目标模型在干净测试集上的准确率")
    print("Adv Acc：目标模型在对抗样本上的准确率")
    print("ASR：原本分类正确的样本中，被攻击后分类错误的比例")


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
        "--max_samples",
        type=int,
        default=None,
        help="快速测试时可设置为 1000；正式实验建议不设置，使用全部 10000 张测试图"
    )

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("CIFAR-10 对抗攻击实验：白盒攻击 + 迁移攻击")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Model A checkpoint: {args.checkpoint_A}")
    print(f"Model B checkpoint: {args.checkpoint_B}")
    print(f"eps = {args.eps:.6f} ≈ {args.eps * 255:.1f}/255")
    print(f"alpha = {args.alpha:.6f} ≈ {args.alpha * 255:.1f}/255")
    print(f"steps = {args.steps}")
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

    results = []

    # ==========================================================
    # 实验 1：白盒攻击
    # source = Model B, target = Model B
    # ==========================================================
    whitebox_attacks = ["fgsm", "pgd"]

    for attack_name in whitebox_attacks:
        print("\n" + "-" * 100)
        print(f"白盒攻击：{attack_name.upper()} | source = Model B | target = Model B")
        print("-" * 100)

        result = evaluate_attack(
            attack_name=attack_name,
            source_model=model_B,
            target_model=model_B,
            test_loader=test_loader,
            device=device,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            max_samples=args.max_samples
        )

        result["experiment"] = "white-box"
        result["source_model"] = "Model B SimpleCNN"
        result["target_model"] = "Model B SimpleCNN"

        results.append(result)

    # ==========================================================
    # 实验 2：迁移攻击
    # source = Model A, target = Model B
    # ==========================================================
    transfer_attacks = ["fgsm", "pgd", "mi-fgsm"]

    for attack_name in transfer_attacks:
        print("\n" + "-" * 100)
        print(f"迁移攻击：{attack_name.upper()} | source = Model A | target = Model B")
        print("-" * 100)

        result = evaluate_attack(
            attack_name=attack_name,
            source_model=model_A,
            target_model=model_B,
            test_loader=test_loader,
            device=device,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
            max_samples=args.max_samples
        )

        result["experiment"] = "transfer"
        result["source_model"] = "Model A ResNet18"
        result["target_model"] = "Model B SimpleCNN"

        results.append(result)

    print_result_table(results)

    save_path = os.path.join(args.result_dir, "attack_results_whitebox_transfer.csv")
    pd.DataFrame(results).to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n结果已保存到：")
    print(save_path)


if __name__ == "__main__":
    main()