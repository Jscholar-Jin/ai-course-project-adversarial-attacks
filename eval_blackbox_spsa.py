# eval_blackbox_spsa.py
# 功能：实现 SPSA 黑盒攻击
# 说明：
# 1. SPSA 不使用模型梯度 backward()
# 2. 只通过查询目标模型输出估计梯度
# 3. 作为黑盒攻击 baseline

import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import load_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clamp(x, min_value=0.0, max_value=1.0):
    return torch.clamp(x, min_value, max_value)


@torch.no_grad()
def spsa_attack(
    model,
    images,
    labels,
    eps=8 / 255,
    alpha=2 / 255,
    steps=10,
    delta=0.01,
    spsa_samples=16
):
    """
    SPSA 黑盒攻击

    参数说明：
    eps: 最大扰动范围，默认 8/255
    alpha: 每一步更新步长，默认 2/255
    steps: 迭代步数
    delta: SPSA 有限差分扰动大小
    spsa_samples: 每一步用于估计梯度的随机方向数量

    注意：
    这里没有使用 loss.backward()
    梯度是通过查询 model(x + delta*u) 和 model(x - delta*u) 估计出来的
    """

    model.eval()

    ori_images = images.clone().detach()
    labels = labels.clone().detach()

    adv_images = ori_images.clone().detach()

    batch_size = images.size(0)

    for _ in range(steps):
        grad_est = torch.zeros_like(adv_images)

        for _ in range(spsa_samples):
            # Rademacher 随机方向，取值为 {-1, +1}
            u = torch.empty_like(adv_images).bernoulli_(0.5) * 2 - 1

            x_plus = clamp(adv_images + delta * u, 0.0, 1.0)
            x_minus = clamp(adv_images - delta * u, 0.0, 1.0)

            logits_plus = model(x_plus)
            logits_minus = model(x_minus)

            loss_plus = F.cross_entropy(
                logits_plus,
                labels,
                reduction="none"
            )

            loss_minus = F.cross_entropy(
                logits_minus,
                labels,
                reduction="none"
            )

            # 每个样本各自估计梯度
            diff = (loss_plus - loss_minus).view(batch_size, 1, 1, 1)
            grad_est += diff * u / (2.0 * delta)

        grad_est = grad_est / float(spsa_samples)

        # 非定向攻击：增大分类损失
        adv_images = adv_images + alpha * grad_est.sign()

        # 投影回 eps 范围内
        perturb = torch.clamp(
            adv_images - ori_images,
            min=-eps,
            max=eps
        )

        adv_images = clamp(ori_images + perturb, 0.0, 1.0)

    return adv_images.detach()


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


def evaluate_spsa(
    target_model,
    test_loader,
    device,
    eps,
    alpha,
    steps,
    delta,
    spsa_samples,
    max_samples
):
    target_model.eval()

    total = 0

    clean_correct = 0
    adv_correct = 0

    clean_correct_total = 0
    attack_success = 0

    pbar = tqdm(test_loader, desc="Black-box SPSA", leave=True)

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

        with torch.no_grad():
            clean_outputs = target_model(images)
            clean_preds = clean_outputs.argmax(dim=1)

        adv_images = spsa_attack(
            model=target_model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps,
            delta=delta,
            spsa_samples=spsa_samples
        )

        with torch.no_grad():
            adv_outputs = target_model(adv_images)
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

    # 每个样本理论查询次数
    queries_per_sample = 2 * spsa_samples * steps

    result = {
        "experiment": "black-box",
        "attack_name": "spsa",
        "source_model": "Query Model B",
        "target_model": "Model B SimpleCNN",
        "eps": eps,
        "eps_255": eps * 255,
        "alpha": alpha,
        "alpha_255": alpha * 255,
        "steps": steps,
        "delta": delta,
        "spsa_samples": spsa_samples,
        "queries_per_sample": queries_per_sample,
        "total_samples": total,
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "asr": asr,
        "clean_correct_total": clean_correct_total,
        "attack_success": attack_success
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_B", type=str, default="./checkpoints/model_B_simplecnn_best.pth")
    parser.add_argument("--result_dir", type=str, default="./results")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)

    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--spsa_samples", type=int, default=16)

    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("CIFAR-10 黑盒攻击实验：SPSA")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Target checkpoint: {args.checkpoint_B}")
    print(f"eps = {args.eps:.6f} ≈ {args.eps * 255:.1f}/255")
    print(f"alpha = {args.alpha:.6f} ≈ {args.alpha * 255:.1f}/255")
    print(f"steps = {args.steps}")
    print(f"delta = {args.delta}")
    print(f"spsa_samples = {args.spsa_samples}")
    print(f"queries_per_sample = {2 * args.spsa_samples * args.steps}")
    print(f"max_samples = {args.max_samples}")

    test_loader = get_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print("\n加载目标模型 Model B...")
    model_B = load_model("simplecnn", args.checkpoint_B, device)
    print("模型加载完成")

    result = evaluate_spsa(
        target_model=model_B,
        test_loader=test_loader,
        device=device,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        delta=args.delta,
        spsa_samples=args.spsa_samples,
        max_samples=args.max_samples
    )

    df = pd.DataFrame([result])

    print("\n" + "=" * 100)
    print("黑盒攻击 SPSA 结果")
    print("=" * 100)

    show_cols = [
        "experiment",
        "attack_name",
        "source_model",
        "target_model",
        "eps_255",
        "steps",
        "spsa_samples",
        "queries_per_sample",
        "clean_acc",
        "adv_acc",
        "asr"
    ]

    print(df[show_cols].to_string(index=False))

    save_path = os.path.join(args.result_dir, "attack_results_blackbox_spsa.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n结果已保存到：")
    print(save_path)


if __name__ == "__main__":
    main()