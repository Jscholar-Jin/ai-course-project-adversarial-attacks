import argparse
import os

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from attacks import make_adv
from models import load_model


def test_loader(data_dir, batch_size, workers, download):
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transforms.ToTensor(),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def score_attack(name, source, target, loader, args, device):
    source.eval()
    target.eval()

    total = 0
    clean_ok = 0
    adv_ok = 0
    attack_ok = 0
    clean_base = 0

    for images, labels in tqdm(loader, desc=name):
        images = images.to(device)
        labels = labels.to(device)

        if args.max_samples and total >= args.max_samples:
            break
        if args.max_samples and total + images.size(0) > args.max_samples:
            keep = args.max_samples - total
            images = images[:keep]
            labels = labels[:keep]

        with torch.no_grad():
            clean_pred = target(images).argmax(1)

        adv_images = make_adv(
            name,
            source,
            images,
            labels,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
        )

        with torch.no_grad():
            adv_pred = target(adv_images).argmax(1)

        clean_mask = clean_pred.eq(labels)
        adv_mask = adv_pred.eq(labels)
        batch = labels.size(0)

        total += batch
        clean_ok += clean_mask.sum().item()
        adv_ok += adv_mask.sum().item()
        clean_base += clean_mask.sum().item()
        attack_ok += (clean_mask & ~adv_mask).sum().item()

    return {
        "attack": name,
        "eps": args.eps,
        "eps_255": args.eps * 255,
        "alpha": args.alpha,
        "alpha_255": args.alpha * 255,
        "steps": args.steps,
        "samples": total,
        "clean_acc": 100.0 * clean_ok / total,
        "adv_acc": 100.0 * adv_ok / total,
        "asr": 100.0 * attack_ok / max(clean_base, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--resnet", default="./checkpoints/resnet.pt")
    parser.add_argument("--cnn", default="./checkpoints/cnn.pt")
    parser.add_argument("--result-dir", default="./results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet = load_model("resnet", args.resnet, device)
    cnn = load_model("cnn", args.cnn, device)
    loader = test_loader(args.data_dir, args.batch_size, args.workers, args.download)

    jobs = [
        ("white", "fgsm", cnn, cnn),
        ("white", "pgd", cnn, cnn),
        ("transfer", "fgsm", resnet, cnn),
        ("transfer", "pgd", resnet, cnn),
        ("transfer", "mifgsm", resnet, cnn),
    ]

    rows = []
    for kind, attack, source, target in jobs:
        row = score_attack(attack, source, target, loader, args, device)
        row["type"] = kind
        row["source"] = "cnn" if source is cnn else "resnet"
        row["target"] = "cnn"
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(args.result_dir, "attacks.csv")
    df.to_csv(path, index=False)
    print(df[["type", "attack", "source", "target", "clean_acc", "adv_acc", "asr"]])
    print(f"saved {path}")


if __name__ == "__main__":
    main()
