import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from attacks import make_adv
from data_utils import build_cifar10_loaders, seed_all
from models import get_model


def train_epoch(model, loader, loss_fn, optimizer, device, args):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if args.training == "adv":
            model.eval()
            adv_images = make_adv(
                args.attack,
                model,
                images,
                labels,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                max_deepfool_steps=args.deepfool_steps,
            )
            model.train()
        else:
            adv_images = images

        optimizer.zero_grad(set_to_none=True)
        if args.training == "adv":
            clean_logits = model(images)
            adv_logits = model(adv_images)
            clean_loss = loss_fn(clean_logits, labels)
            adv_loss = loss_fn(adv_logits, labels)
            loss = args.clean_weight * clean_loss + args.adv_weight * adv_loss
            logits = adv_logits
        else:
            logits = model(adv_images)
            loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        size = labels.size(0)
        loss_sum += loss.item() * size
        correct += logits.argmax(1).eq(labels).sum().item()
        total += size

    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def test(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="test", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        size = labels.size(0)
        loss_sum += loss.item() * size
        correct += logits.argmax(1).eq(labels).sum().item()
        total += size

    return loss_sum / total, 100.0 * correct / total


def fit(name, train_loader, test_loader, args, device):
    model = get_model(name).to(device)
    if args.training == "adv" and args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = []
    suffix = "" if args.training == "standard" else f"_{args.training}"
    ckpt_path = os.path.join(args.checkpoint_dir, f"{name}{suffix}.pt")
    log_path = os.path.join(args.log_dir, f"{name}{suffix}.csv")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device, args
        )
        test_loss, test_acc = test(model, test_loader, loss_fn, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "training": args.training,
            "attack": args.attack if args.training == "adv" else "none",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"{name} {args.training} epoch {epoch}/{args.epochs} "
            f"train={train_acc:.2f}% test={test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "name": name,
                    "training": args.training,
                    "attack": args.attack if args.training == "adv" else "none",
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "acc": test_acc,
                    "args": vars(args),
                },
                ckpt_path,
            )

    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"{name} {args.training} best={best_acc:.2f}% checkpoint={ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--model", default="both", choices=["resnet", "cnn", "both"])
    parser.add_argument(
        "--training",
        default="standard",
        choices=["standard", "adv"],
    )
    parser.add_argument("--attack", default="pgd", choices=["fgsm", "pgd", "deepfool"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--deepfool-steps", type=int, default=15)
    parser.add_argument("--init-checkpoint", default="./checkpoints/cnn.pt")
    parser.add_argument("--clean-weight", type=float, default=0.5)
    parser.add_argument("--adv-weight", type=float, default=0.5)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.download,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )

    names = ["resnet", "cnn"] if args.model == "both" else [args.model]
    for name in names:
        fit(name, train_loader, test_loader, args, device)


if __name__ == "__main__":
    main()
