import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import torch

from attacks import make_adv
from data_utils import CIFAR10_CLASSES, build_cifar10_loaders
from models import load_model


ATTACKS = ("fgsm", "pgd", "deepfool")
ATTACK_LABELS = {"fgsm": "FGSM", "pgd": "PGD", "deepfool": "DeepFool"}
ATTACK_COLORS = {"fgsm": "#4C78A8", "pgd": "#F58518", "deepfool": "#54A24B"}


def style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.titlesize": 16,
        }
    )


def attack_title(name: str) -> str:
    return ATTACK_LABELS.get(name, name.upper())


def collect_examples(model, loader, device, args):
    model.eval()
    selected = {attack: [] for attack in ATTACKS}
    feature_store = []
    feature_labels = []
    feature_kinds = []
    fragile_rows = []
    fallback_batch = None

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_logits = model(images)
            clean_pred = clean_logits.argmax(1)
            clean_feat = model.forward_features(images)

        if fallback_batch is None:
            fallback_batch = (images.detach().cpu(), labels.detach().cpu(), clean_pred.detach().cpu())

        feature_store.append(clean_feat.cpu())
        feature_labels.extend(labels.cpu().tolist())
        feature_kinds.extend(["clean"] * labels.size(0))

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
                adv_pred = adv_logits.argmax(1)
                adv_feat = model.forward_features(adv_images)

            feature_store.append(adv_feat.cpu())
            feature_labels.extend(labels.cpu().tolist())
            feature_kinds.extend([attack] * labels.size(0))

            margin_drop = clean_logits.gather(1, labels.unsqueeze(1)).squeeze(1) - adv_logits.gather(
                1, labels.unsqueeze(1)
            ).squeeze(1)
            fool_mask = clean_pred.eq(labels) & adv_pred.ne(labels)

            for idx in torch.where(fool_mask)[0].tolist():
                selected[attack].append(
                    {
                        "attack": attack,
                        "image": images[idx].cpu(),
                        "adv_image": adv_images[idx].cpu(),
                        "label": labels[idx].item(),
                        "clean_pred": clean_pred[idx].item(),
                        "adv_pred": adv_pred[idx].item(),
                        "margin_drop": margin_drop[idx].item(),
                        "linf": (adv_images[idx] - images[idx]).abs().max().item(),
                    }
                )
                fragile_rows.append(
                    {
                        "attack": attack_title(attack),
                        "label": CIFAR10_CLASSES[labels[idx].item()],
                        "adv_pred": CIFAR10_CLASSES[adv_pred[idx].item()],
                        "margin_drop": margin_drop[idx].item(),
                        "linf": (adv_images[idx] - images[idx]).abs().max().item(),
                    }
                )

        enough = all(len(selected[attack]) >= args.num_examples for attack in ATTACKS)
        if enough and len(feature_labels) >= args.tsne_samples:
            break

    for attack in ATTACKS:
        selected[attack].sort(key=lambda item: item["margin_drop"], reverse=True)
        if len(selected[attack]) >= args.num_examples:
            selected[attack] = selected[attack][: args.num_examples]
            continue

        if fallback_batch is None:
            continue
        images, labels, clean_pred = fallback_batch
        need = args.num_examples - len(selected[attack])
        for idx in range(min(need, images.size(0))):
            adv_image = make_adv(
                attack,
                model,
                images[idx : idx + 1].to(device),
                labels[idx : idx + 1].to(device),
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                max_deepfool_steps=args.deepfool_steps,
            )[0].cpu()
            with torch.no_grad():
                adv_pred = model(adv_image.unsqueeze(0).to(device)).argmax(1).item()
            selected[attack].append(
                {
                    "attack": attack,
                    "image": images[idx],
                    "adv_image": adv_image,
                    "label": labels[idx].item(),
                    "clean_pred": clean_pred[idx].item(),
                    "adv_pred": adv_pred,
                    "margin_drop": 0.0,
                    "linf": (adv_image - images[idx]).abs().max().item(),
                }
            )

    feature_tensor = torch.cat(feature_store, dim=0)[: args.tsne_samples]
    feature_labels = feature_labels[: args.tsne_samples]
    feature_kinds = feature_kinds[: args.tsne_samples]
    return selected, feature_tensor, feature_labels, feature_kinds, fragile_rows


def save_examples(selected, output_path, args):
    rows = len(ATTACKS)
    cols = min(args.num_examples, max(len(items) for items in selected.values()))
    fig, axes = plt.subplots(rows * 3, cols, figsize=(2.7 * cols, 2.6 * rows * 3))
    if cols == 1:
        axes = axes.reshape(rows * 3, 1)

    fig.suptitle(
        f"Adversarial Examples on CIFAR-10 | model={args.model.upper()} | eps={args.eps * 255:.0f}/255 | steps={args.steps}",
        y=0.995,
    )

    for row_idx, attack in enumerate(ATTACKS):
        examples = selected[attack]
        for col_idx in range(cols):
            clean_ax = axes[row_idx * 3 + 0, col_idx]
            adv_ax = axes[row_idx * 3 + 1, col_idx]
            delta_ax = axes[row_idx * 3 + 2, col_idx]

            if col_idx >= len(examples):
                clean_ax.axis("off")
                adv_ax.axis("off")
                delta_ax.axis("off")
                continue

            example = examples[col_idx]
            image = example["image"].permute(1, 2, 0).numpy()
            adv_image = example["adv_image"].permute(1, 2, 0).numpy()
            delta = adv_image - image
            delta_vis = (delta * args.magnify + 0.5).clip(0.0, 1.0)

            clean_ax.imshow(image)
            adv_ax.imshow(adv_image)
            delta_ax.imshow(delta_vis)

            clean_ax.set_title(
                f"{attack_title(attack)} sample {col_idx + 1}\ntrue={CIFAR10_CLASSES[example['label']]} | clean={CIFAR10_CLASSES[example['clean_pred']]}",
                fontsize=9,
            )
            adv_ax.set_title(
                f"adv={CIFAR10_CLASSES[example['adv_pred']]} | Linf={example['linf'] * 255:.1f}/255",
                fontsize=9,
                color=ATTACK_COLORS[attack],
            )
            delta_ax.set_title(f"perturbation x{args.magnify:.0f}", fontsize=9)

            if col_idx == 0:
                clean_ax.set_ylabel("Clean", fontsize=11)
                adv_ax.set_ylabel(attack_title(attack), fontsize=11)
                delta_ax.set_ylabel("Delta", fontsize=11)

            for ax in (clean_ax, adv_ax, delta_ax):
                ax.set_xticks([])
                ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_tsne(features, labels, kinds, output_path):
    perplexity = max(5, min(30, features.size(0) - 1))
    embed = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(features.numpy())

    color_map = {
        "clean": "#222222",
        "fgsm": ATTACK_COLORS["fgsm"],
        "pgd": ATTACK_COLORS["pgd"],
        "deepfool": ATTACK_COLORS["deepfool"],
    }
    marker_map = {"clean": "o", "fgsm": "^", "pgd": "s", "deepfool": "x"}

    frame = pd.DataFrame({"x": embed[:, 0], "y": embed[:, 1], "kind": kinds, "label": labels})
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for kind in ["clean", "fgsm", "pgd", "deepfool"]:
        group = frame[frame["kind"] == kind]
        if group.empty:
            continue
        ax.scatter(
            group["x"],
            group["y"],
            s=16,
            c=color_map[kind],
            marker=marker_map[kind],
            alpha=0.65,
            label="Clean" if kind == "clean" else attack_title(kind),
        )

    ax.set_title("Feature Space Shift: Clean vs Adversarial Samples")
    ax.text(
        0.02,
        0.02,
        "Adversarial samples occupy shifted regions in representation space.",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.legend(loc="upper right", frameon=True)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_fragile(frame: pd.DataFrame, output_csv: str, output_png: str):
    stats = (
        frame.groupby("label")
        .agg(
            fooled_count=("label", "size"),
            mean_margin_drop=("margin_drop", "mean"),
            mean_linf=("linf", "mean"),
        )
        .sort_values(["fooled_count", "mean_margin_drop"], ascending=[False, False])
    )
    stats.to_csv(output_csv)

    attack_breakdown = (
        frame.groupby(["label", "attack"])
        .size()
        .unstack(fill_value=0)
        .reindex(stats.index)
        .fillna(0)
    )
    top_labels = stats.head(8).index
    attack_breakdown = attack_breakdown.loc[top_labels]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    bottom = None
    for attack in ["FGSM", "PGD", "DeepFool"]:
        values = attack_breakdown[attack] if attack in attack_breakdown.columns else 0
        ax.bar(
            attack_breakdown.index,
            values,
            bottom=bottom,
            label=attack,
            color=ATTACK_COLORS[attack.lower()],
        )
        bottom = values if bottom is None else bottom + values

    totals = attack_breakdown.sum(axis=1)
    for idx, total in enumerate(totals):
        ax.text(idx, total + 0.5, f"{int(total)}", ha="center", fontsize=9)
    ax.set_ylabel("Successful Attacks Count")
    ax.set_title("Most Fragile Classes Under Adversarial Attack")
    ax.text(
        0.5,
        0.02,
        "Stack colors show which attack contributes most to each fragile class.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=True)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/cnn.pt")
    parser.add_argument("--model", default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--figure-dir", default="./figures")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--deepfool-steps", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--magnify", type=float, default=12.0)
    parser.add_argument("--max-samples", type=int, default=384)
    parser.add_argument("--tsne-samples", type=int, default=384)
    parser.add_argument("--prefix", default="vis")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    style()
    args = parse_args()
    os.makedirs(args.figure_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.checkpoint, device)
    _, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.download,
        test_samples=args.max_samples,
    )

    selected, features, labels, kinds, fragile_rows = collect_examples(model, test_loader, device, args)
    save_examples(selected, os.path.join(args.figure_dir, f"{args.prefix}_examples.png"), args)
    save_tsne(features, labels, kinds, os.path.join(args.figure_dir, f"{args.prefix}_tsne.png"))

    if fragile_rows:
        frame = pd.DataFrame(fragile_rows)
        save_fragile(
            frame,
            os.path.join(args.figure_dir, f"{args.prefix}_fragile.csv"),
            os.path.join(args.figure_dir, f"{args.prefix}_fragile.png"),
        )


if __name__ == "__main__":
    main()
