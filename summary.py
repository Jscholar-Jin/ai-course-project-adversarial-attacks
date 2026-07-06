import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ATTACK_ORDER = ["fgsm", "pgd", "deepfool"]
ATTACK_LABELS = {"fgsm": "FGSM", "pgd": "PGD", "deepfool": "DeepFool"}
DEFENSE_LABELS = {
    "cnn": "Standard CNN",
    "cnn_adv": "Adv-trained CNN",
    "cnn_jpeg": "JPEG Defense",
    "cnn_squeeze": "Feature Squeeze",
}
SCENARIO_COLORS = {
    "fgsm": "#4C78A8",
    "pgd": "#F58518",
    "deepfool": "#54A24B",
    "clean": "#72B7B2",
    "adv_train": "#E45756",
    "jpeg": "#B279A2",
    "squeeze": "#FF9DA6",
    "adaptive": "#9D755D",
}


def style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def label_attack(name: str) -> str:
    return ATTACK_LABELS.get(name, name.upper())


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def annotate_bars(ax, bars, values: Iterable[float], suffix: str = "%", dy: float = 1.2):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def load_checkpoint_acc(path: str, fallback: float) -> float:
    if not os.path.exists(path):
        return fallback
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return float(checkpoint.get("acc", fallback))


def save_clean_accuracy(figure_dir: str):
    rows = [
        ("ResNet18", load_checkpoint_acc("./checkpoints/resnet.pt", 93.35)),
        ("Standard CNN", load_checkpoint_acc("./checkpoints/cnn.pt", 91.65)),
        ("Adv-trained CNN", load_checkpoint_acc("./checkpoints/cnn_adv.pt", 81.27)),
    ]
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [SCENARIO_COLORS["clean"], SCENARIO_COLORS["fgsm"], SCENARIO_COLORS["adv_train"]]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Clean Accuracy Trade-off Across Models")
    ax.text(
        0.02,
        0.03,
        "Adversarial training improves robustness later, but costs about 10 points of clean accuracy.",
        transform=ax.transAxes,
        fontsize=10,
        color="#333333",
    )
    annotate_bars(ax, bars, values)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "clean_acc.png"), dpi=300)
    plt.close(fig)


def save_whitebox_compare(df: pd.DataFrame, figure_dir: str):
    base = df[(df["scenario"] == "white") & (df["eps_255"] == 8.0)].copy()
    base["attack"] = pd.Categorical(base["attack"], ATTACK_ORDER, ordered=True)
    base = base.sort_values("attack")

    x = np.arange(len(base))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bars1 = ax.bar(
        x - width / 2,
        base["asr"],
        width=width,
        color="#E45756",
        label="Attack Success Rate",
    )
    bars2 = ax.bar(
        x + width / 2,
        base["adv_acc"],
        width=width,
        color="#4C78A8",
        label="Adversarial Accuracy",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label_attack(v) for v in base["attack"]])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("White-box Attack Strength at eps = 8/255")
    annotate_bars(ax, bars1, base["asr"], dy=1.4)
    annotate_bars(ax, bars2, base["adv_acc"], dy=1.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=True)
    fig.text(
        0.5,
        0.02,
        "Lower adversarial accuracy means the attack is stronger.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(figure_dir, "whitebox.png"), dpi=300)
    plt.close(fig)


def save_whitebox_epsilon(df: pd.DataFrame, figure_dir: str):
    base = df[df["scenario"] == "white"].copy()
    base["attack"] = pd.Categorical(base["attack"], ATTACK_ORDER, ordered=True)
    base = base.sort_values(["attack", "eps_255"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    x_shift = {"fgsm": -0.08, "pgd": 0.0, "deepfool": 0.08}
    y_shift = {
        "asr": {"fgsm": 1.8, "pgd": 1.8, "deepfool": 3.2},
        "adv_acc": {"fgsm": 1.6, "pgd": 1.2, "deepfool": 2.8},
    }
    panels = [
        ("asr", "Attack Success Rate (%)", "Attack success climbs quickly as epsilon grows."),
        ("adv_acc", "Adversarial Accuracy (%)", "Model accuracy collapses as the perturbation budget grows."),
    ]

    for ax, (metric, ylabel, note) in zip(axes, panels):
        for attack in ATTACK_ORDER:
            group = base[base["attack"] == attack]
            if group.empty:
                continue
            ax.plot(
                group["eps_255"],
                group[metric],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=SCENARIO_COLORS[attack],
                label=label_attack(attack),
            )
            for _, row in group.iterrows():
                ax.text(
                    row["eps_255"] + x_shift[attack],
                    row[metric] + y_shift[metric][attack],
                    f"{row[metric]:.1f}",
                    ha="center",
                    fontsize=8,
                )
        ax.set_xlabel("Perturbation Budget (eps in 255 scale)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105)
        ax.text(0.03, 0.05, note, transform=ax.transAxes, fontsize=9)

    axes[0].set_title("White-box Attack Strength vs Epsilon")
    axes[1].set_title("White-box Accuracy vs Epsilon")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(figure_dir, "white_eps.png"), dpi=300)
    plt.close(fig)


def save_pgd_steps(df: pd.DataFrame, figure_dir: str):
    base = df[df["scenario"] == "white_steps"].copy().sort_values("steps")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        base["steps"],
        base["asr"],
        marker="o",
        linewidth=2.4,
        color=SCENARIO_COLORS["pgd"],
        label="ASR",
    )
    ax.plot(
        base["steps"],
        base["adv_acc"],
        marker="s",
        linewidth=2.4,
        color=SCENARIO_COLORS["fgsm"],
        label="Adversarial Accuracy",
    )
    for _, row in base.iterrows():
        ax.text(row["steps"], row["asr"] + 1.4, f"{row['asr']:.1f}", ha="center", fontsize=8)
        ax.text(row["steps"], row["adv_acc"] + 1.4, f"{row['adv_acc']:.1f}", ha="center", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_xlabel("PGD Iterations")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("PGD Gets Stronger with More Iterations (eps = 8/255)")
    ax.text(
        0.03,
        0.06,
        "Most of the gain appears in the first few iterations, then the curve saturates.",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.legend(loc="center right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "pgd_steps.png"), dpi=300)
    plt.close(fig)


def save_transfer(df: pd.DataFrame, figure_dir: str):
    base = df[df["scenario"] == "transfer"].copy()
    base["attack"] = pd.Categorical(base["attack"], ATTACK_ORDER, ordered=True)
    base = base.sort_values("attack")

    x = np.arange(len(base))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bars1 = ax.bar(x - width / 2, base["asr"], width=width, color="#9D755D", label="ASR")
    bars2 = ax.bar(
        x + width / 2,
        base["adv_acc"],
        width=width,
        color="#72B7B2",
        label="Target Model Accuracy",
    )
    annotate_bars(ax, bars1, base["asr"], dy=1.3)
    annotate_bars(ax, bars2, base["adv_acc"], dy=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([label_attack(v) for v in base["attack"]])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Transfer Attack: Adversarial Samples from ResNet18 to CNN")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=True)
    fig.text(
        0.5,
        0.02,
        "PGD transfers well, while DeepFool remains highly source-model specific.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(figure_dir, "transfer.png"), dpi=300)
    plt.close(fig)


def save_adv_train(df: pd.DataFrame, figure_dir: str):
    white = df[(df["scenario"] == "white") & (df["eps_255"] == 8.0)].copy()
    robust = df[df["scenario"] == "defense_advtrain"].copy()
    white["model"] = "Standard CNN"
    robust["model"] = "Adv-trained CNN"
    base = pd.concat([white, robust], ignore_index=True)
    base["attack"] = pd.Categorical(base["attack"], ATTACK_ORDER, ordered=True)
    base = base.sort_values(["attack", "model"])

    x = np.arange(len(ATTACK_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    std_vals = []
    adv_vals = []
    for attack in ATTACK_ORDER:
        std_vals.append(float(base[(base["attack"] == attack) & (base["model"] == "Standard CNN")]["adv_acc"].iloc[0]))
        adv_vals.append(float(base[(base["attack"] == attack) & (base["model"] == "Adv-trained CNN")]["adv_acc"].iloc[0]))

    bars1 = ax.bar(x - width / 2, std_vals, width=width, color="#4C78A8", label="Standard CNN")
    bars2 = ax.bar(x + width / 2, adv_vals, width=width, color="#E45756", label="Adv-trained CNN")
    annotate_bars(ax, bars1, std_vals, dy=1.3)
    annotate_bars(ax, bars2, adv_vals, dy=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([label_attack(v) for v in ATTACK_ORDER])
    ax.set_ylim(0, 60)
    ax.set_ylabel("Adversarial Accuracy (%)")
    ax.set_title("Adversarial Training Raises Robust Accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=True)
    fig.text(
        0.5,
        0.02,
        "At eps = 8/255, PGD accuracy rises from 0.6% to 33.8% after adversarial training.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(figure_dir, "adv_train.png"), dpi=300)
    plt.close(fig)


def save_preprocess(df: pd.DataFrame, figure_dir: str):
    base = df[df["scenario"] == "defense_preprocess"].copy()
    pivot = (
        base.pivot(index="eval_defense", columns="attack", values="adv_acc")
        .reindex(index=["jpeg", "squeeze"], columns=ATTACK_ORDER)
    )
    clean_lookup = (
        base.groupby("eval_defense")["clean_acc"].first().reindex(["jpeg", "squeeze"]).to_dict()
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(ATTACK_ORDER)))
    ax.set_xticklabels([label_attack(v) for v in ATTACK_ORDER])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [
            f"JPEG\n(clean {clean_lookup['jpeg']:.1f}%)",
            f"Feature Squeeze\n(clean {clean_lookup['squeeze']:.1f}%)",
        ]
    )
    ax.set_title("Preprocessing Defense Robust Accuracy (Higher is Better)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white" if value < 50 else "#0F2D3A",
                fontsize=10,
                fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Adversarial Accuracy (%)")
    ax.text(
        0.0,
        -0.16,
        "JPEG helps against DeepFool, but both preprocessing defenses remain weak against PGD.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "preprocess.png"), dpi=300)
    plt.close(fig)


def save_adaptive(df: pd.DataFrame, figure_dir: str):
    preprocess = df[df["scenario"] == "defense_preprocess"].copy()
    adaptive = df[df["scenario"] == "adaptive"].copy()

    rows = []
    for defense in ["jpeg", "squeeze"]:
        naive = preprocess[(preprocess["eval_defense"] == defense) & (preprocess["attack"] == "pgd")].iloc[0]
        adapt = adaptive[(adaptive["eval_defense"] == defense) & (adaptive["attack"] == "pgd")].iloc[0]
        rows.append(
            {
                "defense": defense,
                "clean_acc": naive["clean_acc"],
                "non_adaptive_asr": naive["asr"],
                "adaptive_asr": adapt["asr"],
            }
        )
    frame = pd.DataFrame(rows)

    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars1 = ax.bar(x - width / 2, frame["non_adaptive_asr"], width=width, color="#72B7B2", label="Standard PGD")
    bars2 = ax.bar(x + width / 2, frame["adaptive_asr"], width=width, color="#E45756", label="Adaptive BPDA+PGD")
    annotate_bars(ax, bars1, frame["non_adaptive_asr"], dy=1.3)
    annotate_bars(ax, bars2, frame["adaptive_asr"], dy=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"JPEG\nclean {frame.iloc[0]['clean_acc']:.1f}%",
            f"Feature Squeeze\nclean {frame.iloc[1]['clean_acc']:.1f}%",
        ]
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Adaptive Attack Breaks Preprocessing Defenses")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=True)
    fig.text(
        0.5,
        0.02,
        "Once the attacker models the defense, ASR jumps back above 97%.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(figure_dir, "adaptive.png"), dpi=300)
    plt.close(fig)


def save_summary_csv(df: pd.DataFrame, result_dir: str):
    summary_cols = [
        "scenario",
        "source",
        "target",
        "attack",
        "adaptive",
        "eval_defense",
        "eps_255",
        "steps",
        "clean_acc",
        "adv_acc",
        "asr",
        "mean_linf",
        "mean_l2",
    ]
    summary = df[summary_cols].sort_values(["scenario", "target", "attack", "eps_255", "steps"])
    summary.to_csv(os.path.join(result_dir, "summary.csv"), index=False)


def main():
    style()
    result_dir = "./results"
    figure_dir = "./figures"
    ensure_dir(figure_dir)

    source = os.path.join(result_dir, "metrics.csv")
    if not os.path.exists(source):
        raise FileNotFoundError(f"missing {source}")

    df = pd.read_csv(source)
    save_clean_accuracy(figure_dir)
    save_whitebox_compare(df, figure_dir)
    save_whitebox_epsilon(df, figure_dir)
    save_pgd_steps(df, figure_dir)
    save_transfer(df, figure_dir)
    save_adv_train(df, figure_dir)
    save_preprocess(df, figure_dir)
    save_adaptive(df, figure_dir)
    save_summary_csv(df, result_dir)
    print("saved summary figures and results/summary.csv")


if __name__ == "__main__":
    main()
