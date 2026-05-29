import os

import matplotlib.pyplot as plt
import pandas as pd


def label(row):
    prefix = "White" if row["type"] == "white" else "Transfer"
    attack = "MI-FGSM" if row["attack"] == "mifgsm" else row["attack"].upper()
    return f"{prefix} {attack}"


def plot(df, column, ylabel, filename, figure_dir):
    plt.figure(figsize=(9, 4))
    plt.bar(df["label"], df[column])
    plt.ylabel(ylabel)
    plt.ylim(0, 100)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, filename), dpi=300)
    plt.close()


def main():
    result_dir = "./results"
    figure_dir = "./figures"
    os.makedirs(figure_dir, exist_ok=True)

    source = os.path.join(result_dir, "attacks.csv")
    if not os.path.exists(source):
        raise FileNotFoundError(f"missing {source}")

    df = pd.read_csv(source)
    df["label"] = df.apply(label, axis=1)

    order = [
        "White FGSM",
        "White PGD",
        "Transfer FGSM",
        "Transfer PGD",
        "Transfer MI-FGSM",
    ]
    df["order"] = df["label"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("order")

    keep = [
        "type",
        "attack",
        "label",
        "source",
        "target",
        "eps_255",
        "steps",
        "samples",
        "clean_acc",
        "adv_acc",
        "asr",
    ]
    out = df[keep]

    csv_path = os.path.join(result_dir, "summary.csv")
    out.to_csv(csv_path, index=False)
    print(out)

    plot(out, "adv_acc", "Adversarial Accuracy (%)", "adv_acc.png", figure_dir)
    plot(out, "asr", "Attack Success Rate (%)", "asr.png", figure_dir)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
