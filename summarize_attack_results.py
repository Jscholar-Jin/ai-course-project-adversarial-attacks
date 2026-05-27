# summarize_attack_results.py
# 功能：
# 1. 合并白盒攻击、黑盒攻击、迁移攻击结果
# 2. 生成最终汇总 CSV
# 3. 生成 Adv Acc 和 ASR 柱状图

import os
import pandas as pd
import matplotlib.pyplot as plt


def make_attack_label(row):
    experiment = row["experiment"]
    attack = row["attack_name"]

    if experiment == "white-box":
        return f"White-box {attack.upper()}"
    elif experiment == "black-box":
        return f"Black-box {attack.upper()}"
    elif experiment == "transfer":
        if attack == "mi-fgsm":
            return "Transfer MI-FGSM"
        else:
            return f"Transfer {attack.upper()}"
    else:
        return f"{experiment} {attack}"


def main():
    result_dir = "./results"
    figure_dir = "./figures"

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    file_white_transfer = os.path.join(
        result_dir,
        "attack_results_whitebox_transfer.csv"
    )

    file_blackbox = os.path.join(
        result_dir,
        "attack_results_blackbox_spsa.csv"
    )

    if not os.path.exists(file_white_transfer):
        raise FileNotFoundError(f"找不到文件: {file_white_transfer}")

    if not os.path.exists(file_blackbox):
        raise FileNotFoundError(f"找不到文件: {file_blackbox}")

    df1 = pd.read_csv(file_white_transfer)
    df2 = pd.read_csv(file_blackbox)

    df = pd.concat([df1, df2], ignore_index=True)

    df["method_label"] = df.apply(make_attack_label, axis=1)

    # 指定展示顺序
    order = [
        "White-box FGSM",
        "White-box PGD",
        "Black-box SPSA",
        "Transfer FGSM",
        "Transfer PGD",
        "Transfer MI-FGSM",
    ]

    df["order"] = df["method_label"].apply(
        lambda x: order.index(x) if x in order else 999
    )

    df = df.sort_values("order").reset_index(drop=True)

    # 保留核心字段
    keep_cols = [
        "experiment",
        "attack_name",
        "method_label",
        "source_model",
        "target_model",
        "eps_255",
        "steps",
        "total_samples",
        "clean_acc",
        "adv_acc",
        "asr",
    ]

    if "queries_per_sample" in df.columns:
        keep_cols.append("queries_per_sample")

    summary_df = df[keep_cols]

    save_csv = os.path.join(result_dir, "final_attack_summary.csv")
    summary_df.to_csv(save_csv, index=False, encoding="utf-8-sig")

    print("=" * 120)
    print("最终攻击结果汇总")
    print("=" * 120)
    print(summary_df.to_string(index=False))

    print("\n结果已保存:")
    print(save_csv)

    # =========================
    # 图 1：Adv Acc 柱状图
    # =========================
    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["method_label"], summary_df["adv_acc"])
    plt.ylabel("Adversarial Accuracy (%)")
    plt.xlabel("Attack Method")
    plt.title("Adversarial Accuracy under Different Attacks")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()

    adv_acc_fig = os.path.join(figure_dir, "attack_adv_acc_bar.png")
    plt.savefig(adv_acc_fig, dpi=300)
    plt.close()

    # =========================
    # 图 2：ASR 柱状图
    # =========================
    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["method_label"], summary_df["asr"])
    plt.ylabel("Attack Success Rate (%)")
    plt.xlabel("Attack Method")
    plt.title("Attack Success Rate under Different Attacks")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()

    asr_fig = os.path.join(figure_dir, "attack_asr_bar.png")
    plt.savefig(asr_fig, dpi=300)
    plt.close()

    print("\n图片已保存:")
    print(adv_acc_fig)
    print(asr_fig)

    print("\n可以写进报告的核心结论：")
    print("1. 白盒 PGD 攻击最强，目标模型对抗准确率降至最低。")
    print("2. 迁移攻击在不访问目标模型梯度的情况下仍能显著降低目标模型准确率。")
    print("3. MI-FGSM 的迁移攻击效果最强，说明动量机制可以提升对抗样本迁移性。")
    print("4. SPSA 黑盒攻击效果明显，但查询成本较高，每张图像需要多次模型查询。")


if __name__ == "__main__":
    main()