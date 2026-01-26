# plot_val_acc_and_entropy.py
# 用法：
#   python plot_val_acc_and_entropy.py --dir /path/to/csvs --out_acc val_acc.png --out_entropy entropy.png
#
# 需要的文件（在 --dir 目录下）：
#   # accuracy
#   grpo.csv
#   entropyinloss.csv
#   entropyinloss_cont.csv
#   entropyinreward.csv
#   entropyinreward_cont.csv
#
#   # entropy (suffix _H)
#   grpo_H.csv
#   entropyinloss_H.csv
#   entropyinloss_H_cont.csv
#   entropyinreward_H.csv
#   entropyinreward_H_cont.csv

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def read_step_and_metric(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Step" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 'Step' 列，实际列名：{list(df.columns)}")
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} 至少需要 2 列（Step + metric），但只有 {df.shape[1]} 列")

    out = pd.DataFrame(
        {
            "Step": pd.to_numeric(df["Step"], errors="coerce"),
            "Metric": pd.to_numeric(df.iloc[:, 1], errors="coerce"),  # 取第 2 列
        }
    ).dropna()

    out = out.sort_values("Step")
    return out


def stitch_base_and_cont(base: pd.DataFrame, cont: pd.DataFrame) -> pd.DataFrame:
    # base: 取 [0, 500)；cont: 取 [500, 1000]
    base_part = base[(base["Step"] >= 0) & (base["Step"] < 500)]
    cont_part = cont[(cont["Step"] >= 500) & (cont["Step"] <= 1000)]

    stitched = pd.concat([base_part, cont_part], ignore_index=True)
    stitched = stitched.sort_values("Step")

    # 防止 step 重复（若有重复，保留后出现的那个）
    stitched = stitched.drop_duplicates(subset=["Step"], keep="last")

    # 最终只保留 0..1000
    stitched = stitched[(stitched["Step"] >= 0) & (stitched["Step"] <= 1000)]
    return stitched


def load_three_series(dir_path: str, prefix: str):
    """
    prefix: "" for accuracy files, "_H" for entropy files
    """
    def p(name: str) -> str:
        return os.path.join(dir_path, name)

    grpo = read_step_and_metric(p(f"grpo{prefix}.csv"))
    eil = read_step_and_metric(p(f"entropyinloss{prefix}.csv"))
    eil_c = read_step_and_metric(p(f"entropyinloss{prefix}_cont.csv"))
    eir = read_step_and_metric(p(f"entropyinreward{prefix}.csv"))
    eir_c = read_step_and_metric(p(f"entropyinreward{prefix}_cont.csv"))

    if prefix == "_H" and (not eir_c.empty) and (eir_c["Step"].max() < 500):
        eir_c = eir_c.copy()
        eir_c["Step"] = eir_c["Step"] + 500
        eir_c = eir_c.sort_values("Step")

    grpo_0_1000 = (
        grpo[(grpo["Step"] >= 0) & (grpo["Step"] <= 1000)]
        .drop_duplicates(subset=["Step"])
        .sort_values("Step")
    )
    eil_0_1000 = stitch_base_and_cont(eil, eil_c)
    eir_0_1000 = stitch_base_and_cont(eir, eir_c)

    return grpo_0_1000, eil_0_1000, eir_0_1000


def plot_three(
    series,
    y_label: str,
    out_path: str,
    show: bool = False,
):
    grpo, eil, eir = series

    plt.figure(figsize=(8, 4.8))
    plt.plot(grpo["Step"], grpo["Metric"], label="GRPO")
    plt.plot(eil["Step"], eil["Metric"], label="Entropy in Loss")
    plt.plot(eir["Step"], eir["Metric"], label="Entropy in Reward")

    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved to: {out_path}")

    if show:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="recipe/GARO", help="csv 文件所在目录")
    ap.add_argument("--out_acc", type=str, default="val_acc.png", help="accuracy 图输出路径")
    ap.add_argument("--out_entropy", type=str, default="entropy.png", help="entropy 图输出路径")
    ap.add_argument("--show", action="store_true", help="是否弹窗显示图")
    args = ap.parse_args()

    # 1) Validation Accuracy vs Step
    acc_series = load_three_series(args.dir, prefix="")
    plot_three(
        acc_series,
        y_label="Validation Accuracy",
        out_path=args.out_acc,
        show=args.show,
    )

    # 2) Entropy vs Step (suffix _H)
    ent_series = load_three_series(args.dir, prefix="_H")
    plot_three(
        ent_series,
        y_label="Entropy",
        out_path=args.out_entropy,
        show=args.show,
    )


if __name__ == "__main__":
    main()
