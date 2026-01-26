# plot_early_intervention.py
#
# Usage:
#   python recipe/GARO/plot_early_intervention.py --dir recipe/GARO
#
# Expected files under --dir:
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_step_and_metric(csv_path: str) -> pd.DataFrame:
    """Return a DataFrame with columns: Step (int/float), Metric (float).

    Metric is taken as the 2nd column of the csv (index 1), as requested.
    """
    df = pd.read_csv(csv_path)
    if "Step" not in df.columns:
        raise ValueError(f"{csv_path} missing 'Step' column. Columns: {list(df.columns)}")
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} must have at least 2 columns (Step + metric), got {df.shape[1]}")

    out = pd.DataFrame(
        {
            "Step": pd.to_numeric(df["Step"], errors="coerce"),
            "Metric": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
        }
    ).dropna()
    return out.sort_values("Step")


def stitch_base_and_cont(base: pd.DataFrame, cont: pd.DataFrame) -> pd.DataFrame:
    """Stitch a base run and a continuation run into a single 0..1000 curve.

    Requirement (your spec):
      - base uses its own Step in [0, 500] as steps 0..500
      - cont uses its own Step in [0, 500] as steps 500..1000
        (If cont is already logged in absolute steps, normalize to start at 500 anyway.)
    """
    base_part = base[(base["Step"] >= 0) & (base["Step"] <= 500)].copy()

    cont_part = cont.sort_values("Step").copy()
    if not cont_part.empty:
        cont_start = float(cont_part["Step"].min())
        cont_part = cont_part[(cont_part["Step"] >= cont_start) & (cont_part["Step"] <= cont_start + 500)].copy()
        cont_part["Step"] = cont_part["Step"] - cont_start + 500

    stitched = pd.concat([base_part, cont_part], ignore_index=True)
    stitched = stitched.sort_values("Step")

    # If a step repeats, keep the later one (continuation should win at step=500 if both exist)
    stitched = stitched.drop_duplicates(subset=["Step"], keep="last")

    return stitched[(stitched["Step"] >= 0) & (stitched["Step"] <= 1000)]


def ema_smooth(y: np.ndarray, ema_strength: float) -> np.ndarray:
    """Exponential moving average.

    Convention used here:
      out[t] = (1-s)*y[t] + s*out[t-1]
    s in [0, 1]. Larger means smoother (more inertia).

    Note: s=1 makes the series constant at y[0]. This matches the requested 'ema_strength=1' plot.
    """
    if ema_strength <= 0.0 or len(y) == 0:
        return y.astype(float, copy=False)

    s = float(ema_strength)
    out = np.empty_like(y, dtype=float)
    out[0] = float(y[0])
    for i in range(1, len(y)):
        out[i] = (1.0 - s) * float(y[i]) + s * float(out[i - 1])
    return out


def load_three_series(dir_path: str, prefix: str):
    """Load GRPO and two stitched series (entropy-in-loss / entropy-in-reward).

    prefix: "" for accuracy files, "_H" for entropy files.
    """
    def p(name: str) -> str:
        return os.path.join(dir_path, name)

    grpo = read_step_and_metric(p(f"grpo{prefix}.csv"))
    eil = read_step_and_metric(p(f"entropyinloss{prefix}.csv"))
    eil_c = read_step_and_metric(p(f"entropyinloss{prefix}_cont.csv"))
    eir = read_step_and_metric(p(f"entropyinreward{prefix}.csv"))
    eir_c = read_step_and_metric(p(f"entropyinreward{prefix}_cont.csv"))

    grpo_0_1000 = (
        grpo[(grpo["Step"] >= 0) & (grpo["Step"] <= 1000)]
        .drop_duplicates(subset=["Step"])
        .sort_values("Step")
    )
    eil_0_1000 = stitch_base_and_cont(eil, eil_c)
    eir_0_1000 = stitch_base_and_cont(eir, eir_c)

    return grpo_0_1000, eil_0_1000, eir_0_1000


def plot_acc(series, out_path: str, show: bool = False):
    grpo, eil, eir = series

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(grpo["Step"], grpo["Metric"], label="GRPO")
    ax.plot(eil["Step"], eil["Metric"], label="Entropy in Loss")
    ax.plot(eir["Step"], eir["Metric"], label="Entropy in Reward")

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved to: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_entropy_dual(series, out_path: str, ema_strength: float = 0.9, show: bool = False):
    """Entropy figure: for each run, plot raw (dashed) and EMA-smoothed (solid)."""
    grpo, eil, eir = series

    fig, ax = plt.subplots(figsize=(8, 4.8))

    def plot_one(df: pd.DataFrame, name: str):
        df = df.sort_values("Step")
        x = df["Step"].to_numpy()
        y = df["Metric"].to_numpy()

        # Raw (dashed) — slightly transparent so it doesn't overpower the EMA curve
        ln_raw = ax.plot(x, y, linestyle="--", alpha=0.35, label=f"{name} raw")[0]

        # EMA (solid) — same color as raw
        y_ema = ema_smooth(y, ema_strength=ema_strength)
        ax.plot(x, y_ema, linestyle="-", color=ln_raw.get_color(), label=f"{name} EMA({ema_strength:g})")

    plot_one(grpo, "GRPO")
    plot_one(eil, "Entropy in Loss")
    plot_one(eir, "Entropy in Reward")

    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved to: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="recipe/GARO", help="csv directory")
    ap.add_argument("--out_acc", type=str, default="val_acc.png", help="output path for accuracy plot")
    ap.add_argument("--out_entropy", type=str, default="entropy.png", help="output path for entropy plot")
    ap.add_argument(
        "--ema_strength",
        type=float,
        default=0.9,
        help="Entropy plot: EMA smoothing strength s in [0,1]. Solid line uses this s; dashed line is raw.",
    )
    ap.add_argument("--show", action="store_true", help="show the figures interactively")
    args = ap.parse_args()

    # 1) Validation Accuracy vs Step
    acc_series = load_three_series(args.dir, prefix="")
    plot_acc(acc_series, out_path=args.out_acc, show=args.show)

    # 2) Entropy vs Step (suffix _H)
    ent_series = load_three_series(args.dir, prefix="_H")
    plot_entropy_dual(ent_series, out_path=args.out_entropy, ema_strength=args.ema_strength, show=args.show)


if __name__ == "__main__":
    main()
