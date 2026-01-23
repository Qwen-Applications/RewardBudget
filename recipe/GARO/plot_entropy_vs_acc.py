import os
import re
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# python recipe/GARO/plot_entropy_vs_acc.py --horizon 10 --coef_inset --plot_step_acc

# mapping from file group id (1/2/5) to true kappa value
GROUPID_TO_KAPPA = {
    1: 200,    # 1/0.005
    2: 2000,   # 2/0.001
    5: 5000,   # 5/0.001
}


def read_last_col_with_step(path: str) -> pd.DataFrame:
    """Return columns: Step(int), value(float). 'value' is last column."""
    df = pd.read_csv(path)
    if "Step" not in df.columns:
        raise ValueError(f'Missing "Step" column in {path}')
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have at least 2 columns (Step + metric): {path}")

    last_col = df.columns[-1]
    out = df[["Step", last_col]].copy().rename(columns={last_col: "value"})
    out["Step"] = pd.to_numeric(out["Step"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["Step", "value"]).copy()
    out["Step"] = out["Step"].astype(int)
    return out


def parse_group_id_from_filename(fname: str) -> int:
    """Parse group id from drgrpo1.csv / drgrpo1_H.csv -> 1"""
    m = re.search(r"drgrpo(\d+)", fname.lower())
    if not m:
        raise ValueError(f"Cannot parse group id from filename: {fname}")
    gid = int(m.group(1))
    if gid not in GROUPID_TO_KAPPA:
        raise ValueError(f"Unknown group id {gid} in {fname}. Expected one of {list(GROUPID_TO_KAPPA.keys())}")
    return gid


def ols_line(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray):
    """Fit y = a + b x and return yhat on x_grid."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    y_hat = model.predict(sm.add_constant(x_grid))
    return y_hat, model


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd


def standardized_ols_abs_coefs(df: pd.DataFrame):
    """
    Standardize V(=acc), H, kappa (numeric), then compare:
      (1) V ~ H
      (2) V ~ H + kappa
    Return abs coefficients for inset bars.
    """
    tmp = df.copy()
    tmp["V_z"] = zscore(tmp["acc"])
    tmp["H_z"] = zscore(tmp["H"])
    tmp["kappa_z"] = zscore(tmp["kappa"])

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["V_z", "H_z", "kappa_z"]).copy()
    if len(tmp) < 5:
        return None

    X1 = sm.add_constant(tmp[["H_z"]])
    m1 = sm.OLS(tmp["V_z"], X1).fit()

    X2 = sm.add_constant(tmp[["H_z", "kappa_z"]])
    m2 = sm.OLS(tmp["V_z"], X2).fit()

    return {
        "abs_beta_H_m1": float(abs(m1.params["H_z"])),
        "abs_beta_H_m2": float(abs(m2.params["H_z"])),
        "abs_beta_kappa_m2": float(abs(m2.params["kappa_z"])),
        "m1": m1,
        "m2": m2,
    }


def save_with_suffix(path: str, suffix: str) -> str:
    """out.png + '_step' -> out_step.png ; if no extension, append suffix"""
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{path}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="recipe/GARO/plot")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Predictive horizon in Step units (t -> t+horizon). Default: 10.")
    parser.add_argument("--save", type=str, default="recipe/GARO/plot", help="Optional output path for main plot (e.g. out.png)")

    # optional inset bars for standardized abs coefficients
    parser.add_argument("--coef_inset", action="store_true",
                        help="Add inset bars comparing standardized abs coefficients.")
    parser.add_argument("--inset_loc", type=str, default="upper right",
                        choices=["upper right", "upper left", "lower right", "lower left"])

    # NEW: optional Step vs acc plot (no predictive horizon)
    parser.add_argument("--plot_step_acc", action="store_true",
                        help="If set, also plot Step vs (scaled) acc without predictive horizon alignment.")
    args = parser.parse_args()

    base_dir = args.dir
    horizon = args.horizon

    files = [
        "drgrpo1.csv", "drgrpo2.csv", "drgrpo5.csv",
        "drgrpo1_H.csv", "drgrpo2_H.csv", "drgrpo5_H.csv",
    ]
    for f in files:
        p = os.path.join(base_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")

    # --------- Build paired dataset for main plot: x=H_{kappa,t}, y=acc_{t+horizon} (scaled) ----------
    rows = []
    for acc_fname in ["drgrpo1.csv", "drgrpo2.csv", "drgrpo5.csv"]:
        gid = parse_group_id_from_filename(acc_fname)   # 1/2/5
        kappa_val = GROUPID_TO_KAPPA[gid]               # 200/2000/5000
        ent_fname = f"drgrpo{gid}_H.csv"

        acc_path = os.path.join(base_dir, acc_fname)
        ent_path = os.path.join(base_dir, ent_fname)

        acc = read_last_col_with_step(acc_path).rename(columns={"value": "acc"})
        ent = read_last_col_with_step(ent_path).rename(columns={"value": "H"})

        # scale acc by dividing by group id (1/2/5)
        acc["acc"] = acc["acc"] / float(gid)

        # align: ent at step t, acc at step t+horizon
        acc_shifted = acc.copy()
        acc_shifted["Step"] = acc_shifted["Step"] - horizon

        merged = ent.merge(acc_shifted, on="Step", how="inner")
        merged["group_id"] = gid
        merged["kappa"] = kappa_val
        rows.append(merged[["Step", "group_id", "kappa", "H", "acc"]])

    data = pd.concat(rows, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["H", "acc"]).copy()
    if data.empty:
        raise RuntimeError("No aligned (H_t, acc_{t+horizon}) pairs found. Check Step & horizon.")

    # --------- Main plot: Entropy vs acc_{t+h} ----------
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))

    ax = sns.scatterplot(
        data=data,
        x="H",
        y="acc",
        hue="kappa",          # shows 200/2000/5000
        palette="deep",
        alpha=0.75,
        s=45,
        edgecolor=None,
    )

    x_min, x_max = float(np.nanmin(data["H"])), float(np.nanmax(data["H"]))
    x_grid = np.linspace(x_min, x_max, 200)

    # Overall OLS (black)
    overall = ols_line(data["H"].values, data["acc"].values, x_grid)
    if overall is not None:
        yhat, _ = overall
        ax.plot(x_grid, yhat, color="black", linewidth=2.5, label="Overall OLS")

    # Group OLS lines
    kappa_sorted = sorted(data["kappa"].unique())
    palette = sns.color_palette("deep", n_colors=len(kappa_sorted))
    color_map = {k: palette[i] for i, k in enumerate(kappa_sorted)}
    for k in kappa_sorted:
        sub = data[data["kappa"] == k]
        fit = ols_line(sub["H"].values, sub["acc"].values, x_grid)
        if fit is None:
            continue
        yhat_k, _ = fit
        ax.plot(x_grid, yhat_k, color=color_map[k], linewidth=2.0, alpha=0.95)

    ax.set_title(f"Entropy vs Validation Accuracy (horizon = {horizon})")
    ax.set_xlabel(r"Entropy  $H_{\kappa, t}$")
    ax.set_ylabel(r"Scaled val acc  $P_{\kappa, t+\mathrm{horizon}}$  (acc / group_id)")
    ax.legend(title="kappa", loc="best")

    # inset: standardized |coef|
    if args.coef_inset:
        coefs = standardized_ols_abs_coefs(data)
        if coefs is not None:
            bars = [
                ("|β_H|  (V~H)", coefs["abs_beta_H_m1"]),
                ("|β_H|  (V~H+κ)", coefs["abs_beta_H_m2"]),
                ("|β_κ|  (V~H+κ)", coefs["abs_beta_kappa_m2"]),
            ]
            loc_map = {
                "upper right": (0.62, 0.58, 0.36, 0.36),
                "upper left":  (0.02, 0.58, 0.36, 0.36),
                "lower right": (0.62, 0.05, 0.36, 0.36),
                "lower left":  (0.02, 0.05, 0.36, 0.36),
            }
            x0, y0, w, h = loc_map[args.inset_loc]
            ax_in = ax.inset_axes([x0, y0, w, h])

            labels = [b[0] for b in bars]
            values = [b[1] for b in bars]
            ax_in.bar(range(len(values)), values)
            ax_in.set_xticks(range(len(values)))
            ax_in.set_xticklabels(labels, rotation=45, ha="right")
            ax_in.set_title("Std. |coef|", fontsize=10)
            ax_in.tick_params(axis="both", labelsize=8)
            ax_in.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved main figure to: {args.save}")
        # keep it open only if you also want second plot shown; otherwise close to avoid overlap
        if not args.plot_step_acc:
            plt.close()

    else:
        plt.show()

    # --------- NEW optional plot: Step vs acc (no horizon), still scaled by / (1,2,5) ----------
    if args.plot_step_acc:
        step_rows = []
        for acc_fname in ["drgrpo1.csv", "drgrpo2.csv", "drgrpo5.csv"]:
            gid = parse_group_id_from_filename(acc_fname)  # 1/2/5
            kappa_val = GROUPID_TO_KAPPA[gid]

            acc_path = os.path.join(base_dir, acc_fname)
            acc = read_last_col_with_step(acc_path).rename(columns={"value": "acc"})
            acc["acc"] = acc["acc"] / float(gid)  # scale
            acc["group_id"] = gid
            acc["kappa"] = kappa_val
            step_rows.append(acc[["Step", "group_id", "kappa", "acc"]])

        step_data = pd.concat(step_rows, ignore_index=True)
        step_data = step_data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Step", "acc"]).copy()
        if step_data.empty:
            raise RuntimeError("No data for Step vs acc plot.")

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(9, 6))

        # line plot per kappa (no aggregation)
        ax2 = sns.lineplot(
            data=step_data.sort_values(["kappa", "Step"]),
            x="Step",
            y="acc",
            hue="kappa",
            palette="deep",
            estimator=None,   # plot raw lines
            lw=2.0,
        )
        # optional: also show points
        sns.scatterplot(
            data=step_data,
            x="Step",
            y="acc",
            hue="kappa",
            palette="deep",
            s=30,
            alpha=0.6,
            legend=False,
            ax=ax2,
        )

        ax2.set_title("Step vs Validation Accuracy (no horizon)")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Scaled val acc (acc / group_id)")
        ax2.legend(title="kappa", loc="best")

        plt.tight_layout()

        if args.save:
            out2 = save_with_suffix(args.save, "_step")
            plt.savefig(out2, dpi=200)
            print(f"Saved Step-vs-acc figure to: {out2}")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    main()
