import os
import re
import argparse
import math

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


def standardized_ols_abs_coefs(df: pd.DataFrame, use_log_kappa: bool = False):
    """
    Standardize V(=acc), H, kappa (numeric), then compare:
      (1) V ~ H
      (2) V ~ H + kappa
    Return abs coefficients for inset bars.
    """
    tmp = df.copy()
    tmp["V_z"] = zscore(tmp["acc"])
    tmp["H_z"] = zscore(tmp["H"])
    if use_log_kappa:
        # kappa is positive (200/2000/5000), so log is safe
        tmp["kappa_feat"] = np.log(tmp["kappa"].astype(float))
    else:
        tmp["kappa_feat"] = tmp["kappa"].astype(float)
    tmp["kappa_z"] = zscore(tmp["kappa_feat"])

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

def lowess_smooth_and_inflections(
    x: np.ndarray,
    y: np.ndarray,
    *,
    frac: float = 0.25,
    n_grid: int = 400,
    max_points: int = 6,
    min_sep_frac: float = 0.05,
):
    '''
    LOWESS-smooth a 1D curve and mark inflection points (approx. where 2nd derivative crosses zero).

    Returns:
        xg, yg: dense grid and smoothed curve values
        inf_x, inf_y: lists of inflection points on the smoothed curve
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 5:
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]

    # statsmodels LOWESS returns an (n,2) array sorted by x
    lo = sm.nonparametric.lowess(y, x, frac=frac, it=0, return_sorted=True)
    xs, ys = lo[:, 0], lo[:, 1]

    # de-duplicate x for safe interpolation
    ux, inv = np.unique(xs, return_inverse=True)
    if ux.size != xs.size:
        # average ys for duplicate xs
        import numpy as _np
        sums = _np.bincount(inv, weights=ys)
        cnts = _np.bincount(inv)
        uy = sums / _np.maximum(cnts, 1)
    else:
        uy = ys

    if ux.size < 4:
        return None

    x_min, x_max = float(ux.min()), float(ux.max())
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return None

    n_grid = int(max(50, n_grid))
    xg = np.linspace(x_min, x_max, n_grid)
    yg = np.interp(xg, ux, uy)

    # numerical derivatives on the dense grid
    d1 = np.gradient(yg, xg)
    d2 = np.gradient(d1, xg)

    # sign with simple zero-handling (ffill)
    s = np.sign(d2)
    # forward-fill zeros
    for i in range(1, s.size):
        if s[i] == 0:
            s[i] = s[i - 1]
    # backward-fill leading zeros
    if s.size > 0 and s[0] == 0:
        j = 0
        while j < s.size and s[j] == 0:
            j += 1
        if j < s.size:
            s[:j] = s[j]

    # candidate indices where sign changes
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size == 0:
        return xg, yg, [], []

    # rank by curvature jump, then keep at most max_points and enforce separation
    strengths = np.abs(d2[idx + 1] - d2[idx])
    order = np.argsort(strengths)[::-1]

    sep = float(min_sep_frac) * (x_max - x_min)
    if sep <= 0:
        sep = 0.0

    inf_x, inf_y = [], []
    for t in order:
        i = int(idx[t])
        x0, x1 = xg[i], xg[i + 1]
        y0, y1 = d2[i], d2[i + 1]
        # linear interpolation for root of d2
        if y1 == y0:
            continue
        xr = x0 - y0 * (x1 - x0) / (y1 - y0)
        if not np.isfinite(xr):
            continue
        if (xr < x_min) or (xr > x_max):
            continue
        if sep > 0 and any(abs(xr - px) < sep for px in inf_x):
            continue
        yr = float(np.interp(xr, xg, yg))
        inf_x.append(float(xr))
        inf_y.append(yr)
        if len(inf_x) >= int(max_points):
            break

    # sort inflection points by x (for nicer visuals)
    if inf_x:
        p = np.argsort(inf_x)
        inf_x = [inf_x[i] for i in p]
        inf_y = [inf_y[i] for i in p]

    return xg, yg, inf_x, inf_y

def polyfit_smooth_and_turning_point(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int = 2,
    n_grid: int = 400,
):
    """Polynomial-fit a 1D curve and return a dashed fit line + ONE turning point.

    Turning point is chosen as a real root of p'(x)=0 within [min(x), max(x)].
    If multiple roots exist (degree > 2), pick the one with the largest |p''(x)|.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < max(5, int(degree) + 1):
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]

    x_min, x_max = float(x.min()), float(x.max())
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return None

    degree = int(max(1, degree))
    try:
        p = np.poly1d(np.polyfit(x, y, deg=degree))
    except Exception:
        return None

    n_grid = int(max(50, n_grid))
    xg = np.linspace(x_min, x_max, n_grid)
    yg = p(xg)

    turn_x = None
    turn_y = None
    if degree >= 2:
        dp = p.deriv(1)
        roots = dp.r
        roots = roots[np.isreal(roots)].real
        roots = roots[(roots >= x_min) & (roots <= x_max)]
        if roots.size > 0:
            ddp = p.deriv(2)
            curv = np.abs(ddp(roots))
            j = int(np.argmax(curv))
            turn_x = float(roots[j])
            turn_y = float(p(turn_x))

    return xg, yg, turn_x, turn_y

def polyfit_smooth_and_convergence_point(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int = 5,
    n_grid: int = 400,
    tail_frac: float = 0.10,
    eps_abs: float = 0.005,
    eps_rel: float = 0.02,
    window_frac: float = 0.10,
    resid_frac: float = 0.10,
    sustain_frac: float = 0.06,
):
    """Polynomial-fit a 1D curve and return (fit line) + a *convergence* point.

    "Converge" here means: **future improvement becomes negligible** (not an inflection).

    Operationally:
      1) Fit p(x) to (x,y) and evaluate on a dense grid (xg, yg).
      2) Define a forward window of length `window_frac` of the x-range.
      3) Mark a point as "converged" if BOTH hold:
           (a) local gain over the next window <= eps_local,
           (b) remaining gain to the future max is small (<= eps_remain).
         where eps_local = max(eps_abs, eps_rel * total_gain) and
               eps_remain = max(eps_abs, resid_frac * total_gain).
      4) The convergence point is the earliest x where it is converged and stays
         converged for at least `sustain_frac` of the grid.

    We also compute a plateau guide value (median of the last `tail_frac` of yg)
    for visualization.

    Returns: xg, yg, conv_x, conv_y, plateau_y
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < max(10, int(degree) + 1):
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]
    x_min, x_max = float(x.min()), float(x.max())
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return None

    degree = int(max(1, degree))
    n_grid = int(max(80, n_grid))
    try:
        p = np.poly1d(np.polyfit(x, y, deg=degree))
    except Exception:
        return None

    xg = np.linspace(x_min, x_max, n_grid)
    yg = np.asarray(p(xg), dtype=float)

    tail_frac = float(np.clip(tail_frac, 0.02, 0.50))
    tail_n = max(20, int(round(tail_frac * n_grid)))
    plateau_y = float(np.nanmedian(yg[-tail_n:]))

    # total dynamic range on the fitted curve
    total_gain = float(np.nanmax(yg) - np.nanmin(yg))
    total_gain = max(1e-12, total_gain)
    eps_local = max(float(eps_abs), float(eps_rel) * total_gain)
    eps_remain = max(float(eps_abs), float(resid_frac) * total_gain)

    window_frac = float(np.clip(window_frac, 0.02, 0.50))
    win_n = max(5, int(round(window_frac * n_grid)))
    if win_n >= n_grid:
        win_n = n_grid - 1

    # forward gain over the next window (local flatness) + remaining gain to future max
    forward_gain = np.abs(yg[win_n:] - yg[:-win_n])  # length n_grid - win_n
    future_max = np.maximum.accumulate(yg[::-1])[::-1]
    remaining_gain = np.abs(future_max - yg)

    flat_local = forward_gain <= eps_local
    flat_global = remaining_gain[: flat_local.size] <= eps_remain
    flat = flat_local & flat_global

    sustain_frac = float(np.clip(sustain_frac, 0.02, 0.50))
    sustain_n = max(10, int(round(sustain_frac * n_grid)))
    sustain_n = min(sustain_n, flat.size)

    conv_x, conv_y = None, None
    if flat.size > 0 and sustain_n > 0:
        for i in range(0, flat.size - sustain_n + 1):
            if bool(np.all(flat[i : i + sustain_n])):
                conv_x = float(xg[i])
                conv_y = float(yg[i])
                break

    return xg, yg, conv_x, conv_y, plateau_y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="recipe/GARO/plot")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Predictive horizon in Step units (t -> t+horizon). Default: 10.")
    parser.add_argument("--save", type=str, default="recipe/GARO/plot", help="Optional output path for main plot (e.g. out.png)")

    # optional inset bars for standardized abs coefficients
    parser.add_argument("--coef_inset", action="store_true",
                        help="Add inset bars comparing standardized abs coefficients.")
    parser.add_argument("--inset_loc", type=str, default="lower left",
                         choices=["upper right", "upper left", "lower right", "lower left"])
    parser.add_argument("--log_kappa", action="store_true",
                        help="Use log(kappa) (then z-score) in the regression/inset instead of raw kappa.")
    parser.add_argument("--font_scale", type=float, default=2.0,
                        help="Global font scale multiplier (default: 2.0).")

    # optional Step vs acc plot (no predictive horizon)
    parser.add_argument("--plot_step_acc", action="store_true",
                        help="If set, also plot Step vs (scaled) acc without predictive horizon alignment.")

    # smoothing / turning-point markers for the Step-vs-acc plot
    parser.add_argument("--step_max", type=int, default=1200,
                        help="Only show Step <= step_max on the Step-vs-acc plot (default: 1200).")

    # polynomial fit for the dashed curve on the Step-vs-acc plot
    parser.add_argument("--fit_degree", type=int, default=5,
                        help="Polynomial degree for the dashed fit on Step-vs-acc (default: 5).")
    parser.add_argument("--smooth_grid", type=int, default=400,
                        help="Number of points used to draw the dashed fitted curve (default: 400).")

    # convergence marker configuration
    parser.add_argument("--conv_tail_frac", type=float, default=0.10,
                        help="Tail fraction (of fitted grid) used to estimate plateau for convergence (default: 0.10).")
    parser.add_argument("--conv_eps_abs", type=float, default=0.005,
                        help="Absolute epsilon band for convergence (default: 0.005).")
    parser.add_argument("--conv_eps_rel", type=float, default=0.02,
                        help="Relative epsilon as a fraction of total fitted gain (default: 0.02).")
    parser.add_argument("--conv_window_frac", type=float, default=0.10,
                        help="Forward window length (fraction of fitted grid) used to judge convergence (default: 0.10).")
    parser.add_argument("--conv_resid_frac", type=float, default=0.10,
                        help="Remaining-gain threshold as a fraction of total fitted gain (default: 0.10).")
    parser.add_argument("--conv_sustain_frac", type=float, default=0.06,
                        help="How long (fraction of fitted grid) the curve must stay within the band to count as converged (default: 0.06).")

    args = parser.parse_args()

    # Keep fonts consistent across normal text and mathtext (avoid mixed-looking annotations).
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"

    # Keep fonts consistent across normal text and mathtext (avoid mixed-looking annotations).
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"

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
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    sns.set_context("talk", font_scale=args.font_scale)
    plt.figure(figsize=(12, 8))

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
        ax.plot(x_grid, yhat, color="black", linewidth=2.5, label="_nolegend_")

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

    ax.set_xlabel(r"Entropy")
    ax.set_ylabel(r"Validation Accuracy")
    ax.legend(title="kappa", loc="lower right")

    # inset: standardized |coef|
    if args.coef_inset:
        coefs = standardized_ols_abs_coefs(data, use_log_kappa=args.log_kappa)
        if coefs is not None:
            kappa_name = "logκ" if args.log_kappa else "κ"
            bars = [
                (r"$|\beta_H|$" + "\n" + r"$(V\sim H)$", coefs["abs_beta_H_m1"]),
                (r"$|\beta_H|$" + "\n" + rf"$(V\sim H+{kappa_name})$", coefs["abs_beta_H_m2"]),
                (rf"$|\beta_{{{kappa_name}}}|$" + "\n" + rf"$(V\sim H+{kappa_name})$", coefs["abs_beta_kappa_m2"]),
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
            rects = ax_in.bar(range(len(values)), values)
            ax_in.set_xticks(range(len(values)))
            # remove messy x tick labels/ticks; we annotate on bars instead
            ax_in.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            # remove y-axis tick labels like 0.0/0.25/0.50 (keep grid if you want)
            ax_in.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax_in.set_title("Std. |coef|")
            ax_in.grid(True, axis="y", alpha=0.25)
            y_max = float(np.nanmax(values)) if len(values) else 0.0
            dy = 0.02 * y_max if y_max > 0 else 0.01
            if y_max > 0:
                ax_in.set_ylim(0.0, y_max + 0.35 * y_max + 6.0 * dy)
            anno_fs = float(plt.rcParams.get("font.size", 12)) * 0.75
            for rect, txt in zip(rects, labels):
                x = rect.get_x() + rect.get_width() / 2.0
                y = rect.get_height()
                ax_in.text(x, y + dy, txt, ha="center", va="bottom", fontsize=anno_fs, clip_on=True)

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
        if args.step_max is not None:
            step_data = step_data[step_data["Step"] <= int(args.step_max)].copy()
        if step_data.empty:
            raise RuntimeError("No data for Step vs acc plot.")

        sns.set_theme(style="whitegrid", font="DejaVu Sans")
        sns.set_context("talk", font_scale=args.font_scale)
        plt.figure(figsize=(12, 8))

        kappa_sorted = sorted(step_data["kappa"].unique())
        palette = sns.color_palette("deep", n_colors=len(kappa_sorted))
        color_map = {k: palette[i] for i, k in enumerate(kappa_sorted)}

        ax2 = plt.gca()
        conv_points = []  # (kappa, x_conv, y_conv, plateau_y)
        x_right = float(step_data["Step"].max())
        for k in kappa_sorted:
            sub = step_data[step_data["kappa"] == k].sort_values("Step")
            x = sub["Step"].to_numpy(dtype=float)
            y = sub["acc"].to_numpy(dtype=float)
            # dashed fitted curve (polynomial) + convergence point
            fit = polyfit_smooth_and_convergence_point(
                x,
                y,
                degree=int(args.fit_degree),
                n_grid=int(args.smooth_grid),
                tail_frac=float(args.conv_tail_frac),
                eps_abs=float(args.conv_eps_abs),
                eps_rel=float(args.conv_eps_rel),
                window_frac=float(args.conv_window_frac),
                resid_frac=float(args.conv_resid_frac),
                sustain_frac=float(args.conv_sustain_frac),
             )
            if fit is not None:
                xg, yg, conv_x, conv_y, plateau_y = fit
                ax2.plot(
                    xg, yg,
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.9,
                    color=color_map[k],
                    zorder=1,
                    label="_nolegend_",
                )
                if (conv_x is not None) and (conv_y is not None):
                    conv_points.append((k, float(conv_x), float(conv_y), float(plateau_y)))

            # raw curve + points
            ax2.plot(x, y, linewidth=2.0, color=color_map[k], zorder=3, label=str(k))
            ax2.scatter(x, y, s=30, alpha=0.6, color=color_map[k], zorder=4, label="_nolegend_")

        # Mark convergence: vertical dashed line where fitted curve enters and stays near plateau.
        if conv_points:
            ymin, _ = ax2.get_ylim()
            for kk, cx, cy, py in conv_points:
                if (not np.isfinite(cx)) or (not np.isfinite(cy)) or (not np.isfinite(py)):
                    continue
                ax2.vlines(
                    cx,
                    ymin=ymin,
                    ymax=cy,
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.9,
                    color=color_map[kk],
                    zorder=2,
                    label="_nolegend_",
                )
                # plateau guide segment (helps visually justify the 'converged to what')
                ax2.hlines(
                    py,
                    xmin=cx,
                    xmax=x_right,
                    linestyle=":",
                    linewidth=2.0,
                    alpha=0.6,
                    color=color_map[kk],
                    zorder=2,
                    label="_nolegend_",
                )

        ax2.set_xlabel("Step")
        ax2.set_ylabel("Validation Accuracy")
        ax2.legend(title="kappa", loc="lower right")

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
