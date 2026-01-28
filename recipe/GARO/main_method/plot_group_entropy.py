#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot mean entropy curves for groups of columns from a CSV.

Supports manual grouping (list[list[str]]) or automatic grouping ("auto"):

- Manual:
    --groups  '[["col_a","col_b"],["col_c"]]'
    --legends '[["Group AB"],["Group C"]]'
  Each group curve is the row-wise mean of its columns.

- Auto:
    --groups auto
  Treat every column (except "Step") as an individual run curve, extract shape features
  (non-parametric resampling + high-degree polynomial fit), then cluster the runs into
  a few clusters without specifying K (AffinityPropagation by default). Each cluster is
  plotted as the mean curve across its member columns. Legends are auto-generated from
  column names (joined by newlines).

Smoothing:
  --ema_strength > 0:
    raw curve: dashed + transparent
    EMA curve: solid (legend label uses only the group label)

Notes:
- We set seaborn theme/context as requested.
- Some seaborn versions can error with certain pandas versions when calling sns.lineplot.
  This script draws lines with matplotlib for compatibility, while keeping seaborn styling.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _parse_list_of_list_str(s: str, name: str) -> List[List[str]]:
    """Parse a string into list[list[str]]; accepts JSON or Python literal."""
    if s is None:
        raise ValueError(f"{name} is required.")

    # Try JSON first
    try:
        obj = json.loads(s)
    except Exception:
        # Fallback: Python literal
        try:
            obj = ast.literal_eval(s)
        except Exception as e:
            raise ValueError(
                f"Failed to parse {name}. Provide JSON like "
                f"'[[\"col1\",\"col2\"],[\"col3\"]]'. Original error: {e}"
            )

    if not isinstance(obj, list) or any(not isinstance(x, list) for x in obj):
        raise ValueError(f"{name} must be a list of list of strings.")

    out: List[List[str]] = []
    for i, inner in enumerate(obj):
        if any(not isinstance(v, str) for v in inner):
            raise ValueError(f"{name}[{i}] must be a list of strings.")
        out.append(inner)
    return out


def _ema_smooth(y: pd.Series, strength: float) -> pd.Series:
    """
    EMA with 'strength' interpreted as weight on previous EMA:
      ema[t] = strength * ema[t-1] + (1-strength) * y[t]
    Equivalent to pandas ewm(alpha=1-strength, adjust=False).mean()
    """
    alpha = 1.0 - strength
    # ignore_na=True helps when some points are NaN (not available in some older pandas)
    try:
        return y.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    except TypeError:
        return y.ewm(alpha=alpha, adjust=False).mean()


def _interp_to_grid(step: np.ndarray, y: np.ndarray, n_grid: int) -> np.ndarray:
    """Resample (step, y) to a fixed grid using linear interpolation."""
    m = np.isfinite(step) & np.isfinite(y)
    step2 = step[m]
    y2 = y[m]
    if step2.size == 0:
        return np.full((n_grid,), np.nan, dtype=float)

    # Ensure ascending step
    order = np.argsort(step2)
    step2 = step2[order]
    y2 = y2[order]

    # Unique steps for interp stability
    step_u, idx = np.unique(step2, return_index=True)
    y_u = y2[idx]

    if step_u.size == 1:
        return np.full((n_grid,), y_u[0], dtype=float)

    grid = np.linspace(step_u.min(), step_u.max(), n_grid)
    return np.interp(grid, step_u, y_u)


def _poly_features(step: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """
    Fit a high-degree polynomial to capture curve shape.
    Returns coefficients (degree+1). Uses normalized step in [0,1] for stability.
    """
    m = np.isfinite(step) & np.isfinite(y)
    step2 = step[m]
    y2 = y[m]
    if step2.size < degree + 1:
        return np.zeros((degree + 1,), dtype=float)

    # Normalize step to [0, 1]
    smin, smax = float(step2.min()), float(step2.max())
    if smax <= smin:
        return np.zeros((degree + 1,), dtype=float)
    x = (step2 - smin) / (smax - smin)

    # Normalize y (shape-only)
    y2 = y2.astype(float)
    mu = float(np.nanmean(y2))
    sd = float(np.nanstd(y2))
    if not np.isfinite(sd) or sd < 1e-8:
        y_norm = y2 - mu
    else:
        y_norm = (y2 - mu) / sd

    try:
        coeff = np.polyfit(x, y_norm, degree)
        # polyfit returns highest degree first; keep as-is
        return coeff.astype(float, copy=False)
    except Exception:
        return np.zeros((degree + 1,), dtype=float)


def _build_run_features(
    df: pd.DataFrame,
    step_col: str,
    run_cols: List[str],
    n_grid: int,
    poly_degree: int,
    use_grid: bool = True,
    use_poly: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix for each run column. Concatenates:
      - non-parametric resampled curve values on a fixed grid (z-scored)
      - high-degree polynomial coefficients (already based on z-scored y)
    """
    step = pd.to_numeric(df[step_col], errors="coerce").to_numpy(dtype=float)
    feats: List[np.ndarray] = []
    kept_cols: List[str] = []

    for c in run_cols:
        y = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)

        parts = []
        if use_grid:
            grid_y = _interp_to_grid(step, y, n_grid=n_grid)
            # z-score for shape (ignore NaNs)
            mu = np.nanmean(grid_y)
            sd = np.nanstd(grid_y)
            if np.isfinite(sd) and sd >= 1e-8:
                grid_y = (grid_y - mu) / sd
            else:
                grid_y = grid_y - (mu if np.isfinite(mu) else 0.0)
            # Replace NaN with 0 after normalization to keep dimensions stable
            grid_y = np.nan_to_num(grid_y, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(grid_y.astype(float, copy=False))

        if use_poly:
            poly = _poly_features(step, y, degree=poly_degree)
            poly = np.nan_to_num(poly, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(poly.astype(float, copy=False))

        if not parts:
            continue

        feat = np.concatenate(parts, axis=0)
        feats.append(feat)
        kept_cols.append(c)

    if not feats:
        raise ValueError("No valid run columns found to build features.")
    X = np.stack(feats, axis=0)  # (n_runs, n_features)
    return X, kept_cols


def _cluster_runs_affinity_propagation(X: np.ndarray, damping: float = 0.8) -> np.ndarray:
    """
    Cluster with Affinity Propagation (no need to specify K).
    Uses a precomputed similarity matrix S_ij = -||xi-xj||^2.
    """
    # Standardize features (manual, to avoid sklearn dependency here)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd

    # Similarity matrix: negative squared euclidean distances
    # S_ij = -||zi-zj||^2
    # Compute efficiently with (a-b)^2 = a^2 + b^2 - 2ab
    sq = np.sum(Z * Z, axis=1, keepdims=True)  # (n,1)
    S = -(sq + sq.T - 2.0 * (Z @ Z.T))
    # Preference controls number of clusters; median similarity is a common heuristic
    preference = float(np.median(S))

    try:
        from sklearn.cluster import AffinityPropagation  # type: ignore
    except Exception as e:
        raise ImportError(
            "Auto clustering requires scikit-learn (for AffinityPropagation). "
            "Install scikit-learn or run with manual --groups.\n"
            f"Import error: {e}"
        )

    ap = AffinityPropagation(affinity="precomputed", damping=damping, preference=preference, random_state=0)
    labels = ap.fit_predict(S)
    return labels.astype(int, copy=False)


def _cluster_runs_fallback_threshold(X: np.ndarray) -> np.ndarray:
    """
    Fallback non-parametric-ish clustering without sklearn:
    build a graph by connecting pairs whose distance is below an automatically
    chosen threshold, then take connected components as clusters.
    """
    # Standardize
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd

    # Pairwise distances
    # D_ij = ||zi-zj||
    sq = np.sum(Z * Z, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * (Z @ Z.T)
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)

    # Choose threshold from distance distribution (robust)
    tri = D[np.triu_indices(D.shape[0], k=1)]
    if tri.size == 0:
        return np.zeros((Z.shape[0],), dtype=int)
    q25 = float(np.percentile(tri, 25))
    q50 = float(np.percentile(tri, 50))
    q75 = float(np.percentile(tri, 75))
    iqr = q75 - q25
    # A conservative threshold; tends to create a few clusters if structure exists
    thr = max(q25, q50 - 0.5 * iqr)

    # Build adjacency
    n = D.shape[0]
    adj = (D <= thr) & (~np.eye(n, dtype=bool))

    # Connected components (DFS)
    labels = -np.ones((n,), dtype=int)
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            u = stack.pop()
            nbrs = np.where(adj[u])[0]
            for v in nbrs:
                if labels[v] == -1:
                    labels[v] = cid
                    stack.append(v)
        cid += 1
    return labels


def _auto_groups_from_csv(
    df: pd.DataFrame,
    step_col: str = "Step",
    n_grid: int = 60,
    poly_degree: int = 12,
    feature_mode: str = "grid+poly",
    ap_damping: float = 0.8,
) -> Tuple[List[List[str]], List[str]]:
    """
    Build groups and auto legends from CSV columns (excluding Step).

    Returns:
      groups: list of clusters, each is list of column names
      group_labels: list of legend labels (joined column names with newlines)
    """
    run_cols = [c for c in df.columns if c != step_col]
    if not run_cols:
        raise ValueError(f"No run columns found (everything except '{step_col}').")

    use_grid = feature_mode in ("grid", "grid+poly")
    use_poly = feature_mode in ("poly", "grid+poly")

    X, kept_cols = _build_run_features(
        df=df,
        step_col=step_col,
        run_cols=run_cols,
        n_grid=n_grid,
        poly_degree=poly_degree,
        use_grid=use_grid,
        use_poly=use_poly,
    )

    # Cluster
    try:
        labels = _cluster_runs_affinity_propagation(X, damping=ap_damping)
    except Exception:
        # Fallback (no sklearn or AP failure)
        labels = _cluster_runs_fallback_threshold(X)

    # Collect clusters (stable order by size desc, then name)
    clusters: Dict[int, List[str]] = {}
    for col, lab in zip(kept_cols, labels.tolist()):
        clusters.setdefault(int(lab), []).append(col)

    # Sort members inside cluster for stable legends
    for k in clusters:
        clusters[k] = sorted(clusters[k])

    # Order clusters by size desc
    ordered_keys = sorted(clusters.keys(), key=lambda k: (-len(clusters[k]), k))
    groups = [clusters[k] for k in ordered_keys]
    group_labels = ["\n".join(cols) for cols in groups]
    return groups, group_labels


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=r"recipe/GARO/main_method/entropy_llama.csv",
        help="Input CSV path",
    )

    parser.add_argument(
        "--groups",
        type=str,
        default="auto",
        help=(
            "Either 'auto' or JSON/Python literal list[list[str]] of column names, "
            "e.g. '[[\"a\",\"b\"],[\"c\"]]'."
        ),
    )

    # Accept both --legends and --legend (alias)
    parser.add_argument(
        "--legends",
        type=str,
        default=None,
        help="JSON/Python literal list[list[str]] labels, same length as groups (ignored when groups=auto).",
    )
    parser.add_argument(
        "--legend",
        type=str,
        default=None,
        help="Alias of --legends (ignored when groups=auto).",
    )

    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.4,
        help="Seaborn font scale multiplier",
    )
    parser.add_argument(
        "--ema_strength",
        type=float,
        default=0.9,
        help="EMA smoothing strength in [0, 1). 0 disables smoothing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="If set, save figure to this path; otherwise plt.show().",
    )

    # Auto-clustering knobs (optional)
    parser.add_argument(
        "--auto_feature_mode",
        type=str,
        default="grid+poly",
        choices=["grid", "poly", "grid+poly"],
        help="Feature extraction mode for groups=auto.",
    )
    parser.add_argument(
        "--auto_grid",
        type=int,
        default=60,
        help="Number of grid points used for non-parametric resampling (groups=auto).",
    )
    parser.add_argument(
        "--auto_poly_degree",
        type=int,
        default=12,
        help="Polynomial degree used for curve-shape fitting (groups=auto).",
    )
    parser.add_argument(
        "--auto_ap_damping",
        type=float,
        default=0.8,
        help="AffinityPropagation damping (groups=auto).",
    )

    args = parser.parse_args()

    if not (0.0 <= args.ema_strength < 1.0):
        raise ValueError("--ema_strength must be in [0, 1).")

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)

    if "Step" not in df.columns:
        raise KeyError(f'"Step" column not found in {in_path}. Available columns: {list(df.columns)[:30]} ...')

    # Clean and sort by Step
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df = df[df["Step"].notna()].copy()
    df.sort_values("Step", inplace=True)

    groups_arg = (args.groups or "").strip()
    if groups_arg.lower() == "auto":
        groups, auto_labels = _auto_groups_from_csv(
            df=df,
            step_col="Step",
            n_grid=args.auto_grid,
            poly_degree=args.auto_poly_degree,
            feature_mode=args.auto_feature_mode,
            ap_damping=args.auto_ap_damping,
        )
        group_labels = auto_labels  # legend labels from column names
    else:
        groups = _parse_list_of_list_str(groups_arg, "groups")
        legends_str = args.legends if args.legends is not None else args.legend
        if legends_str is None:
            raise ValueError("For manual groups, you must provide --legends (or --legend).")
        legends = _parse_list_of_list_str(legends_str, "legends")
        if len(groups) != len(legends):
            raise ValueError(f"len(legends) must equal len(groups). Got {len(legends)} vs {len(groups)}.")
        group_labels = [", ".join(parts) if parts else f"group_{i}" for i, parts in enumerate(legends)]

    # Seaborn styling (as requested)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    sns.set_context("talk", font_scale=args.font_scale)
    plt.figure(figsize=(12, 8))

    palette = sns.color_palette(n_colors=len(groups))

    for i, cols in enumerate(groups):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Group {i} has missing columns: {missing}. "
                f"Available columns example: {list(df.columns)[:30]} ..."
            )

        tmp = df[cols].apply(pd.to_numeric, errors="coerce")
        y_raw = tmp.mean(axis=1, skipna=True)

        color = palette[i]
        label = group_labels[i] if i < len(group_labels) else f"group_{i}"

        if args.ema_strength > 0.0:
            # raw curve: dashed + transparent, no legend entry
            plt.plot(df["Step"].values, y_raw.values, linestyle="--", alpha=0.35, color=color)
            # smoothed curve: solid, with legend label
            y_ema = _ema_smooth(y_raw, args.ema_strength)
            plt.plot(df["Step"].values, y_ema.values, linestyle="-", alpha=1.0, color=color, label=label)
        else:
            plt.plot(df["Step"].values, y_raw.values, linestyle="-", alpha=1.0, color=color, label=label)

    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.legend()  # no title requested

    plt.tight_layout()
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved figure to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
