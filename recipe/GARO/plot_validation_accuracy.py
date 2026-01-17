# python recipe/GARO/plot_wandb_multi_val_accuracy.py --run_ids cvui4zn7, 654u0res, fc5r64vw, ilqbp27a --max_steps 940
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import wandb

def infer_step_from_filename(name: str):
    nums = re.findall(r"\d+", name)
    if not nums:
        return None
    return int(max(nums, key=lambda x: int(x)))


def load_table_json(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy_from_table(table_json: dict):
    columns = table_json.get("columns")
    data = table_json.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        raise ValueError("Unexpected table json format: missing 'columns' or 'data'")

    score_idx = [i for i, c in enumerate(columns) if isinstance(c, str) and c.startswith("score_")]
    if not score_idx:
        raise ValueError("No columns starting with 'score_' found")

    vals = []
    for row in data:
        if not isinstance(row, list):
            continue
        for i in score_idx:
            if i >= len(row):
                continue
            v = row[i]
            if v is None:
                continue
            if isinstance(v, bool):
                v = int(v)
            elif isinstance(v, (int, float)):
                pass
            elif isinstance(v, str) and v.strip() in ("0", "1"):
                v = int(v.strip())
            else:
                continue
            vals.append(float(v))

    if not vals:
        raise ValueError("Found score_ columns but no numeric values")

    return sum(vals) / len(vals), len(vals)


def fetch_one_run_curve(api: wandb.Api, entity: str, project: str, run_id: str,
                        prefix: str, outdir: Path, fallback_every: int):
    run = api.run(f"{entity}/{project}/{run_id}")

    files = []
    for f in run.files():
        name = getattr(f, "name", "")
        if name.startswith(prefix) and name.endswith(".table.json"):
            files.append(f)
    if not files:
        raise RuntimeError(f"[{run_id}] No files found under prefix='{prefix}' ending with .table.json")

    dl_dir = outdir / "downloaded_tables" / run_id
    dl_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        name = f.name
        local_path = Path(f.download(root=str(dl_dir), replace=True).name)

        tj = load_table_json(local_path)
        acc, n = compute_accuracy_from_table(tj)

        step = infer_step_from_filename(name)
        updated_at = getattr(f, "updatedAt", None) or getattr(f, "updated_at", None)

        rows.append({
            "run_id": run_id,
            "file": name,
            "updated_at": updated_at,
            "step_inferred": step,
            "acc": acc,
            "n_scores_used": n,
        })

    df = pd.DataFrame(rows)

    inferred_ratio = df["step_inferred"].notna().mean()
    if inferred_ratio >= 0.6:
        df = df.sort_values(["step_inferred", "updated_at"], na_position="last")
        steps = df["step_inferred"].tolist()
        filled = []
        last = None
        for i, s in enumerate(steps):
            if pd.notna(s):
                last = int(s)
                filled.append(last)
            else:
                if last is None:
                    filled.append(i * fallback_every)
                else:
                    last = last + fallback_every
                    filled.append(last)
        df["step"] = filled
    else:
        df = df.sort_values(["updated_at", "file"])
        df["step"] = [i * fallback_every for i in range(len(df))]

    # save per-run FULL csv (never truncated)
    csv_path = outdir / f"val_accuracy_curve_{run_id}.csv"
    df[["run_id", "step", "acc", "file", "updated_at", "n_scores_used"]].to_csv(csv_path, index=False)

    label = run.name if getattr(run, "name", None) else run_id
    return df[["run_id", "step", "acc"]].copy(), label, csv_path


def parse_run_ids(items):
    run_ids = []
    for item in (items or []):
        parts = [p.strip() for p in item.split(",") if p.strip()]
        run_ids.extend(parts)
    seen = set()
    uniq = []
    for r in run_ids:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="sheet")
    ap.add_argument("--project", default="GARO")
    ap.add_argument("--run_id", action="append",
                    help="Can be provided multiple times; each can also be comma-separated")
    ap.add_argument("--run_ids", default=None,
                    help="Comma-separated run ids (alternative to --run_id)")
    ap.add_argument("--prefix", default="media/table/val")
    ap.add_argument("--outdir", default="wandb_val_acc_out")
    ap.add_argument("--fallback_every", type=int, default=5)

    # NEW: only affects plotting (and an optional plotting-csv)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Max step to INCLUDE IN PLOT only (data beyond this is still saved to csv). "
                         "E.g. 1000 keeps steps <= 1000.")

    args = ap.parse_args()

    run_id_inputs = []
    if args.run_id:
        run_id_inputs.extend(args.run_id)
    if args.run_ids:
        run_id_inputs.append(args.run_ids)
    run_ids = parse_run_ids(run_id_inputs)
    if not run_ids:
        raise SystemExit("Please provide at least one run id via --run_id or --run_ids")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()

    all_curves = []
    labels = {}
    for rid in run_ids:
        df_curve, label, csv_path = fetch_one_run_curve(
            api=api,
            entity=args.entity,
            project=args.project,
            run_id=rid,
            prefix=args.prefix,
            outdir=outdir,
            fallback_every=args.fallback_every,
        )
        all_curves.append(df_curve)
        labels[rid] = label
        print(f"[OK] {rid} -> {csv_path}")

    merged = pd.concat(all_curves, ignore_index=True)
    merged_csv = outdir / "val_accuracy_curve_ALL.csv"
    merged.to_csv(merged_csv, index=False)

    # NEW: plotting view (truncated by max_steps)
    if args.max_steps is not None:
        merged_plot = merged[merged["step"] <= args.max_steps].copy()
        merged_plot_csv = outdir / f"val_accuracy_curve_ALL_for_plot_upto_{args.max_steps}.csv"
        merged_plot.to_csv(merged_plot_csv, index=False)
        print(f"[OK] saved (plot-truncated): {merged_plot_csv}")
    else:
        merged_plot = merged

    # Plot all runs on one figure
    plt.figure()
    for rid in run_ids:
        sub = merged_plot[merged_plot["run_id"] == rid].sort_values("step")
        plt.plot(sub["step"], sub["acc"], label=labels.get(rid, rid))

    plt.xlabel("Step")
    plt.ylabel("Validation Accuracy")
    title = f"W&B val accuracy curves ({args.entity}/{args.project})"
    if args.max_steps is not None:
        title += f" [<= {args.max_steps}]"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    fig_path = outdir / "val_accuracy_curves_multi.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)

    print(f"[OK] saved: {fig_path}")
    print(f"[OK] saved: {merged_csv}")


if __name__ == "__main__":
    main()