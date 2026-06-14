#!/usr/bin/env python
"""Aggregate prediction files and draw reproduction/extension figures."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

DEFAULT_DATASETS = ["agnews", "emotion", "rottentomatoes", "sst5", "twitter"]
DEFAULT_LAYERS = [1, 6, 12]
ORDINAL_DATASETS = {"sst5"}


def parse_scores(value: str) -> List[float]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, dict):
        return [float(parsed[k]) for k in sorted(parsed, key=lambda x: int(x))]
    return [float(x) for x in parsed]


def ece_mce(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        ece += mask.mean() * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}


def unimodal_rate(probs: np.ndarray, eps: float = 1e-9) -> float:
    ok = []
    for row in probs:
        peak = int(np.argmax(row))
        left_ok = np.all(np.diff(row[: peak + 1]) >= -eps)
        right_ok = np.all(np.diff(row[peak:]) <= eps)
        ok.append(left_ok and right_ok)
    return float(np.mean(ok))


def read_run(path: Path) -> Optional[Dict[str, object]]:
    pred_path = path / "predictions.csv"
    args_path = path / "args.json"
    if not pred_path.exists() or not args_path.exists():
        return None
    args = json.loads(args_path.read_text())
    df = pd.read_csv(pred_path)
    y_true = df["ground_truth"].astype(int).to_numpy()
    y_pred = df["predicted_label"].astype(int).to_numpy()
    probs = np.asarray([parse_scores(x) for x in df["scores"]])

    row = {
        "model": args["model"],
        "dataset": args["dataset"],
        "sample_size": int(args["sample_size"]),
        "seed": int(args["seed"]),
        "layers": int(args["layers"]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    row.update(ece_mce(y_true, probs))
    if args["dataset"] in ORDINAL_DATASETS:
        row["mae"] = float(mean_absolute_error(y_true, y_pred))
        row["mse"] = float(mean_squared_error(y_true, y_pred))
        row["um"] = unimodal_rate(probs)
    return row


def collect(output_root: Path) -> pd.DataFrame:
    rows = []
    for args_path in output_root.rglob("args.json"):
        row = read_run(args_path.parent)
        if row:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No completed runs found under {output_root}")
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, out_path: Path, datasets: Optional[List[str]] = None) -> None:
    datasets = datasets or ["agnews", "sst5"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    colors = {"ar": "#2E8B57", "enc": "#F28E2B", "mlm": "#1F77B4"}
    labels = {"ar": "AR", "enc": "ENC", "mlm": "MLM"}
    for ax, dataset in zip(axes, datasets):
        sub = summary[summary["dataset"] == dataset]
        for model in ["enc", "ar", "mlm"]:
            cur = sub[sub["model"] == model].sort_values("sample_size")
            if cur.empty or f"{metric}_mean" not in cur.columns:
                continue
            ax.plot(cur["sample_size"], cur[f"{metric}_mean"], marker="o", label=labels[model], color=colors[model])
            if f"{metric}_std" in cur:
                lo = cur[f"{metric}_mean"] - cur[f"{metric}_std"].fillna(0)
                hi = cur[f"{metric}_mean"] + cur[f"{metric}_std"].fillna(0)
                ax.fill_between(cur["sample_size"], lo, hi, color=colors[model], alpha=0.16)
        ax.set_xscale("log", base=2)
        ax.set_title(dataset.upper() if dataset == "agnews" else "SST-5")
        ax.set_xlabel("sample size")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_grid(summary: pd.DataFrame, metric: str, out_path: Path, datasets: List[str], layers: List[int]) -> None:
    colors = {"ar": "#2E8B57", "enc": "#F28E2B", "mlm": "#1F77B4"}
    labels = {"ar": "AR", "enc": "ENC", "mlm": "MLM"}
    fig, axes = plt.subplots(len(layers), len(datasets), figsize=(3.1 * len(datasets), 2.55 * len(layers)), sharex=True)
    if len(layers) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(datasets) == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, layer in enumerate(layers):
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]
            sub = summary[(summary["dataset"] == dataset) & (summary["layers"] == layer)]
            for model in ["enc", "ar", "mlm"]:
                cur = sub[sub["model"] == model].sort_values("sample_size")
                if cur.empty or f"{metric}_mean" not in cur.columns:
                    continue
                ax.plot(cur["sample_size"], cur[f"{metric}_mean"], marker="o", linewidth=1.8, markersize=3.5, label=labels[model], color=colors[model])
                if f"{metric}_std" in cur:
                    lo = cur[f"{metric}_mean"] - cur[f"{metric}_std"].fillna(0)
                    hi = cur[f"{metric}_mean"] + cur[f"{metric}_std"].fillna(0)
                    ax.fill_between(cur["sample_size"], lo, hi, color=colors[model], alpha=0.15)
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(dataset)
            if c == 0:
                ax.set_ylabel(f"Layer {layer}\n{metric}")
            if r == len(layers) - 1:
                ax.set_xlabel("sample size")
            if r == 0 and c == len(datasets) - 1:
                ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro")
    parser.add_argument("--layers", type=int, nargs="+", default=[12])
    parser.add_argument("--datasets", nargs="+", default=["agnews", "sst5"])
    parser.add_argument("--full_grid", action="store_true", help="Draw a Figure-2-like layers x datasets grid.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = collect(output_root)
    runs.to_csv(output_root / "all_run_metrics.csv", index=False)

    runs = runs[runs["layers"].isin(args.layers)]
    runs = runs[runs["dataset"].isin(args.datasets)]
    agg_cols = ["model", "dataset", "layers", "sample_size"]
    summary = runs.groupby(agg_cols).agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([x for x in col if x]) for col in summary.columns]
    layer_tag = "_".join(map(str, args.layers))
    summary.to_csv(output_root / f"summary_layers_{layer_tag}.csv", index=False)

    for metric in ["weighted_f1", "ece", "mce"]:
        if args.full_grid:
            plot_grid(summary, metric, fig_dir / f"{metric}_grid_layers_{layer_tag}.png", args.datasets, args.layers)
        elif len(args.layers) == 1:
            one_layer = summary[summary["layers"] == args.layers[0]]
            plot_metric(one_layer, metric, fig_dir / f"{metric}_layers_{layer_tag}.png", args.datasets)

    sst5 = summary[summary["dataset"].isin(sorted(ORDINAL_DATASETS))]
    for metric in ["mae", "mse", "um"]:
        if f"{metric}_mean" in sst5.columns and sst5[f"{metric}_mean"].notna().any():
            if args.full_grid:
                ordinal_datasets = [d for d in args.datasets if d in ORDINAL_DATASETS]
                if ordinal_datasets:
                    plot_grid(sst5, metric, fig_dir / f"ordinal_{metric}_grid_layers_{layer_tag}.png", ordinal_datasets, args.layers)
            elif len(args.layers) == 1:
                ordinal_datasets = [d for d in args.datasets if d in ORDINAL_DATASETS]
                if ordinal_datasets:
                    plot_metric(sst5[sst5["layers"] == args.layers[0]], metric, fig_dir / f"ordinal_{metric}_layers_{layer_tag}.png", ordinal_datasets)

    print(f"Wrote metrics and figures to {output_root}")


if __name__ == "__main__":
    main()
