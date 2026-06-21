#!/usr/bin/env python
"""Aggregate prediction files and draw reproduction/extension figures."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

DEFAULT_DATASETS = [
    "imdb",
    "agnews",
    "emotion",
    "hatespeech",
    "multiclasssentiment",
    "rottentomatoes",
    "sst2",
    "sst5",
    "twitter",
]
DEFAULT_LAYERS = [1, 6, 12]
ORDINAL_DATASETS = {"hatespeech", "multiclasssentiment", "sst5", "twitter"}
ORDINAL_LABEL_ORDER = {
    "hatespeech": [0, 1, 2],
    "multiclasssentiment": [0, 1, 2],
    "sst5": [0, 1, 2, 3, 4],
    # Raw labels are Bearish=0, Bullish=1, Neutral=2.
    "twitter": [0, 2, 1],
}


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


def probability_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    true_probs = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    one_hot = np.eye(probs.shape[1], dtype=float)[y_true]
    return {
        "nll": float(-np.log(true_probs).mean()),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
    }


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
        "initialization": args.get("initialization", "scratch"),
        "sample_size": int(args["sample_size"]),
        "seed": int(args["seed"]),
        "layers": int(args["layers"]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    probabilities_available = bool(
        args.get("probabilities_available", args.get("model") != "diff")
    )
    if probabilities_available:
        row.update(ece_mce(y_true, probs))
        row.update(probability_metrics(y_true, probs))
    if probabilities_available and args["dataset"] in ORDINAL_DATASETS:
        order = ORDINAL_LABEL_ORDER[args["dataset"]]
        rank_by_raw_label = {raw_label: rank for rank, raw_label in enumerate(order)}
        ordered_true = np.asarray([rank_by_raw_label[int(x)] for x in y_true])
        ordered_pred = np.asarray([rank_by_raw_label[int(x)] for x in y_pred])
        ordered_probs = probs[:, order]
        expected_label = (
            ordered_probs * np.arange(ordered_probs.shape[1])
        ).sum(axis=1)
        row["mae"] = float(mean_absolute_error(ordered_true, ordered_pred))
        row["mse"] = float(mean_squared_error(ordered_true, ordered_pred))
        row["expected_mae"] = float(
            mean_absolute_error(ordered_true, expected_label)
        )
        row["expected_mse"] = float(
            mean_squared_error(ordered_true, expected_label)
        )
        row["um"] = unimodal_rate(ordered_probs)
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
    colors = {
        "ar": "#2E8B57",
        "arpseudo": "#9467BD",
        "enc": "#F28E2B",
        "mlm": "#1F77B4",
        "diff": "#D62728",
    }
    labels = {
        "ar": "AR",
        "arpseudo": "AR-pseudo",
        "enc": "ENC",
        "mlm": "MLM",
        "diff": "DIFF",
    }
    for ax, dataset in zip(axes, datasets):
        sub = summary[summary["dataset"] == dataset]
        for model in ["enc", "diff", "ar", "arpseudo", "mlm"]:
            cur = sub[sub["model"] == model].copy()
            if cur.empty or f"{metric}_mean" not in cur.columns:
                continue
            cur["plot_x"] = sample_plot_positions(cur["sample_size"])
            cur = cur.sort_values("plot_x")
            ax.plot(cur["plot_x"], cur[f"{metric}_mean"], marker="o", label=labels[model], color=colors[model])
            if f"{metric}_std" in cur:
                lo = cur[f"{metric}_mean"] - cur[f"{metric}_std"].fillna(0)
                hi = cur[f"{metric}_mean"] + cur[f"{metric}_std"].fillna(0)
                ax.fill_between(cur["plot_x"], lo, hi, color=colors[model], alpha=0.16)
        ax.set_xscale("log", base=2)
        set_sample_ticks(ax, sub["sample_size"])
        ax.set_title(dataset)
        ax.set_xlabel("sample size")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_grid(summary: pd.DataFrame, metric: str, out_path: Path, datasets: List[str], layers: List[int]) -> None:
    colors = {
        "ar": "#2E8B57",
        "arpseudo": "#9467BD",
        "enc": "#F28E2B",
        "mlm": "#1F77B4",
        "diff": "#D62728",
    }
    labels = {
        "ar": "AR",
        "arpseudo": "AR-pseudo",
        "enc": "ENC",
        "mlm": "MLM",
        "diff": "DIFF",
    }
    fig, axes = plt.subplots(len(layers), len(datasets), figsize=(3.1 * len(datasets), 2.55 * len(layers)), sharex=True)
    if len(layers) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(datasets) == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, layer in enumerate(layers):
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]
            sub = summary[(summary["dataset"] == dataset) & (summary["layers"] == layer)]
            for model in ["enc", "diff", "ar", "arpseudo", "mlm"]:
                cur = sub[sub["model"] == model].copy()
                if cur.empty or f"{metric}_mean" not in cur.columns:
                    continue
                cur["plot_x"] = sample_plot_positions(cur["sample_size"])
                cur = cur.sort_values("plot_x")
                ax.plot(cur["plot_x"], cur[f"{metric}_mean"], marker="o", linewidth=1.8, markersize=3.5, label=labels[model], color=colors[model])
                if f"{metric}_std" in cur:
                    lo = cur[f"{metric}_mean"] - cur[f"{metric}_std"].fillna(0)
                    hi = cur[f"{metric}_mean"] + cur[f"{metric}_std"].fillna(0)
                    ax.fill_between(cur["plot_x"], lo, hi, color=colors[model], alpha=0.15)
            ax.set_xscale("log", base=2)
            set_sample_ticks(ax, sub["sample_size"])
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


def sample_plot_positions(sample_sizes: Iterable[int]) -> np.ndarray:
    values = np.asarray(list(sample_sizes), dtype=int)
    regular = values[values > 0]
    full_position = int(regular.max() * 2) if len(regular) else 8192
    return np.where(values < 0, full_position, values)


def set_sample_ticks(ax, sample_sizes: Iterable[int]) -> None:
    values = sorted(set(int(x) for x in sample_sizes))
    regular = [x for x in values if x > 0]
    ticks = regular[:]
    labels = [str(x) for x in regular]
    if any(x < 0 for x in values):
        full_position = regular[-1] * 2 if regular else 8192
        ticks.append(full_position)
        labels.append("full")
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro")
    parser.add_argument(
        "--additional_output_roots",
        nargs="*",
        default=[],
        help="Optional extra roots, e.g. the separate DIFF output directory.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[12])
    parser.add_argument("--datasets", nargs="+", default=["agnews", "sst5"])
    parser.add_argument("--initialization", choices=["scratch", "pretrained"], default="scratch")
    parser.add_argument("--full_grid", action="store_true", help="Draw a Figure-2-like layers x datasets grid.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    frames = [collect(output_root)]
    frames.extend(collect(Path(root)) for root in args.additional_output_roots)
    runs = pd.concat(frames, ignore_index=True)
    runs.to_csv(output_root / "all_run_metrics.csv", index=False)

    runs = runs[runs["layers"].isin(args.layers)]
    runs = runs[runs["dataset"].isin(args.datasets)]
    runs = runs[runs["initialization"] == args.initialization]
    agg_cols = ["initialization", "model", "dataset", "layers", "sample_size"]
    metric_cols = [
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "ece",
        "mce",
        "nll",
        "brier",
        "mae",
        "mse",
        "expected_mae",
        "expected_mse",
        "um",
    ]
    metric_cols = [column for column in metric_cols if column in runs.columns]
    summary = runs.groupby(agg_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([x for x in col if x]) for col in summary.columns]
    layer_tag = "_".join(map(str, args.layers))
    summary.to_csv(output_root / f"summary_layers_{layer_tag}.csv", index=False)

    for metric in ["weighted_f1", "ece", "mce", "nll", "brier"]:
        if args.full_grid:
            plot_grid(summary, metric, fig_dir / f"{metric}_grid_layers_{layer_tag}.png", args.datasets, args.layers)
        elif len(args.layers) == 1:
            one_layer = summary[summary["layers"] == args.layers[0]]
            plot_metric(one_layer, metric, fig_dir / f"{metric}_layers_{layer_tag}.png", args.datasets)

    sst5 = summary[summary["dataset"].isin(sorted(ORDINAL_DATASETS))]
    for metric in ["mae", "mse", "expected_mae", "expected_mse", "um"]:
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
