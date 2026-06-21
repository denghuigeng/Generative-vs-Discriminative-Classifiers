#!/usr/bin/env python
"""Plot reliability diagrams for selected completed runs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_scores(value: str):
    parsed = ast.literal_eval(value)
    if isinstance(parsed, dict):
        return [float(parsed[key]) for key in sorted(parsed, key=lambda x: int(x))]
    return [float(x) for x in parsed]


def reliability_points(y_true, probs, n_bins):
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = prediction == y_true
    xs, ys, counts = [], [], []
    for index in range(n_bins):
        low, high = index / n_bins, (index + 1) / n_bins
        mask = (confidence >= low) & (
            confidence <= high if index == n_bins - 1 else confidence < high
        )
        if mask.any():
            xs.append(float(confidence[mask].mean()))
            ys.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
    return xs, ys, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--sample_size", type=int, default=-1)
    parser.add_argument(
        "--models", nargs="+", default=["enc", "ar", "arpseudo", "mlm"]
    )
    parser.add_argument("--initialization", default="scratch")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    grouped = {}
    for args_path in Path(args.output_root).rglob("args.json"):
        run_args = json.loads(args_path.read_text())
        if run_args.get("dataset") != args.dataset:
            continue
        if int(run_args.get("layers", 0)) != args.layers:
            continue
        if int(run_args.get("sample_size", 0)) != args.sample_size:
            continue
        if run_args.get("model") not in args.models:
            continue
        if run_args.get("initialization", "scratch") != args.initialization:
            continue
        if not run_args.get("probabilities_available", True):
            continue
        prediction_path = args_path.parent / "predictions.csv"
        if not prediction_path.exists():
            continue
        frame = pd.read_csv(prediction_path)
        y_true = frame["ground_truth"].astype(int).to_numpy()
        probs = np.asarray([parse_scores(x) for x in frame["scores"]])
        grouped.setdefault(run_args["model"], []).append((y_true, probs))

    if not grouped:
        raise FileNotFoundError("No matching probabilistic prediction files found.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="perfect")
    for model, chunks in grouped.items():
        y_true = np.concatenate([chunk[0] for chunk in chunks])
        probs = np.concatenate([chunk[1] for chunk in chunks])
        confidence, accuracy, _ = reliability_points(y_true, probs, args.bins)
        ax.plot(confidence, accuracy, marker="o", label=model)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(
        f"Reliability: {args.dataset}, {args.layers}L, sample={args.sample_size}"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = (
        Path(args.out)
        if args.out
        else Path(args.output_root)
        / "figures"
        / f"reliability_{args.dataset}_{args.layers}L_{args.sample_size}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    print(output_path)


if __name__ == "__main__":
    main()
