#!/usr/bin/env python
"""Aggregate Q2 robustness summaries across models, datasets and seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows = []
    for summary_path in output_root.rglob("robustness/*/summary.csv"):
        run_dir = summary_path.parents[2]
        args_path = run_dir / "args.json"
        if not args_path.exists():
            continue
        run_args = json.loads(args_path.read_text())
        frame = pd.read_csv(summary_path)
        for key in [
            "model",
            "dataset",
            "sample_size",
            "seed",
            "layers",
            "initialization",
        ]:
            frame[key] = run_args.get(key)
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No robustness summaries found under {output_root}")

    all_results = pd.concat(rows, ignore_index=True)
    all_results.to_csv(output_root / "robustness_all_runs.csv", index=False)

    grouped = (
        all_results.groupby(["model", "dataset", "layers", "noise", "rate"])[
            ["weighted_f1", "relative_f1_drop"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = ["_".join(x for x in col if x) for col in grouped.columns]
    grouped.to_csv(output_root / "robustness_summary.csv", index=False)

    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted(grouped["dataset"].unique()):
        for noise in ["drop", "substitute"]:
            sub = grouped[
                (grouped["dataset"] == dataset) & (grouped["noise"] == noise)
            ]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for (model, layers), cur in sub.groupby(["model", "layers"]):
                cur = cur.sort_values("rate")
                ax.plot(
                    cur["rate"],
                    cur["weighted_f1_mean"],
                    marker="o",
                    label=f"{model}-{layers}L",
                )
            ax.set_title(f"{dataset}: random token {noise}")
            ax.set_xlabel("noise rate")
            ax.set_ylabel("weighted-F1")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(
                figure_dir / f"robustness_{dataset}_{noise}.png", dpi=240
            )
            plt.close(fig)

    print(f"Wrote robustness tables and figures to {output_root}")


if __name__ == "__main__":
    main()
