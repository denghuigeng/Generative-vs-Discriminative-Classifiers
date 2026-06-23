#!/usr/bin/env python
"""Generate DIFF experiment jobs matching the paper matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(".")

DATASETS = {
    "imdb": "imdb",
    "agnews": "ag_news",
    "emotion": "emotion",
    "hatespeech": "SetFit/hate_speech_offensive",
    "multiclasssentiment": "Sp1786/multiclass-sentiment-analysis-dataset",
    "rottentomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "sst2": "SetFit/sst2",
    "sst5": "SetFit/sst5",
    "twitter": "zeroshot/twitter-financial-news-sentiment",
}
MODELS = ["small", "medium", "large"]
SAMPLES = [128, 256, 512, 1024, 2048, 4096, -1]
SEEDS = [79140, 24561, 54641]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one_seed", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--samples", nargs="+", type=int, default=SAMPLES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--out",
        default=str(ROOT / "repro_fig2" / "jobs_diff.tsv"),
    )
    args = parser.parse_args()

    seeds = args.seeds[:1] if args.one_seed else args.seeds
    rows = []
    for dataset in args.datasets:
        for model in args.models:
            for sample in args.samples:
                for seed in seeds:
                    rows.append(
                        f"{dataset}\t{DATASETS[dataset]}\t{model}\t{sample}\t{seed}\n"
                    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(rows))
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Samples: {args.samples}")
    print(f"Seeds: {seeds}")
    print(f"Wrote {len(rows)} DIFF jobs to {out}")
    print(f"Use SLURM array range: 0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
