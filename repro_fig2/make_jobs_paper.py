#!/usr/bin/env python
"""Generate TSV task matrices for paper-style experiments.

Each row contains:
  model dataset sample_size seed layers heads initialization
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(".")

ALL_DATASETS = [
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
FIGURE2_DATASETS = ["agnews", "emotion", "rottentomatoes", "sst5", "twitter"]
ALL_MODELS = ["enc", "ar", "arpseudo", "mlm"]
FIGURE2_MODELS = ["enc", "ar", "mlm"]
SAMPLES = [128, 256, 512, 1024, 2048, 4096, -1]
SEEDS = [79140, 24561, 54641]
LAYERS = [(1, 1), (6, 6), (12, 12)]


PRESETS = {
    "figure2": (FIGURE2_DATASETS, FIGURE2_MODELS),
    "figure8": (ALL_DATASETS, FIGURE2_MODELS),
    "non_diff_full": (ALL_DATASETS, ALL_MODELS),
    "course_core": (["agnews", "sst5"], ["enc", "ar"]),
}


def csv_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="figure2")
    parser.add_argument("--datasets", type=csv_list, default=None)
    parser.add_argument("--models", type=csv_list, default=None)
    parser.add_argument("--samples", type=csv_list, default=None)
    parser.add_argument("--layers", type=csv_list, default=None)
    parser.add_argument("--seeds", type=csv_list, default=None)
    parser.add_argument("--one_seed", action="store_true")
    parser.add_argument("--initialization", choices=["scratch", "pretrained"], default="scratch")
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "repro_fig2" / "jobs_paper.tsv"),
    )
    args = parser.parse_args()

    preset_datasets, preset_models = PRESETS[args.preset]
    datasets = args.datasets or preset_datasets
    models = args.models or preset_models
    samples = [int(x) for x in (args.samples or SAMPLES)]
    seeds = [int(x) for x in (args.seeds or SEEDS)]
    if args.one_seed:
        seeds = seeds[:1]

    if args.layers:
        layer_pairs = [(int(x), int(x)) for x in args.layers]
    else:
        layer_pairs = LAYERS

    if args.initialization == "pretrained":
        layer_pairs = [(12, 12)]
        models = [model for model in models if model in {"enc", "ar"}]

    rows = []
    for dataset in datasets:
        for model in models:
            for layers, heads in layer_pairs:
                for sample in samples:
                    for seed in seeds:
                        rows.append(
                            f"{model}\t{dataset}\t{sample}\t{seed}\t"
                            f"{layers}\t{heads}\t{args.initialization}\n"
                        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(rows))
    print(f"Preset: {args.preset}")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Samples: {samples}")
    print(f"Seeds: {seeds}")
    print(f"Layer pairs: {layer_pairs}")
    print(f"Initialization: {args.initialization}")
    print(f"Wrote {len(rows)} jobs to {out_path}")
    print(f"Use SLURM array range: 0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
