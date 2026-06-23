#!/usr/bin/env python
"""Compatibility launcher for the canonical paper-style AR pipeline.

The original Lightning implementation is retained in ``train_gpt_legacy.py``
for loading historical checkpoints. New training commands are translated to
``repro_fig2/train_one.py`` so sampling, validation, early stopping, testing,
and result layout are shared with the full reproduction workflow.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


DATASET_ALIASES = {
    "imdb": "imdb",
    "IMDb": "imdb",
    "ag_news": "agnews",
    "emotion": "emotion",
    "SetFit/hate_speech_offensive": "hatespeech",
    "Sp1786/multiclass-sentiment-analysis-dataset": "multiclasssentiment",
    "cornell-movie-review-data/rotten_tomatoes": "rottentomatoes",
    "SetFit/sst2": "sst2",
    "SetFit/sst5": "sst5",
    "zeroshot/twitter-financial-news-sentiment": "twitter",
}
LAYERS_BY_SIZE = {"small": 1, "medium": 6, "full": 12}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_key", required=True, choices=DATASET_ALIASES)
    parser.add_argument(
        "--ckpt_dir",
        default="outputs/paper_repro",
        help="Root directory for structured checkpoints and predictions",
    )
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument(
        "--model_size",
        choices=LAYERS_BY_SIZE,
        default="full",
        help="small=1 layer, medium=6 layers, full=12 layers",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--n_tr_sub",
        type=int,
        default=-1,
        help="Training sample size; -1 uses the full training split",
    )
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val_size", type=int, default=480)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument(
        "--initialization",
        choices=["scratch", "pretrained"],
        default="scratch",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the translated canonical command without starting training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = LAYERS_BY_SIZE[args.model_size]
    visible_devices = torch.cuda.device_count()
    if not args.dry_run and visible_devices != args.n_devices:
        raise RuntimeError(
            f"Requested --n_devices {args.n_devices}, but "
            f"{visible_devices} CUDA device(s) are visible. Set "
            "CUDA_VISIBLE_DEVICES to expose exactly the requested GPUs."
        )

    gradient_accumulation = max(
        1, math.ceil(32 / max(1, args.bsz * args.n_devices))
    )
    project_root = Path(__file__).resolve().parents[1]
    canonical_script = project_root / "repro_fig2" / "train_one.py"
    output_root = Path(args.ckpt_dir).resolve()
    command = [
        sys.executable,
        str(canonical_script),
        "--model",
        "ar",
        "--dataset",
        DATASET_ALIASES[args.data_key],
        "--sample_size",
        str(args.n_tr_sub),
        "--seed",
        str(args.seed),
        "--layers",
        str(layers),
        "--heads",
        str(layers),
        "--initialization",
        args.initialization,
        "--epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.bsz),
        "--gradient_accumulation_steps",
        str(gradient_accumulation),
        "--eval_batch_size",
        str(args.bsz),
        "--inference_batch_size",
        str(args.inference_batch_size),
        "--max_len",
        str(args.max_len),
        "--lr",
        str(args.lr),
        "--val_size",
        str(args.val_size),
        "--precision",
        args.precision,
        "--output_root",
        str(output_root),
    ]
    if args.overwrite:
        command.append("--overwrite")

    sample_tag = "full" if args.n_tr_sub < 0 else str(args.n_tr_sub)
    output_dir = (
        output_root
        / f"init_{args.initialization}"
        / "ar"
        / DATASET_ALIASES[args.data_key]
        / f"layers_{layers}"
        / f"samples_{sample_tag}"
        / f"seed_{args.seed}"
    )

    print(f"Hugging Face endpoint: {os.environ['HF_ENDPOINT']}")
    print("Using the canonical paper-style AR training pipeline.")
    print(
        "Training flow: stratified sample -> independent validation split -> "
        "randomly initialized GPT-2 -> validation weighted-F1 early stopping "
        "-> full test prediction."
    )
    print(
        f"Model={args.model_size} ({layers}L/{layers}H), "
        f"sample_size={args.n_tr_sub}, patience={args.patience}, "
        f"effective_batch≈{args.bsz * args.n_devices * gradient_accumulation}"
    )
    print("Command:", " ".join(command))
    print(f"Run output directory: {output_dir}")
    if not args.dry_run:
        subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
