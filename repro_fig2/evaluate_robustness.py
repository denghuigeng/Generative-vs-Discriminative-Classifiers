#!/usr/bin/env python
"""Evaluate a completed checkpoint under token drop/substitution noise."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from aggregate_and_plot import ece_mce, probability_metrics
from train_one import (
    ar_predict_probs,
    ar_pseudo_predict_probs,
    compute_basic_metrics,
    load_repro_dataset,
    mlm_predict_probs,
    save_predictions,
)


def perturb_text(
    text: str,
    tokenizer,
    noise_type: str,
    rate: float,
    rng: np.random.Generator,
) -> str:
    if rate <= 0:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return text
    special_ids = set(tokenizer.all_special_ids)

    if noise_type == "drop":
        kept = [
            token_id
            for token_id in token_ids
            if token_id in special_ids or rng.random() >= rate
        ]
        token_ids = kept or token_ids[:1]
    elif noise_type == "substitute":
        replaced = []
        vocab_size = int(tokenizer.vocab_size)
        for token_id in token_ids:
            if token_id not in special_ids and rng.random() < rate:
                replacement = int(rng.integers(0, vocab_size))
                while replacement in special_ids:
                    replacement = int(rng.integers(0, vocab_size))
                replaced.append(replacement)
            else:
                replaced.append(token_id)
        token_ids = replaced
    else:
        raise ValueError(noise_type)

    return tokenizer.decode(token_ids, skip_special_tokens=True)


@torch.no_grad()
def enc_predict(model, tokenizer, texts: Sequence[str], max_len: int, batch_size: int):
    device = next(model.parameters()).device
    all_probs = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        all_probs.extend(F.softmax(model(**encoded).logits, dim=-1).cpu().numpy())
    return np.asarray(all_probs)


def load_model_and_tokenizer(run_args: Dict[str, object], model_dir: Path):
    model_name = str(run_args["model"])
    if model_name == "enc":
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    elif model_name == "mlm":
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForMaskedLM.from_pretrained(model_dir)
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def predict(run_args, model, tokenizer, texts, num_labels):
    model_name = str(run_args["model"])
    labels = list(range(num_labels))
    max_len = int(run_args.get("max_len", 512))
    batch_size = int(run_args.get("inference_batch_size", 16))
    if model_name == "enc":
        return enc_predict(model, tokenizer, texts, max_len, batch_size)
    if model_name == "mlm":
        return mlm_predict_probs(model, tokenizer, texts, labels, max_len, batch_size)
    if model_name == "ar":
        return ar_predict_probs(
            model,
            tokenizer,
            texts,
            labels,
            max_len,
            batch_size,
            bool(run_args.get("ar_length_normalize", True)),
        )
    return ar_pseudo_predict_probs(
        model, tokenizer, texts, labels, max_len, batch_size
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--noise", choices=["drop", "substitute"], required=True)
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    )
    parser.add_argument("--noise_seed", type=int, default=2026)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_args = json.loads((run_dir / "args.json").read_text())
    _, _, test_ds, _ = load_repro_dataset(
        str(run_args["dataset"]),
        int(run_args["sample_size"]),
        int(run_args["seed"]),
        int(run_args.get("val_size", 480)),
        str(run_args.get("test_split", "paper")),
    )
    y_true = [int(x) for x in test_ds["label"]]
    num_labels = max(y_true) + 1
    model, tokenizer = load_model_and_tokenizer(run_args, run_dir / "model")

    output_root = run_dir / "robustness" / args.noise
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for rate in args.rates:
        rng = np.random.default_rng(args.noise_seed + int(rate * 10_000))
        noisy_texts = [
            perturb_text(text, tokenizer, args.noise, rate, rng)
            for text in test_ds["text"]
        ]
        probs = predict(run_args, model, tokenizer, noisy_texts, num_labels)
        rate_dir = output_root / f"rate_{rate:.2f}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(rate_dir, y_true, probs)
        metrics = compute_basic_metrics(y_true, probs)
        metrics.update(ece_mce(np.asarray(y_true), probs))
        metrics.update(probability_metrics(np.asarray(y_true), probs))
        metrics["noise"] = args.noise
        metrics["rate"] = rate
        (rate_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        rows.append(metrics)
        print(args.noise, rate, metrics["weighted_f1"])

    summary = pd.DataFrame(rows).sort_values("rate")
    baseline = float(summary.iloc[0]["weighted_f1"])
    summary["relative_f1_drop"] = (
        baseline - summary["weighted_f1"]
    ) / max(baseline, 1e-12)
    summary.to_csv(output_root / "summary.csv", index=False)

    thresholds = {}
    for target in [0.05, 0.10, 0.15, 0.20, 0.30]:
        reached = summary[summary["relative_f1_drop"] >= target]
        thresholds[str(target)] = (
            None if reached.empty else float(reached.iloc[0]["rate"])
        )
    (output_root / "drop_thresholds.json").write_text(
        json.dumps(thresholds, indent=2)
    )


if __name__ == "__main__":
    main()
