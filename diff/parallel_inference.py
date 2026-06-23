#!/usr/bin/env python
"""Distributed classification inference for trained diffusion models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from sklearn.metrics import classification_report
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2TokenizerFast

import sampling_inference
from load_model import load_model


PAPER_TEST_SPLITS = {
    "imdb": "test",
    "ag_news": "test",
    "emotion": "test",
    "SetFit/hate_speech_offensive": "test",
    "Sp1786/multiclass-sentiment-analysis-dataset": "test",
    "cornell-movie-review-data/rotten_tomatoes": "test",
    "SetFit/sst2": "validation",
    "SetFit/sst5": "validation",
    "zeroshot/twitter-financial-news-sentiment": "validation",
}


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def find_subsequence(sequence: torch.Tensor, pattern: torch.Tensor) -> Optional[int]:
    if len(pattern) > len(sequence):
        return None
    for start in range(len(sequence) - len(pattern), -1, -1):
        if torch.equal(sequence[start : start + len(pattern)], pattern):
            return start
    return None


def load_eval_dataset(
    dataset_name: str,
    tokenizer: GPT2TokenizerFast,
    max_length: int,
):
    raw = load_dataset(dataset_name)
    split = PAPER_TEST_SPLITS.get(
        dataset_name, "test" if "test" in raw else "validation"
    )
    if split not in raw:
        raise KeyError(
            f"{dataset_name} has no split '{split}'. Available: {list(raw.keys())}"
        )
    dataset = raw[split]
    if "text" not in dataset.column_names or "label" not in dataset.column_names:
        raise KeyError(
            f"{dataset_name}/{split} must contain text and label columns; "
            f"found {dataset.column_names}"
        )
    row_ids = list(range(len(dataset)))
    dataset = dataset.add_column("_row_id", row_ids)
    num_labels = max(int(label) for label in dataset["label"]) + 1

    def tokenize(batch):
        prompts = [f"{text}. Label:" for text in batch["text"]]
        encoded = tokenizer(
            prompts,
            return_attention_mask=False,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        encoded["ground_truth"] = [int(label) for label in batch["label"]]
        encoded["row_id"] = [int(row_id) for row_id in batch["_row_id"]]
        return encoded

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
    )
    tokenized.set_format("torch")
    return tokenized, split, num_labels


def parse_generated_label(text: str, num_labels: int) -> Optional[int]:
    label_part = text.rsplit("Label:", maxsplit=1)
    if len(label_part) != 2:
        return None
    match = re.search(r"\d+", label_part[1])
    if match is None:
        return None
    label = int(match.group(0))
    return label if 0 <= label < num_labels else None


def save_predictions(
    output_file: Path,
    records: List[Tuple[int, int, int]],
    num_labels: int,
    expected_rows: int,
    dataset_name: str,
    split: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    by_row: Dict[int, Tuple[int, int]] = {}
    for row_id, true_label, predicted_label in records:
        by_row.setdefault(row_id, (true_label, predicted_label))

    missing_rows = sorted(set(range(expected_rows)) - set(by_row))
    manifest = {
        "dataset": dataset_name,
        "split": split,
        "expected_rows": expected_rows,
        "completed_rows": len(by_row),
        "missing_rows": len(missing_rows),
        "probabilities_available": False,
        "note": "Scores are one-hot placeholders; diffusion calibration is not reported.",
    }
    (output_file.parent / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    if missing_rows:
        raise RuntimeError(
            f"Diffusion inference parsed {len(by_row)}/{expected_rows} examples. "
            "No predictions.csv was written; inspect generated labels and rerun."
        )

    ordered = [(row_id, *by_row[row_id]) for row_id in range(expected_rows)]
    with output_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ground_truth", "predicted_label", "scores"])
        for _, true_label, predicted_label in ordered:
            one_hot = [0.0] * num_labels
            one_hot[predicted_label] = 1.0
            writer.writerow([true_label, predicted_label, str(one_hot)])

    true_labels = [true_label for _, true_label, _ in ordered]
    predictions = [predicted_label for _, _, predicted_label in ordered]
    print(classification_report(true_labels, predictions, digits=4, zero_division=0))
    print(f"Saved predictions to {output_file}")


def run_model(
    rank: int,
    world_size: int,
    port: int,
    args: argparse.Namespace,
) -> None:
    initialized = False
    try:
        setup(rank, world_size, port)
        initialized = True
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        model, graph, noise = load_model(args.model_path, device)
        model = DDP(model, device_ids=[rank])
        model.eval()

        tokenizer = GPT2TokenizerFast.from_pretrained(
            "gpt2", truncation_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        eos_token_id = int(tokenizer.eos_token_id)
        label_tokens = torch.tensor(
            tokenizer.encode(" Label:", add_special_tokens=False),
            dtype=torch.long,
            device=device,
        )

        eval_set, split, num_labels = load_eval_dataset(
            args.dataset, tokenizer, args.max_length
        )
        sampler = DistributedSampler(
            eval_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        loader = DataLoader(
            eval_set,
            batch_size=max(1, args.batch_size // world_size),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        local_records: List[Tuple[int, int, int]] = []
        parse_failures = 0
        for batch_index, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            batch_size, sequence_length = input_ids.shape
            label_positions = []
            for row in input_ids:
                start = find_subsequence(row, label_tokens)
                if start is None:
                    raise RuntimeError(
                        "Could not find the ' Label:' prompt after tokenization."
                    )
                label_positions.append(start + len(label_tokens) - 1)

            def projection(sample):
                attention_mask = torch.ones_like(sample, dtype=torch.float32)
                for row_index, label_position in enumerate(label_positions):
                    sample[row_index, : label_position + 1] = input_ids[
                        row_index, : label_position + 1
                    ]
                    if label_position + 2 < sequence_length:
                        sample[row_index, label_position + 2 :] = eos_token_id
                        attention_mask[row_index, label_position + 2 :] = 0
                return sample, attention_mask

            sampler_fn = sampling_inference.get_pc_sampler(
                graph,
                noise,
                (batch_size, sequence_length),
                "analytic",
                args.steps,
                device=device,
                proj_fun=projection,
            )
            generated = sampler_fn(model)
            decoded = tokenizer.batch_decode(generated)

            for row_id, true_label, text in zip(
                batch["row_id"].tolist(),
                batch["ground_truth"].tolist(),
                decoded,
            ):
                prediction = parse_generated_label(text, num_labels)
                if prediction is None:
                    parse_failures += 1
                    continue
                local_records.append(
                    (int(row_id), int(true_label), int(prediction))
                )

            if rank == 0:
                print(
                    f"batch={batch_index + 1}/{len(loader)} "
                    f"parsed={len(local_records)} failures={parse_failures}"
                )

        gathered_records = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_records, gathered_records, dst=0)
        if rank == 0:
            combined: List[Tuple[int, int, int]] = []
            for rank_records in gathered_records:
                combined.extend(rank_records)
            output_file = Path(
                args.output_file
                or os.path.join(args.model_path, "predictions.csv")
            )
            save_predictions(
                output_file,
                combined,
                num_labels,
                len(eval_set),
                args.dataset,
                split,
            )
    finally:
        if initialized:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    print(f"Hugging Face endpoint: {os.environ['HF_ENDPOINT']}")
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("Diffusion inference requires at least one visible CUDA GPU.")
    port = find_free_port()
    print(f"Starting diffusion inference with {world_size} GPU(s)")
    mp.spawn(
        run_model,
        args=(world_size, port, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
