#!/usr/bin/env python
"""Train and evaluate one paper-style text-classification experiment.

Supported approaches:
  - enc: discriminative BERT encoder
  - mlm: masked-language-model classifier
  - ar: label-conditional autoregressive classifier
  - arpseudo: single-pass pseudo-autoregressive classifier

The diffusion experiments remain in ``diff/`` because they require a separate
environment and a different training/inference pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATASET_PATHS = {
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

ORDINAL_DATASETS = {"hatespeech", "multiclasssentiment", "sst5", "twitter"}

PAPER_TEST_SPLITS = {
    "imdb": "test",
    "agnews": "test",
    "emotion": "test",
    "hatespeech": "test",
    "multiclasssentiment": "test",
    "rottentomatoes": "test",
    "sst2": "validation",
    "sst5": "validation",
    "twitter": "validation",
}

PAPER_DEFAULTS = {
    "enc": {"epochs": 30, "batch_size": 16, "grad_accum": 2, "lr": 5e-5},
    "mlm": {"epochs": 200, "batch_size": 16, "grad_accum": 2, "lr": 5e-5},
    "ar": {"epochs": 100, "batch_size": 8, "grad_accum": 4, "lr": 5e-5},
    "arpseudo": {"epochs": 100, "batch_size": 8, "grad_accum": 4, "lr": 5e-5},
}


def normalize_text_columns(dataset: DatasetDict) -> DatasetDict:
    if "text" not in dataset["train"].column_names:
        raise KeyError(
            f"Dataset has no 'text' column. Available columns: {dataset['train'].column_names}"
        )
    return dataset.map(lambda x: {"text": str(x["text"])})


def allocate_stratified_counts(labels: np.ndarray, target_size: int) -> Dict[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    target_size = min(int(target_size), int(counts.sum()))
    raw = counts / counts.sum() * target_size
    allocated = np.floor(raw).astype(int)

    if target_size >= len(classes):
        allocated = np.maximum(allocated, 1)
    allocated = np.minimum(allocated, counts)

    while allocated.sum() > target_size:
        candidates = np.where(allocated > 1)[0]
        if len(candidates) == 0:
            candidates = np.where(allocated > 0)[0]
        idx = candidates[np.argmax(allocated[candidates] - raw[candidates])]
        allocated[idx] -= 1

    while allocated.sum() < target_size:
        capacity = counts - allocated
        candidates = np.where(capacity > 0)[0]
        idx = candidates[np.argmax(raw[candidates] - allocated[candidates])]
        allocated[idx] += 1

    return {int(cls): int(n) for cls, n in zip(classes, allocated)}


def stratified_select(dataset: Dataset, target_size: int, seed: int) -> Dataset:
    if target_size < 0 or target_size >= len(dataset):
        return dataset.shuffle(seed=seed)

    labels = np.asarray(dataset["label"], dtype=int)
    per_class = allocate_stratified_counts(labels, target_size)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for cls, count in per_class.items():
        indices = np.where(labels == cls)[0]
        rng.shuffle(indices)
        selected.extend(indices[:count].tolist())
    rng.shuffle(selected)
    return dataset.select(selected)


def stratified_holdout(dataset: Dataset, holdout_size: int, seed: int) -> Tuple[Dataset, Dataset]:
    labels = np.asarray(dataset["label"], dtype=int)
    per_class = allocate_stratified_counts(labels, min(holdout_size, len(dataset) // 2))
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    holdout_indices: List[int] = []
    for cls, count in per_class.items():
        indices = np.where(labels == cls)[0]
        rng.shuffle(indices)
        holdout_indices.extend(indices[:count].tolist())
        train_indices.extend(indices[count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(holdout_indices)
    return dataset.select(train_indices), dataset.select(holdout_indices)


def load_repro_dataset(
    dataset_name: str,
    sample_size: int,
    seed: int,
    val_size: int,
    test_split: str,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, object]]:
    raw = normalize_text_columns(load_dataset(DATASET_PATHS[dataset_name]))

    selected_test_split = (
        PAPER_TEST_SPLITS[dataset_name] if test_split == "paper" else test_split
    )
    if selected_test_split == "auto":
        selected_test_split = "test" if "test" in raw else "validation"
    if selected_test_split not in raw:
        raise KeyError(
            f"{dataset_name} has no split '{selected_test_split}'. "
            f"Available splits: {list(raw.keys())}"
        )
    test_ds = raw[selected_test_split]

    validation_source = "train_holdout"
    if selected_test_split != "validation" and "validation" in raw:
        train_pool = raw["train"]
        val_ds = raw["validation"]
        validation_source = "validation"
    elif selected_test_split != "test" and "test" in raw:
        train_pool = raw["train"]
        val_ds = raw["test"]
        validation_source = "test"
    else:
        train_pool, val_ds = stratified_holdout(raw["train"], val_size, seed + 10_000)

    train_ds = stratified_select(train_pool, sample_size, seed)
    manifest = {
        "dataset_name": dataset_name,
        "huggingface_path": DATASET_PATHS[dataset_name],
        "requested_sample_size": sample_size,
        "actual_train_size": len(train_ds),
        "validation_size": len(val_ds),
        "test_size": len(test_ds),
        "selected_test_split": selected_test_split,
        "train_label_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(train_ds["label"], return_counts=True))
        },
        "validation_source": validation_source,
    }
    return train_ds, val_ds, test_ds, manifest


def bert_config(num_labels: int, layers: int, heads: int):
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.num_labels = num_labels
    config.num_hidden_layers = layers
    config.num_attention_heads = heads
    config.hidden_size = int((768 * heads) // 12)
    config.intermediate_size = config.hidden_size * 4
    return config


def gpt2_config(layers: int, heads: int):
    config = GPT2Config.from_pretrained("gpt2")
    config.n_layer = layers
    config.n_head = heads
    config.n_embd = int((768 * heads) // 12)
    config.n_inner = config.n_embd * 4
    return config


def compute_basic_metrics(y_true: Sequence[int], probs: np.ndarray) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "weighted_f1": float(f1_score(y_true_array, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true_array, y_pred, average="macro")),
    }


def save_predictions(out_dir: Path, y_true: Sequence[int], probs: np.ndarray) -> None:
    y_pred = probs.argmax(axis=1)
    with (out_dir / "predictions.csv").open("w") as f:
        f.write("ground_truth,predicted_label,scores\n")
        for gt, pred, score in zip(y_true, y_pred, probs):
            score_str = "[" + ", ".join(f"{x:.8f}" for x in score.tolist()) + "]"
            f.write(f'{int(gt)},{int(pred)},"{score_str}"\n')


def precision_flags(precision: str) -> Dict[str, bool]:
    return {
        "fp16": precision == "fp16",
        "bf16": precision == "bf16",
    }


def training_arguments(
    args: argparse.Namespace,
    out_dir: Path,
    metric_for_best_model: str,
    greater_is_better: bool,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        logging_dir=str(out_dir / "logs"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=2,
        logging_steps=20,
        report_to=["tensorboard"],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        **precision_flags(args.precision),
    )


def callbacks(patience: int) -> List[EarlyStoppingCallback]:
    if patience <= 0:
        return []
    return [EarlyStoppingCallback(early_stopping_patience=patience)]


def train_enc(
    args: argparse.Namespace,
    out_dir: Path,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    num_labels: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if args.initialization == "pretrained":
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
    else:
        model = AutoModelForSequenceClassification.from_config(
            bert_config(num_labels, args.layers, args.heads)
        )

    def tokenize(dataset: Dataset) -> Dataset:
        def tok(batch):
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=args.max_len,
            )
            encoded["labels"] = batch["label"]
            return encoded

        result = dataset.map(tok, batched=True, remove_columns=dataset.column_names)
        result.set_format("torch")
        return result

    train_tok = tokenize(train_ds)
    val_tok = tokenize(val_ds)
    test_tok = tokenize(test_ds)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        return compute_basic_metrics(labels.tolist(), probs)

    trainer = Trainer(
        model=model,
        args=training_arguments(args, out_dir, "eval_loss", False),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks(args.patience),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    prediction = trainer.predict(test_tok)
    return torch.softmax(torch.tensor(prediction.predictions), dim=-1).numpy()


def train_mlm(
    args: argparse.Namespace,
    out_dir: Path,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    num_labels: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.truncation_side = "left"
    if args.initialization == "pretrained":
        model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    else:
        model = AutoModelForMaskedLM.from_config(
            bert_config(num_labels, args.layers, args.heads)
        )

    def tokenize(dataset: Dataset) -> Dataset:
        def tok(batch):
            texts = [
                f"{text} Label:{label}"
                for text, label in zip(batch["text"], batch["label"])
            ]
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=args.max_len,
            )

        result = dataset.map(tok, batched=True, remove_columns=dataset.column_names)
        result.set_format("torch")
        return result

    train_tok = tokenize(train_ds)
    val_tok = tokenize(val_ds)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    trainer = Trainer(
        model=model,
        args=training_arguments(args, out_dir, "eval_loss", False),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks(args.patience),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    return mlm_predict_probs(
        trainer.model,
        tokenizer,
        test_ds["text"],
        list(range(num_labels)),
        args.max_len,
        args.inference_batch_size,
    )


@torch.no_grad()
def mlm_predict_probs(
    model: AutoModelForMaskedLM,
    tokenizer,
    texts: Sequence[str],
    labels: Sequence[int],
    max_len: int,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    label_token_ids = [
        tokenizer.encode(str(label), add_special_tokens=False)[0] for label in labels
    ]
    all_probs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = [
            f"{text} Label:{tokenizer.mask_token}"
            for text in texts[start : start + batch_size]
        ]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits
        mask_positions = (encoded.input_ids == tokenizer.mask_token_id).nonzero(
            as_tuple=False
        )
        masked_logits = logits[mask_positions[:, 0], mask_positions[:, 1]][
            :, label_token_ids
        ]
        all_probs.extend(F.softmax(masked_logits, dim=-1).cpu().numpy())
    if was_training:
        model.train()
    return np.asarray(all_probs)


def tokenize_ar_dataset(
    dataset: Dataset,
    tokenizer: GPT2TokenizerFast,
    max_len: int,
    mode: str,
) -> Dataset:
    def tok(batch):
        if mode == "ar":
            texts = [
                f"Label:{label},Text:{text}"
                for text, label in zip(batch["text"], batch["label"])
            ]
        else:
            texts = [
                f"text: {text}, label:{label}"
                for text, label in zip(batch["text"], batch["label"])
            ]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        encoded["labels"] = [
            [token_id if mask else -100 for token_id, mask in zip(ids, attn)]
            for ids, attn in zip(encoded["input_ids"], encoded["attention_mask"])
        ]
        return encoded

    tokenized = dataset.map(tok, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    return tokenized


@torch.no_grad()
def ar_predict_probs(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    texts: Sequence[str],
    labels: Sequence[int],
    max_len: int,
    batch_size: int,
    length_normalize: bool,
) -> np.ndarray:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    all_probs: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        text_batch = texts[start : start + batch_size]
        candidates: List[str] = []
        prefix_lengths: List[int] = []
        for text in text_batch:
            for label in labels:
                prefix = f"Label:{label},Text:"
                candidates.append(prefix + text)
                prefix_lengths.append(len(prefix))

        encoded = tokenizer(
            candidates,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits[:, :-1, :]
        targets = encoded["input_ids"][:, 1:]
        attention = encoded["attention_mask"][:, 1:].float()
        target_offsets = offsets[:, 1:, 1].to(device)
        prefix_tensor = torch.tensor(prefix_lengths, device=device).unsqueeze(1)
        text_mask = (target_offsets > prefix_tensor).float() * attention

        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        scores = (token_log_probs * text_mask).sum(dim=1)
        if length_normalize:
            scores = scores / text_mask.sum(dim=1).clamp_min(1.0)
        scores = scores.view(len(text_batch), len(labels))
        all_probs.extend(F.softmax(scores, dim=-1).cpu().numpy())

    if was_training:
        model.train()
    return np.asarray(all_probs)


@torch.no_grad()
def ar_pseudo_predict_probs(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    texts: Sequence[str],
    labels: Sequence[int],
    max_len: int,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    label_token_ids = [
        tokenizer.encode(str(label), add_special_tokens=False)[0] for label in labels
    ]
    all_probs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        prompts = [
            f"text: {text}, label:" for text in texts[start : start + batch_size]
        ]
        encoded = tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits
        last_positions = encoded["attention_mask"].sum(dim=1) - 1
        row_indices = torch.arange(len(prompts), device=device)
        label_logits = logits[row_indices, last_positions][:, label_token_ids]
        all_probs.extend(F.softmax(label_logits, dim=-1).cpu().numpy())
    if was_training:
        model.train()
    return np.asarray(all_probs)


class GenerativeClassificationTrainer(Trainer):
    """Trainer that adds classification metrics to AR/ARpseudo evaluation."""

    def __init__(self, *args, classification_eval: Callable, **kwargs):
        super().__init__(*args, **kwargs)
        self.classification_eval = classification_eval

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        probs, labels = self.classification_eval(self.model)
        cls_metrics = compute_basic_metrics(labels, probs)
        metrics.update({f"eval_{key}": value for key, value in cls_metrics.items()})
        self.log(metrics)
        return metrics


def train_generative(
    args: argparse.Namespace,
    out_dir: Path,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    num_labels: int,
) -> np.ndarray:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right" if args.model == "ar" else "left"

    if args.initialization == "pretrained":
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        config = gpt2_config(args.layers, args.heads)
        config.pad_token_id = tokenizer.pad_token_id
        model = GPT2LMHeadModel(config)

    train_tok = tokenize_ar_dataset(
        train_ds, tokenizer, args.max_len, mode=args.model
    )
    val_tok = tokenize_ar_dataset(val_ds, tokenizer, args.max_len, mode=args.model)
    class_labels = list(range(num_labels))

    def predict(model_to_eval, texts):
        if args.model == "ar":
            return ar_predict_probs(
                model_to_eval,
                tokenizer,
                texts,
                class_labels,
                args.max_len,
                args.inference_batch_size,
                args.ar_length_normalize,
            )
        return ar_pseudo_predict_probs(
            model_to_eval,
            tokenizer,
            texts,
            class_labels,
            args.max_len,
            args.inference_batch_size,
        )

    def validation_classification(model_to_eval):
        return predict(model_to_eval, val_ds["text"]), [
            int(x) for x in val_ds["label"]
        ]

    trainer = GenerativeClassificationTrainer(
        model=model,
        args=training_arguments(args, out_dir, "eval_weighted_f1", True),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        callbacks=callbacks(args.patience),
        classification_eval=validation_classification,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    return predict(trainer.model, test_ds["text"])


def apply_paper_defaults(args: argparse.Namespace) -> None:
    defaults = PAPER_DEFAULTS[args.model]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = defaults["grad_accum"]
    if args.lr is None:
        args.lr = defaults["lr"]


def validate_args(args: argparse.Namespace) -> None:
    if args.initialization == "pretrained" and (args.layers, args.heads) != (12, 12):
        raise ValueError("Pretrained experiments only support the base 12-layer/12-head models.")
    if args.initialization == "pretrained" and args.model not in {"enc", "ar"}:
        raise ValueError("The paper's pretrained extension compares only ENC and AR.")
    if args.layers != args.heads:
        raise ValueError("Paper configurations use matching layers/heads: 1/1, 6/6, 12/12.")
    if (args.layers, args.heads) not in {(1, 1), (6, 6), (12, 12)}:
        raise ValueError("Supported paper configurations are 1/1, 6/6 and 12/12.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ar", "arpseudo", "enc", "mlm"], required=True)
    parser.add_argument("--dataset", choices=list(DATASET_PATHS), required=True)
    parser.add_argument(
        "--sample_size",
        type=int,
        required=True,
        help="128..4096, or -1 for the full training split.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--initialization", choices=["scratch", "pretrained"], default="scratch")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val_size", type=int, default=480)
    parser.add_argument(
        "--test_split",
        choices=["paper", "auto", "test", "validation"],
        default="paper",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--ar_length_normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--run_suffix",
        type=str,
        default="",
        help="Optional path suffix for hyperparameter-search runs.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/paper_repro",
    )
    args = parser.parse_args()

    apply_paper_defaults(args)
    if args.patience is None:
        args.patience = 10 if args.model in {"ar", "arpseudo"} else 0
    validate_args(args)

    set_seed(args.seed)
    print(f"Hugging Face endpoint: {os.environ['HF_ENDPOINT']}")
    torch.set_float32_matmul_precision("medium")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"CUDA available={torch.cuda.is_available()}, count={torch.cuda.device_count()}")

    sample_tag = "full" if args.sample_size < 0 else str(args.sample_size)
    out_dir = (
        Path(args.output_root)
        / f"init_{args.initialization}"
        / args.model
        / args.dataset
        / f"layers_{args.layers}"
        / f"samples_{sample_tag}"
        / f"seed_{args.seed}"
    )
    if args.run_suffix:
        out_dir = out_dir / args.run_suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    if (
        not args.overwrite
        and (out_dir / "predictions.csv").exists()
        and (out_dir / "metrics.json").exists()
    ):
        print(f"Completed output already exists; skipping: {out_dir}")
        return

    train_ds, val_ds, test_ds, manifest = load_repro_dataset(
        args.dataset,
        args.sample_size,
        args.seed,
        args.val_size,
        args.test_split,
    )
    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (out_dir / "dataset_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    num_labels = len(set(train_ds["label"]))
    y_true = [int(x) for x in test_ds["label"]]

    if args.model == "enc":
        probs = train_enc(args, out_dir, train_ds, val_ds, test_ds, num_labels)
    elif args.model == "mlm":
        probs = train_mlm(args, out_dir, train_ds, val_ds, test_ds, num_labels)
    else:
        probs = train_generative(
            args, out_dir, train_ds, val_ds, test_ds, num_labels
        )

    save_predictions(out_dir, y_true, probs)
    metrics = compute_basic_metrics(y_true, probs)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
