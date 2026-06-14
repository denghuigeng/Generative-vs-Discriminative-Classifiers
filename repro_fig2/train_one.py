#!/usr/bin/env python
"""Train/evaluate one AR or ENC experiment for the Figure 2 reproduction.

Outputs:
  <output_root>/<model>/<dataset>/layers_<L>/samples_<N>/seed_<S>/
    model/
    predictions.csv
    metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATASET_PATHS = {
    "agnews": "ag_news",
    "emotion": "emotion",
    "rottentomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "sst5": "SetFit/sst5",
    "twitter": "zeroshot/twitter-financial-news-sentiment",
}

ORDINAL_DATASETS = {"sst5"}


def normalize_text_columns(dataset: DatasetDict) -> DatasetDict:
    return dataset.map(lambda x: {"text": str(x["text"])})


def get_test_split(dataset: DatasetDict) -> str:
    return "test" if "test" in dataset else "validation"


def stratified_subset(train_ds: Dataset, sample_size: int, seed: int) -> Dataset:
    if sample_size < 0 or sample_size >= len(train_ds):
        return train_ds.shuffle(seed=seed)

    labels = np.asarray(train_ds["label"])
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    raw = counts / total * sample_size
    per_class = np.floor(raw).astype(int)

    remainder = sample_size - int(per_class.sum())
    if remainder > 0:
        order = np.argsort(-(raw - per_class))
        for idx in order[:remainder]:
            per_class[idx] += 1

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for cls, n_cls in zip(classes, per_class):
        idxs = np.where(labels == cls)[0]
        rng.shuffle(idxs)
        selected.extend(idxs[:n_cls].tolist())
    rng.shuffle(selected)
    return train_ds.select(selected)


def load_repro_dataset(dataset_name: str, sample_size: int, seed: int, val_size: int) -> Tuple[Dataset, Dataset, Dataset]:
    raw = normalize_text_columns(load_dataset(DATASET_PATHS[dataset_name], trust_remote_code=True))
    test_key = get_test_split(raw)
    train_subset = stratified_subset(raw["train"], sample_size, seed)
    test_ds = raw[test_key]
    val_ds = test_ds.shuffle(seed=seed).select(range(min(val_size, len(test_ds))))
    return train_subset, val_ds, test_ds


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
    return config


def compute_basic_metrics(y_true: List[int], probs: np.ndarray) -> Dict[str, float]:
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_predictions(out_dir: Path, y_true: List[int], probs: np.ndarray) -> None:
    y_pred = probs.argmax(axis=1)
    with (out_dir / "predictions.csv").open("w") as f:
        f.write("ground_truth,predicted_label,scores\n")
        for gt, pred, score in zip(y_true, y_pred, probs):
            score_str = "[" + ", ".join(f"{x:.8f}" for x in score.tolist()) + "]"
            f.write(f"{int(gt)},{int(pred)},\"{score_str}\"\n")


def train_enc(args, out_dir: Path, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, num_labels: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = bert_config(num_labels, args.layers, args.heads)
    model = AutoModelForSequenceClassification.from_config(config)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

    train_tok = train_ds.map(tok, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    val_tok = val_ds.map(tok, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    test_tok = test_ds.map(tok, batched=True, remove_columns=["text"]).rename_column("label", "labels")

    train_tok.set_format("torch")
    val_tok.set_format("torch")
    test_tok.set_format("torch")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        return compute_basic_metrics(labels.tolist(), probs)

    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_weighted_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=20,
        report_to="tensorboard",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    pred = trainer.predict(test_tok)
    return torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()


def tokenize_ar_dataset(ds: Dataset, tokenizer: GPT2Tokenizer, max_len: int) -> Dataset:
    def tok(batch):
        texts = [f"Label:{label},Text:{text}" for text, label in zip(batch["text"], batch["label"])]
        enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
        labels = []
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
            row = [tok_id if m == 1 else -100 for tok_id, m in zip(ids, mask)]
            labels.append(row)
        enc["labels"] = labels
        return enc

    tokenized = ds.map(tok, batched=True, remove_columns=["text", "label"])
    tokenized.set_format("torch")
    return tokenized


def train_mlm(args, out_dir: Path, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, num_labels: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = bert_config(num_labels, args.layers, args.heads)
    model = AutoModelForMaskedLM.from_config(config)

    def tok(batch):
        texts = [f"{text} Label:{label}" for text, label in zip(batch["text"], batch["label"])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=args.max_len)

    train_tok = train_ds.map(tok, batched=True, remove_columns=["text", "label"])
    val_tok = val_ds.map(tok, batched=True, remove_columns=["text", "label"])
    train_tok.set_format("torch")
    val_tok.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=20,
        report_to="tensorboard",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    return mlm_predict_probs(model, tokenizer, test_ds["text"], list(range(num_labels)), args.max_len)


@torch.no_grad()
def mlm_predict_probs(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    labels: List[int],
    max_len: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    label_token_ids = [tokenizer.encode(str(label), add_special_tokens=False)[0] for label in labels]
    all_probs: List[np.ndarray] = []
    mask = tokenizer.mask_token
    for start in range(0, len(texts), 32):
        batch_texts = [f"{text} Label:{mask}" for text in texts[start : start + 32]]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
        logits = model(**enc).logits
        mask_pos = (enc.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)
        masked_logits = logits[mask_pos[:, 0], mask_pos[:, 1]][:, label_token_ids]
        probs = F.softmax(masked_logits, dim=-1).detach().cpu().numpy()
        all_probs.extend(probs)
    return np.asarray(all_probs)


@torch.no_grad()
def ar_predict_probs(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: List[str],
    labels: List[int],
    device: torch.device,
    max_len: int,
) -> np.ndarray:
    model.eval()
    all_probs: List[np.ndarray] = []
    label_values = sorted(labels)
    for text in texts:
        candidates = [f"Label:{label},Text:{text}" for label in label_values]
        enc = tokenizer(candidates, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = enc["input_ids"][:, 1:]
        shift_mask = enc["attention_mask"][:, 1:].float()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        seq_logp = (token_logp * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1.0)
        probs = F.softmax(seq_logp, dim=0).detach().cpu().numpy()
        all_probs.append(probs)
    return np.asarray(all_probs)


def train_ar(args, out_dir: Path, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, num_labels: int) -> np.ndarray:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = gpt2_config(args.layers, args.heads)
    config.pad_token_id = tokenizer.pad_token_id
    model = GPT2LMHeadModel(config)

    train_tok = tokenize_ar_dataset(train_ds, tokenizer, args.max_len)
    val_tok = tokenize_ar_dataset(val_ds, tokenizer, args.max_len)

    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=20,
        report_to="tensorboard",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return ar_predict_probs(
        model=model,
        tokenizer=tokenizer,
        texts=test_ds["text"],
        labels=list(range(num_labels)),
        device=device,
        max_len=args.max_len,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ar", "enc", "mlm"], required=True)
    parser.add_argument("--dataset", choices=list(DATASET_PATHS), required=True)
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val_size", type=int, default=480)
    parser.add_argument("--output_root", type=str, default="/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    out_dir = (
        Path(args.output_root)
        / args.model
        / args.dataset
        / f"layers_{args.layers}"
        / f"samples_{args.sample_size}"
        / f"seed_{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    train_ds, val_ds, test_ds = load_repro_dataset(args.dataset, args.sample_size, args.seed, args.val_size)
    num_labels = len(set(train_ds["label"]))
    y_true = [int(x) for x in test_ds["label"]]

    if args.model == "enc":
        probs = train_enc(args, out_dir, train_ds, val_ds, test_ds, num_labels)
    elif args.model == "ar":
        probs = train_ar(args, out_dir, train_ds, val_ds, test_ds, num_labels)
    else:
        probs = train_mlm(args, out_dir, train_ds, val_ds, test_ds, num_labels)

    save_predictions(out_dir, y_true, probs)
    metrics = compute_basic_metrics(y_true, probs)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
