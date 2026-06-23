#!/usr/bin/env python
"""Download/cache datasets, tokenizers, configs and optional pretrained weights.

Run this once on the server before submitting long training jobs. Models are
initialized from scratch, but we still use Hugging Face configs/tokenizers.
"""

import argparse

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include_pretrained_weights",
        action="store_true",
        help="Also cache BERT/GPT-2 weights for the optional pretrained extension.",
    )
    parser.add_argument(
        "--include_diffusion_eval_model",
        action="store_true",
        help="Also cache gpt2-large, used by the original diffusion perplexity evaluation.",
    )
    args = parser.parse_args()

    print("Caching Hugging Face datasets")
    for name, path in DATASETS.items():
        ds = load_dataset(path)
        print(f"- {name}: {path} -> {ds}")

    print("\nCaching tokenizer/config files")
    AutoTokenizer.from_pretrained("bert-base-uncased")
    AutoConfig.from_pretrained("bert-base-uncased")
    GPT2Tokenizer.from_pretrained("gpt2")
    GPT2Config.from_pretrained("gpt2")
    if args.include_pretrained_weights:
        AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        GPT2LMHeadModel.from_pretrained("gpt2")
    if args.include_diffusion_eval_model:
        GPT2LMHeadModel.from_pretrained("gpt2-large")
    print("Done.")


if __name__ == "__main__":
    main()
