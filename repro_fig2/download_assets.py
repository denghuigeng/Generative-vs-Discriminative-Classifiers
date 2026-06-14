#!/usr/bin/env python
"""Download/cache the datasets and tokenizer/config files used by the reproduction.

Run this once on the server before submitting long training jobs. Models are
initialized from scratch, but we still use Hugging Face configs/tokenizers.
"""

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, GPT2Config, GPT2Tokenizer

DATASETS = {
    "agnews": "ag_news",
    "emotion": "emotion",
    "rottentomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "sst5": "SetFit/sst5",
    "twitter": "zeroshot/twitter-financial-news-sentiment",
}


def main() -> None:
    print("Caching Hugging Face datasets")
    for name, path in DATASETS.items():
        ds = load_dataset(path, trust_remote_code=True)
        print(f"- {name}: {path} -> {ds}")

    print("\nCaching tokenizer/config files")
    AutoTokenizer.from_pretrained("bert-base-uncased")
    AutoConfig.from_pretrained("bert-base-uncased")
    GPT2Tokenizer.from_pretrained("gpt2")
    GPT2Config.from_pretrained("gpt2")
    print("Done.")


if __name__ == "__main__":
    main()
