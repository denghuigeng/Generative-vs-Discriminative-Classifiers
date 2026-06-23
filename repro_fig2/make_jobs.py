#!/usr/bin/env python
"""Create a TSV job list for SLURM array submission."""

import argparse
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "repro_fig2" / "jobs_12layer.tsv"

DATASETS = ["agnews", "sst5"]
MODELS = ["enc", "ar"]
SAMPLES = [128, 256, 512, 1024, 2048, 4096, -1]
SEEDS = [79140, 24561, 54641]


def sample_first_order(samples):
    return sorted(samples, key=lambda sample: (sample < 0, sample if sample >= 0 else 10**18))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(OUT))
    args = parser.parse_args()

    rows = []
    for sample in sample_first_order(SAMPLES):
        for dataset in DATASETS:
            for model in MODELS:
                for seed in SEEDS:
                    rows.append(
                        f"{model}\t{dataset}\t{sample}\t{seed}\t12\t12\tscratch\n"
                    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(rows))
    print(f"Wrote {len(rows)} jobs to {out}")
    print(f"Use array range: 0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
