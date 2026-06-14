#!/usr/bin/env python
"""Create a TSV job list for SLURM array submission."""

from pathlib import Path

ROOT = Path("/data/gdh/Generative-vs-Discriminative-Classifiers")
OUT = ROOT / "repro_fig2" / "jobs_12layer.tsv"

DATASETS = ["agnews", "sst5"]
MODELS = ["enc", "ar"]
SAMPLES = [128, 256, 512, 1024, 2048, 4096]
SEEDS = [79140, 24561, 54641]


def main() -> None:
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for sample in SAMPLES:
                for seed in SEEDS:
                    rows.append(f"{model}\t{dataset}\t{sample}\t{seed}\t12\t12\n")
    OUT.write_text("".join(rows))
    print(f"Wrote {len(rows)} jobs to {OUT}")
    print(f"Use array range: 0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
