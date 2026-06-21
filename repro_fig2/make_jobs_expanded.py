#!/usr/bin/env python
"""Create expanded job lists for a Figure-2-like reproduction.

Default preset:
  5 datasets x 3 models x 3 layer sizes x 7 sample sizes x 3 seeds = 945 jobs

Use --one_seed for a cheaper first pass:
  5 x 3 x 3 x 7 x 1 = 315 jobs
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path("/data/gdh/Generative-vs-Discriminative-Classifiers")

DATASETS = ["agnews", "emotion", "rottentomatoes", "sst5", "twitter"]
MODELS = ["enc", "ar", "mlm"]
SAMPLES = [128, 256, 512, 1024, 2048, 4096, -1]
SEEDS = [79140, 24561, 54641]
LAYERS = [(1, 1), (6, 6), (12, 12)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one_seed", action="store_true", help="Only use the first seed for a cheaper first pass.")
    parser.add_argument("--out", type=str, default=str(ROOT / "repro_fig2" / "jobs_expanded.tsv"))
    args = parser.parse_args()

    seeds = SEEDS[:1] if args.one_seed else SEEDS
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for layers, heads in LAYERS:
                for sample in SAMPLES:
                    for seed in seeds:
                        rows.append(
                            f"{model}\t{dataset}\t{sample}\t{seed}\t"
                            f"{layers}\t{heads}\tscratch\n"
                        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(rows))
    print(f"Wrote {len(rows)} jobs to {out}")
    print(f"Use array range: 0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
