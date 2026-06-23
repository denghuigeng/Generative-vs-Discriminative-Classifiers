#!/usr/bin/env python
"""Find completed full-data checkpoints and create Q2 robustness jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 12])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["enc", "ar", "arpseudo", "mlm"],
    )
    parser.add_argument(
        "--out",
        default="./repro_fig2/jobs_robustness.tsv",
    )
    args = parser.parse_args()

    rows = []
    for args_path in Path(args.output_root).rglob("args.json"):
        run_args = json.loads(args_path.read_text())
        if int(run_args.get("sample_size", 0)) >= 0:
            continue
        if int(run_args.get("layers", 0)) not in args.layers:
            continue
        if run_args.get("model") not in args.models:
            continue
        run_dir = args_path.parent
        if not (run_dir / "model").exists():
            continue
        for noise in ["drop", "substitute"]:
            rows.append(f"{run_dir}\t{noise}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(rows))
    print(f"Wrote {len(rows)} robustness jobs to {out_path}")
    if rows:
        print(f"Use SLURM array range: 0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
