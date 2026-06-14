#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/gdh/Generative-vs-Discriminative-Classifiers"
OUT="$ROOT/outputs/fig2_repro"

cd "$ROOT"

DATASETS=("agnews" "sst5")
MODELS=("enc" "ar")
SAMPLES=(128 256 512 1024 2048 4096)
SEEDS=(79140 24561 54641)

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for sample in "${SAMPLES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "Running model=$model dataset=$dataset sample=$sample seed=$seed"
        python repro_fig2/train_one.py \
          --model "$model" \
          --dataset "$dataset" \
          --sample_size "$sample" \
          --seed "$seed" \
          --layers 12 \
          --heads 12 \
          --epochs 50 \
          --batch_size 16 \
          --eval_batch_size 32 \
          --max_len 256 \
          --output_root "$OUT"
      done
    done
  done
done

python repro_fig2/aggregate_and_plot.py --output_root "$OUT" --layers 12
