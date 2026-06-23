#!/usr/bin/env bash
set -euo pipefail

ROOT="."
OUT="./outputs/fig2_repro"
GPU="${GPU:-0}"
PRECISION="${PRECISION:-bf16}"

cd .

DATASETS=("agnews" "sst5")
MODELS=("enc" "ar")
SAMPLES=(128 256 512 1024 2048 4096 -1)
SEEDS=(79140 24561 54641)

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for sample in "${SAMPLES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "Running model=$model dataset=$dataset sample=$sample seed=$seed"
        CUDA_VISIBLE_DEVICES="$GPU" python repro_fig2/train_one.py \
          --model "$model" \
          --dataset "$dataset" \
          --sample_size "$sample" \
          --seed "$seed" \
          --layers 12 \
          --heads 12 \
          --eval_batch_size 32 \
          --max_len 512 \
          --precision "$PRECISION" \
          --output_root "$OUT"
      done
    done
  done
done

python repro_fig2/aggregate_and_plot.py \
  --output_root "$OUT" \
  --layers 12 \
  --datasets agnews sst5
