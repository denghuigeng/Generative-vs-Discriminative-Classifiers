#!/usr/bin/env bash
# Run a non-DIFF TSV job file across selected local GPUs.
#
# Usage:
#   bash repro_fig2/run_job_file_local.sh JOBS.tsv OUTPUT_ROOT 0,1,2,3

set -euo pipefail

ROOT="/data/gdh/Generative-vs-Discriminative-Classifiers"
JOBS="${1:?job TSV is required}"
OUT="${2:?output root is required}"
GPU_CSV="${3:-0}"
PRECISION="${PRECISION:-bf16}"
MAX_LEN="${MAX_LEN:-512}"

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
WORKER_COUNT="${#GPUS[@]}"
mkdir -p "$OUT/local_logs"

worker() {
  local worker_index="$1"
  local gpu="$2"
  local line_index=0
  while IFS=$'\t' read -r model dataset sample seed layers heads initialization; do
    if (( line_index % WORKER_COUNT == worker_index )); then
      initialization="${initialization:-scratch}"
      log="$OUT/local_logs/${model}_${dataset}_${sample}_${seed}_${layers}.log"
      echo "[GPU $gpu] $model $dataset sample=$sample seed=$seed layers=$layers"
      CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/repro_fig2/train_one.py" \
        --model "$model" \
        --dataset "$dataset" \
        --sample_size "$sample" \
        --seed "$seed" \
        --layers "$layers" \
        --heads "$heads" \
        --initialization "$initialization" \
        --precision "$PRECISION" \
        --max_len "$MAX_LEN" \
        --output_root "$OUT" >"$log" 2>&1
    fi
    line_index=$((line_index + 1))
  done < "$JOBS"
}

cd "$ROOT"
for index in "${!GPUS[@]}"; do
  worker "$index" "${GPUS[$index]}" &
done
wait
