#!/usr/bin/env bash
#SBATCH --job-name=gendisc_fig2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm/%x_%A_%a.out
#SBATCH --error=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm/%x_%A_%a.err

set -euo pipefail

ROOT="/data/gdh/Generative-vs-Discriminative-Classifiers"
OUT="${OUT:-$ROOT/outputs/fig2_repro}"
JOBS="${JOBS:-$ROOT/repro_fig2/jobs_12layer.tsv}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gendisc-transformers

export HF_HOME="/data/gdh/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/outputs/slurm"
cd "$ROOT"

line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS")"
IFS=$'\t' read -r model dataset sample seed layers heads initialization <<< "$line"
initialization="${initialization:-scratch}"

echo "model=$model dataset=$dataset sample=$sample seed=$seed layers=$layers heads=$heads initialization=$initialization"

python repro_fig2/train_one.py \
  --model "$model" \
  --dataset "$dataset" \
  --sample_size "$sample" \
  --seed "$seed" \
  --layers "$layers" \
  --heads "$heads" \
  --initialization "$initialization" \
  --precision "${PRECISION:-bf16}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-32}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE:-16}" \
  --max_len "${MAX_LEN:-512}" \
  --val_size "${VAL_SIZE:-480}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --output_root "$OUT"
