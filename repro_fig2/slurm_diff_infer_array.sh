#!/usr/bin/env bash
#SBATCH --job-name=gendisc_diff_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm/%x_%A_%a.out
#SBATCH --error=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm/%x_%A_%a.err

set -euo pipefail

ROOT="/data/gdh/Generative-vs-Discriminative-Classifiers"
JOBS="${JOBS:-$ROOT/repro_fig2/jobs_diff.tsv}"
DIFF_OUT="${DIFF_OUT:-$ROOT/outputs/diff_repro}"
STEPS="${STEPS:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sedd

export HF_HOME="/data/gdh/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

cd "$ROOT/diff"
line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS")"
IFS=$'\t' read -r dataset_key dataset_path model_size sample seed <<< "$line"
sample_tag="$sample"
if [[ "$sample" == "-1" ]]; then sample_tag="full"; fi
run_dir="$DIFF_OUT/$dataset_key/$model_size/samples_$sample_tag/seed_$seed"

python parallel_inference.py \
  --model_path "$run_dir" \
  --dataset "$dataset_path" \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --max_length "$MAX_LENGTH" \
  --num_workers "$NUM_WORKERS" \
  --output_file "$run_dir/predictions.csv"
