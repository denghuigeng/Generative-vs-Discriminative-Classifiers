#!/usr/bin/env bash
#SBATCH --job-name=gendisc_robust
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm/%x_%A_%a.out
#SBATCH --error=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm/%x_%A_%a.err

set -euo pipefail

ROOT="/data/gdh/Generative-vs-Discriminative-Classifiers"
JOBS="${JOBS:-$ROOT/repro_fig2/jobs_robustness.tsv}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gendisc-transformers

export HF_HOME="/data/gdh/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

cd "$ROOT"
line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS")"
IFS=$'\t' read -r run_dir noise <<< "$line"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python repro_fig2/evaluate_robustness.py \
  --run_dir "$run_dir" \
  --noise "$noise" \
  --rates 0 0.05 0.10 0.15 0.20 0.30 0.40 0.50
