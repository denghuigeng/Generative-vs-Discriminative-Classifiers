#!/usr/bin/env bash
#SBATCH --job-name=gendisc_diff
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm/%x_%A_%a.out
#SBATCH --error=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm/%x_%A_%a.err

set -euo pipefail

ROOT="/data/gdh/Generative-vs-Discriminative-Classifiers"
JOBS="${JOBS:-$ROOT/repro_fig2/jobs_diff.tsv}"
DIFF_OUT="${DIFF_OUT:-$ROOT/outputs/diff_repro}"
NGPUS="${NGPUS:-1}"
N_ITERS="${N_ITERS:-200000}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sedd

export HF_HOME="/data/gdh/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/outputs/slurm" "$DIFF_OUT"
cd "$ROOT/diff"

line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS")"
IFS=$'\t' read -r dataset_key dataset_path model_size sample seed <<< "$line"

sample_tag="$sample"
train_size_args=()
if [[ "$sample" == "-1" ]]; then
  sample_tag="full"
else
  train_size_args=("TRAIN_SIZE=$sample")
fi

run_dir="$DIFF_OUT/$dataset_key/$model_size/samples_$sample_tag/seed_$seed"
mkdir -p "$run_dir"
layers=1
if [[ "$model_size" == "medium" ]]; then layers=6; fi
if [[ "$model_size" == "large" ]]; then layers=12; fi
cat > "$run_dir/args.json" <<EOF
{
  "model": "diff",
  "dataset": "$dataset_key",
  "sample_size": $sample,
  "seed": $seed,
  "layers": $layers,
  "heads": $layers,
  "initialization": "scratch",
  "probabilities_available": false
}
EOF

echo "dataset=$dataset_key model=$model_size sample=$sample_tag seed=$seed ngpus=$NGPUS"

if [[ "$sample" == "-1" ]]; then
  DATASET_NAME="$dataset_path" N_ITERS="$N_ITERS" SEED="$seed" \
    python train.py "model=$model_size" "ngpus=$NGPUS" \
      "work_dir=$run_dir" "hydra.run.dir=$run_dir"
else
  DATASET_NAME="$dataset_path" TRAIN_SIZE="$sample" N_ITERS="$N_ITERS" SEED="$seed" \
    python train.py "model=$model_size" "ngpus=$NGPUS" \
      "work_dir=$run_dir" "hydra.run.dir=$run_dir"
fi
