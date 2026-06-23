#!/usr/bin/env bash
# Run a non-DIFF TSV job file across selected local GPUs.
#
# Usage:
#   bash repro_fig2/run_job_file_local.sh JOBS.tsv OUTPUT_ROOT 0,1,2,3

set -euo pipefail

ROOT="${ROOT:-.}"
JOBS="${1:?job TSV is required}"
OUT="${2:?output root is required}"
GPU_CSV="${3:-0}"
PRECISION="${PRECISION:-bf16}"
MAX_LEN="${MAX_LEN:-512}"

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
WORKER_COUNT="${#GPUS[@]}"
mkdir -p "$OUT/local_logs"

TOTAL_JOBS="$(awk 'NF {n++} END {print n + 0}' "$JOBS")"
START_TS="$(date +%s)"
PROGRESS_FILE="$OUT/local_logs/.progress_count"
LOCK_FILE="$OUT/local_logs/.progress.lock"
LOCK_DIR="$OUT/local_logs/.progress.lockdir"
printf '0\n' > "$PROGRESS_FILE"

format_seconds() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

with_progress_lock() {
  if command -v flock >/dev/null 2>&1; then
    (
      flock 9
      "$@"
    ) 9>"$LOCK_FILE"
    return
  fi

  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    sleep 0.1
  done
  "$@"
  rmdir "$LOCK_DIR"
}

increment_progress() {
  local done_count
  done_count="$(cat "$PROGRESS_FILE" 2>/dev/null || printf '0')"
  done_count=$((done_count + 1))
  printf '%s\n' "$done_count" > "$PROGRESS_FILE"
  printf '%s\n' "$done_count"
}

print_progress() {
  local done_count="$1"
  local gpu="$2"
  local model="$3"
  local dataset="$4"
  local sample="$5"
  local seed="$6"
  local layers="$7"
  local job_seconds="$8"
  local now elapsed avg remaining eta

  now="$(date +%s)"
  elapsed=$((now - START_TS))
  if ((done_count > 0)); then
    avg=$((elapsed / done_count))
  else
    avg=0
  fi
  remaining=$((TOTAL_JOBS - done_count))
  if ((remaining < 0)); then
    remaining=0
  fi
  eta=$((avg * remaining))

  printf '[%s] [GPU %s] DONE %s/%s %s %s sample=%s seed=%s layers=%s job=%s elapsed=%s avg/job=%s ETA=%s\n' \
    "$(date '+%F %T')" \
    "$gpu" \
    "$done_count" \
    "$TOTAL_JOBS" \
    "$model" \
    "$dataset" \
    "$sample" \
    "$seed" \
    "$layers" \
    "$(format_seconds "$job_seconds")" \
    "$(format_seconds "$elapsed")" \
    "$(format_seconds "$avg")" \
    "$(format_seconds "$eta")"
}

echo "Job file: $JOBS"
echo "Output root: $OUT"
echo "GPUs: $GPU_CSV  workers=$WORKER_COUNT"
echo "Precision: $PRECISION  max_len=$MAX_LEN"
echo "Total jobs: $TOTAL_JOBS"
echo "Logs: $OUT/local_logs"
echo "Optional batch overrides: BATCH_SIZE/GRAD_ACCUM or ENC_BATCH_SIZE/ENC_GRAD_ACCUM, AR_BATCH_SIZE/AR_GRAD_ACCUM, MLM_BATCH_SIZE/MLM_GRAD_ACCUM"

get_override() {
  local specific_var="$1"
  local global_var="$2"
  local value=""
  if [[ -n "${!global_var-}" ]]; then
    value="${!global_var}"
  fi
  if [[ -n "${!specific_var-}" ]]; then
    value="${!specific_var}"
  fi
  printf '%s' "$value"
}

worker() {
  local worker_index="$1"
  local gpu="$2"
  local line_index=0
  local job_number start_ts end_ts duration status done_count log
  local upper_model batch_size grad_accum extra_args
  while IFS=$'\t' read -r model dataset sample seed layers heads initialization; do
    if [[ -z "${model:-}" ]]; then
      continue
    fi
    if (( line_index % WORKER_COUNT == worker_index )); then
      initialization="${initialization:-scratch}"
      log="$OUT/local_logs/${model}_${dataset}_${sample}_${seed}_${layers}.log"
      upper_model="$(printf '%s' "$model" | tr '[:lower:]' '[:upper:]')"
      batch_size="$(get_override "${upper_model}_BATCH_SIZE" "BATCH_SIZE")"
      grad_accum="$(get_override "${upper_model}_GRAD_ACCUM" "GRAD_ACCUM")"
      extra_args=()
      if [[ -n "$batch_size" ]]; then
        extra_args+=(--batch_size "$batch_size")
      fi
      if [[ -n "$grad_accum" ]]; then
        extra_args+=(--gradient_accumulation_steps "$grad_accum")
      fi
      if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
        extra_args+=(--eval_batch_size "$EVAL_BATCH_SIZE")
      fi
      if [[ -n "${INFERENCE_BATCH_SIZE:-}" ]]; then
        extra_args+=(--inference_batch_size "$INFERENCE_BATCH_SIZE")
      fi
      job_number=$((line_index + 1))
      start_ts="$(date +%s)"
      printf '[%s] [GPU %s] START job=%s/%s %s %s sample=%s seed=%s layers=%s batch=%s grad_accum=%s log=%s\n' \
        "$(date '+%F %T')" \
        "$gpu" \
        "$job_number" \
        "$TOTAL_JOBS" \
        "$model" \
        "$dataset" \
        "$sample" \
        "$seed" \
        "$layers" \
        "${batch_size:-paper-default}" \
        "${grad_accum:-paper-default}" \
        "$log"
      if CUDA_VISIBLE_DEVICES="$gpu" python "./repro_fig2/train_one.py" \
        --model "$model" \
        --dataset "$dataset" \
        --sample_size "$sample" \
        --seed "$seed" \
        --layers "$layers" \
        --heads "$heads" \
        --initialization "$initialization" \
        --precision "$PRECISION" \
        --max_len "$MAX_LEN" \
        --output_root "$OUT" \
        "${extra_args[@]}" >"$log" 2>&1; then
        status=0
      else
        status=$?
      fi
      end_ts="$(date +%s)"
      duration=$((end_ts - start_ts))
      if ((status != 0)); then
        printf '[%s] [GPU %s] FAILED %s %s sample=%s seed=%s layers=%s job=%s log=%s\n' \
          "$(date '+%F %T')" \
          "$gpu" \
          "$model" \
          "$dataset" \
          "$sample" \
          "$seed" \
          "$layers" \
          "$(format_seconds "$duration")" \
          "$log"
        tail -n 40 "$log" || true
        exit "$status"
      fi
      done_count="$(with_progress_lock increment_progress)"
      print_progress "$done_count" "$gpu" "$model" "$dataset" "$sample" "$seed" "$layers" "$duration"
    fi
    line_index=$((line_index + 1))
  done < "$JOBS"
}

cd .
for index in "${!GPUS[@]}"; do
  worker "$index" "${GPUS[$index]}" &
done
wait
