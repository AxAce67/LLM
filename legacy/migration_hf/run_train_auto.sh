#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_TEXT="${TRAIN_TEXT:-dataset/hf/train.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-models/hf_lora}"

echo "[HF Migration] base_model=${BASE_MODEL}"
echo "[HF Migration] train_text=${TRAIN_TEXT}"
echo "[HF Migration] output_dir=${OUTPUT_DIR}"
echo "[HF Migration] host=$(hostname) role=${SYSTEM_ROLE:-master}"

python3 migration_hf/train_lora.py \
  --base_model "${BASE_MODEL}" \
  --train_text "${TRAIN_TEXT}" \
  --output_dir "${OUTPUT_DIR}" \
  --auto_tune
