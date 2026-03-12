#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

CONFIG_PATH="${1:-configs/dpo_qwen3_4b_lora.json}"

echo "============================================="
echo "Starting DPO training"
echo "Config Path: ${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "============================================="

python train_dpo.py --config_path "${CONFIG_PATH}"
