#!/bin/bash

# 云端 4090 单卡 SFT 训练启动脚本示例

# 指定 CUDA 设备为单卡 0
export CUDA_VISIBLE_DEVICES=0

# 防止 tokenizer 冲突警告
export TOKENIZERS_PARALLELISM=false

# 数据集和输出路径
DATA_PATH="data/SFT_data/exports/sft_train_v1.jsonl"
OUTPUT_DIR="./output/qwen-sft-v1"

echo "============================================="
echo "Starting SFT Training on Single 4090 GPU ...."
echo "Data Path: $DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "============================================="

python train_sft.py \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --logging_steps 5 \
    --max_seq_length 2048 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05
