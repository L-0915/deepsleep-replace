#!/bin/bash
# Qwen2.5-0.5B SFT (identical hyperparameters to DeepSleep SFT)
#
# Quick start:
#   bash scripts/run/run_sft_qwen.sh
#
# Reads configs/sft_qwen.yaml for training params.
# Logs: out/sft_qwen_train.log

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------- Performance ----------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export TRANSFORMERS_VERBOSITY=error

# ---------- Model ----------
MODEL_PATH=${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen2.5-0.5B}

# ---------- Data (same as DeepSleep SFT) ----------
DATA=${DATA:-data/sft/xiaoxi/xiaoxi_sft.jsonl}
OUTPUT=${OUTPUT:-out/sft_qwen}

# ---------- Command (reads sft_qwen.yaml for Qwen-specific params) ----------
CMD="python trainer/train_sft_qwen.py \
  --config configs/sft_qwen.yaml \
  --model_path $MODEL_PATH \
  --data_path $DATA"

echo "========================================"
echo " Qwen2.5-0.5B SFT"
echo "========================================"
echo " Config:  configs/sft_qwen.yaml"
echo " Model:   $MODEL_PATH"
echo " Data:    $DATA"
echo "========================================"

nohup $CMD > out/sft_qwen_train.log 2>&1 &
echo "PID: $!"
echo "Log: tail -f out/sft_qwen_train.log"
