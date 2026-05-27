#!/bin/bash
# DeepSleep SFT (Supervised Fine-Tuning)
#
# Quick start:
#   bash scripts/run/run_sft.sh
#
# Custom checkpoint:
#   FROM_WEIGHT=out/cpt/final/model.pth bash scripts/run/run_sft.sh
#
# Logs: out/sft_train.log
# TensorBoard: tensorboard --logdir out/sft/runs

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------- Performance ----------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export TRANSFORMERS_VERBOSITY=error

# ---------- Overrides ----------
CONFIG=${CONFIG:-configs/sft.yaml}
TOKENIZER=${TOKENIZER:-checkpoints/tokenizer}
DATA=${DATA:-data/sft/xiaoxi/xiaoxi_sft.jsonl}
FROM_WEIGHT=${FROM_WEIGHT:-out/cpt/final/model.pth}
OUTPUT=${OUTPUT:-out/sft}

# ---------- Command ----------
CMD="python trainer/train_sft.py \
  --config $CONFIG \
  --data_path $DATA \
  --tokenizer_path $TOKENIZER \
  --from_weight $FROM_WEIGHT \
  --save_dir $OUTPUT"

echo "========================================"
echo " DeepSleep SFT"
echo "========================================"
echo " Config:  $CONFIG"
echo " Data:    $DATA"
echo " From:    $FROM_WEIGHT"
echo " Output:  $OUTPUT"
echo "========================================"

$CMD 2>&1 | tee out/sft_train.log
