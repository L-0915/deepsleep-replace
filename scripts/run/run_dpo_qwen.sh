#!/bin/bash
# Qwen2.5-0.5B DPO (Direct Preference Optimization)
#
# Quick start:
#   bash scripts/run/run_dpo_qwen.sh
#
# Custom beta:
#   BETA=0.5 bash scripts/run/run_dpo_qwen.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------- Overrides ----------
CONFIG=${CONFIG:-configs/dpo_qwen.yaml}
DATA=${DATA:-data/dpo/xiaoxi_dpo.jsonl}
SFT_MODEL=${SFT_MODEL:-out/sft_qwen/final_model}
OUTPUT=${OUTPUT:-out/dpo_qwen}
EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-4}
LR=${LR:-5e-7}
BETA=${BETA:-0.1}
SEED=${SEED:-42}
SEQ_LEN=${SEQ_LEN:-2048}

# ---------- Command ----------
CMD="python trainer/train_dpo_qwen.py \
  --config $CONFIG \
  --data_path $DATA \
  --sft_model_path $SFT_MODEL \
  --save_dir $OUTPUT \
  --epochs $EPOCHS \
  --batch_size $BATCH \
  --learning_rate $LR \
  --dpo_beta $BETA \
  --seed $SEED \
  --max_seq_len $SEQ_LEN"

echo "========================================"
echo " Qwen2.5-0.5B DPO"
echo "========================================"
echo " Config:  $CONFIG"
echo " Data:    $DATA"
echo " From:    $SFT_MODEL"
echo " Output:  $OUTPUT"
echo " Beta:    $BETA"
echo " Seed:    $SEED"
echo "========================================"

eval $CMD
