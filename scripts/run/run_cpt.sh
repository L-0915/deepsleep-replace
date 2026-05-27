#!/bin/bash
# DeepSleep CPT — Medical QA Continued Pretraining
#
# Quick start:
#   bash scripts/run/run_cpt.sh
#
# Resume from checkpoint:
#   RESUME=out/cpt/checkpoint-1000 bash scripts/run/run_cpt.sh
#
# Logs: out/cpt_train.log
# TensorBoard: tensorboard --logdir out/cpt/runs

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------- Environment ----------
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------- Config ----------
CONFIG=${CONFIG:-configs/cpt.yaml}
TOKENIZER=${TOKENIZER:-checkpoints/tokenizer}
PRETRAINED=${PRETRAINED:-out/pretrain/final}
RESUME=${RESUME:-}

# ---------- Run ----------
mkdir -p out/cpt

CMD="python trainer/train_cpt.py --config $CONFIG --tokenizer_path $TOKENIZER --pretrained_model_path $PRETRAINED"
[ -n "$RESUME" ] && CMD="$CMD --resume_from_checkpoint $RESUME"

echo "========================================"
echo " DeepSleep CPT — Medical QA"
echo "========================================"
echo " Config:     $CONFIG"
echo " Tokenizer:  $TOKENIZER"
echo " Pretrained: $PRETRAINED"
echo " Resume:     ${RESUME:-none}"
echo " Log:        out/cpt_train.log"
echo "========================================"

$CMD 2>&1 | tee out/cpt_train.log
