#!/bin/bash
# 依次执行全部6组Qwen高LR (1e-6) DPO实验
# batch_size=1, accumulation=16, 有效batch=16
# 预计耗时: ~1.5h on RTX 4090 D
# 用法: bash scripts/run/exp/run_qwen_lr1e-6.sh

set -e
cd /root/dslm/deepsleep

EXPERIMENTS=(
    qwen_b0.1_lr1e-6_s42
    qwen_b0.1_lr1e-6_s123
    qwen_b0.1_lr1e-6_s7
    qwen_b0.5_lr1e-6_s42
    qwen_b0.5_lr1e-6_s123
    qwen_b0.5_lr1e-6_s7
)

TOTAL=${#EXPERIMENTS[@]}
START=$(date +%s)

echo "=========================================="
echo "  Running $TOTAL Qwen DPO experiments (LR=1e-6)"
echo "  batch_size=1, accumulation=16"
echo "  Started: $(date)"
echo "=========================================="

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NUM=$((i + 1))
    echo ""
    echo ">>> [$NUM/$TOTAL] $EXP ($(date +%H:%M:%S))"
    echo "=========================================="
    bash "scripts/run/exp/${EXP}.sh"
    echo ">>> [$NUM/$TOTAL] $EXP DONE"
done

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "=========================================="
echo "  All $TOTAL Qwen experiments completed!"
echo "  Elapsed: ${ELAPSED} minutes"
echo "  Finished: $(date)"
echo "=========================================="
echo ""
echo "Run analysis:"
echo "  python scripts/analysis/analyze_factorial.py"
