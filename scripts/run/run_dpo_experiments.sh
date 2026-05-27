#!/bin/bash
# ====================================================================
# 2² 全因子 DPO 实验 — 自动运行全部 8 组
#
# 实验设计:
#   Factor A: 模型架构 (DeepSleep MoE vs Qwen Dense)
#   Factor B: DPO Beta (0.1 vs 0.5)
#   每组重复 2 次 (seed=42, seed=123)
#   共 4 × 2 = 8 runs
#
# 用法:
#   bash scripts/run/run_dpo_experiments.sh          # 运行全部 8 组
#   bash scripts/run/run_dpo_experiments.sh 3        # 只运行第 3 组
#   bash scripts/run/run_dpo_experiments.sh 5 6      # 运行第 5、6 组
# ====================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------- 固定参数 ----------
DATA="data/dpo/xiaoxi_dpo.jsonl"
TOKENIZER="checkpoints/tokenizer"
DS_SFT="out/sft/final_model.pth"
QWEN_SFT="out/sft_qwen/final_model"
EPOCHS=1
BATCH=4
LR="5e-7"
ACCUM=4
SEQ_LEN=2048

# ---------- 实验矩阵 ----------
declare -a MODELS=("ds" "ds" "qwen" "qwen")
declare -a BETAS=("0.1" "0.5" "0.1" "0.5")
declare -a LABELS=("DeepSleep-MoE" "DeepSleep-MoE" "Qwen2.5-0.5B" "Qwen2.5-0.5B")
SEEDS=(42 123)

# ---------- 选择运行哪些 ----------
if [ $# -eq 0 ]; then
    RUNS=(1 2 3 4 5 6 7 8)
else
    RUNS=("$@")
fi

TOTAL=${#RUNS[@]}
CURRENT=0

echo "============================================================"
echo "  2² 全因子 DPO 实验 — 共 ${TOTAL} 组"
echo "============================================================"
echo "  Run | 模型              | Beta | Seed | 输出目录"
echo "  ----+-------------------+------+------+--------------------"
for i in 1 2 3 4; do
    IDX=$((i - 1))
    for S in "${SEEDS[@]}"; do
        RUN_NUM=$(((IDX) * 2 + (S == 42 ? 1 : 2)))
        printf "  %-4s| %-17s | %-4s | %-4s | /root/blockdata/dpo_exp/%s_b%s_s%s\n" \
            "$RUN_NUM" "${LABELS[$IDX]}" "${BETAS[$IDX]}" "$S" \
            "${MODELS[$IDX]}" "${BETAS[$IDX]}" "$S"
    done
done
echo "============================================================"

for RUN in "${RUNS[@]}"; do
    # 计算 group index 和 seed
    GROUP=$(( (RUN - 1) / 2 ))       # 0,1,2,3
    SEED_IDX=$(( (RUN - 1) % 2 ))    # 0 or 1
    SEED=${SEEDS[$SEED_IDX]}
    MODEL="${MODELS[$GROUP]}"
    BETA="${BETAS[$GROUP]}"
    LABEL="${LABELS[$GROUP]}"
    OUTDIR="/root/blockdata/dpo_exp/${MODEL}_b${BETA}_s${SEED}"

    CURRENT=$((CURRENT + 1))

    echo ""
    echo "============================================================"
    echo "  [${CURRENT}/${TOTAL}] Run ${RUN}: ${LABEL} | beta=${BETA} | seed=${SEED}"
    echo "  Output: ${OUTDIR}"
    echo "============================================================"

    if [ -f "${OUTDIR}/report.json" ]; then
        echo "  ⏭  已完成，跳过"
        continue
    fi

    START_TIME=$(date +%s)

    if [ "$MODEL" = "ds" ]; then
        python trainer/train_dpo.py \
            --data_path "$DATA" \
            --tokenizer_path "$TOKENIZER" \
            --sft_checkpoint "$DS_SFT" \
            --save_dir "$OUTDIR" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH" \
            --learning_rate "$LR" \
            --accumulation_steps "$ACCUM" \
            --dpo_beta "$BETA" \
            --seed "$SEED" \
            --max_seq_len "$SEQ_LEN" \
            --log_interval 50 \
            --save_interval 200 \
            --num_workers 0
    else
        python trainer/train_dpo_qwen.py \
            --data_path "$DATA" \
            --sft_model_path "$QWEN_SFT" \
            --save_dir "$OUTDIR" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH" \
            --learning_rate "$LR" \
            --accumulation_steps "$ACCUM" \
            --dpo_beta "$BETA" \
            --seed "$SEED" \
            --max_seq_len "$SEQ_LEN" \
            --log_interval 50 \
            --save_interval 200 \
            --num_workers 0
    fi

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "  ✅ Run ${RUN} 完成 (${ELAPSED}s)"
done

echo ""
echo "============================================================"
echo "  全部实验完成！结果汇总："
echo "============================================================"
echo ""

# 收集所有 report.json
echo "  Run | 模型              | Beta | Seed | Loss     | Acc    | Time"
echo "  ----+-------------------+------+------+----------+--------+-------"
for i in 1 2 3 4 5 6 7 8; do
    GROUP=$(( (i - 1) / 2 ))
    SEED_IDX=$(( (i - 1) % 2 ))
    SEED=${SEEDS[$SEED_IDX]}
    MODEL="${MODELS[$GROUP]}"
    BETA="${BETAS[$GROUP]}"
    LABEL="${LABELS[$GROUP]}"
    OUTDIR="/root/blockdata/dpo_exp/${MODEL}_b${BETA}_s${SEED}"

    if [ -f "${OUTDIR}/report.json" ]; then
        LOSS=$(python3 -c "import json; r=json.load(open('${OUTDIR}/report.json')); print(f\"{r.get('final_loss',0):.4f}\")")
        ACC=$(python3 -c "import json; r=json.load(open('${OUTDIR}/report.json')); print(f\"{r.get('final_accuracy',0):.4f}\")")
        TIME=$(python3 -c "import json; r=json.load(open('${OUTDIR}/report.json')); print(f\"{r.get('total_time_hours',0):.2f}h\")")
        printf "  %-4s| %-17s | %-4s | %-4s | %s | %s | %s\n" "$i" "$LABEL" "$BETA" "$SEED" "$LOSS" "$ACC" "$TIME"
    else
        printf "  %-4s| %-17s | %-4s | %-4s | %-8s | %-6s | %s\n" "$i" "$LABEL" "$BETA" "$SEED" "N/A" "N/A" "N/A"
    fi
done
echo ""
echo "详细数据见: /root/blockdata/dpo_exp/*/report.json"
