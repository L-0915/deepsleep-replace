#!/bin/bash
# Run lm-evaluation-harness on 5 benchmarks for 8 models
# Usage: bash scripts/eval/run_benchmark.sh

set -e

export PYTHONPATH="/root/dslm/lm-evaluation-harness:$PYTHONPATH"
export HF_ENDPOINT="https://hf-mirror.com"

DS_PRETRAIN="/root/dslm/deepsleep/out/pretrain/final"
DS_B01_HF="/root/dslm/deepsleep/out/ds_b0.1_hf"
DS_B05_HF="/root/dslm/deepsleep/out/ds_b0.5_hf"
MINIMIND3="/root/dslm/minimind-3"
MEDICAL_GPT2="Dominic0406/medical_gpt2"
QW_BASE="/public/huggingface-models/Qwen/Qwen2.5-0.5B"
QW_B01_HF="/root/blockdata/dpo_exp/qwen_b0.1_s42/final_model"
QW_B05_HF="/root/blockdata/dpo_exp/qwen_b0.5_s42/final_model"
RESULTS_DIR="/root/dslm/deepsleep/data/eval/benchmark_results"
mkdir -p "$RESULTS_DIR"

ALL_TASKS="pubmedqa_local,medqa_4options,arc_easy,piqa,openbookqa"
NEW_TASK="openbookqa"

echo "=========================================="
echo "Phase 4: lm-evaluation-harness Benchmark"
echo "=========================================="

# === New models: run all 5 benchmarks ===

echo ""
echo "[1/8] DeepSleep Pretrain ..."
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$DS_PRETRAIN,trust_remote_code=True" \
    --tasks "$ALL_TASKS" \
    --batch_size auto \
    --output_path "$RESULTS_DIR/ds_pretrain.json" \
    --trust_remote_code \
    --log_samples

echo ""
echo "[2/8] MiniMind-3 ..."
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MINIMIND3,trust_remote_code=True" \
    --tasks "$ALL_TASKS" \
    --batch_size auto \
    --output_path "$RESULTS_DIR/minimind3.json" \
    --trust_remote_code \
    --log_samples

echo ""
echo "[3/8] Medical-GPT2 ..."
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MEDICAL_GPT2" \
    --tasks "$ALL_TASKS" \
    --batch_size auto \
    --output_path "$RESULTS_DIR/medical_gpt2.json" \
    --trust_remote_code \
    --log_samples

echo ""
echo "[4/8] Qwen2.5-0.5B Base ..."
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$QW_BASE" \
    --tasks "$ALL_TASKS" \
    --batch_size auto \
    --output_path "$RESULTS_DIR/qw_base.json" \
    --trust_remote_code \
    --log_samples

# === Already ran models: only run OpenBookQA, then merge ===

echo ""
echo "[5/8] DeepSleep MoE beta=0.1 (OpenBookQA only) ..."
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$DS_B01_HF,trust_remote_code=True" \
    --tasks "$NEW_TASK" \
    --batch_size auto \
    --output_path "$RESULTS_DIR/ds_b0.1_obqa.json" \
    --trust_remote_code \
    --log_samples

echo ""
echo "[6/8] DeepSleep MoE beta=0.5 (OpenBookQA only) ..."
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$DS_B05_HF,trust_remote_code=True" \
    --tasks "$NEW_TASK" \
    --batch_size auto \
    --output_path "$RESULTS_DIR/ds_b0.5_obqa.json" \
    --trust_remote_code \
    --log_samples

echo ""
echo "[7/8] Qwen Dense beta=0.1 (OpenBookQA only) ..."
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$QW_B01_HF" \
    --tasks "$NEW_TASK" \
    --batch_size auto \
    --output_path "$RESULTS_DIR/qw_b0.1_obqa.json" \
    --trust_remote_code \
    --log_samples

echo ""
echo "[8/8] Qwen Dense beta=0.5 (OpenBookQA only) ..."
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$QW_B05_HF" \
    --tasks "$NEW_TASK" \
    --batch_size auto \
    --output_path "$RESULTS_DIR/qw_b0.5_obqa.json" \
    --trust_remote_code \
    --log_samples

# === Merge OpenBookQA results into existing files ===
echo ""
echo "Merging OpenBookQA results into existing files ..."
python3 scripts/eval/merge_results.py

echo ""
echo "=========================================="
echo "Done! Results saved to $RESULTS_DIR/"
echo "=========================================="
