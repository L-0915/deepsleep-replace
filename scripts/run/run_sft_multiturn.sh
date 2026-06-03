#!/bin/bash
# 多轮 SFT 训练 (在 DPO 权重基础上)
# Flow: Base → SFT → DPO → 多轮SFT (3000条)
# 训练完成后自动覆盖原 DPO 权重

set -e
cd "$(dirname "$0")/../.."

echo "=========================================="
echo " Step 1/2: 多轮 SFT (β=0.1)"
echo "=========================================="
python trainer/train_sft_qwen.py --data_path data/sft/xiaoxi/xiaoxi_sft_multiturn.jsonl --config configs/sft_qwen_multiturn_b0.1.yaml

echo ""
echo "=========================================="
echo " Step 2/2: 多轮 SFT (β=0.5)"
echo "=========================================="
python trainer/train_sft_qwen.py --data_path data/sft/xiaoxi/xiaoxi_sft_multiturn.jsonl --config configs/sft_qwen_multiturn_b0.5.yaml

echo ""
echo "=========================================="
echo " 覆盖原 DPO 权重"
echo "=========================================="

# 备份原 DPO 权重
echo "备份原 DPO 权重..."
cp -r /root/blockdata/dpo_exp/qwen_b0.1_s42/final_model /root/blockdata/dpo_exp/qwen_b0.1_s42/final_model.bak
cp -r /root/blockdata/dpo_exp/qwen_b0.5_s42/final_model /root/blockdata/dpo_exp/qwen_b0.5_s42/final_model.bak

# 用多轮 SFT 权重覆盖
echo "覆盖 β=0.1..."
cp -r out/sft_qwen_multiturn_b0.1/final_model/* /root/blockdata/dpo_exp/qwen_b0.1_s42/final_model/

echo "覆盖 β=0.5..."
cp -r out/sft_qwen_multiturn_b0.5/final_model/* /root/blockdata/dpo_exp/qwen_b0.5_s42/final_model/

echo ""
echo "✅ 完成！原 DPO 权重已备份为 final_model.bak"
echo "  β=0.1: /root/blockdata/dpo_exp/qwen_b0.1_s42/final_model/"
echo "  β=0.5: /root/blockdata/dpo_exp/qwen_b0.5_s42/final_model/"
