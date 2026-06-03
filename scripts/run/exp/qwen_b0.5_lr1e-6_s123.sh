#!/bin/bash
# Qwen Dense | beta=0.5 | lr=1e-6 | seed=123
cd /root/dslm/deepsleep
python trainer/train_dpo_qwen.py \
    --data_path data/dpo/xiaoxi_dpo.jsonl \
    --sft_model_path out/sft_qwen/final_model \
    --save_dir /root/blockdata/dpo_exp/qwen_b0.5_lr1e-6_s123 \
    --epochs 1 --batch_size 1 --learning_rate 1e-6 --accumulation_steps 16 \
    --dpo_beta 0.5 --seed 123 --max_seq_len 2048 --num_workers 0
