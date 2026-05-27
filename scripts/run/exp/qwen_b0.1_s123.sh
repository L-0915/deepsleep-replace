#!/bin/bash
# Run 6: Qwen Dense | beta=0.1 | seed=123
cd /root/dslm/deepsleep
python trainer/train_dpo_qwen.py \
    --data_path data/dpo/xiaoxi_dpo.jsonl \
    --sft_model_path out/sft_qwen/final_model \
    --save_dir /root/blockdata/dpo_exp/qwen_b0.1_s123 \
    --epochs 1 --batch_size 2 --learning_rate 5e-7 --accumulation_steps 8 \
    --dpo_beta 0.1 --seed 123 --max_seq_len 3072 --num_workers 0
