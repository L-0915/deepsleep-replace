#!/bin/bash
# DeepSleep MoE | beta=0.1 | lr=1e-6 | seed=123
cd /root/dslm/deepsleep
python trainer/train_dpo.py \
    --data_path data/dpo/xiaoxi_dpo.jsonl \
    --tokenizer_path checkpoints/tokenizer \
    --sft_checkpoint out/sft/deepsleep-sft.pth \
    --save_dir /root/blockdata/dpo_exp/ds_b0.1_lr1e-6_s123 \
    --epochs 1 --batch_size 4 --learning_rate 1e-6 --accumulation_steps 4 \
    --dpo_beta 0.1 --seed 123 --max_seq_len 3072 --num_workers 0
