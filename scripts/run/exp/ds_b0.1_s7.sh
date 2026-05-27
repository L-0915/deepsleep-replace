#!/bin/bash
# Run 9: DeepSleep MoE | beta=0.1 | seed=7
cd /root/dslm/deepsleep
python trainer/train_dpo.py \
    --data_path data/dpo/xiaoxi_dpo.jsonl \
    --tokenizer_path checkpoints/tokenizer \
    --sft_checkpoint out/sft/final_model.pth \
    --save_dir /root/blockdata/dpo_exp/ds_b0.1_s7 \
    --epochs 1 --batch_size 4 --learning_rate 5e-7 --accumulation_steps 4 \
    --dpo_beta 0.1 --seed 7 --max_seq_len 3072 --num_workers 0
