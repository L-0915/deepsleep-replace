"""交互式测试DPO模型对话效果。

用法:
  # DeepSleep模型
  python tests/test_chat.py --model ds --beta 0.1 --seed 42

  # Qwen模型
  python tests/test_chat.py --model qwen --beta 0.5 --seed 7

  # 不传参则列出所有可用模型选择
  python tests/test_chat.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

DPO_BASE = "/root/blockdata/dpo_exp"


def load_deepsleep_model(ckpt_path, tokenizer_path, device):
    import torch
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    config = DeepSleepConfig(
        d_model=768, n_layers=8, use_moe=True, num_experts=8,
        num_shared_experts=0, top_k=2, vocab_size=7200, max_position_embeddings=3072,
    )
    model = DeepSleepForCausalLM(config)
    weights = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=False)
    model = model.to(device).eval()
    return model, tokenizer


def load_qwen_model(model_path, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device).eval()
    return model, tokenizer


def generate(model, tokenizer, messages, device, model_type, max_new_tokens=512):
    import torch
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        if model_type == "ds":
            output = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
        else:
            output = model.generate(
                input_ids=inputs, max_new_tokens=max_new_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
    text = tokenizer.decode(output[0], skip_special_tokens=False)
    if "<|im_start|>assistant\n" in text:
        reply = text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    else:
        reply = text[len(prompt):].strip()
    return reply


def list_available_models():
    """列出所有可用的DPO模型。"""
    models = []
    for mt in ["ds", "qwen"]:
        for beta in [0.1, 0.5]:
            for seed in [42, 123, 7]:
                exp_name = f"{mt}_b{beta}_s{seed}"
                exp_dir = os.path.join(DPO_BASE, exp_name)
                if model_type == "ds":
                    ckpt = os.path.join(exp_dir, "final_model.pth")
                    exists = os.path.exists(ckpt)
                else:
                    ckpt = os.path.join(exp_dir, "final_model")
                    exists = os.path.exists(os.path.join(ckpt, "config.json"))
                if exists:
                    models.append((mt, beta, seed, exp_name))
    return models


def interactive(model_type, beta, seed, device="cuda:0"):
    import torch

    exp_name = f"{model_type}_b{beta}_s{seed}"
    exp_dir = os.path.join(DPO_BASE, exp_name)

    if model_type == "ds":
        model, tokenizer = load_deepsleep_model(
            os.path.join(exp_dir, "final_model.pth"), "checkpoints/tokenizer", device)
        label = "DeepSleep-MoE"
    else:
        model, tokenizer = load_qwen_model(os.path.join(exp_dir, "final_model"), device)
        label = "Qwen2.5-0.5B"

    print(f"\n{'='*50}")
    print(f"  {label} | beta={beta} | seed={seed}")
    print(f"  输入 quit 退出")
    print(f"{'='*50}\n")

    system_msg = "你是星辰曦（小曦），一位温暖、有趣的睡眠健康助手。你擅长用亲切的语气帮助用户改善睡眠问题，也会在需要时展现专业的一面。"

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_input},
        ]
        reply = generate(model, tokenizer, messages, device, model_type)
        print(f"\n小曦: {reply}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="交互式测试DPO模型")
    parser.add_argument("--model", choices=["ds", "qwen"])
    parser.add_argument("--beta", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.model and args.beta and args.seed:
        interactive(args.model, args.beta, args.seed, args.device)
    else:
        # 列出可用模型让用户选择
        print("\n可用模型:")
        print(f"  {'#':<4} {'模型':<20} {'Beta':<6} {'Seed':<6}")
        print(f"  {'─'*40}")
        available = []
        idx = 1
        for mt in ["ds", "qwen"]:
            for beta in [0.1, 0.5]:
                for seed in [42, 123, 7]:
                    exp_name = f"{mt}_b{beta}_s{seed}"
                    exp_dir = os.path.join(DPO_BASE, exp_name)
                    if mt == "ds":
                        ok = os.path.exists(os.path.join(exp_dir, "final_model.pth"))
                    else:
                        ok = os.path.exists(os.path.join(exp_dir, "final_model", "config.json"))
                    if ok:
                        label = "DeepSleep-MoE" if mt == "ds" else "Qwen2.5-0.5B"
                        print(f"  {idx:<4} {label:<20} {beta:<6} {seed:<6}")
                        available.append((mt, beta, seed))
                        idx += 1

        if not available:
            print("  没有找到已完成的模型")
        else:
            choice = input("\n选择编号: ").strip()
            try:
                i = int(choice) - 1
                mt, beta, seed = available[i]
                interactive(mt, beta, seed, args.device)
            except (ValueError, IndexError):
                print("无效选择")
