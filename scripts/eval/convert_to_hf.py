"""Convert DeepSleep .pth checkpoint to HuggingFace format for lm-evaluation-harness."""

import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM


def convert(pth_path: str, output_dir: str, tokenizer_path: str):
    os.makedirs(output_dir, exist_ok=True)

    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)

    config = DeepSleepConfig()
    model = DeepSleepForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    model.save_pretrained(output_dir)

    for f in os.listdir(tokenizer_path):
        src = os.path.join(tokenizer_path, f)
        dst = os.path.join(output_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    print(f"Done: {output_dir}")
    print(f"  Config: {output_dir}/config.json")
    print(f"  Model:  {output_dir}/model.safetensors (or .bin)")
    print(f"  Tokenizer: copied from {tokenizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="/root/dslm/deepsleep/checkpoints/tokenizer")
    args = parser.parse_args()
    convert(args.pth, args.out, args.tokenizer)
