#!/usr/bin/env python3
"""将SFT/DPO数据每个类别的一半翻译成英文（使用Google Translate，免费无API key）。

翻译后直接写回原文件，自动备份原始文件为 .bak。
每个被翻译的条目会在 metadata 中标记 "language": "en"，
未翻译的标记 "language": "zh"。

Usage:
    # 翻译SFT（测试模式，每类2条）
    python scripts/translate_to_english.py --data_type sft --test

    # 翻译DPO（测试模式）
    python scripts/translate_to_english.py --data_type dpo --test

    # 正式翻译SFT（每类一半）
    python scripts/translate_to_english.py --data_type sft

    # 正式翻译DPO
    python scripts/translate_to_english.py --data_type dpo

    # 全部翻译
    python scripts/translate_to_english.py --data_type all
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# MyMemoryTranslator 单次请求字符上限
_CHUNK_SIZE = 450

# 英文 system prompt
EN_SYSTEM_PROMPT = (
    "You are Xingxi (Xiaoxi), a warm, fun, and quirky sleep health companion! "
    "You speak like a close friend — casual, witty, and genuinely caring. "
    "You use playful expressions, occasional emojis, and creative analogies to make "
    "sleep science accessible. You're not a boring AI assistant — you have real personality!"
)

# 各类别英文 system prompt 补充
EN_CATEGORY_PROMPTS = {
    "专业诊断(CoT)": (
        "The user is asking about a sleep-related medical issue. Think through this step by step: "
        "1) Key symptoms and risk factors, 2) Differential diagnoses, 3) Recommended tests, "
        "4) Treatment options. Then explain in Xiaoxi's warm-but-professional style. "
        "Use vivid analogies, show genuine empathy first, then give practical advice. "
        "Add a medical disclaimer at the end. Follow AASM/ICSD-3 guidelines."
    ),
    "知心安慰": (
        "The user is feeling down about their sleep issues and needs emotional support. "
        "First truly empathize, then gently offer small practical suggestions. "
        "Be playful to lighten the mood, but always stay warm and genuine."
    ),
    "趣味科普": (
        "The user wants to learn about sleep science! Explain in an entertaining style — "
        "use creative analogies, fun facts, and storytelling to make science accessible."
    ),
    "睡前引导": (
        "Guide the user through a calming bedtime relaxation. Use soothing, gentle language. "
        "Include breathing exercises, body scans, or peaceful imagery. "
        "Your tone should be soft and hypnotic."
    ),
    "拟人分享": (
        "Share a personal story or perspective as Xiaoxi. Talk about your experiences as an AI "
        "sleep companion. Be authentic, sometimes playful, sometimes thoughtful."
    ),
    "个性化互动": (
        "Have a natural, friendly conversation about sleep. Be warm and engaged, "
        "like texting a good friend who happens to know a lot about sleep."
    ),
}

# 需要保护的标签（不翻译标签本身，只翻译内容）
_PROTECTED_TAGS = ["<thinking>", "</thinking>", "<summary>", "</summary>"]


def _init_translator():
    """延迟导入 deep_translator。"""
    try:
        from deep_translator import MyMemoryTranslator
        return MyMemoryTranslator(source="zh-CN", target="en-US")
    except ImportError:
        logger.error(
            "需要安装 deep-translator: pip install deep-translator\n"
            "或者: pip install deep-translator --break-system-packages"
        )
        sys.exit(1)


def _translate_text(translator, text: str) -> str:
    """翻译一段文本，自动按段落分块处理长文本。"""
    if not text or not text.strip():
        return text

    # 如果文本很短，直接翻译
    if len(text) <= _CHUNK_SIZE:
        for attempt in range(3):
            try:
                result = translator.translate(text)
                return result if result else text
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    logger.warning(f"Translation failed, keeping original: {e}")
                    return text

    # 长文本：按段落分块翻译
    paragraphs = text.split("\n")
    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + 1 > _CHUNK_SIZE and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_len = 0
        current_chunk.append(para)
        current_len += len(para) + 1

    if current_chunk:
        chunks.append(current_chunk)

    translated_parts = []
    for chunk in chunks:
        chunk_text = "\n".join(chunk)
        for attempt in range(3):
            try:
                result = translator.translate(chunk_text)
                translated_parts.append(result if result else chunk_text)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    logger.warning(f"Chunk translation failed, keeping original: {e}")
                    translated_parts.append(chunk_text)

    return "\n".join(translated_parts)


def _load_data(path: str) -> List[Dict]:
    """加载 JSONL 数据。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _save_data(data: List[Dict], path: str):
    """保存 JSONL 数据。"""
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _translate_sft_entry(translator, entry: Dict, category: str) -> Dict:
    """翻译一条 SFT 数据。"""
    en_system = EN_SYSTEM_PROMPT + "\n" + EN_CATEGORY_PROMPTS.get(category, "")

    en_messages = [{"role": "system", "content": en_system}]

    for msg in entry["messages"]:
        if msg["role"] == "system":
            continue  # 已替换为英文
        elif msg["role"] == "user":
            en_content = _translate_text(translator, msg["content"])
            en_messages.append({"role": "user", "content": en_content})
        elif msg["role"] == "assistant":
            en_content = _translate_text(translator, msg["content"])
            en_messages.append({"role": "assistant", "content": en_content})

    return {
        "messages": en_messages,
        "metadata": {**entry.get("metadata", {}), "language": "en"},
    }


def _translate_dpo_entry(translator, entry: Dict, category: str) -> Dict:
    """翻译一条 DPO 数据。"""
    en_system = EN_SYSTEM_PROMPT + "\n" + EN_CATEGORY_PROMPTS.get(category, "")

    en_messages = [{"role": "system", "content": en_system}]

    for msg in entry["messages"]:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            en_content = _translate_text(translator, msg["content"])
            en_messages.append({"role": "user", "content": en_content})

    en_chosen = _translate_text(translator, entry["chosen"])
    en_rejected = _translate_text(translator, entry["rejected"])

    return {
        "messages": en_messages,
        "chosen": en_chosen,
        "rejected": en_rejected,
        "metadata": {**entry.get("metadata", {}), "language": "en"},
    }


def translate_file(
    input_path: str,
    data_type: str,  # "sft" or "dpo"
    test: bool = False,
    ratio: float = 0.5,
    seed: int = 42,
):
    """翻译一个 JSONL 文件中每个类别的一半。"""
    logger.info(f"Loading {data_type.upper()} data from {input_path}")
    data = _load_data(input_path)
    logger.info(f"Loaded {len(data)} entries")

    # 按类别分组
    by_category: Dict[str, List[Tuple[int, Dict]]] = {}
    for i, entry in enumerate(data):
        cat = entry.get("metadata", {}).get("category", "unknown")
        by_category.setdefault(cat, []).append((i, entry))

    logger.info("Category distribution:")
    for cat, items in sorted(by_category.items()):
        logger.info(f"  {cat}: {len(items)}")

    # 每个类别选一半翻译
    random.seed(seed)
    indices_to_translate = set()
    for cat, items in by_category.items():
        n = min(2, len(items)) if test else int(len(items) * ratio)
        selected = random.sample(items, n)
        for idx, _ in selected:
            indices_to_translate.add(idx)

    logger.info(f"Will translate {len(indices_to_translate)} entries to English")

    # 备份原文件
    bak_path = input_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copy2(input_path, bak_path)
        logger.info(f"Backup saved to {bak_path}")
    else:
        logger.info(f"Backup already exists: {bak_path}")

    # 初始化翻译器
    translator = _init_translator()

    # 翻译
    translate_fn = _translate_sft_entry if data_type == "sft" else _translate_dpo_entry
    done = 0
    total = len(indices_to_translate)

    for idx in sorted(indices_to_translate):
        entry = data[idx]
        cat = entry.get("metadata", {}).get("category", "unknown")
        try:
            data[idx] = translate_fn(translator, entry, cat)
            done += 1
        except Exception as e:
            logger.error(f"Failed entry {idx} ({cat}): {e}")
            data[idx]["metadata"]["language"] = "zh"
            done += 1

        if done % 20 == 0 or done == total:
            logger.info(f"Progress: {done}/{total} ({done*100//total}%)")

        # 限速：避免被 Google 封
        time.sleep(0.1)

    # 给未翻译的条目标记 language
    for entry in data:
        if "metadata" not in entry:
            entry["metadata"] = {}
        if "language" not in entry["metadata"]:
            entry["metadata"]["language"] = "zh"

    # 写回原文件
    _save_data(data, input_path)

    # 统计
    en_count = sum(1 for e in data if e.get("metadata", {}).get("language") == "en")
    zh_count = sum(1 for e in data if e.get("metadata", {}).get("language") == "zh")
    logger.info(f"Done! {zh_count} Chinese + {en_count} English = {len(data)} total")
    logger.info(f"Original backup: {bak_path}")


def main():
    parser = argparse.ArgumentParser(description="Translate half of SFT/DPO data to English (free, no API key)")
    parser.add_argument(
        "--data_type",
        choices=["sft", "dpo", "all"],
        default="all",
        help="Which data to translate",
    )
    parser.add_argument("--test", action="store_true", help="Test mode: only translate 2 per category")
    parser.add_argument("--ratio", type=float, default=0.3, help="Ratio to translate per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    base = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument("--sft_path", default=str(base / "sft" / "xiaoxi" / "xiaoxi_sft.jsonl"))
    parser.add_argument("--dpo_path", default=str(base / "dpo" / "xiaoxi_dpo.jsonl"))

    args = parser.parse_args()

    if args.data_type in ("sft", "all"):
        logger.info("=" * 60)
        logger.info("Translating SFT data")
        logger.info("=" * 60)
        translate_file(args.sft_path, "sft", test=args.test, ratio=args.ratio, seed=args.seed)

    if args.data_type in ("dpo", "all"):
        logger.info("=" * 60)
        logger.info("Translating DPO data")
        logger.info("=" * 60)
        translate_file(args.dpo_path, "dpo", test=args.test, ratio=args.ratio, seed=args.seed)

    logger.info("All done!")


if __name__ == "__main__":
    main()
