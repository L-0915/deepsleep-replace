#!/usr/bin/env python3
"""
多轮 SFT 数据生成器 — 从现有单轮数据拼接生成多轮对话

用法:
  python scripts/generate_multiturn_sft.py --step generate   # 生成多轮数据
  python scripts/generate_multiturn_sft.py --step stats      # 查看统计
  python scripts/generate_multiturn_sft.py --step merge       # 合并单轮+多轮
  python scripts/generate_multiturn_sft.py --step sample     # 抽查3条数据
"""

import argparse
import json
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sft", "xiaoxi")
INPUT_FILE = os.path.join(DATA_DIR, "xiaoxi_sft.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "xiaoxi_sft_multiturn.jsonl")
MERGED_FILE = os.path.join(DATA_DIR, "xiaoxi_sft_mixed.jsonl")

SYSTEM_PROMPT = "你是星辰曦（小曦），一个温暖有趣的睡眠健康伙伴。"

# ── 追问模板: 第二轮 user 消息 ──
FOLLOWUP_TEMPLATES = [
    "谢谢你！我还想问一下，{q}",
    "对了，{q}",
    "嗯嗯~那{q}",
    "我试试你说的方法！另外{q}",
    "好的，那如果是{q}呢？",
    "明白了！还有个小问题，{q}",
    "谢谢小曦~我还想知道，{q}",
    "了解了，那关于{q}有什么建议吗？",
    "原来如此。那如果遇到{q}该怎么办？",
    "好嘞！再问一下，{q}",
]

# ── 第三轮追问模板 ──
THIRD_TURN_TEMPLATES = [
    "好的好的，最后一个问题：{q}",
    "最后一个问题哈，{q}",
    "差点忘了问，{q}",
    "哦对，还有个事：{q}",
    "顺便问一下，{q}",
]

# ── 承接句模板: 加在 assistant 回复前面 ──
BRIDGE_TEMPLATES = [
    "关于你刚才提到的，我再补充一下。",
    "嗯嗯，结合你之前说的情况，",
    "我理解你的感受。",
    "针对你新提的问题，",
    "好的，针对这个问题，",
    "这是个好问题！",
    "你说的这个情况很常见，",
    "刚才聊到了相关的话题，正好展开说说。",
]


def load_single_turn_data(path):
    """加载单轮 SFT 数据，按类别分组"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            messages = sample.get("messages", [])
            category = sample.get("metadata", {}).get("category", "未知")
            if len(messages) >= 3:
                data.append({"messages": messages, "category": category})
    return data


def group_by_category(data):
    """按类别分组"""
    groups = {}
    for item in data:
        cat = item["category"]
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(item)
    return groups


def extract_turn(single_turn_messages):
    """从单轮数据中提取 user 和 assistant 内容"""
    user_content = ""
    assistant_content = ""
    for m in single_turn_messages:
        if m["role"] == "user":
            user_content = m["content"]
        elif m["role"] == "assistant":
            assistant_content = m["content"]
    return user_content, assistant_content


def make_followup_question(original_question):
    """用模板包装原始问题成自然追问"""
    template = random.choice(FOLLOWUP_TEMPLATES)
    q = original_question.rstrip("。？！")
    return template.format(q=q)


def make_third_turn_question(original_question):
    """用模板包装第三轮追问"""
    template = random.choice(THIRD_TURN_TEMPLATES)
    q = original_question.rstrip("。？！")
    return template.format(q=q)


def make_bridge_response(original_response):
    """在原始回复前加承接句"""
    bridge = random.choice(BRIDGE_TEMPLATES)
    # 如果原始回复以 <thinking> 开头，把承接句放在 thinking 之前还是之后？
    # 保持 thinking 标签不变，在正文前加承接句
    if original_response.startswith("<thinking>"):
        # 找到 </thinking> 后面的正文
        close_idx = original_response.find("</thinking>")
        if close_idx != -1:
            think_part = original_response[:close_idx + len("</thinking>")]
            content_part = original_response[close_idx + len("</thinking>"):].lstrip("\n")
            if content_part:
                return f"{think_part}\n\n{bridge} {content_part}"
            else:
                return f"{think_part}\n\n{bridge}"
    # 没有 thinking 标签
    return f"{bridge} {original_response}"


def generate_multiturn(groups, num_2turn=2000, num_3turn=1000, seed=42):
    """生成多轮数据"""
    random.seed(seed)
    results = []
    used_combos = set()
    categories = list(groups.keys())

    def pick_different(pool, exclude_indices):
        """从池中选一个不同于已选的"""
        candidates = [i for i in range(len(pool)) if i not in exclude_indices]
        if not candidates:
            return None
        return random.choice(candidates)

    # ── 生成两轮对话 ──
    count_2 = 0
    attempts = 0
    while count_2 < num_2turn and attempts < num_2turn * 5:
        attempts += 1
        cat = random.choice(categories)
        pool = groups[cat]
        if len(pool) < 2:
            continue

        idx_a = random.randint(0, len(pool) - 1)
        idx_b = pick_different(pool, {idx_a})
        if idx_b is None:
            continue

        combo = (cat, idx_a, idx_b)
        if combo in used_combos:
            continue
        used_combos.add(combo)

        user_a, asst_a = extract_turn(pool[idx_a]["messages"])
        user_b, asst_b = extract_turn(pool[idx_b]["messages"])

        if not user_a or not asst_a or not user_b or not asst_b:
            continue

        # 第2轮 user: 追问模板
        followup_q = make_followup_question(user_b)
        # 第2轮 assistant: 承接句 + 原始回复
        bridge_response = make_bridge_response(asst_b)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_a},
            {"role": "assistant", "content": asst_a},
            {"role": "user", "content": followup_q},
            {"role": "assistant", "content": bridge_response},
        ]

        results.append({
            "messages": messages,
            "metadata": {
                "category": cat,
                "turns": 2,
                "source": "multiturn_stitched",
            },
        })
        count_2 += 1

    # ── 生成三轮对话 ──
    count_3 = 0
    attempts = 0
    while count_3 < num_3turn and attempts < num_3turn * 5:
        attempts += 1
        cat = random.choice(categories)
        pool = groups[cat]
        if len(pool) < 3:
            continue

        idx_a = random.randint(0, len(pool) - 1)
        idx_b = pick_different(pool, {idx_a})
        if idx_b is None:
            continue
        idx_c = pick_different(pool, {idx_a, idx_b})
        if idx_c is None:
            continue

        combo = (cat, idx_a, idx_b, idx_c)
        if combo in used_combos:
            continue
        used_combos.add(combo)

        user_a, asst_a = extract_turn(pool[idx_a]["messages"])
        user_b, asst_b = extract_turn(pool[idx_b]["messages"])
        user_c, asst_c = extract_turn(pool[idx_c]["messages"])

        if not user_a or not asst_a or not user_b or not asst_b or not user_c or not asst_c:
            continue

        followup_q2 = make_followup_question(user_b)
        bridge_resp2 = make_bridge_response(asst_b)
        followup_q3 = make_third_turn_question(user_c)
        bridge_resp3 = make_bridge_response(asst_c)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_a},
            {"role": "assistant", "content": asst_a},
            {"role": "user", "content": followup_q2},
            {"role": "assistant", "content": bridge_resp2},
            {"role": "user", "content": followup_q3},
            {"role": "assistant", "content": bridge_resp3},
        ]

        results.append({
            "messages": messages,
            "metadata": {
                "category": cat,
                "turns": 3,
                "source": "multiturn_stitched",
            },
        })
        count_3 += 1

    return results


def step_generate():
    """生成多轮数据"""
    print("加载单轮 SFT 数据...")
    data = load_single_turn_data(INPUT_FILE)
    print(f"  加载 {len(data)} 条")

    groups = group_by_category(data)
    for cat, items in groups.items():
        print(f"  {cat}: {len(items)} 条")

    print("\n生成多轮数据...")
    results = generate_multiturn(groups)

    # 按类别统计
    cat_counts = {}
    turn_counts = {2: 0, 3: 0}
    for r in results:
        cat = r["metadata"]["category"]
        turns = r["metadata"]["turns"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        turn_counts[turns] = turn_counts.get(turns, 0) + 1

    print(f"\n生成完成: {len(results)} 条多轮数据")
    print(f"  两轮: {turn_counts.get(2, 0)} 条")
    print(f"  三轮: {turn_counts.get(3, 0)} 条")
    print("  按类别:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    # 写入文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已保存到 {OUTPUT_FILE}")


def step_stats():
    """查看统计"""
    if not os.path.exists(OUTPUT_FILE):
        print(f"多轮数据文件不存在: {OUTPUT_FILE}")
        print("请先运行: python scripts/generate_multiturn_sft.py --step generate")
        return

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cat_counts = {}
    turn_counts = {}
    for line in lines:
        d = json.loads(line)
        meta = d.get("metadata", {})
        cat = meta.get("category", "?")
        turns = meta.get("turns", 0)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        turn_counts[turns] = turn_counts.get(turns, 0) + 1

    print(f"多轮数据统计: {len(lines)} 条")
    print(f"  两轮: {turn_counts.get(2, 0)} 条")
    print(f"  三轮: {turn_counts.get(3, 0)} 条")
    print("  按类别:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    # 合并后统计
    if os.path.exists(INPUT_FILE):
        single_count = sum(1 for _ in open(INPUT_FILE))
        total = single_count + len(lines)
        print(f"\n合并后总计: {single_count} 单轮 + {len(lines)} 多轮 = {total} 条")


def step_sample():
    """抽查数据"""
    if not os.path.exists(OUTPUT_FILE):
        print(f"多轮数据文件不存在: {OUTPUT_FILE}")
        return

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    samples = random.sample(lines, min(3, len(lines)))
    for i, line in enumerate(samples):
        d = json.loads(line)
        meta = d.get("metadata", {})
        print(f"\n{'='*60}")
        print(f"样本 {i+1}: {meta.get('category', '?')} | {meta.get('turns', '?')} 轮")
        print(f"{'='*60}")
        for m in d["messages"]:
            role = m["role"].upper()
            content = m["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"\n[{role}]")
            print(content)


def step_merge():
    """合并单轮 + 多轮数据"""
    if not os.path.exists(OUTPUT_FILE):
        print(f"多轮数据文件不存在: {OUTPUT_FILE}")
        print("请先运行: python scripts/generate_multiturn_sft.py --step generate")
        return

    single_count = 0
    multi_count = 0

    with open(MERGED_FILE, "w", encoding="utf-8") as out:
        # 单轮
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                out.write(line)
                single_count += 1
        # 多轮
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                out.write(line)
                multi_count += 1

    print(f"合并完成: {single_count} 单轮 + {multi_count} 多轮 = {single_count + multi_count} 条")
    print(f"保存到: {MERGED_FILE}")


def main():
    parser = argparse.ArgumentParser(description="多轮 SFT 数据生成器")
    parser.add_argument("--step", choices=["generate", "stats", "sample", "merge"],
                        required=True, help="执行步骤")
    parser.add_argument("--num_2turn", type=int, default=2000, help="两轮对话数量")
    parser.add_argument("--num_3turn", type=int, default=1000, help="三轮对话数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    if args.step == "generate":
        step_generate()
    elif args.step == "stats":
        step_stats()
    elif args.step == "sample":
        step_sample()
    elif args.step == "merge":
        step_merge()


if __name__ == "__main__":
    main()
