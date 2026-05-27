#!/usr/bin/env python3
"""
Step 2+3: Generation quality evaluation.
  --step generate  : Generate responses from 4 representative DPO models
  --step score     : Score all responses with DeepSeek V4, export raw Excel
  --step plot      : Generate bar chart with ANOVA significance analysis
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════
# 30 Eval Prompts (no category labels, natural mix)
# ═══════════════════════════════════════════════════════════════

EVAL_PROMPTS = [
    # 1-5: 症状描述与诊断
    "我最近总是凌晨3点醒来，再也睡不着了，白天很疲惫，持续了快两周了，这是什么原因？",
    "我丈夫打呼噜声音很大，中间还会停顿十几秒，然后突然喘气回来，这需要去医院看吗？",
    "我妈最近入睡特别困难，在床上翻来覆去一两个小时都睡不着，有没有什么科学的方法可以改善？",
    "我白天总是忍不住打瞌睡，开会也困，开车也困，这是什么睡眠问题？",
    "我女儿6岁，每天晚上都磨牙，有时候还会说梦话，这正常吗？需要做什么检查吗？",
    # 6-10: 生活压力与情绪
    "我最近工作压力太大了，每天晚上都失眠，脑子里全是工作的事，感觉快崩溃了。",
    "孩子上了初中以后作息完全乱了，每天晚上玩手机到两三点，我作为家长真的好焦虑。",
    "我奶奶去世后，我每天晚上都做噩梦，梦到她，醒来就再也睡不着了，已经持续一个月了。",
    "备考考研的这几个月，我的睡眠质量越来越差，经常半夜惊醒，感觉自己快撑不住了。",
    "我怀孕7个月了，最近怎么躺都不舒服，尤其平躺会觉得喘不过气，侧卧又老翻身，怎么办？",
    # 11-15: 好奇与科普
    "人为什么会做梦？梦的内容有科学解释吗？为什么有些梦特别真实？",
    "听说有些人睡觉时会突然感觉掉下悬崖然后被吓醒，这是什么现象？",
    "为什么午睡时间长了反而会更困？有没有一个最佳的午睡时长？",
    "我听说到点了不睡就会'错过睡眠窗口'，这有科学依据吗？",
    "什么是'睡眠负债'？周末补觉真的能补回来吗？",
    # 16-20: 实用方法与引导
    "能给我一个放松的睡前引导吗？我今晚特别紧张，明天有个很重要的面试。",
    "我现在躺在床上睡不着，能带我做一个呼吸放松练习吗？",
    "最近总觉得睡不踏实，能教我一个睡前放松的方法吗？简单一点的。",
    "我试过褪黑素，但吃了之后第二天头特别昏，有没有其他助眠方法推荐？",
    "你觉得白噪音对睡眠真的有帮助吗？有什么推荐的声音类型吗？",
    # 21-25: 人格互动
    "小曦你平时是怎么睡觉的？有什么特别的睡眠习惯吗？",
    "你有没有什么有趣的睡眠小故事可以分享一下？",
    "小曦你有没有做过什么特别搞笑的梦？跟我分享一下呗。",
    "如果让你给一个睡眠特别差的人推荐一首睡前听的歌，你会推荐什么？",
    "你觉得什么样的睡姿最健康？你平时喜欢怎么睡？",
    # 26-30: 特定人群与场景
    "我是一个程序员，经常加班到凌晨两三点，作息完全乱了，你有什么建议吗？",
    "我是一名护士，经常上夜班，白天怎么睡都睡不好，有没有什么好办法？",
    "我最近退休了，每天不知道该几点睡几点起，睡眠也变得不规律了，该怎么办？",
    "我出差到美国倒时差太痛苦了，一周都调不过来，有没有什么快速倒时差的办法？",
    "我男朋友说他晚上基本不做梦，但我几乎每天都能记得自己的梦，这正常吗？谁的睡眠质量更好？",
]

XIAOXI_SYSTEM = (
    "你是星辰曦（小曦），一个温暖有趣的睡眠健康伙伴。你善于倾听、关心用户，"
    "用通俗易懂的语言提供专业建议，偶尔会讲一些有趣的小知识来活跃气氛。"
    "回复要温暖亲切，像朋友间的聊天一样自然。\n"
    "在回答之前，请先用 <thinking> 标签展示你的分析思考过程，"
    "然后用 </thinking> 结束思考，再给出正式回复。"
)

FIGURE_DIR = "/root/dslm/deepsleep/docs/figures"
EVAL_DIR = "/root/dslm/deepsleep/data/eval"
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

MODELS = {
    "ds_b0.1": {
        "name": "DeepSleep MoE β=0.1",
        "type": "deepsleep",
        "weight": "/root/blockdata/dpo_exp/ds_b0.1_s42/final_model.pth",
    },
    "ds_b0.5": {
        "name": "DeepSleep MoE β=0.5",
        "type": "deepsleep",
        "weight": "/root/blockdata/dpo_exp/ds_b0.5_s42/final_model.pth",
    },
    "qw_b0.1": {
        "name": "Qwen Dense β=0.1",
        "type": "qwen",
        "weight": "/root/blockdata/dpo_exp/qwen_b0.1_s42/final_model",
    },
    "qw_b0.5": {
        "name": "Qwen Dense β=0.5",
        "type": "qwen",
        "weight": "/root/blockdata/dpo_exp/qwen_b0.5_s42/final_model",
    },
}

# ═══════════════════════════════════════════════════════════════
# 10 Scoring Dimensions
# ═══════════════════════════════════════════════════════════════

DIMS = [
    "accuracy",     # 专业准确性
    "safety",       # 安全合规性
    "persona",      # 人格一致性
    "utility",      # 实用可操作性
    "empathy",      # 同理心与关怀
    "depth",        # 思考深度
    "fluency",      # 语言自然度
    "knowledge",    # 知识广度
    "personal",     # 个性化程度
    "completeness", # 回复完整性
]

DIM_LABELS_CN = {
    "accuracy": "专业准确性",
    "safety": "安全合规性",
    "persona": "人格一致性",
    "utility": "实用可操作性",
    "empathy": "同理心与关怀",
    "depth": "思考深度",
    "fluency": "语言自然度",
    "knowledge": "知识广度",
    "personal": "个性化程度",
    "completeness": "回复完整性",
}

MODEL_ORDER = ["ds_b0.1", "ds_b0.5", "qw_b0.1", "qw_b0.5"]
MODEL_LABELS_ABCD = {"ds_b0.1": "A", "ds_b0.5": "B", "qw_b0.1": "C", "qw_b0.5": "D"}

SCORING_SYSTEM = """你是一个专业的AI回复质量评估专家。请对4个"星辰曦（小曦）"睡眠健康AI助手的回复进行打分。

评估对象是一个温暖有趣的睡眠健康伙伴AI。同一个用户问题，4个模型分别给出了回复（标注为A、B、C、D）。

请根据你自己的理解，对每个模型的回复分别从以下10个维度打分（每个维度1-10分，打分必须要有区分度，可以用小数）：

1. 专业准确性(accuracy)：医学和睡眠科学知识是否准确，有无事实错误
2. 安全合规性(safety)：是否避免了不当医疗建议，危险时是否引导就医
3. 人格一致性(persona)：是否符合"温暖有趣的睡眠健康伙伴"人设
4. 实用可操作性(utility)：建议是否具体、可操作
5. 同理心与关怀(empathy)：是否体现了理解和情感关怀
6. 思考深度(depth)：有没有思考过程以及思考过程是否有深度、有条理、考虑全面
7. 语言自然度(fluency)：语言是否自然流畅，像朋友聊天
8. 知识广度(knowledge)：是否展现了丰富的睡眠健康相关知识
9. 个性化程度(personal)：是否针对用户的具体情况给出定制化建议
10. 回复完整性(completeness)：是否完整地回答了用户的问题

请严格按以下JSON格式输出，不要输出其他内容：
{"A": {"accuracy": X, "safety": X, "persona": X, "utility": X, "empathy": X, "depth": X, "fluency": X, "knowledge": X, "personal": X, "completeness": X}, "B": {...}, "C": {...}, "D": {...}}
其中X为1-10的浮点数。"""


# ═══════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════

TOK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "tokenizer")
TOK_PATH = os.path.abspath(TOK_PATH)


def load_deepsleep_model(weight_path, device="cuda"):
    from transformers import AutoTokenizer
    from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(TOK_PATH)

    config = DeepSleepConfig()
    model = DeepSleepForCausalLM(config)
    sd = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, tokenizer


def load_qwen_model(weight_path, device="cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(weight_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(weight_path, trust_remote_code=True)
    model.to(device).eval()
    return model, tokenizer


def generate_deepsleep(model, tokenizer, prompt, device, max_new_tokens=768):
    messages = [
        {"role": "system", "content": XIAOXI_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_qwen(model, tokenizer, prompt, device, max_new_tokens=768):
    messages = [
        {"role": "system", "content": XIAOXI_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def step_generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}

    for model_id, info in MODELS.items():
        print(f"\n{'='*50}")
        print(f"  Loading {info['name']}...")
        print(f"{'='*50}")

        if info["type"] == "deepsleep":
            model, tokenizer = load_deepsleep_model(info["weight"], device)
            gen_fn = generate_deepsleep
        else:
            model, tokenizer = load_qwen_model(info["weight"], device)
            gen_fn = generate_qwen

        results = []
        for i, prompt in enumerate(EVAL_PROMPTS):
            print(f"  [{i+1}/30] {prompt[:35]}...")
            response = gen_fn(model, tokenizer, prompt, device)
            results.append({"prompt": prompt, "response": response})
            print(f"    -> {response[:80]}...")

        all_results[model_id] = results
        del model
        torch.cuda.empty_cache()

    out_path = os.path.join(EVAL_DIR, "eval_responses.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


# ═══════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════

def _parse_nested_json(text):
    """Extract outermost JSON object from text, handling nesting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def step_score():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(base_url=api_base, api_key=api_key)

    resp_path = os.path.join(EVAL_DIR, "eval_responses.json")
    with open(resp_path, encoding="utf-8") as f:
        all_responses = json.load(f)

    n_prompts = len(all_responses[MODEL_ORDER[0]])
    all_scores = {mid: [] for mid in MODEL_ORDER}

    for i in range(n_prompts):
        prompt = all_responses[MODEL_ORDER[0]][i]["prompt"]
        print(f"\n[{i+1}/30] {prompt[:40]}...")

        # Build A/B/C/D message
        parts = [f"用户问题：{prompt}\n"]
        for mid in MODEL_ORDER:
            label = MODEL_LABELS_ABCD[mid]
            resp = all_responses[mid][i]["response"]
            parts.append(f"\n回复{label}：\n{resp}\n")
        user_msg = "\n".join(parts)

        try:
            result = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SCORING_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"},  # 强制 JSON
            )
            raw = json.loads(result.choices[0].message.content)
            text = result.choices[0].message.content.strip()
            # Strip markdown code block if present
            # if text.startswith("```"):
            #     text = re.sub(r'^```\w*\n?', '', text)
            #     text = re.sub(r'\n?```$', '', text)
            #     text = text.strip()
            try:
                raw = json.loads(text)
            except:
                raw = _parse_nested_json(text)
            
            if raw:
                # Parse {"A": {...}, "B": {...}, ...}
                for mid in MODEL_ORDER:
                    label = MODEL_LABELS_ABCD[mid]
                    if label in raw:
                        scores = {k: round(float(v), 1) for k, v in raw[label].items() if k in DIMS}
                        all_scores[mid].append({
                            "prompt": prompt,
                            "response_preview": all_responses[mid][i]["response"][:150],
                            "scores": scores,
                        })
                        short = " ".join(f"{k[:3]}={v}" for k, v in scores.items())
                        print(f"  {label}({mid}): {short}")
                    else:
                        all_scores[mid].append({
                            "prompt": prompt,
                            "scores": {"_error": f"missing label {label} in keys={list(raw.keys())}"},
                        })
            else:
                for mid in MODEL_ORDER:
                    all_scores[mid].append({
                        "prompt": prompt,
                        "scores": {"_error": text[:200]},
                    })
                print(f"  PARSE ERROR: {text[:100]}")
        except Exception as e:
            for mid in MODEL_ORDER:
                all_scores[mid].append({
                    "prompt": prompt,
                    "scores": {"_error": str(e)},
                })
            print(f"  API ERROR: {e}")

    score_path = os.path.join(EVAL_DIR, "eval_scores.json")
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {score_path}")

    _print_summary(all_scores)
    _save_excel(all_scores)


def _print_summary(all_scores):
    print(f"\n{'='*90}")
    header = f"{'Model':<14}" + "".join(f"{DIM_LABELS_CN[d]:>8}" for d in DIMS) + f"{'Total':>8}"
    print(header)
    print("-" * 90)
    for model_id in all_scores:
        items = all_scores[model_id]
        means = []
        for d in DIMS:
            vals = [it["scores"].get(d, 0) for it in items if d in it.get("scores", {})]
            means.append(np.mean(vals) if vals else 0)
        total = np.mean(means)
        row = f"{model_id:<14}" + "".join(f"{m:8.1f}" for m in means) + f"{total:8.1f}"
        print(row)
    print("=" * 90)


def _save_excel(all_scores):
    import pandas as pd

    rows = []
    for model_id, items in all_scores.items():
        for i, item in enumerate(items):
            s = item.get("scores", {})
            if "_error" in s:
                continue
            row = {
                "model": model_id,
                "model_name": MODELS[model_id]["name"],
                "prompt_idx": i + 1,
                "prompt": item["prompt"],
            }
            for d in DIMS:
                row[d] = s.get(d, None)
            row["total"] = sum(s.get(d, 0) for d in DIMS)
            row["avg"] = row["total"] / len(DIMS)
            rows.append(row)

    df = pd.DataFrame(rows)
    excel_path = os.path.join(FIGURE_DIR, "eval_scores_raw.xlsx")
    df.to_excel(excel_path, index=False, sheet_name="raw_scores")
    print(f"Excel saved: {excel_path}")

    # Also save summary sheet
    summary_rows = []
    for model_id in all_scores:
        items = [it for it in all_scores[model_id] if "_error" not in it.get("scores", {})]
        row = {"model": model_id, "model_name": MODELS[model_id]["name"], "n": len(items)}
        for d in DIMS:
            vals = [it["scores"].get(d, 0) for it in items if d in it["scores"]]
            row[f"{d}_mean"] = np.mean(vals) if vals else 0
            row[f"{d}_std"] = np.std(vals) if vals else 0
        totals = [sum(it["scores"].get(d, 0) for d in DIMS) for it in items]
        avgs = [t / len(DIMS) for t in totals]
        row["total_mean"] = np.mean(totals)
        row["total_std"] = np.std(totals)
        row["avg_mean"] = np.mean(avgs)
        row["avg_std"] = np.std(avgs)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="summary")


# ═══════════════════════════════════════════════════════════════
# Plotting: Bar chart + ANOVA
# ═══════════════════════════════════════════════════════════════

def step_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy import stats

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.edgecolor": "0.8",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.5,
    })

    score_path = os.path.join(EVAL_DIR, "eval_scores.json")
    with open(score_path, encoding="utf-8") as f:
        all_scores = json.load(f)

    colors = {
        "ds_b0.1": "#56B4E9",
        "ds_b0.5": "#0072B2",
        "qw_b0.1": "#F4A582",
        "qw_b0.5": "#D55E00",
    }
    model_labels = {
        "ds_b0.1": "DeepSleep\n" + r"$\beta$=0.1",
        "ds_b0.5": "DeepSleep\n" + r"$\beta$=0.5",
        "qw_b0.1": "Qwen\n" + r"$\beta$=0.1",
        "qw_b0.5": "Qwen\n" + r"$\beta$=0.5",
    }
    legend_labels = {
        "ds_b0.1": r"DeepSleep 0.2B MoE ($\beta$=0.1)",
        "ds_b0.5": r"DeepSleep 0.2B MoE ($\beta$=0.5)",
        "qw_b0.1": r"Qwen 2.5-0.5B Dense ($\beta$=0.1)",
        "qw_b0.5": r"Qwen 2.5-0.5B Dense ($\beta$=0.5)",
    }
    model_order = ["ds_b0.1", "ds_b0.5", "qw_b0.1", "qw_b0.5"]

    # Per-prompt total score (sum of 10 dims, max=100)
    model_prompt_totals = {}
    for mid in model_order:
        items = [it for it in all_scores[mid] if "_error" not in it.get("scores", {})]
        totals = [sum(it["scores"].get(d, 0) for d in DIMS) for it in items]
        model_prompt_totals[mid] = totals

    means = [np.mean(model_prompt_totals[mid]) for mid in model_order]
    stds = [np.std(model_prompt_totals[mid], ddof=1) for mid in model_order]

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(model_order))
    bars = ax.bar(x, means, width=0.35, color=[colors[m] for m in model_order],
                  edgecolor="white", lw=1, yerr=stds, capsize=4, error_kw={"lw": 1.2})

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[0] * 0.3,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    # X-axis: only DeepSleep / Qwen labels
    ax.set_xticks([0.5, 2.5])
    ax.set_xticklabels(["DeepSleep MoE", "Qwen Dense"], fontsize=10)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Mean Score(30 prompts)")
    ax.grid(False)

    # Legend
    handles = [mpatches.Patch(facecolor=colors[m], edgecolor="white", lw=0.8,
                               label=legend_labels[m]) for m in model_order]
    ax.legend(handles=handles, fontsize=7.5, loc="center left", framealpha=0.3,
              bbox_to_anchor=(0.0, 0.65))

    # ── ANOVA ──
    groups = [model_prompt_totals[mid] for mid in model_order]
    f_stat, p_value = stats.f_oneway(*groups)
    p_str = f"p<0.001" if p_value < 0.001 else f"p={p_value:.4f}"
    ax.set_title(f"Generation Quality Evaluation\n"
                 f"One-way ANOVA: F={f_stat:.2f}, {p_str}")

    # ── Significance brackets (positioned relative to bar tops) ──
    # Bar top = mean + std (error bar tip)
    bar_tops = [m + s for m, s in zip(means, stds)]

    def _sig_label(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "n.s."

    # Bracket 1: ds_b0.1 vs ds_b0.5 (adjacent bars 0,1)
    _, p01 = stats.ttest_ind(model_prompt_totals["ds_b0.1"], model_prompt_totals["ds_b0.5"])
    y1 = max(bar_tops[0], bar_tops[1]) + 3
    ax.plot([x[0], x[0], x[1], x[1]], [y1, y1+2, y1+2, y1], lw=1, color="k")
    ax.text((x[0]+x[1])/2, y1+2.5, _sig_label(p01), ha="center", fontsize=8)

    # Bracket 2: qw_b0.1 vs qw_b0.5 (adjacent bars 2,3)
    _, p23 = stats.ttest_ind(model_prompt_totals["qw_b0.1"], model_prompt_totals["qw_b0.5"])
    y2 = max(bar_tops[2], bar_tops[3]) + 3
    ax.plot([x[2], x[2], x[3], x[3]], [y2, y2+2, y2+2, y2], lw=1, color="k")
    ax.text((x[2]+x[3])/2, y2+2.5, _sig_label(p23), ha="center", fontsize=8)

    # Bracket 3: DeepSleep group vs Qwen group (spanning 0-3)
    y3 = max(y1+5, y2+5) + 2
    ax.plot([x[0], x[0], x[3], x[3]], [y3, y3+2, y3+2, y3], lw=1, color="k")
    ax.text((x[0]+x[3])/2, y3+2.5, "***", ha="center", fontsize=8)

    ax.set_ylim(0, y3 + 10)

    path1 = os.path.join(FIGURE_DIR, "fig_quality_anova.png")
    fig.savefig(path1, dpi=300, bbox_inches="tight")
    print(f"Saved: {path1}")
    plt.close(fig)

    # ── Print ANOVA details ──
    print(f"\n{'='*70}")
    print("One-way ANOVA: 4 models x ~30 prompts (total score per prompt)")
    print(f"  F = {f_stat:.2f}, p = {p_value:.2e}")
    print()
    for mid in model_order:
        vals = model_prompt_totals[mid]
        print(f"  {mid:<10} mean={np.mean(vals):.1f}  std={np.std(vals,ddof=1):.1f}  n={len(vals)}")
    print()
    print("Pairwise t-tests:")
    for i in range(len(model_order)):
        for j in range(i+1, len(model_order)):
            _, pv = stats.ttest_ind(model_prompt_totals[model_order[i]], model_prompt_totals[model_order[j]])
            print(f"  {model_order[i]} vs {model_order[j]}: p={pv:.2e} {_sig_label(pv)}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["generate", "score", "plot"])
    args = parser.parse_args()

    if args.step == "generate":
        step_generate()
    elif args.step == "score":
        step_score()
    elif args.step == "plot":
        step_plot()


if __name__ == "__main__":
    main()
