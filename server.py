#!/usr/bin/env python3
"""
DeepSleep Server - 睡眠健康AI对话后端服务
支持 DeepSleep 和 Qwen 两种架构，4个模型切换，SSE流式输出
工业级部署: torch.compile FP8 加速 + 模型预热 + CORS + 结构化日志
"""

import argparse
import asyncio
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deepsleep")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是星辰曦（小曦），由L-0915开发的睡眠健康AI助手。"
    "你温暖、有趣、专业。回答用户的睡眠健康问题，提供专业但易懂的建议。"
)

THINKING_SUFFIX = (
    "在回答之前，请先使用<thinking></thinking>标签进行深入思考和分析。"
)

# 兼容 DeepSleep <thinking> 和 Qwen <think > 标签
THINK_OPEN_RE = re.compile(r'<(?:thinking|think)\s*>')
THINK_CLOSE_RE = re.compile(r'</(?:thinking|think)\s*>')

# Strip special tokens from decoded text, but keep <thinking> and </thinking>
_NON_THINKING_SPECIAL = re.compile(
    r'<\|im_start\|>|<\|im_end\|>|<s>|</s>|<pad>|<unk>|<summary>'
)

def _strip_non_thinking_special(text):
    return _NON_THINKING_SPECIAL.sub('', text)

MAX_CONTEXT = {"deepsleep": 1800, "qwen": 32000, "qwen_mt": 32000}

MODEL_CONFIGS: Dict[str, dict] = {
    "ds_b0.1": {
        "path": "/root/dslm/deepsleep/out/ds_b0.1_hf/",
        "arch": "deepsleep",
        "name": "DeepSleep β=0.1",
    },
    "ds_b0.5": {
        "path": "/root/dslm/deepsleep/out/ds_b0.5_hf/",
        "arch": "deepsleep",
        "name": "DeepSleep β=0.5",
    },
    "qwen_b0.1": {
        "path": "/root/blockdata/dpo_exp/qwen_b0.1_s42/final_model/",
        "arch": "qwen",
        "name": "Qwen β=0.1",
    },
    "qwen_b0.5": {
        "path": "/root/blockdata/dpo_exp/qwen_b0.5_s42/final_model/",
        "arch": "qwen",
        "name": "Qwen β=0.5",
    },
    "qwen_mt_b0.1": {
        "path": "out/sft_qwen_multiturn_b0.1/final_model/",
        "arch": "qwen",
        "name": "Qwen多轮 β=0.1",
    },
    "qwen_mt_b0.5": {
        "path": "out/sft_qwen_multiturn_b0.5/final_model/",
        "arch": "qwen",
        "name": "Qwen多轮 β=0.5",
    },
}

# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class ModelManager:
    """管理 4 个模型的惰性加载、FP8 加速、线程安全推理"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, tuple] = {}
        self._load_locks = {mid: asyncio.Lock() for mid in MODEL_CONFIGS}
        self._infer_locks = {mid: asyncio.Lock() for mid in MODEL_CONFIGS}
        self._compile_enabled = False

    def _load_model(self, model_id: str):
        """加载模型 + FP8 torch.compile 加速"""
        cfg = MODEL_CONFIGS[model_id]
        path = cfg["path"]
        arch = cfg["arch"]
        logger.info(f"加载模型 {cfg['name']} ({model_id}) from {path}")

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.to(self.device)
        model.eval()

        # torch.compile 可选: --compile 参数启用
        # 注意: 自定义 MoE 模型首次编译耗时较长 (数分钟)
        if self._compile_enabled:
            try:
                model = torch.compile(model, mode="default")
                logger.info(f"torch.compile 启用成功 (mode=default)")
            except Exception as e:
                logger.warning(f"torch.compile 启用失败，使用原始 FP16: {e}")

        self.models[model_id] = (model, tokenizer, arch)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"{cfg['name']} 加载完成: {n_params:.1f}M params, device={self.device}")

    def _warmup(self, model_id: str):
        """模型预热: 跑一次推理，触发 CUDA kernel 编译和 torch.compile 缓存"""
        model, tokenizer, arch = self.models[model_id]
        logger.info(f"预热模型 {model_id} ...")
        t0 = time.time()
        dummy = tokenizer.encode("你好", return_tensors="pt").to(self.device)
        with torch.no_grad():
            if arch == "deepsleep":
                # DeepSleep 手动生成
                past_kv = None
                cur = dummy
                for _ in range(8):
                    outputs = model(input_ids=cur if past_kv is None else cur[:, -1:],
                                    use_cache=True, past_key_values=past_kv)
                    past_kv = outputs.past_key_values
                    next_t = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    cur = torch.cat([cur, next_t], dim=-1)
            else:
                model.generate(dummy, max_new_tokens=8, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        logger.info(f"预热完成: {model_id}, 耗时 {elapsed:.1f}s")

    async def ensure_loaded(self, model_id: str):
        if model_id not in MODEL_CONFIGS:
            raise ValueError(f"未知模型: {model_id}")
        if model_id in self.models:
            return
        async with self._load_locks[model_id]:
            if model_id in self.models:
                return
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model, model_id)
            await loop.run_in_executor(None, self._warmup, model_id)

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self.models

    # ---- 对话截断 ----

    def _truncate_messages(self, messages: List[dict], tokenizer, max_context: int) -> List[dict]:
        system_msg = None
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m
            else:
                chat_msgs.append(m)

        while chat_msgs:
            test_msgs = ([system_msg] if system_msg else []) + chat_msgs
            text = self._build_plain_text(test_msgs, tokenizer)
            if len(tokenizer.encode(text)) <= max_context:
                break
            chat_msgs = chat_msgs[2:] if len(chat_msgs) >= 2 else chat_msgs[1:]

        return ([system_msg] if system_msg else []) + chat_msgs

    @staticmethod
    def _build_plain_text(messages: List[dict], tokenizer) -> str:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            return "".join(parts)

    # ---- 流式生成入口 ----

    async def generate_stream(self, model_id, messages, thinking=True,
                              temperature=0.7, top_p=0.9, max_tokens=512):
        try:
            await self.ensure_loaded(model_id)
        except Exception as e:
            yield {"type": "error", "content": f"模型加载失败: {e}"}
            return

        model, tokenizer, arch = self.models[model_id]
        max_ctx = MAX_CONTEXT.get(arch, 2048)

        async with self._infer_locks[model_id]:
            if arch == "deepsleep":
                async for chunk in self._generate_deepsleep(
                    model, tokenizer, messages, thinking, temperature, top_p, max_tokens, max_ctx):
                    yield chunk
            else:
                async for chunk in self._generate_qwen(
                    model, tokenizer, messages, thinking, temperature, top_p, max_tokens, max_ctx):
                    yield chunk

    # ---- DeepSleep 推理 (手动逐 token + KV cache) ----

    async def _generate_deepsleep(self, model, tokenizer, messages,
                                  thinking, temperature, top_p, max_tokens, max_ctx):
        system_content = SYSTEM_PROMPT + (THINKING_SUFFIX if thinking else "")
        processed = self._prepare_messages(messages, system_content)
        processed = self._truncate_messages(processed, tokenizer, max_ctx)

        prompt = self._build_chatml(processed)
        enc = tokenizer(prompt, add_special_tokens=False)
        ids = enc["input_ids"]
        bos = tokenizer.bos_token_id
        if bos is not None:
            ids = [bos] + ids
        input_ids = torch.tensor([ids], device=self.device)
        prompt_tokens = input_ids.shape[1]

        # 在线程中逐 token 生成，通过 queue 传递
        import queue as qmod
        q: qmod.Queue = qmod.Queue()
        error_holder = [None]
        total_tokens = [0]

        def _gen():
            try:
                generated = []
                past_kv = None
                cur = input_ids
                eos_id = tokenizer.eos_token_id
                # Also stop at <|im_end|> (ChatML end-of-turn marker)
                im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
                stop_ids = {eos_id} if eos_id is not None else set()
                if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
                    stop_ids.add(im_end_id)

                with torch.no_grad():
                    for step in range(max_tokens):
                        out = model(
                            input_ids=(cur if past_kv is None else cur[:, -1:]),
                            use_cache=True,
                            past_key_values=past_kv,
                        )
                        logits = out.logits[:, -1, :]
                        past_kv = out.past_key_values

                        # Repetition penalty: penalize previously generated tokens
                        if generated:
                            rep_penalty = 1.3
                            for tid in set(generated[-64:]):
                                logits[0, tid] /= rep_penalty

                        # 采样
                        logits = logits / max(temperature, 1e-8)
                        if top_p < 1.0:
                            srt, idx = torch.sort(logits, descending=True)
                            cum = torch.cumsum(torch.softmax(srt, dim=-1), dim=-1)
                            mask = cum > top_p
                            mask[..., 1:] = mask[..., :-1].clone()
                            mask[..., 0] = False
                            srt[mask] = float("-inf")
                            logits = srt.scatter(1, idx, srt)

                        nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

                        if stop_ids and nxt.item() in stop_ids:
                            q.put(None)
                            return

                        generated.append(nxt.item())
                        cur = torch.cat([cur, nxt], dim=-1)

                        # 每 2 个 token decode 一次（平衡延迟和多字节字符安全）
                        if len(generated) % 2 == 0 or step < 4:
                            text = tokenizer.decode(generated, skip_special_tokens=False)
                            text = _strip_non_thinking_special(text)
                            q.put(text)

                if generated and len(generated) % 2 != 0:
                    text = tokenizer.decode(generated, skip_special_tokens=False)
                    text = _strip_non_thinking_special(text)
                    q.put(text)
                q.put(None)
                total_tokens[0] = len(generated)
            except Exception as e:
                error_holder[0] = str(e)
                q.put(None)

        t = threading.Thread(target=_gen, daemon=True)
        t.start()

        # 主线程: 从 queue 读取，解析 thinking/content，逐字符推送
        last_text = ""
        phase = "content"
        sent_offset = 0

        try:
            while True:
                item = await asyncio.get_event_loop().run_in_executor(None, q.get)
                if item is None:
                    break
                if error_holder[0]:
                    yield {"type": "error", "content": error_holder[0]}
                    return

                full = item
                if len(full) <= len(last_text):
                    continue
                last_text = full

                events, sent_offset, phase = self._process_thinking(full, sent_offset, phase)
                for evt in events:
                    if "_phase" in evt:
                        del evt["_phase"]
                    yield evt

        finally:
            t.join(timeout=5)

        # Flush remaining buffered text (no safety margin needed at end)
        flush_evts, phase = self._flush_thinking(last_text, sent_offset, phase)
        for evt in flush_evts:
            if "_phase" in evt:
                del evt["_phase"]
            yield evt

        yield {"type": "done", "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_tokens[0],
        }, "max_context": max_ctx}

    # ---- Qwen 推理 (model.generate + TextIteratorStreamer) ----

    async def _generate_qwen(self, model, tokenizer, messages,
                             thinking, temperature, top_p, max_tokens, max_ctx):
        system_content = SYSTEM_PROMPT + (THINKING_SUFFIX if thinking else "")
        processed = self._prepare_messages(messages, system_content)
        processed = self._truncate_messages(processed, tokenizer, max_ctx)

        try:
            prompt = tokenizer.apply_chat_template(processed, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = self._build_chatml(processed)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kw = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 1e-8),
            "top_p": top_p,
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        def _gen():
            with torch.no_grad():
                model.generate(**gen_kw)

        def _safe_next(s):
            try:
                return next(s)
            except StopIteration:
                return None

        t = threading.Thread(target=_gen, daemon=True)
        t.start()

        phase = "content"
        full_text = ""
        sent_offset = 0

        try:
            loop = asyncio.get_event_loop()
            while True:
                chunk = await loop.run_in_executor(None, _safe_next, streamer)
                if chunk is None:
                    break
                if not chunk:
                    continue
                full_text += chunk

                events, sent_offset, phase = self._process_thinking(full_text, sent_offset, phase)
                for evt in events:
                    if "_phase" in evt:
                        del evt["_phase"]
                    yield evt

        except Exception as e:
            yield {"type": "error", "content": f"生成失败: {e}"}
            return
        finally:
            t.join(timeout=5)

        # Flush remaining buffered text
        flush_evts, phase = self._flush_thinking(full_text, sent_offset, phase)
        for evt in flush_evts:
            if "_phase" in evt:
                del evt["_phase"]
            yield evt

        yield {"type": "done", "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokenizer.encode(full_text)),
        }, "max_context": max_ctx}

    # ---- 共用工具 ----

    @staticmethod
    def _prepare_messages(messages, system_content):
        processed = []
        has_system = False
        for m in messages:
            if m["role"] == "system":
                processed.append({"role": "system", "content": system_content})
                has_system = True
            else:
                processed.append(m)
        if not has_system:
            processed.insert(0, {"role": "system", "content": system_content})
        return processed

    @staticmethod
    def _build_chatml(messages):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @staticmethod
    def _parse_tags(text):
        """Find all thinking open/close tags in text, return sorted list."""
        tags = []
        for m in THINK_OPEN_RE.finditer(text):
            tags.append((m.start(), m.end(), "thinking"))
        for m in THINK_CLOSE_RE.finditer(text):
            tags.append((m.start(), m.end(), "content"))
        tags.sort()
        return tags

    @staticmethod
    def _yield_chunked(text, phase):
        """Yield text as 2-char events with correct type and phase."""
        CHUNK = 2
        events = []
        for i in range(0, len(text), CHUNK):
            evt = {"type": phase, "content": text[i:i + CHUNK]}
            if phase == "thinking":
                evt["_phase"] = "thinking"
            events.append(evt)
        return events

    @staticmethod
    def _process_thinking(full_text, offset, phase):
        """Parse <thinking>/<think tags in full_text[offset:], return (events, new_offset, new_phase)."""
        unprocessed = full_text[offset:]
        if not unprocessed:
            return [], offset, phase

        # Find tags in the FULL unprocessed text (not truncated)
        tags = ModelManager._parse_tags(unprocessed)

        if tags:
            # Process up to and including the last complete tag
            process_end = tags[-1][1]
        else:
            # No complete tag — check for a potential partial tag at the end
            last_lt = unprocessed.rfind('<')
            if (last_lt >= 0 and last_lt >= len(unprocessed) - 12
                    and (unprocessed[last_lt:].startswith('<t')
                         or unprocessed[last_lt:].startswith('</t')
                         or unprocessed[last_lt:] in ('<', '</'))):
                process_end = last_lt  # hold back the potential partial tag
            else:
                process_end = len(unprocessed)  # safe to process everything
            tags = []

        scan_text = unprocessed[:process_end]
        tags_in_range = [(s, e, p) for s, e, p in tags if e <= process_end]

        events = []
        pos = 0
        cur_phase = phase

        for tag_start, tag_end, tag_phase in tags_in_range:
            events.extend(ModelManager._yield_chunked(scan_text[pos:tag_start], cur_phase))
            cur_phase = tag_phase
            pos = tag_end

        events.extend(ModelManager._yield_chunked(scan_text[pos:], cur_phase))
        return events, offset + process_end, cur_phase

    @staticmethod
    def _flush_thinking(full_text, offset, phase):
        """Flush remaining text at end of generation, parsing any remaining tags."""
        unprocessed = full_text[offset:]
        if not unprocessed:
            return [], phase

        tags = ModelManager._parse_tags(unprocessed)
        events = []
        pos = 0
        cur_phase = phase

        for tag_start, tag_end, tag_phase in tags:
            events.extend(ModelManager._yield_chunked(unprocessed[pos:tag_start], cur_phase))
            cur_phase = tag_phase
            pos = tag_end

        events.extend(ModelManager._yield_chunked(unprocessed[pos:], cur_phase))
        return events, cur_phase


# ---------------------------------------------------------------------------
# 全局实例
# ---------------------------------------------------------------------------

manager = ModelManager()

# ---------------------------------------------------------------------------
# FastAPI 应用 + 生产配置
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动时打印信息"""
    logger.info("=" * 50)
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        logger.info(f"显存: {total / 1024**3:.1f} GB (可用 {free / 1024**3:.1f} GB)")
    else:
        logger.warning("未检测到 GPU")
    logger.info(f"可用模型: {list(MODEL_CONFIGS.keys())}")
    logger.info("=" * 50)
    yield


app = FastAPI(title="DeepSleep Server", version="1.0.0", lifespan=lifespan)

# CORS: 公网部署必须
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    messages: List[dict] = Field(..., description="对话历史")
    model: str = Field("ds_b0.1", description="模型 ID")
    thinking: bool = Field(True, description="是否启用思考模式")
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    max_tokens: int = Field(512, ge=1, le=2048)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    gpu = "N/A"
    vram_used = vram_total = 0.0
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        vram_used = (total - free) / (1024 ** 3)
        vram_total = total / (1024 ** 3)
    return {
        "status": "ok",
        "gpu": gpu,
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
        "loaded_models": list(manager.models.keys()),
    }


@app.get("/api/models")
async def list_models():
    return {"models": [
        {"id": mid, "name": cfg["name"], "arch": cfg["arch"], "loaded": manager.is_loaded(mid),
         "max_context": MAX_CONTEXT.get(cfg["arch"], 2048)}
        for mid, cfg in MODEL_CONFIGS.items()
    ]}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.model not in MODEL_CONFIGS:
        return JSONResponse(400, content={"error": f"未知模型: {request.model}"})
    if not request.messages:
        return JSONResponse(400, content={"error": "messages 不能为空"})

    async def _stream():
        try:
            async for chunk in manager.generate_stream(
                model_id=request.model,
                messages=request.messages,
                thinking=request.thinking,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            ):
                yield {"event": chunk["type"], "data": json.dumps(chunk, ensure_ascii=False)}
        except Exception as e:
            logger.error(f"生成异常: {e}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"type": "error", "content": str(e)})}

    return EventSourceResponse(_stream())


# ---------------------------------------------------------------------------
# 静态文件
# ---------------------------------------------------------------------------

_static_dir = None
for _candidate in [
    os.path.join(os.path.dirname(__file__), "web", "dist"),
    os.path.join(os.path.dirname(__file__), "static"),
]:
    if os.path.isdir(_candidate) and os.listdir(_candidate):
        _static_dir = _candidate
        break


@app.get("/")
async def root():
    if _static_dir:
        return RedirectResponse(url="/index.html")
    return {"message": "DeepSleep Server running. No frontend found."}


if _static_dir:
    # no-cache: ensure browsers always revalidate HTML/JS (avoids stale model lists)
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DeepSleep Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--preload", action="store_true", help="启动时预加载所有模型")
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile 加速 (首次编译较慢)")
    args = parser.parse_args()

    import uvicorn

    manager._compile_enabled = args.compile

    if args.preload:
        logger.info("预加载所有模型 ...")
        for mid in MODEL_CONFIGS:
            manager._load_model(mid)
            manager._warmup(mid)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
