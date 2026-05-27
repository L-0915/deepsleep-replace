#!/usr/bin/env python3
"""
DeepSleep Server - 睡眠健康AI对话后端服务
支持 DeepSleep 和 Qwen 两种架构，4个模型切换，SSE流式输出
"""

import argparse
import asyncio
import json
import os
import queue
import re
import threading
from typing import AsyncGenerator, Dict, List

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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

# DeepSleep 上下文约 2048 tokens（为输入留余量），Qwen 可用更多
MAX_CONTEXT = {"deepsleep": 1800, "qwen": 32000}

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
}

# ---------------------------------------------------------------------------
# 中文 decode 修复
# ---------------------------------------------------------------------------

_CJK = r"一-鿿㐀-䶿豈-﫿"
_CJK_RE = re.compile(f"(?<=[{_CJK}])\\s+(?=[{_CJK}])")
_PUNCT_SPACE_RE = re.compile(r"\s+([，。！？；：、）】」』])")
_PRE_PUNCT_SPACE_RE = re.compile(r"([（【「『])\s+")


def clean_decode(text: str) -> str:
    """清理 tokenizer decode 后中文间的多余空格"""
    text = _CJK_RE.sub("", text)
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _PRE_PUNCT_SPACE_RE.sub(r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# ModelManager — 惰性加载、线程安全推理
# ---------------------------------------------------------------------------


class ModelManager:
    """管理 4 个模型的惰性加载和推理，每个模型一把异步锁防止并发冲突"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 已加载的模型和 tokenizer
        self.models: Dict[str, tuple] = {}  # model_id -> (model, tokenizer, arch)
        # 加载锁（防止同一模型并发加载）
        self._load_locks: Dict[str, asyncio.Lock] = {
            mid: asyncio.Lock() for mid in MODEL_CONFIGS
        }
        # 推理锁（防止同一模型并发推理）
        self._infer_locks: Dict[str, asyncio.Lock] = {
            mid: asyncio.Lock() for mid in MODEL_CONFIGS
        }

    # ---- 加载 ----

    def _load_model(self, model_id: str):
        """同步加载模型到 GPU（首次调用时触发）"""
        cfg = MODEL_CONFIGS[model_id]
        path = cfg["path"]
        arch = cfg["arch"]
        print(f"[ModelManager] 正在加载 {cfg['name']} ({model_id}) from {path} ...")

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

        self.models[model_id] = (model, tokenizer, arch)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(
            f"[ModelManager] {cfg['name']} 加载完成: {n_params:.1f}M params, "
            f"device={self.device}"
        )

    async def ensure_loaded(self, model_id: str):
        """确保模型已加载，惰性加载带锁"""
        if model_id not in MODEL_CONFIGS:
            raise ValueError(f"未知模型: {model_id}")
        if model_id in self.models:
            return
        async with self._load_locks[model_id]:
            # double-check
            if model_id in self.models:
                return
            # 在线程池中执行同步加载
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model, model_id)

    def get_model(self, model_id: str):
        """获取已加载的模型和 tokenizer，返回 (model, tokenizer, arch)"""
        return self.models[model_id]

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self.models

    # ---- 对话历史截断 ----

    def _truncate_messages(
        self, messages: List[dict], tokenizer, max_context: int
    ) -> List[dict]:
        """从最早的对话开始截断，保留 system 和最近的消息"""
        # 始终保留 system（第一条）
        system_msg = None
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m
            else:
                chat_msgs.append(m)

        # 从头开始删，直到 token 数在限制内
        while chat_msgs:
            test_msgs = ([system_msg] if system_msg else []) + chat_msgs
            text = self._build_plain_text(test_msgs, tokenizer)
            n_tokens = len(tokenizer.encode(text))
            if n_tokens <= max_context:
                break
            # 删最早的一对 user-assistant（或单条 user）
            chat_msgs = chat_msgs[2:] if len(chat_msgs) >= 2 else chat_msgs[1:]

        return ([system_msg] if system_msg else []) + chat_msgs

    @staticmethod
    def _build_plain_text(messages: List[dict], tokenizer) -> str:
        """用 tokenizer 的 chat_template（如果有）构建纯文本用于计数"""
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # fallback: 手动拼接
            parts = []
            for m in messages:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            return "".join(parts)

    # ---- 流式生成 ----

    async def generate_stream(
        self,
        model_id: str,
        messages: List[dict],
        thinking: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> AsyncGenerator[dict, None]:
        """流式生成回复，yield SSE 事件字典"""
        try:
            await self.ensure_loaded(model_id)
        except Exception as e:
            yield {"type": "error", "content": f"模型加载失败: {e}"}
            return

        model, tokenizer, arch = self.models[model_id]
        max_ctx = MAX_CONTEXT.get(arch, 2048)

        # 异步推理锁
        async with self._infer_locks[model_id]:
            if arch == "deepsleep":
                async for chunk in self._generate_deepsleep(
                    model, tokenizer, messages, thinking, temperature, top_p, max_tokens, max_ctx
                ):
                    yield chunk
            else:
                async for chunk in self._generate_qwen(
                    model, tokenizer, messages, thinking, temperature, top_p, max_tokens, max_ctx
                ):
                    yield chunk

    # ---- DeepSleep 推理 ----

    async def _generate_deepsleep(
        self, model, tokenizer, messages, thinking, temperature, top_p, max_tokens, max_ctx
    ) -> AsyncGenerator[dict, None]:
        """DeepSleep 模型推理：逐 token 流式输出"""

        # 构建 system prompt
        system_content = SYSTEM_PROMPT
        if thinking:
            system_content += THINKING_SUFFIX

        # 替换 messages 中的 system
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

        # 截断
        processed = self._truncate_messages(processed, tokenizer, max_ctx)

        # 构建 ChatML prompt
        prompt_parts = []
        for m in processed:
            prompt_parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt_text = "".join(prompt_parts)

        # tokenize
        enc = tokenizer(prompt_text, add_special_tokens=False)
        input_ids_list = enc["input_ids"]
        bos_id = tokenizer.bos_token_id
        if bos_id is not None:
            input_ids_list = [bos_id] + input_ids_list

        input_ids = torch.tensor([input_ids_list], device=self.device)
        prompt_tokens = input_ids.shape[1]

        # queue: 每个元素是一个增量文本片段 (str) 或 None (结束)
        q: queue.Queue = queue.Queue()
        error_holder: list = [None]

        def _generate_thread():
            """逐 token 生成，立即 decode 并推送全量文本（主线程算增量）"""
            try:
                generated_ids = []
                past_kv = None
                cur_ids = input_ids

                with torch.no_grad():
                    for step in range(max_tokens):
                        if past_kv is None:
                            outputs = model(input_ids=cur_ids, use_cache=True)
                        else:
                            outputs = model(input_ids=cur_ids[:, -1:], past_key_values=past_kv, use_cache=True)

                        logits = outputs.logits[:, -1, :]
                        past_kv = outputs.past_key_values

                        logits = logits / max(temperature, 1e-8)
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            mask = cumulative_probs > top_p
                            mask[..., 1:] = mask[..., :-1].clone()
                            mask[..., 0] = False
                            sorted_logits[mask] = float("-inf")
                            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                        eos_id = tokenizer.eos_token_id
                        if eos_id is not None and next_token.item() == eos_id:
                            q.put(None)
                            return

                        tid = next_token.item()
                        generated_ids.append(tid)
                        cur_ids = torch.cat([cur_ids, next_token], dim=-1)

                        # 逐 token 全量 decode，主线程通过比较算增量
                        q.put(tokenizer.decode(generated_ids, skip_special_tokens=True))

                q.put(None)
            except Exception as e:
                error_holder[0] = str(e)
                q.put(None)

        t = threading.Thread(target=_generate_thread, daemon=True)
        t.start()

        # 解析 thinking/content，逐 token 推送
        last_text = ""
        phase = "content"

        try:
            while True:
                item = await asyncio.get_event_loop().run_in_executor(None, q.get)
                if item is None:
                    break
                if error_holder[0] is not None:
                    yield {"type": "error", "content": error_holder[0]}
                    return

                full_text: str = item
                if len(full_text) <= len(last_text):
                    continue
                delta = full_text[len(last_text):]
                last_text = full_text

                if phase == "content":
                    think_idx = full_text.find("<thinking>")
                    if think_idx != -1:
                        phase = "thinking"
                        before = full_text[:think_idx].strip()
                        if before:
                            yield {"type": "content", "content": before}
                        # delta 中 thinking> 之后的部分作为 thinking 开始
                        tag_in_delta = delta.find("<thinking>")
                        if tag_in_delta != -1:
                            think_delta = delta[tag_in_delta + len("<thinking>"):]
                            if think_delta:
                                yield {"type": "thinking", "content": think_delta}
                    else:
                        yield {"type": "content", "content": delta}

                elif phase == "thinking":
                    close_idx = full_text.find("</thinking>")
                    if close_idx == -1:
                        yield {"type": "thinking", "content": delta}
                    else:
                        phase = "thinking_done"
                        # delta 中 </thinking> 之前的部分仍是 thinking
                        tag_in_delta = delta.find("</thinking>")
                        think_delta = delta[:tag_in_delta] if tag_in_delta != -1 else ""
                        if think_delta:
                            yield {"type": "thinking", "content": think_delta}
                        content_delta = delta[tag_in_delta + len("</thinking>"):] if tag_in_delta != -1 else delta
                        if content_delta:
                            yield {"type": "content", "content": content_delta}

                elif phase == "thinking_done":
                    yield {"type": "content", "content": delta}

        finally:
            t.join(timeout=5)

        completion_tokens = len(generated_ids) if generated_ids else len(tokenizer.encode(last_text))
        yield {
            "type": "done",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    # ---- Qwen 推理 ----

    async def _generate_qwen(
        self, model, tokenizer, messages, thinking, temperature, top_p, max_tokens, max_ctx
    ) -> AsyncGenerator[dict, None]:
        """Qwen 模型推理：apply_chat_template + TextIteratorStreamer"""

        system_content = SYSTEM_PROMPT
        if thinking:
            system_content += THINKING_SUFFIX

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

        processed = self._truncate_messages(processed, tokenizer, max_ctx)

        try:
            prompt = tokenizer.apply_chat_template(
                processed, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_parts = []
            for m in processed:
                prompt_parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            prompt_parts.append("<|im_start|>assistant\n")
            prompt = "".join(prompt_parts)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
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

        def _generate():
            with torch.no_grad():
                model.generate(**gen_kwargs)

        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        # 逐 chunk 流式输出，DeepSeek 风格
        phase = "content"
        full_text = ""

        try:
            loop = asyncio.get_event_loop()
            while True:
                # 逐个读取 streamer chunk，立即推送
                try:
                    new_text = await loop.run_in_executor(None, next, streamer)
                except StopIteration:
                    break

                if not new_text:
                    continue
                full_text += new_text

                if phase == "content":
                    if "<thinking>" in full_text or "<think" in full_text:
                        phase = "thinking"
                    else:
                        yield {"type": "content", "content": new_text}

                if phase == "thinking":
                    if "</thinking>" in full_text or "</think" in full_text:
                        phase = "thinking_done"
                    yield {"type": "thinking", "content": new_text}

                elif phase == "thinking_done":
                    yield {"type": "content", "content": new_text}

        except Exception as e:
            yield {"type": "error", "content": f"生成失败: {e}"}
            return
        finally:
            t.join(timeout=5)

        completion_tokens = len(tokenizer.encode(full_text))
        yield {
            "type": "done",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }


# ---------------------------------------------------------------------------
# 全局 ModelManager 实例
# ---------------------------------------------------------------------------

manager = ModelManager()

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(title="DeepSleep Server", version="1.0.0")

# ---------------------------------------------------------------------------
# Pydantic 请求模型
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    messages: List[dict] = Field(..., description="对话历史")
    model: str = Field("ds_b0.1", description="模型 ID")
    thinking: bool = Field(True, description="是否启用思考模式")
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    max_tokens: int = Field(512, ge=1, le=2048)


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    """健康检查，返回 GPU 信息"""
    gpu_name = "N/A"
    vram_used = 0.0
    vram_total = 0.0

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        vram_used = (total - free) / (1024 ** 3)
        vram_total = total / (1024 ** 3)

    return {
        "status": "ok",
        "gpu": gpu_name,
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
        "loaded_models": list(manager.models.keys()),
    }


@app.get("/api/models")
async def list_models():
    """返回模型列表和加载状态"""
    models = []
    for mid, cfg in MODEL_CONFIGS.items():
        models.append(
            {
                "id": mid,
                "name": cfg["name"],
                "arch": cfg["arch"],
                "loaded": manager.is_loaded(mid),
            }
        )
    return {"models": models}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """SSE 流式对话端点"""
    model_id = request.model
    if model_id not in MODEL_CONFIGS:
        return JSONResponse(
            status_code=400,
            content={"error": f"未知模型: {model_id}，可选: {list(MODEL_CONFIGS.keys())}"},
        )

    if not request.messages:
        return JSONResponse(status_code=400, content={"error": "messages 不能为空"})

    async def _stream():
        try:
            async for chunk in manager.generate_stream(
                model_id=model_id,
                messages=request.messages,
                thinking=request.thinking,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            ):
                yield {"event": chunk["type"], "data": json.dumps(chunk, ensure_ascii=False)}
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"type": "error", "content": str(e)}, ensure_ascii=False
                ),
            }

    return EventSourceResponse(_stream())


# ---------------------------------------------------------------------------
# 静态文件托管（前端）
# ---------------------------------------------------------------------------

# 尝试挂载 web/dist 或 static 目录
_static_dir = None
_web_dist = os.path.join(os.path.dirname(__file__), "web", "dist")
_local_static = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(_web_dist) and os.listdir(_web_dist):
    _static_dir = _web_dist
elif os.path.isdir(_local_static) and os.listdir(_local_static):
    _static_dir = _local_static


@app.get("/")
async def root():
    """根路径重定向到静态文件"""
    if _static_dir:
        return RedirectResponse(url="/index.html")
    return {"message": "DeepSleep Server is running. No frontend files found."}


if _static_dir:
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")

# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------


def print_gpu_info():
    """启动时打印 GPU 信息"""
    print("=" * 60)
    print("DeepSleep Server - 睡眠健康AI对话后端")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"显存: {total / 1024**3:.1f} GB (可用 {free / 1024**3:.1f} GB)")
    else:
        print("警告: 未检测到 GPU，将使用 CPU（速度很慢）")
    print(f"可用模型: {list(MODEL_CONFIGS.keys())}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DeepSleep Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="监听端口 (默认: 7860)")
    args = parser.parse_args()

    print_gpu_info()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
