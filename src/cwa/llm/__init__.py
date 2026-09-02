"""LLM 客户端：OpenAI 兼容接口，支持 Ollama 本地降级。"""

from __future__ import annotations

from llama_index.core.llms import LLM

from ..config import Config


def build_llm(cfg: Config) -> LLM:
    """按配置构建 LLM。未配 key 时降级到本地 Ollama（若可用）。"""
    if cfg.configured:
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=cfg.model,
            api_base=cfg.base_url,
            api_key=cfg.api_key,
            is_chat_model=True,
        )

    # 无 key：尝试本地 Ollama
    try:
        from llama_index.llms.ollama import Ollama

        return Ollama(model=cfg.ollama_model, request_timeout=120.0)
    except Exception:
        raise RuntimeError(
            "未配置 API key，且本地 Ollama 不可用。\n"
            "请二选一：\n"
            "  1) 运行 `cwa config set api_key sk-xxx` 配置密钥\n"
            "  2) 安装并启动本地 Ollama（https://ollama.com）"
        )
