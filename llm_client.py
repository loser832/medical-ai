"""Factory for the online OpenAI-compatible Qwen client."""

from openai import OpenAI

from config import (
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    SERVE_URL,
)


def create_llm_client() -> OpenAI:
    """Create a configured client and fail early when the API key is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "未配置 Qwen API Key。请先设置环境变量 MODAGENT_API_KEY，"
            "例如：$env:MODAGENT_API_KEY='<YOUR_API_KEY>'（PowerShell）。"
        )

    return OpenAI(
        base_url=SERVE_URL,
        api_key=OPENAI_API_KEY,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
