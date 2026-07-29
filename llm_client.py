"""Factory for the online OpenAI-compatible Qwen client."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

from config import (
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    SERVE_URL,
)


def create_llm_client() -> "OpenAI":
    """Create a configured client and fail early when the API key is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "未配置 Qwen API Key。请先设置环境变量 MODAGENT_API_KEY，"
            "例如：$env:MODAGENT_API_KEY='<YOUR_API_KEY>'（PowerShell）。"
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "未安装 openai 依赖。请先运行 `pip install -r requirements.txt`。"
        ) from error

    return OpenAI(
        base_url=SERVE_URL,
        api_key=OPENAI_API_KEY,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
