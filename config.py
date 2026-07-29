"""Central configuration for the medical multi-agent system."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Online OpenAI-compatible Qwen service.
# Keep secrets in environment variables rather than committing them to Git.
SERVE_URL = os.getenv("LLM_BASE_URL", "https://api.modagent-homing.com/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("MODAGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "Qwen3-32B")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_ENABLE_THINKING = _env_flag("LLM_ENABLE_THINKING", False)

# Local retrieval resources. Prefer the checked-in/downloaded local directories so
# service startup never depends on Hugging Face network availability.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "bge-m3"),
)
RERANK_MODEL = os.getenv(
    "RERANK_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "reranker"),
)
FAISS_INDEX_REPO = "literary123/faiss_index_A_v4"
FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    str(PROJECT_ROOT / "models" / "faiss_index_A_v4"),
)
HF_LOCAL_FILES_ONLY = _env_flag("HF_LOCAL_FILES_ONLY", True)

# Optional request-scoped web grounding. The UI switch is off by default, and
# the server-side flag provides an independent deployment-wide kill switch.
WEB_SEARCH_ENABLED = _env_flag("WEB_SEARCH_ENABLED", True)
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "auto")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_MAX_CONTEXT_CHARS = int(
    os.getenv("WEB_SEARCH_MAX_CONTEXT_CHARS", "8000")
)
WEB_SEARCH_TIMEOUT_SECONDS = float(
    os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "12")
)
WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", "wt-wt")
WEB_SEARCH_USER_AGENT = os.getenv(
    "WEB_SEARCH_USER_AGENT",
    "MedScope-AI/1.0 (+server-side medical research prototype)",
)
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
WEB_SEARCH_BRAVE_ENDPOINT = os.getenv(
    "WEB_SEARCH_BRAVE_ENDPOINT",
    "https://api.search.brave.com/res/v1/web/search",
)
WEB_SEARCH_DDG_ENDPOINT = os.getenv(
    "WEB_SEARCH_DDG_ENDPOINT",
    "https://html.duckduckgo.com/html/",
)

# Multiple FAISS index versions can be registered here when available.
DEFAULT_FAISS_VERSION = "v4"
FAISS_INDEX_VERSIONS = {
    "v4": FAISS_INDEX_PATH,
}

# Retrieval defaults.
RETRIEVER_MAIN_TOPK = 3
RETRIEVER_SUB_TOPK = 3
RETRIEVER_MIN_SCORE = 0.9
STREAM_RETRIEVER_MIN_SCORE = 0.95
RERANK_TOP_N = 40
VECTOR_RETRIEVER_TOP_K = 200

# FastAPI service.
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 50042
SSE_HEARTBEAT_SECONDS = float(os.getenv("SSE_HEARTBEAT_SECONDS", "10"))
CORS_ALLOW_ORIGINS = ["*"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
MAX_CONVERSATION_TURNS = 500000000000000000
MAX_CONVERSATION_HISTORY = 20000000000000000

# Generation defaults for OpenAI-compatible chat completion APIs.
GENERATION_CONFIG_BASE = {
    "model": MODEL_NAME,
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 8192,
    "stream": True,
}

GENERATION_CONFIG_GREEDY = {
    "model": MODEL_NAME,
    "temperature": 0.2,
    "top_p": 0.8,
    "max_tokens": 2048,
    "stream": False,
}

RETRIEVER_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 4096,
    "stream": False,
}

TEMP_RESPONSE_TEMPERATURES = [0.1]
TEMP_RESPONSE_MAX_TOKENS = 8192
TEMP_RESPONSE_STREAM = True

# Runtime medical skill. The checked-in skill is validated before prompt injection.
MEDICAL_SKILL_ENABLED = _env_flag("MEDICAL_SKILL_ENABLED", True)
MEDICAL_SKILL_DIR = Path(
    os.getenv(
        "MEDICAL_SKILL_DIR",
        str(PROJECT_ROOT / "skills" / "medical-multi-agent"),
    )
).resolve()
MEDICAL_SKILL_MAX_CHARS = int(os.getenv("MEDICAL_SKILL_MAX_CHARS", "16000"))

# Stroke questions use an authoritative expert registry and deterministic
# validation before the legacy recruitment text is exposed to the UI.
STROKE_HARD_RECRUITMENT_ENABLED = _env_flag(
    "STROKE_HARD_RECRUITMENT_ENABLED",
    True,
)
STROKE_EXPERT_REGISTRY_PATH = Path(
    os.getenv(
        "STROKE_EXPERT_REGISTRY_PATH",
        str(PROJECT_ROOT / "skills" / "medical-multi-agent" / "experts.json"),
    )
).resolve()

# Trace2Skill is an offline workflow. Trace capture is disabled by default and
# content capture requires a second explicit opt-in for reviewed/test cases.
TRACE2SKILL_ENABLED = _env_flag("TRACE2SKILL_ENABLED", False)
TRACE2SKILL_CAPTURE_CONTENT = _env_flag("TRACE2SKILL_CAPTURE_CONTENT", False)
TRACE2SKILL_TRACE_DIR = Path(
    os.getenv(
        "TRACE2SKILL_TRACE_DIR",
        str(PROJECT_ROOT / "data" / "trace2skill" / "traces"),
    )
).resolve()
TRACE2SKILL_REPO = Path(
    os.getenv(
        "TRACE2SKILL_REPO",
        str(PROJECT_ROOT / "third_party" / "Trace2Skill"),
    )
).resolve()
TRACE2SKILL_ARTIFACT_DIR = Path(
    os.getenv(
        "TRACE2SKILL_ARTIFACT_DIR",
        str(PROJECT_ROOT / "artifacts" / "trace2skill"),
    )
).resolve()
