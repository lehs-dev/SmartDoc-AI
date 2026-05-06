from pathlib import Path
import json

from core.env import (
    load_env_file,
    get_env,
    get_env_bool,
    get_env_int,
    get_env_float,
    get_env_list,
)

load_env_file()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / raw_path


def _parse_json_list(raw_value, key):
    try:
        parsed = json.loads(raw_value)
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON list for {key}: {exc}") from exc
    if not isinstance(parsed, list):
        raise RuntimeError(f"Invalid JSON list for {key}: expected list")
    return parsed


VECTOR_DB_BASE_PATH = str(_resolve_path(get_env("SMARTDOC_VECTOR_DB_BASE_PATH", required=True)))

HF_HUB_OFFLINE = get_env("HF_HUB_OFFLINE", required=True)
OMP_NUM_THREADS = get_env("OMP_NUM_THREADS", required=True)
OPENBLAS_NUM_THREADS = get_env("OPENBLAS_NUM_THREADS", required=True)
MKL_NUM_THREADS = get_env("MKL_NUM_THREADS", required=True)

DEFAULT_LLM_MODEL = get_env("SMARTDOC_LLM_MODEL", required=True)
SUPPORTED_LLM_MODELS = get_env_list("SMARTDOC_LLM_MODELS", required=True)
if DEFAULT_LLM_MODEL not in SUPPORTED_LLM_MODELS:
    SUPPORTED_LLM_MODELS = [DEFAULT_LLM_MODEL] + SUPPORTED_LLM_MODELS

DEFAULT_EMBEDDING_MODEL = get_env("SMARTDOC_EMBEDDING_MODEL", required=True)
SUPPORTED_EMBEDDING_MODELS = get_env_list("SMARTDOC_EMBEDDING_MODELS", required=True)
if DEFAULT_EMBEDDING_MODEL not in SUPPORTED_EMBEDDING_MODELS:
    SUPPORTED_EMBEDDING_MODELS = [DEFAULT_EMBEDDING_MODEL] + SUPPORTED_EMBEDDING_MODELS

DEFAULT_VECTOR_DB_KEY = get_env("SMARTDOC_VECTOR_DB_KEY", required=True)
SUPPORTED_VECTOR_DB_KEYS = get_env_list("SMARTDOC_VECTOR_DB_KEYS", required=True)
if DEFAULT_VECTOR_DB_KEY not in SUPPORTED_VECTOR_DB_KEYS:
    SUPPORTED_VECTOR_DB_KEYS = [DEFAULT_VECTOR_DB_KEY] + SUPPORTED_VECTOR_DB_KEYS

VECTOR_DB_CONFIG = {
    key: {
        "path": str(Path(VECTOR_DB_BASE_PATH) / key),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }
    for key in SUPPORTED_VECTOR_DB_KEYS
}

MEMORY_VECTOR_DB_KEY = get_env("SMARTDOC_MEMORY_VECTOR_DB_KEY", required=True)

FAST_MODE = get_env_bool("SMARTDOC_FAST_MODE", required=True)
MAX_HISTORY_MESSAGES = get_env_int("SMARTDOC_HISTORY_MESSAGES", required=True)
MAX_HISTORY_CHARS = get_env_int("SMARTDOC_HISTORY_CHARS", required=True)
MAX_CONTEXT_CHARS = get_env_int("SMARTDOC_CONTEXT_CHARS", required=True)
MAX_RAG_CHUNKS = get_env_int("SMARTDOC_RAG_CHUNKS", required=True)
MAX_MEMORY_CONTEXT_CHARS = get_env_int("SMARTDOC_MEMORY_CONTEXT_CHARS", required=True)
MAX_MEMORY_RAG_CHUNKS = get_env_int("SMARTDOC_MEMORY_RAG_CHUNKS", required=True)

OLLAMA_KEEP_ALIVE = get_env("SMARTDOC_OLLAMA_KEEP_ALIVE", required=True)
NUM_CTX = get_env_int("SMARTDOC_NUM_CTX", required=True)
NUM_PREDICT = get_env_int("SMARTDOC_NUM_PREDICT", required=True)
TEMPERATURE = get_env_float("SMARTDOC_TEMPERATURE", required=True)
OLLAMA_MODE = get_env("SMARTDOC_OLLAMA_MODE", required=True).strip().lower()
if OLLAMA_MODE not in ("chat", "generate"):
    OLLAMA_MODE = "generate"

LOG_PROMPT = get_env_bool("SMARTDOC_LOG_PROMPT", required=True)
LOG_VERBOSE = get_env_bool("SMARTDOC_LOG_VERBOSE", required=True)
LOG_RAW = get_env_bool("SMARTDOC_LOG_RAW", required=True)
LOG_RAW_LIMIT = get_env_int("SMARTDOC_LOG_RAW_LIMIT", required=True)
FALLBACK_NUM_PREDICT = get_env_int("SMARTDOC_FALLBACK_NUM_PREDICT", required=True)
RAW_PROMPT = get_env_bool("SMARTDOC_RAW_PROMPT", required=True)
ASSISTANT_NAME = get_env("SMARTDOC_ASSISTANT_NAME", required=True)
LOG_LEVEL = get_env("SMARTDOC_LOG_LEVEL", required=True).upper()

STREAM_TEXT_CHUNK_SIZE = get_env_int("SMARTDOC_STREAM_TEXT_CHUNK_SIZE", required=True)
TEXT_CHUNK_SIZE = get_env_int("SMARTDOC_TEXT_CHUNK_SIZE", required=True)
TEXT_CHUNK_OVERLAP = get_env_int("SMARTDOC_TEXT_CHUNK_OVERLAP", required=True)
TEXT_CHUNK_SEPARATORS = _parse_json_list(
    get_env("SMARTDOC_TEXT_CHUNK_SEPARATORS_JSON", required=True),
    "SMARTDOC_TEXT_CHUNK_SEPARATORS_JSON",
)

RETRIEVE_MAX_K = get_env_int("SMARTDOC_RETRIEVE_MAX_K", required=True)
RECENT_HISTORY_LIMIT = get_env_int("SMARTDOC_RECENT_HISTORY_LIMIT", required=True)
MEMORY_SUMMARY_MAX_CHARS = get_env_int("SMARTDOC_MEMORY_SUMMARY_MAX_CHARS", required=True)
MEMORY_TEXT_MAX_CHARS = get_env_int("SMARTDOC_MEMORY_TEXT_MAX_CHARS", required=True)
MEMORY_SUMMARY_SOURCE_CHARS = get_env_int("SMARTDOC_MEMORY_SUMMARY_SOURCE_CHARS", required=True)

UPLOAD_MAX_MB = get_env_int("SMARTDOC_UPLOAD_MAX_MB", required=True)
