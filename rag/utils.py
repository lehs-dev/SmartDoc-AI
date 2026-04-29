import os
import re
import json
import logging
import threading
from datetime import datetime
import unicodedata
import pdfplumber
import docx
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from .models import ChatMessage, ChatSession, ConversationMemory

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

try:
    import fitz as _fitz
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

VECTOR_DB_BASE_PATH = "vector_store"

DEFAULT_LLM_MODEL = os.getenv("SMARTDOC_LLM_MODEL", "gemma4:e4b")
DEFAULT_EMBEDDING_MODEL = os.getenv("SMARTDOC_EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_VECTOR_DB_KEY = os.getenv("SMARTDOC_VECTOR_DB_KEY", "nomic_v1_db")

SUPPORTED_LLM_MODELS = [DEFAULT_LLM_MODEL]
SUPPORTED_EMBEDDING_MODELS = [DEFAULT_EMBEDDING_MODEL]

VECTOR_DB_CONFIG = {
    DEFAULT_VECTOR_DB_KEY: {
        "path": os.path.join(VECTOR_DB_BASE_PATH, DEFAULT_VECTOR_DB_KEY),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    },
}

_FAST_MODE = os.getenv("SMARTDOC_FAST_MODE", "0") != "0"
_MAX_HISTORY_MESSAGES = int(os.getenv("SMARTDOC_HISTORY_MESSAGES", "2"))
_MAX_HISTORY_CHARS = int(os.getenv("SMARTDOC_HISTORY_CHARS", "600"))
_MAX_CONTEXT_CHARS = int(os.getenv("SMARTDOC_CONTEXT_CHARS", "1200"))
_MAX_RAG_CHUNKS = int(os.getenv("SMARTDOC_RAG_CHUNKS", "1"))
_KEEP_ALIVE = os.getenv("SMARTDOC_OLLAMA_KEEP_ALIVE", "10m")
_NUM_CTX = int(os.getenv("SMARTDOC_NUM_CTX", "1024"))
_NUM_PREDICT = int(os.getenv("SMARTDOC_NUM_PREDICT", "192"))
_TEMPERATURE = float(os.getenv("SMARTDOC_TEMPERATURE", "0.2"))
_OLLAMA_MODE = os.getenv("SMARTDOC_OLLAMA_MODE", "generate").strip().lower()
_LOG_PROMPT = os.getenv("SMARTDOC_LOG_PROMPT", "0") == "1"
_LOG_VERBOSE = os.getenv("SMARTDOC_LOG_VERBOSE", "0") == "1"
_LOG_RAW = os.getenv("SMARTDOC_LOG_RAW", "0") == "1"
_LOG_RAW_LIMIT = int(os.getenv("SMARTDOC_LOG_RAW_LIMIT", "3"))
_FALLBACK_NUM_PREDICT = int(os.getenv("SMARTDOC_FALLBACK_NUM_PREDICT", "1024"))
_RAW_PROMPT = os.getenv("SMARTDOC_RAW_PROMPT", "1") == "1"
ASSISTANT_NAME = os.getenv("SMARTDOC_ASSISTANT_NAME", "Gemma 4")
if _OLLAMA_MODE not in ("chat", "generate"):
    _OLLAMA_MODE = "generate"

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USER_INFO_PATH = os.getenv("SMARTDOC_USER_INFO_PATH", os.path.join(_PROJECT_ROOT, "user_info.json"))

_LOG_LEVEL = os.getenv("SMARTDOC_LOG_LEVEL", "INFO").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=_LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(message)s",
    )

_logger = logging.getLogger("smartdoc.rag")
_logger.setLevel(_LOG_LEVEL)
if _LOG_VERBOSE or _LOG_RAW:
    _logger.setLevel(logging.DEBUG)

_embedding_model_cache = {}
_vector_store_cache = {}
_installed_ollama_models_cache = None
_user_info_lock = threading.Lock()


def _log_debug(message, *args):
    _logger.debug(message, *args)


def _log_info(message, *args):
    _logger.info(message, *args)


def _log_warning(message, *args):
    _logger.warning(message, *args)


def _load_user_info_unlocked():
    if not os.path.exists(USER_INFO_PATH):
        return {"version": 1, "users": {}}

    try:
        with open(USER_INFO_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {"version": 1, "users": {}}
        data.setdefault("version", 1)
        data.setdefault("users", {})
        if not isinstance(data["users"], dict):
            data["users"] = {}
        return data
    except Exception as exc:
        _log_warning("Cannot read user_info.json: %s", exc)
        return {"version": 1, "users": {}}


def _save_user_info_unlocked(data):
    directory = os.path.dirname(USER_INFO_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = USER_INFO_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)
    os.replace(tmp_path, USER_INFO_PATH)


def _split_preferences(raw_text):
    if not raw_text:
        return []
    parts = re.split(r",|;|\bva\b|\bvà\b", raw_text, flags=re.IGNORECASE)
    cleaned = [part.strip(" .\t\n\r").strip() for part in parts]
    return [item for item in cleaned if item][:5]


def _extract_user_info_from_text(text):
    if not text:
        return {}

    info = {}

    name_match = re.search(
        r"(?:tôi|toi|mình|minh|em|ta)\s+tên\s+([A-Za-zÀ-ỹ\s]{2,60})",
        text,
        flags=re.IGNORECASE,
    )
    if name_match:
        raw_name = name_match.group(1)
        raw_name = re.split(r"[\n\r,.!?;:]+", raw_name)[0].strip()
        if 1 < len(raw_name) <= 60:
            info["name"] = raw_name

    age_match = re.search(r"(\d{1,3})\s*(?:tuoi|tuổi)\b", text, flags=re.IGNORECASE)
    if age_match:
        age = int(age_match.group(1))
        if 0 < age < 120:
            info["age"] = age

    pref_match = re.search(
        r"(?:sở\s*thích|so\s*thich)(?:\s+của\s+tôi|\s+cua\s+toi)?\s*(?:là|la|:)\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if pref_match:
        raw_pref = pref_match.group(1).strip()
        prefs = _split_preferences(raw_pref)
        if prefs:
            info["preferences"] = prefs
    else:
        like_match = re.search(
            r"(?:tôi|toi|mình|minh|em)\s+thích\s+([^\n]+)",
            text,
            flags=re.IGNORECASE,
        )
        if like_match:
            raw_like = like_match.group(1).strip()
            prefs = _split_preferences(raw_like)
            if prefs:
                info["preferences"] = prefs

    return info


def update_user_profile(user_key, text):
    if not user_key:
        return {}

    updates = _extract_user_info_from_text(text)
    if not updates:
        return {}

    with _user_info_lock:
        data = _load_user_info_unlocked()
        users = data.get("users", {})
        user_data = users.get(user_key, {})

        if "name" in updates:
            user_data["name"] = updates["name"]
        if "age" in updates:
            user_data["age"] = updates["age"]
        if "preferences" in updates:
            existing = user_data.get("preferences", [])
            combined = existing + [item for item in updates["preferences"] if item not in existing]
            user_data["preferences"] = combined[:8]

        user_data["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        users[user_key] = user_data
        data["users"] = users
        _save_user_info_unlocked(data)

    return user_data


def get_user_profile_summary(user_key):
    if not user_key:
        return ""

    with _user_info_lock:
        data = _load_user_info_unlocked()
        user_data = data.get("users", {}).get(user_key)

    if not user_data:
        return ""

    parts = ["Nguoi dung"]
    name = user_data.get("name")
    age = user_data.get("age")
    prefs = user_data.get("preferences") or []

    if name:
        parts.append(f"Ten: {name}")
    if age:
        parts.append(f"Tuoi: {age}")
    if prefs:
        parts.append("So thich: " + ", ".join(prefs))

    return "; ".join(parts)


def _log_request(prompt, model_name, mode, options):
    prompt_len = len(prompt or "")
    _log_info("LLM request model=%s mode=%s prompt_len=%s options=%s", model_name, mode, prompt_len, options)
    if _LOG_PROMPT:
        _log_info("LLM prompt preview: %s", _truncate_text(prompt, 400))


def _summarize_chunk(chunk):
    if isinstance(chunk, dict):
        message = chunk.get("message") or {}
        content = ""
        if isinstance(message, dict):
            content = message.get("content") or ""
        return {
            "keys": list(chunk.keys()),
            "response_len": len(chunk.get("response") or ""),
            "content_len": len(content),
            "done": chunk.get("done"),
            "error": chunk.get("error"),
        }

    return {
        "type": type(chunk).__name__,
        "text_len": len(str(chunk)),
    }


def _truncate_text(text, max_chars):
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _strip_accents(text):
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_text(text):
    text = _strip_accents(text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_ai_identity_question(text):
    normalized = _normalize_text(text)
    if not normalized:
        return False
    patterns = [
        "ban ten gi",
        "ten ban la gi",
        "ban la ai",
        "ban ten la gi",
        "gioi thieu ban",
        "gioi thieu ban than",
    ]
    return any(pattern in normalized for pattern in patterns)


def _normalize_session_id(session_id):
    try:
        return int(session_id)
    except (TypeError, ValueError):
        return None


def _format_recent_messages(messages, max_chars=600):
    lines = []
    total_chars = 0

    for msg in messages:
        role = "U" if msg.role == "user" else "A"
        line = f"{role}: {msg.content}"
        if lines and total_chars + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def _merge_options(base, override):
    if not override:
        return base
    merged = dict(base)
    merged.update(override)
    return merged


def _get_ollama_options(override=None):
    if not _FAST_MODE:
        return _merge_options({}, override)

    base = {
        "num_ctx": max(256, _NUM_CTX),
        "num_predict": max(64, _NUM_PREDICT),
        "temperature": max(0.0, _TEMPERATURE),
    }
    return _merge_options(base, override)


def _extract_ollama_text(response_item, strip_text=True):
    if isinstance(response_item, bytes):
        response_item = response_item.decode("utf-8", errors="ignore")
    if isinstance(response_item, str):
        return response_item.strip() if strip_text else response_item

    def _normalize(value):
        if value is None:
            return ""
        text = str(value)
        return text.strip() if strip_text else text

    if isinstance(response_item, dict):
        message = response_item.get("message") or {}
        if isinstance(message, dict):
            content = _normalize(message.get("content"))
            if content:
                return content
        else:
            content = _normalize(getattr(message, "content", None))
            if content:
                return content

        response_text = response_item.get("response") or response_item.get("content")
        return _normalize(response_text)

    message = getattr(response_item, "message", None)
    if isinstance(message, dict):
        content = _normalize(message.get("content"))
        if content:
            return content

    if message is not None:
        content = _normalize(getattr(message, "content", None))
        if content:
            return content

    response_text = getattr(response_item, "response", None) or getattr(response_item, "content", None)
    return _normalize(response_text)


def _ollama_stream(prompt, model_name, mode=None, options_override=None):
    mode = (mode or _OLLAMA_MODE).strip().lower()
    options = _get_ollama_options(options_override)
    if _LOG_VERBOSE:
        _log_request(prompt, model_name, mode, options)

    raw_logged = 0
    chunk_count = 0

    if mode == "generate":
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options=options,
            keep_alive=_KEEP_ALIVE,
            stream=True,
        )
    else:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            keep_alive=_KEEP_ALIVE,
            stream=True,
        )

    if isinstance(response, (dict, str, bytes)):
        _log_warning("Ollama stream returned non-iterable response type=%s", type(response))
        content = _extract_ollama_text(response, strip_text=False)
        if content:
            yield content
        return

    for chunk in response:
        chunk_count += 1
        if _LOG_VERBOSE and chunk_count <= _LOG_RAW_LIMIT:
            _log_info("Stream chunk type=%s", type(chunk))
            _log_info("Stream chunk summary=%s", _summarize_chunk(chunk))
        if _LOG_RAW and raw_logged < _LOG_RAW_LIMIT:
            _log_info("Stream chunk raw=%s", _truncate_text(str(chunk), 400))
            raw_logged += 1
        content = _extract_ollama_text(chunk, strip_text=False)
        if content:
            yield content
        else:
            _log_debug("Empty stream chunk: %r", chunk)

    if _LOG_VERBOSE:
        _log_info("Stream chunk count=%s", chunk_count)
    if chunk_count == 0:
        _log_warning("Ollama stream had zero chunks")


def _ollama_invoke(prompt, model_name, mode=None, options_override=None):
    mode = (mode or _OLLAMA_MODE).strip().lower()
    options = _get_ollama_options(options_override)
    if _LOG_VERBOSE:
        _log_request(prompt, model_name, mode, options)

    if mode == "generate":
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options=options,
            keep_alive=_KEEP_ALIVE,
            stream=False,
        )
    else:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            keep_alive=_KEEP_ALIVE,
            stream=False,
        )

    content = _extract_ollama_text(response)
    if not content:
        _log_debug("Empty invoke response: %r", response)
    if _LOG_VERBOSE:
        _log_debug("Invoke response summary=%s", _summarize_chunk(response))
        if _LOG_RAW:
            _log_debug("Invoke response raw=%s", _truncate_text(str(response), 400))
    return content


def _stream_text_chunks(text, chunk_size=120):
    text = text or ""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def _stream_with_fallback(prompt, model_name, fallback_prompt=None):
    has_output = False
    primary_mode = _OLLAMA_MODE
    fallback_mode = "chat" if primary_mode == "generate" else "generate"
    retry_prompt = fallback_prompt or prompt

    if _LOG_VERBOSE:
        _log_info("LLM stream start model=%s primary=%s fallback=%s", model_name, primary_mode, fallback_mode)

    try:
        for chunk in _ollama_stream(prompt, model_name, mode=primary_mode):
            if not chunk:
                continue
            has_output = True
            yield chunk
    except Exception as exc:
        _log_warning("LLM stream error: %s", exc)

    if has_output:
        return

    try:
        response_text = _ollama_invoke(prompt, model_name, mode=primary_mode)
    except Exception as exc:
        _log_warning("LLM invoke error: %s", exc)
        response_text = ""

    if not response_text and _FALLBACK_NUM_PREDICT > 0:
        _log_warning("Empty response, retry with num_predict=%s", _FALLBACK_NUM_PREDICT)
        try:
            response_text = _ollama_invoke(
                retry_prompt,
                model_name,
                mode=primary_mode,
                options_override={"num_predict": _FALLBACK_NUM_PREDICT},
            )
        except Exception as exc:
            _log_warning("LLM retry invoke error: %s", exc)
            response_text = ""

    if not response_text:
        try:
            response_text = _ollama_invoke(retry_prompt, model_name, mode=fallback_mode)
        except Exception as exc:
            _log_warning("LLM fallback invoke error: %s", exc)
            response_text = ""

    if not response_text and _FALLBACK_NUM_PREDICT > 0:
        _log_warning("Empty response, retry fallback with num_predict=%s", _FALLBACK_NUM_PREDICT)
        try:
            response_text = _ollama_invoke(
                retry_prompt,
                model_name,
                mode=fallback_mode,
                options_override={"num_predict": _FALLBACK_NUM_PREDICT},
            )
        except Exception as exc:
            _log_warning("LLM retry fallback error: %s", exc)
            response_text = ""

    if response_text:
        for chunk in _stream_text_chunks(response_text):
            yield chunk
        return

    yield "Xin lỗi, model không phản hồi."


def check_ollama_connection():
    try:
        models = get_installed_ollama_models(refresh=True)
        _log_info("Ollama OK. models=%s", len(models))
        return True
    except Exception as exc:
        _log_warning("Ollama check failed: %s", exc)
        return False


def get_installed_ollama_models(refresh=False):
    global _installed_ollama_models_cache

    if _installed_ollama_models_cache is not None and not refresh:
        return list(_installed_ollama_models_cache)

    try:
        response = ollama.list()
        model_items = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])

        installed_models = []
        for item in model_items:
            if isinstance(item, dict):
                model_name = item.get("model") or item.get("name")
            else:
                model_name = getattr(item, "model", None) or getattr(item, "name", None)
            if model_name:
                installed_models.append(model_name)

        _installed_ollama_models_cache = installed_models
    except Exception as exc:
        _log_warning("Cannot read Ollama models: %s", exc)
        _installed_ollama_models_cache = []

    return list(_installed_ollama_models_cache)


def get_available_llm_models():
    return [DEFAULT_LLM_MODEL]


def get_available_embedding_models():
    return list(SUPPORTED_EMBEDDING_MODELS)


def resolve_llm_model(model_name):
    model_name = model_name if model_name in SUPPORTED_LLM_MODELS else DEFAULT_LLM_MODEL
    installed = set(get_installed_ollama_models())

    if model_name in installed:
        return model_name

    raise ValueError(
        f"Model LLM '{model_name}' chưa có trong Ollama local. "
        "Vui lòng chạy: ollama pull gemma4:e4b"
    )


def _normalize_embedding_model(model_name):
    if model_name not in SUPPORTED_EMBEDDING_MODELS:
        return DEFAULT_EMBEDDING_MODEL
    return model_name


def _normalize_vector_db_key(vector_db_key):
    if vector_db_key not in VECTOR_DB_CONFIG:
        return DEFAULT_VECTOR_DB_KEY
    return vector_db_key


def resolve_vector_db_path(vector_db_key):
    vector_db_key = _normalize_vector_db_key(vector_db_key)
    config = VECTOR_DB_CONFIG.get(vector_db_key)
    return config["path"]


def route_embedding_target(file_size_bytes, has_vietnamese):
    file_size_mb = file_size_bytes / (1024 * 1024)
    vector_db_key = DEFAULT_VECTOR_DB_KEY
    config = VECTOR_DB_CONFIG[vector_db_key]
    return {
        "file_size_mb": round(file_size_mb, 2),
        "vector_db_key": vector_db_key,
        "vector_db_path": config["path"],
        "embedding_model": config["embedding_model"],
    }


def get_embeddings_model(model_name):
    model_name = _normalize_embedding_model(model_name)

    if model_name not in _embedding_model_cache:
        _log_info("Loading embedding model: %s", model_name)
        _embedding_model_cache[model_name] = OllamaEmbeddings(model=model_name)

    return _embedding_model_cache[model_name]


def get_cached_vector_store(vector_db_key, embedding_model_name):
    vector_db_key = _normalize_vector_db_key(vector_db_key)
    embedding_model_name = _normalize_embedding_model(embedding_model_name)
    cache_key = f"{vector_db_key}::{embedding_model_name}"

    if cache_key in _vector_store_cache:
        return _vector_store_cache[cache_key]

    vector_db_path = resolve_vector_db_path(vector_db_key)
    index_path = os.path.join(vector_db_path, "index.faiss")

    if not os.path.exists(index_path):
        _vector_store_cache[cache_key] = None
        return None

    embeddings = get_embeddings_model(embedding_model_name)
    _vector_store_cache[cache_key] = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return _vector_store_cache[cache_key]


def extract_text(file_path, file_extension):
    text = ""

    try:
        if file_extension == "pdf":
            if _HAS_PYMUPDF:
                try:
                    doc = _fitz.open(file_path)
                    for page in doc:
                        extracted = page.get_text()
                        if extracted:
                            text += extracted + "\n"
                    doc.close()
                except Exception as exc:
                    _log_warning("PyMuPDF failed, fallback to pdfplumber: %s", exc)
                    text = ""

            if not text.strip():
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"

        elif file_extension == "docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"

        return text

    except Exception as exc:
        _log_warning("Cannot extract text from %s: %s", file_path, exc)
        return ""


def _split_text_chunks(text, chunk_size, chunk_overlap, separators=None):
    splitter_kwargs = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "length_function": len,
    }
    if separators is not None:
        splitter_kwargs["separators"] = separators
        splitter_kwargs["is_separator_regex"] = False

    text_splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
    return text_splitter.split_text(text)


def get_text_chunks(text):
    return get_text_chunks_optimized(text)


def get_text_chunks_optimized(text, file_size_mb=0, has_vietnamese=False):
    chunk_size = 500
    chunk_overlap = 50

    separators = ["\n\n", "\n", "!", "?", ".", " ", ""]

    chunks = _split_text_chunks(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    _log_debug("Chunking done: %s chunks", len(chunks))
    return chunks


def process_document(file_path, file_extension):
    return process_document_optimized(file_path, file_extension)


def process_document_optimized(file_path, file_extension, file_size_mb=0, has_vietnamese=False):
    raw_text = extract_text(file_path, file_extension)

    if not raw_text.strip():
        return []

    return get_text_chunks_optimized(raw_text, file_size_mb, has_vietnamese)


def get_vector_store(chunks, embedding_model_name, vector_db_key):
    vector_db_key = _normalize_vector_db_key(vector_db_key)
    embedding_model_name = _normalize_embedding_model(embedding_model_name)
    vector_db_path = resolve_vector_db_path(vector_db_key)

    os.makedirs(vector_db_path, exist_ok=True)

    embeddings = get_embeddings_model(embedding_model_name)
    index_path = os.path.join(vector_db_path, "index.faiss")

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(
            vector_db_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_texts(chunks)
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    vector_store.save_local(vector_db_path)

    cache_key = f"{vector_db_key}::{embedding_model_name}"
    _vector_store_cache[cache_key] = vector_store
    return vector_store


def _build_general_chat_history(session_id):
    session_id = _normalize_session_id(session_id)
    if session_id is None:
        return ""

    recent_messages = get_recent_conversation_history(session_id, limit=_MAX_HISTORY_MESSAGES)
    return _truncate_text(_format_recent_messages(recent_messages, max_chars=_MAX_HISTORY_CHARS), _MAX_HISTORY_CHARS)


def _get_memory_summary(session_id):
    session_id = _normalize_session_id(session_id)
    if session_id is None:
        return ""

    memory = ConversationMemory.objects.filter(session_id=session_id).first()
    if not memory or not memory.summary:
        return ""

    summary = _truncate_text(memory.summary, 300)
    if "không phản hồi" in summary.lower():
        return ""
    return summary


def _build_general_prompt(question, chat_history="", memory_summary=""):
    identity_question = _is_ai_identity_question(question)
    safe_memory = "" if identity_question else memory_summary

    if _RAW_PROMPT:
        if safe_memory:
            return (
                f"Ban la tro ly SmartDoc AI, ten ban la {ASSISTANT_NAME}."
                f" Neu hoi ten, tra loi: {ASSISTANT_NAME}."
                " Tra loi truc tiep, khong dua nhieu phuong an."
                f"\nThong tin da biet: {safe_memory}"
                f"\nUser: {question}\nAssistant:"
            )
        return (
            f"Ban la tro ly SmartDoc AI, ten ban la {ASSISTANT_NAME}."
            f" Neu hoi ten, tra loi: {ASSISTANT_NAME}."
            " Tra loi truc tiep, khong dua nhieu phuong an."
            f"\nUser: {question}\nAssistant:"
        )

    prompt_parts = [
        f"Ban la tro ly SmartDoc AI, ten ban la {ASSISTANT_NAME}.",
        "Neu tom tat hoac lich su co thong tin, hay dung de tra loi truc tiep.",
        "Neu nguoi dung hoi ten ban, tra loi bang ten cua ban.",
        "Tra loi truc tiep, khong dua nhieu phuong an.",
    ]

    if safe_memory:
        prompt_parts.append(f"Tom tat nho:\n{safe_memory}")

    if chat_history:
        prompt_parts.append(f"Lich su:\n{chat_history}")

    prompt_parts.append(f"Hoi: {question}\nDap:")
    return "\n\n".join(prompt_parts)


def _build_rag_prompt(question, context, chat_history="", memory_summary=""):
    identity_question = _is_ai_identity_question(question)
    safe_memory = "" if identity_question else memory_summary

    if _RAW_PROMPT:
        if context and safe_memory:
            return (
                f"Ban la tro ly SmartDoc AI, ten ban la {ASSISTANT_NAME}."
                f" Neu hoi ten, tra loi: {ASSISTANT_NAME}."
                " Tra loi truc tiep, khong dua nhieu phuong an."
                f"\n{context}"
                f"\nThong tin da biet: {safe_memory}"
                f"\nUser: {question}\nAssistant:"
            )
        if context:
            return (
                f"Ban la tro ly SmartDoc AI, ten ban la {ASSISTANT_NAME}."
                f" Neu hoi ten, tra loi: {ASSISTANT_NAME}."
                " Tra loi truc tiep, khong dua nhieu phuong an."
                f"\n{context}\nUser: {question}\nAssistant:"
            )
        if safe_memory:
            return (
                f"Ban la tro ly SmartDoc AI, ten ban la {ASSISTANT_NAME}."
                f" Neu hoi ten, tra loi: {ASSISTANT_NAME}."
                " Tra loi truc tiep, khong dua nhieu phuong an."
                f"\nThong tin da biet: {safe_memory}"
                f"\nUser: {question}\nAssistant:"
            )
        return (
            f"Ban la tro ly SmartDoc AI, ten ban la {ASSISTANT_NAME}."
            f" Neu hoi ten, tra loi: {ASSISTANT_NAME}."
            " Tra loi truc tiep, khong dua nhieu phuong an."
            f"\nUser: {question}\nAssistant:"
        )

    prompt_parts = [
        f"Ban la tro ly SmartDoc AI, ten ban la {ASSISTANT_NAME}.",
        "Uu tien ngu canh tai lieu; neu tom tat/lich su co thong tin lien quan thi dung.",
        "Neu nguoi dung hoi ten ban, tra loi bang ten cua ban.",
        "Tra loi truc tiep, khong dua nhieu phuong an.",
    ]

    if safe_memory:
        prompt_parts.append(f"Tom tat nho:\n{safe_memory}")

    if chat_history:
        prompt_parts.append(f"Lich su:\n{chat_history}")

    if context:
        prompt_parts.append(f"Ngu canh:\n{context}")

    prompt_parts.append(f"Hoi: {question}\nDap:")
    return "\n\n".join(prompt_parts)


def _retrieve_context(vector_store, question, k_chunks):
    if not vector_store:
        return ""

    k = max(1, min(k_chunks, 4))
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return _truncate_text(context, _MAX_CONTEXT_CHARS)


def ask_gemma(
    question,
    chat_history="",
    llm_model_name=DEFAULT_LLM_MODEL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    vector_db_key=DEFAULT_VECTOR_DB_KEY,
):
    if not embedding_model_name or not vector_db_key:
        return iter([
            "Xin lỗi, chưa xác định được kho dữ liệu cho tài liệu này."
        ])

    vector_store = get_cached_vector_store(vector_db_key, embedding_model_name)
    if vector_store is None:
        return iter([
            "Xin lỗi, kho dữ liệu của tài liệu này chưa có nội dung."
        ])

    model_name = resolve_llm_model(llm_model_name)
    context = _retrieve_context(vector_store, question, k_chunks=_MAX_RAG_CHUNKS)
    prompt = _build_rag_prompt(question, context, chat_history=chat_history)
    fallback_prompt = prompt if _RAW_PROMPT else (f"{context}\n\n{question}" if context else question)
    return _stream_with_fallback(prompt, model_name, fallback_prompt=fallback_prompt)


# ============================================================================
# GENERAL CHAT MODE - DIRECT LLM (NO RAG)
# ============================================================================

def ask_llm_direct(
    question,
    chat_history="",
    llm_model_name=DEFAULT_LLM_MODEL,
    memory_summary="",
):
    model_name = resolve_llm_model(llm_model_name)
    prompt = _build_general_prompt(question, chat_history=chat_history, memory_summary=memory_summary)
    fallback_prompt = prompt if _RAW_PROMPT else question
    return _stream_with_fallback(prompt, model_name, fallback_prompt=fallback_prompt)


def get_recent_conversation_history(session_id, limit=3):
    session_id = _normalize_session_id(session_id)
    if session_id is None:
        return []

    messages = ChatMessage.objects.filter(session_id=session_id).order_by("-created_at")[:limit]
    return list(reversed(messages))


def get_or_create_conversation_memory(session_id):
    session_id = _normalize_session_id(session_id)
    if session_id is None:
        return None

    try:
        return ConversationMemory.objects.get(session_id=session_id)
    except ConversationMemory.DoesNotExist:
        try:
            session = ChatSession.objects.get(id=session_id)
        except ChatSession.DoesNotExist:
            return None

        return ConversationMemory.objects.create(
            session=session,
            memory_type="summary",
        )


def _extract_key_facts_quick(messages):
    default_facts = {"entities": [], "facts": [], "numbers": []}
    if not messages:
        return default_facts

    entity_pattern = re.compile(r"\b(?:[A-ZÀ-Ỵ][\wÀ-ỹ]+(?:\s+[A-ZÀ-Ỵ][\wÀ-ỹ]+)+)\b")
    number_pattern = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")

    entities = []
    facts = []
    numbers = []
    seen_facts = set()

    for msg in messages:
        text = (msg.content or "").strip()
        if not text:
            continue

        for entity in entity_pattern.findall(text):
            if entity not in entities:
                entities.append(entity)

        for number in number_pattern.findall(text):
            if number not in numbers:
                numbers.append(number)

        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = sentence.strip(" -•\t")
            if len(sentence) < 24 or len(sentence) > 180:
                continue
            normalized = re.sub(r"\s+", " ", sentence)
            if normalized in seen_facts:
                continue
            seen_facts.add(normalized)
            facts.append(normalized)
            if len(facts) >= 5:
                break

        if len(facts) >= 5:
            break

    return {
        "entities": entities[:10],
        "facts": facts,
        "numbers": numbers[:10],
    }


def extract_key_facts_from_conversation(messages, llm_model_name=DEFAULT_LLM_MODEL):
    try:
        if not messages:
            return {"entities": [], "facts": [], "numbers": []}
        user_messages = [msg for msg in messages if msg.role == "user"]
        return _extract_key_facts_quick(user_messages or messages)
    except Exception as exc:
        _log_warning("Extract facts failed: %s", exc)
        return {"entities": [], "facts": [], "numbers": []}


def update_conversation_memory(session_id, force_update=False):
    session_id = _normalize_session_id(session_id)
    if session_id is None:
        return None

    messages = list(
        ChatMessage.objects.filter(session_id=session_id)
        .order_by("-created_at")[:3]
    )
    messages.reverse()

    if not messages:
        return None

    memory = get_or_create_conversation_memory(session_id)
    if memory is None:
        return None

    user_messages = [msg for msg in messages if msg.role == "user"]
    summary_source = user_messages or messages
    summary = _format_recent_messages(summary_source, max_chars=400)
    if summary:
        memory.summary = summary

    facts = extract_key_facts_from_conversation(messages)
    if facts:
        memory.key_facts = json.dumps(facts, ensure_ascii=False)

    memory.save(update_fields=["summary", "key_facts", "last_updated"])
    return memory


def ask_gemma_with_memory(
    question,
    session_id,
    llm_model_name=DEFAULT_LLM_MODEL,
    embedding_model_name="",
    vector_db_key="",
    use_memory_augmentation=True,
    is_rag_mode=False,
    user_key=None,
):
    model_name = resolve_llm_model(llm_model_name)
    user_profile = get_user_profile_summary(user_key) if user_key else ""

    def _merge_memory(memory_summary):
        if user_profile:
            return f"{user_profile}\n{memory_summary}" if memory_summary else user_profile
        return memory_summary

    if not is_rag_mode:
        chat_history = _build_general_chat_history(session_id) if use_memory_augmentation else ""
        memory_summary = _get_memory_summary(session_id) if use_memory_augmentation else ""
        memory_summary = _merge_memory(memory_summary)
        return ask_llm_direct(
            question=question,
            chat_history=chat_history,
            llm_model_name=model_name,
            memory_summary=memory_summary,
        )

    if not embedding_model_name or not vector_db_key:
        chat_history = _build_general_chat_history(session_id) if use_memory_augmentation else ""
        memory_summary = _get_memory_summary(session_id) if use_memory_augmentation else ""
        memory_summary = _merge_memory(memory_summary)
        return ask_llm_direct(
            question=question,
            chat_history=chat_history,
            llm_model_name=model_name,
            memory_summary=memory_summary,
        )

    vector_store = get_cached_vector_store(vector_db_key, embedding_model_name)
    if vector_store is None:
        chat_history = _build_general_chat_history(session_id) if use_memory_augmentation else ""
        memory_summary = _get_memory_summary(session_id) if use_memory_augmentation else ""
        memory_summary = _merge_memory(memory_summary)
        return ask_llm_direct(
            question=question,
            chat_history=chat_history,
            llm_model_name=model_name,
            memory_summary=memory_summary,
        )

    chat_history = _build_general_chat_history(session_id) if use_memory_augmentation else ""
    memory_summary = _get_memory_summary(session_id) if use_memory_augmentation else ""
    memory_summary = _merge_memory(memory_summary)
    context = _retrieve_context(vector_store, question, k_chunks=_MAX_RAG_CHUNKS)
    prompt = _build_rag_prompt(question, context, chat_history=chat_history, memory_summary=memory_summary)
    fallback_prompt = prompt if _RAW_PROMPT else (f"{context}\n\n{question}" if context else question)
    return _stream_with_fallback(prompt, model_name, fallback_prompt=fallback_prompt)
