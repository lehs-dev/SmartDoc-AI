import os
import re
import queue
import time
import threading
import pdfplumber
import docx
import ollama
import json
from collections import OrderedDict
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from .models import ChatMessage, ChatSession, ConversationMemory

os.environ['HF_HUB_OFFLINE'] = '1'

# Cấu hình số luồng cho xử lý Vector/Toán học trên CPU
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# PyMuPDF cho đọc PDF nhanh (fallback: pdfplumber)
try:
    import fitz as _fitz
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

# json_repair cho parse JSON an toàn từ LLM output
try:
    import json_repair as _json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

VECTOR_DB_BASE_PATH = "vector_store"

DEFAULT_LLM_MODEL = "gemma4:e4b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_VECTOR_DB_KEY = "nomic_v1_db"

SUPPORTED_LLM_MODELS = [DEFAULT_LLM_MODEL]
SUPPORTED_EMBEDDING_MODELS = [DEFAULT_EMBEDDING_MODEL]

VECTOR_DB_CONFIG = {
    DEFAULT_VECTOR_DB_KEY: {
        "path": os.path.join(VECTOR_DB_BASE_PATH, DEFAULT_VECTOR_DB_KEY),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    },
}

_embedding_model_cache = {}
_llm_model_cache = {}
_vector_store_cache = {}
_installed_ollama_models_cache = None


def check_ollama_connection():
    """
    Kiểm tra kết nối Ollama
    Memory-Augmented RAG: Health check
    
    Returns:
        bool: True nếu Ollama đang chạy
    """
    try:
        print(f"\n🔍 [OLLAMA] Đang kiểm tra kết nối...")
        response = ollama.list()
        models = get_installed_ollama_models()
        print(f"✅ [OLLAMA] Kết nối thành công!")
        print(f"📦 Số models đã pull: {len(models)}")
        print(f"📋 Models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
        return True
    except Exception as e:
        print(f"❌ [OLLAMA] Không thể kết nối: {str(e)}")
        print(f"💡 Hint: Chạy 'ollama serve' hoặc kiểm tra Ollama đang chạy")
        return False
_memory_cache = OrderedDict()  # LRU cache cho memory
_memory_cache_max_size = 100  # Giới hạn số lượng memory items


def _extract_model_name(model_item):
    if isinstance(model_item, dict):
        return model_item.get("model") or model_item.get("name")

    return getattr(model_item, "model", None) or getattr(model_item, "name", None)


def _normalize_session_id(session_id):
    try:
        return int(session_id)
    except (TypeError, ValueError):
        return None


def _is_small_cpu_model(model_name):
    model_name = (model_name or '').lower()
    return 'e2b' in model_name or 'e4b' in model_name or '0.8b' in model_name


def _is_simple_question(question, max_words=18, max_chars=120):
    question = (question or '').strip()
    if not question:
        return False
    if '\n' in question:
        return False
    return len(question) <= max_chars and len(question.split()) <= max_words


def _truncate_text(text, max_chars):
    text = text or ''
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + '...'


def _format_recent_messages(messages, max_chars=1200):
    lines = []
    total_chars = 0

    for msg in messages:
        line = f"{msg.get_role_display()}: {msg.content}"
        if lines and total_chars + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def _build_small_general_prompt(question, chat_history=''):
    prompt_parts = [
        "Bạn là trợ lý AI tiếng Việt.",
        "Trả lời thật ngắn gọn.",
    ]

    if chat_history:
        prompt_parts.append(f"Ngữ cảnh gần nhất:\n{chat_history}")

    prompt_parts.append(f"Hỏi: {question}\nĐáp:")
    return "\n\n".join(prompt_parts)


def _build_small_rag_prompt(question, context, chat_history=''):
    prompt_parts = [
        "Bạn là SmartDoc AI.",
        "Ưu tiên ngữ cảnh; thiếu thì trả lời ngắn gọn.",
    ]

    if context:
        prompt_parts.append(f"Ngữ cảnh:\n{context}")

    if chat_history:
        prompt_parts.append(f"Lịch sử:\n{chat_history}")

    prompt_parts.append(f"Hỏi: {question}\nĐáp:")
    return "\n\n".join(prompt_parts)


def _build_ollama_options(model_name, *, streaming=False):
    _num_thread = min(10, max(1, (os.cpu_count() or 4) - 2))

    if _is_small_cpu_model(model_name):
        _num_thread = min(6, max(2, (os.cpu_count() or 4) // 2))
        return {
            'temperature': 0.1,
            'repeat_penalty': 1.05,
            'top_k': 30,
            'top_p': 0.8,
            'num_ctx': 1024,
            'num_predict': 192,
            'num_thread': _num_thread,
        }

    return {
        'temperature': 0.4,
        'num_ctx': 2048,
        'num_predict': 192 if streaming else 256,
        'num_thread': _num_thread,
    }


def _build_llm_kwargs(resolved_model_name):
    model_kwargs = {
        'model': resolved_model_name,
        'callbacks': [StreamingStdOutCallbackHandler()],
        'keep_alive': '15m',
    }
    model_kwargs.update(_build_ollama_options(resolved_model_name))
    return model_kwargs


def _extract_ollama_text(response_item, strip_text=True):
    def _normalize(value):
        if value is None:
            return ''
        text = str(value)
        return text.strip() if strip_text else text

    if isinstance(response_item, dict):
        message = response_item.get('message') or {}
        if isinstance(message, dict):
            content = _normalize(message.get('content'))
            if content:
                return content
        else:
            content = _normalize(getattr(message, 'content', None))
            if content:
                return content

        response_text = response_item.get('response') or response_item.get('content')
        return _normalize(response_text)

    message = getattr(response_item, 'message', None)
    if isinstance(message, dict):
        content = _normalize(message.get('content'))
        if content:
            return content

    if message is not None:
        content = _normalize(getattr(message, 'content', None))
        if content:
            return content

    response_text = getattr(response_item, 'response', None) or getattr(response_item, 'content', None)
    return _normalize(response_text)


def _ollama_stream(prompt, model_name, mode='chat'):
    if mode == 'chat':
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options=_build_ollama_options(model_name, streaming=True),
            keep_alive='15m',
            stream=True,
        )
    else:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options=_build_ollama_options(model_name, streaming=True),
            keep_alive='15m',
            stream=True,
        )

    for chunk in response:
        content = _extract_ollama_text(chunk, strip_text=False)
        if content != '':
            yield content


def _ollama_invoke(prompt, model_name, mode='chat'):
    if mode == 'chat':
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options=_build_ollama_options(model_name),
            keep_alive='15m',
            stream=False,
        )
    else:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options=_build_ollama_options(model_name),
            keep_alive='15m',
            stream=False,
        )

    return _extract_ollama_text(response)


def _stream_text_chunks(text, chunk_size=120):
    text = text or ''
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def _stream_with_fallback(primary_stream, fallback_stream_factory=None, fallback_label=''):
    has_output = False

    for chunk in primary_stream:
        if not chunk:
            continue
        has_output = True
        yield chunk

    if not has_output and fallback_stream_factory:
        if fallback_label:
            print(f"⚠️  [LLM] Stream trống, fallback sang {fallback_label}")
        for chunk in fallback_stream_factory():
            if chunk:
                yield chunk


def _stream_with_timeouts(stream_iterator, idle_timeout_seconds=18, max_duration_seconds=90):
    stream_queue = queue.Queue()
    done_sentinel = object()

    def _produce():
        try:
            for chunk in stream_iterator:
                stream_queue.put(('chunk', chunk))
        except Exception as exc:
            stream_queue.put(('error', exc))
        finally:
            stream_queue.put(('done', done_sentinel))

    producer = threading.Thread(target=_produce, daemon=True)
    producer.start()

    started_at = time.monotonic()

    while True:
        elapsed = time.monotonic() - started_at
        remaining = max_duration_seconds - elapsed
        if remaining <= 0:
            raise TimeoutError('Model stream vượt thời gian tối đa cho phép')

        wait_timeout = min(idle_timeout_seconds, remaining)

        try:
            event, payload = stream_queue.get(timeout=wait_timeout)
        except queue.Empty as exc:
            raise TimeoutError('Model stream bị treo do không có token mới') from exc

        if event == 'chunk':
            yield payload
            continue

        if event == 'error':
            raise payload

        if event == 'done':
            break


def _invoke_with_timeout(invoke_fn, timeout_seconds=40):
    result_queue = queue.Queue(maxsize=1)

    def _run_invoke():
        try:
            result_queue.put(('ok', invoke_fn()))
        except Exception as exc:
            result_queue.put(('error', exc))

    worker = threading.Thread(target=_run_invoke, daemon=True)
    worker.start()

    try:
        status, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        return ''

    if status == 'error':
        raise payload

    return payload or ''


def _stream_small_model_with_fallback(prompt, model_name, fallback_model=None):
    has_output = False

    try:
        for chunk in _stream_with_timeouts(
            _ollama_stream(prompt, model_name, mode='generate'),
            idle_timeout_seconds=12,
            max_duration_seconds=60,
        ):
            if not chunk:
                continue
            has_output = True
            yield chunk
    except TimeoutError as exc:
        print(f"⚠️  [LLM] Stream model nhỏ bị timeout: {exc}")
    except Exception as exc:
        print(f"⚠️  [LLM] Stream model nhỏ lỗi, thử fallback: {exc}")

    if has_output:
        return

    print("⚠️  [LLM] Stream trống, thử gọi non-stream...")
    try:
        response_text = _invoke_with_timeout(
            lambda: _ollama_invoke(prompt, model_name, mode='generate'),
            timeout_seconds=25,
        )
    except Exception as exc:
        print(f"⚠️  [LLM] Non-stream model nhỏ lỗi: {exc}")
        response_text = ''

    if response_text:
        for chunk in _stream_text_chunks(response_text):
            yield chunk
        return

    if fallback_model:
        print(f"⚠️  [LLM] Non-stream rỗng, fallback sang {fallback_model}")
        has_output = False
        try:
            for chunk in _stream_with_timeouts(
                _ollama_stream(prompt, fallback_model, mode='chat'),
                idle_timeout_seconds=14,
                max_duration_seconds=70,
            ):
                if not chunk:
                    continue
                has_output = True
                yield chunk
        except TimeoutError as exc:
            print(f"⚠️  [LLM] Fallback stream timeout: {exc}")
        except Exception as exc:
            print(f"⚠️  [LLM] Fallback stream lỗi: {exc}")

        if has_output:
            return

    yield (
        "Xin lỗi, hiện tại model nhỏ chưa phản hồi ổn định. "
        "Vui lòng thử lại với câu ngắn hơn hoặc chuyển tạm sang model mạnh hơn."
    )


def _build_general_chat_history(session_id, small_model=False, simple_question=False):
    history_limit = 1 if small_model else 2
    max_chars = 240 if small_model else 600
    recent_messages = get_recent_conversation_history(session_id, limit=history_limit)
    return _format_recent_messages(recent_messages, max_chars=max_chars)


def _build_small_rag_context(retrieval_result, simple_question=False):
    doc_chunks = retrieval_result.get('document_chunks') or []
    context = "\n\n".join([doc.page_content for doc in doc_chunks])
    return _truncate_text(context, 360)


def _ollama_chat_invoke(prompt, model_name):
    return _ollama_invoke(prompt, model_name, mode='chat')


def _ollama_generate_invoke(prompt, model_name):
    return _ollama_invoke(prompt, model_name, mode='generate')


def _ollama_chat_stream(prompt, model_name):
    return _ollama_stream(prompt, model_name, mode='chat')


def _ollama_generate_stream(prompt, model_name):
    return _ollama_stream(prompt, model_name, mode='generate')


def _extract_key_facts_quick(messages):
    default_facts = {"entities": [], "facts": [], "numbers": []}
    if not messages:
        return default_facts

    entity_pattern = re.compile(r'\b(?:[A-ZÀ-Ỵ][\wÀ-ỹ]+(?:\s+[A-ZÀ-Ỵ][\wÀ-ỹ]+)+)\b')
    number_pattern = re.compile(r'\b\d+(?:[.,:/-]\d+)*\b')

    entities = []
    facts = []
    numbers = []
    seen_facts = set()

    for msg in messages:
        text = (msg.content or '').strip()
        if not text:
            continue

        for entity in entity_pattern.findall(text):
            if entity not in entities:
                entities.append(entity)

        for number in number_pattern.findall(text):
            if number not in numbers:
                numbers.append(number)

        for sentence in re.split(r'(?<=[.!?])\s+|\n+', text):
            sentence = sentence.strip(' -•\t')
            if len(sentence) < 24 or len(sentence) > 180:
                continue
            normalized = re.sub(r'\s+', ' ', sentence)
            if normalized in seen_facts:
                continue
            seen_facts.add(normalized)
            facts.append(normalized)
            if len(facts) >= 5:
                break

        if len(facts) >= 5:
            break

    return {
        'entities': entities[:10],
        'facts': facts,
        'numbers': numbers[:10],
    }


def get_installed_ollama_models(refresh=False):
    global _installed_ollama_models_cache

    if _installed_ollama_models_cache is not None and not refresh:
        return list(_installed_ollama_models_cache)

    try:
        response = ollama.list()
        model_items = []

        if isinstance(response, dict):
            model_items = response.get("models", [])
        else:
            model_items = getattr(response, "models", [])

        installed_models = []
        for item in model_items:
            model_name = _extract_model_name(item)
            if model_name:
                installed_models.append(model_name)

        _installed_ollama_models_cache = installed_models
    except Exception as e:
        print(f"Không lấy được danh sách model Ollama: {e}")
        _installed_ollama_models_cache = []

    return list(_installed_ollama_models_cache)


def _find_available_model(preferred_models):
    installed_models = get_installed_ollama_models()
    if not installed_models:
        return None

    installed_set = set(installed_models)
    for name in preferred_models:
        if name in installed_set:
            return name
    return None


def _get_llm_fallback_model(primary_model):
    return None


def _normalize_llm_model(model_name):
    if model_name not in SUPPORTED_LLM_MODELS:
        print(f"LLM model '{model_name}' không hỗ trợ, dùng mặc định {DEFAULT_LLM_MODEL}")
        return DEFAULT_LLM_MODEL
    return model_name


def _normalize_embedding_model(model_name):
    if model_name not in SUPPORTED_EMBEDDING_MODELS:
        print(f"Embedding model '{model_name}' không hỗ trợ, dùng mặc định {DEFAULT_EMBEDDING_MODEL}")
        return DEFAULT_EMBEDDING_MODEL
    return model_name


def _normalize_vector_db_key(vector_db_key):
    if vector_db_key not in VECTOR_DB_CONFIG:
        print(f"Vector DB '{vector_db_key}' không hỗ trợ, dùng mặc định {DEFAULT_VECTOR_DB_KEY}")
        return DEFAULT_VECTOR_DB_KEY
    return vector_db_key


def resolve_llm_model(model_name):
    model_name = _normalize_llm_model(model_name)
    _validate_llm_model(model_name)

    available = _find_available_model([model_name])
    if available:
        return available

    raise ValueError(
        f"Model LLM '{model_name}' chưa có trong Ollama local. "
        "Vui lòng kiểm tra lại model đã pull về trước khi chat."
    )


def get_available_llm_models():
    installed_models = set(get_installed_ollama_models())
    if not installed_models:
        return list(SUPPORTED_LLM_MODELS)

    available = [model for model in SUPPORTED_LLM_MODELS if model in installed_models]
    return available or list(SUPPORTED_LLM_MODELS)


def get_available_embedding_models():
    return list(SUPPORTED_EMBEDDING_MODELS)


def _validate_llm_model(model_name):
    if model_name not in SUPPORTED_LLM_MODELS:
        raise ValueError(f"LLM model không được hỗ trợ: {model_name}")


def _validate_embedding_model(model_name):
    if model_name not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(f"Embedding model không được hỗ trợ: {model_name}")


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
    """
    Load embedding model từ Ollama với caching
    Memory-Augmented RAG: Embedding model management
    
    Args:
        model_name: Tên embedding model
    
    Returns:
        OllamaEmbeddings instance
    """
    model_name = _normalize_embedding_model(model_name)
    _validate_embedding_model(model_name)
    
    if model_name not in _embedding_model_cache:
        print(f"\n🤖 [EMBEDDING] Đang nạp model vào RAM: {model_name}...")
        try:
            _embedding_model_cache[model_name] = OllamaEmbeddings(model=model_name)
            print(f"✅ [EMBEDDING] Model loaded thành công: {model_name}")
        except Exception as e:
            print(f"❌ [EMBEDDING] Lỗi khi load model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    return _embedding_model_cache[model_name]


def get_llm_model(model_name):
    resolved_model_name = resolve_llm_model(model_name)

    if resolved_model_name not in _llm_model_cache:
        print(f"Khởi tạo kết nối tới Ollama với model: {resolved_model_name}...")
        _llm_model_cache[resolved_model_name] = OllamaLLM(**_build_llm_kwargs(resolved_model_name))
        
    return _llm_model_cache[resolved_model_name]


def get_cached_vector_store(vector_db_key, embedding_model_name):
    """Load FAISS theo từng kho và embedding model, chỉ 1 lần cho mỗi cặp."""
    vector_db_key = _normalize_vector_db_key(vector_db_key)
    embedding_model_name = _normalize_embedding_model(embedding_model_name)
    cache_key = f"{vector_db_key}::{embedding_model_name}"
    if cache_key in _vector_store_cache:
        return _vector_store_cache[cache_key]

    print(f"Đang nạp Vector Database vào RAM: {vector_db_key}...")
    vector_db_path = resolve_vector_db_path(vector_db_key)
    index_path = os.path.join(vector_db_path, "index.faiss")

    if not os.path.exists(index_path):
        _vector_store_cache[cache_key] = None
        return None

    embeddings = get_embeddings_model(embedding_model_name)
    _vector_store_cache[cache_key] = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return _vector_store_cache[cache_key]


def extract_text(file_path, file_extension):
    """
    Extract text từ PDF hoặc DOCX
    Memory-Augmented RAG: Text extraction với logging chi tiết
    
    Args:
        file_path: Đường dẫn file
        file_extension: 'pdf' hoặc 'docx'
    
    Returns:
        Extracted text
    """
    print(f"\n📄 [EXTRACT] Đang trích xuất text từ: {file_path}")
    print(f"📎 File extension: {file_extension}")
    
    text = ""
    try:
        if file_extension == 'pdf':
            # Primary: PyMuPDF (nhanh hơn pdfplumber đáng kể)
            if _HAS_PYMUPDF:
                print(f"📕 Processing PDF với PyMuPDF (fast mode)...")
                try:
                    doc = _fitz.open(file_path)
                    print(f"📊 Số trang: {len(doc)}")
                    for i, page in enumerate(doc):
                        extracted = page.get_text()
                        if extracted:
                            text += extracted + "\n"
                            if i < 5:
                                print(f"  ✅ Trang {i+1}: {len(extracted)} ký tự")
                        else:
                            print(f"  ⚠️  Trang {i+1}: Không extract được text")
                    doc.close()
                except Exception as e:
                    print(f"⚠️  PyMuPDF lỗi, fallback sang pdfplumber: {e}")
                    text = ""  # Reset để fallback xử lý lại
            
            # Fallback: pdfplumber (chính xác hơn cho bảng biểu)
            if not text.strip():
                print(f"📕 Processing PDF với pdfplumber (accurate mode)...")
                with pdfplumber.open(file_path) as pdf:
                    print(f"📊 Số trang: {len(pdf.pages)}")
                    for i, page in enumerate(pdf.pages):
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                            if i < 5:
                                print(f"  ✅ Trang {i+1}: {len(extracted)} ký tự")
                        else:
                            print(f"  ⚠️  Trang {i+1}: Không extract được text")
                        
        elif file_extension == "docx":
            print(f"📘 Processing DOCX...")
            doc = docx.Document(file_path)
            print(f"📊 Số paragraphs: {len(doc.paragraphs)}")
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    text += para.text + "\n"
                    if i < 5:  # Chỉ log 5 paragraphs đầu
                        print(f"  ✅ Paragraph {i+1}: {len(para.text)} ký tự")
        
        total_chars = len(text)
        print(f"\n✅ [EXTRACT] Hoàn thành! Tổng: {total_chars} ký tự")
        
        if total_chars == 0:
            print(f"⚠️  [EXTRACT] Cảnh báo: Không extract được text nào!")
        
        return text
        
    except Exception as e:
        print(f"❌ [EXTRACT] Lỗi khi đọc file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

def _split_text_chunks(text, chunk_size, chunk_overlap, separators=None):
    splitter_kwargs = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'length_function': len,
    }
    if separators is not None:
        splitter_kwargs['separators'] = separators
        splitter_kwargs['is_separator_regex'] = False

    text_splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
    return text_splitter.split_text(text)


def get_text_chunks(text):
    return get_text_chunks_optimized(text)


def get_text_chunks_optimized(text, file_size_mb=0, has_vietnamese=False):
    """
    Chunking cố định cho tài liệu 1 trang A4, ưu tiên tốc độ trên CPU.
    
    Memory-Augmented RAG: Tối ưu chunking cho memory efficiency
    
    Args:
        text: Văn bản cần chunk
        file_size_mb: Kích thước file (MB)
        has_vietnamese: Có chứa tiếng Việt không
    
    Returns:
        list of text chunks
    """
    chunk_size = 600
    chunk_overlap = 60
    
    # Separators tối ưu cho tiếng Việt và English
    separators = [
        "\n\n",
        "\n",
        "!",
        "?",
        ".",
        " ",
        ""
    ]
    
    chunks = _split_text_chunks(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    
    print(f"Fixed chunking: {len(chunks)} chunks | size={chunk_size} | overlap={chunk_overlap}")
    
    return chunks

def process_document(file_path, file_extension):
    return process_document_optimized(file_path, file_extension)


def process_document_optimized(file_path, file_extension, file_size_mb=0, has_vietnamese=False):
    """
    Optimized document processing với fixed chunking cho 1 trang
    
    Args:
        file_path: Đường dẫn file
        file_extension: PDF hoặc DOCX
        file_size_mb: Kích thước file (MB)
        has_vietnamese: Có tiếng Việt không
    
    Returns:
        list of text chunks
    """
    print(f'Đang xử lý file (optimized): {file_path}...')
    raw_text = extract_text(file_path, file_extension)

    if not raw_text.strip():
        print(f"Không tìm thấy nội dung trong tài liệu")
        return []
    
    # Sử dụng fixed chunking
    chunks = get_text_chunks_optimized(raw_text, file_size_mb, has_vietnamese)
    
    print(f"Đã xử lý tài liệu thành {len(chunks)} đoạn (optimized)")
    return chunks

def get_vector_store(chunks, embedding_model_name, vector_db_key):
    """
    Tạo và lưu vector store vào FAISS
    Memory-Augmented RAG: Vectorization với logging chi tiết
    
    Args:
        chunks: List of text chunks
        embedding_model_name: Tên embedding model
        vector_db_key: Key của vector database
    
    Returns:
        FAISS vector store
    """
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU VECTORIZE DOCUMENT")
    print(f"{'='*60}")
    print(f"📊 Số chunks: {len(chunks)}")
    print(f"🤖 Embedding model: {embedding_model_name}")
    print(f"💾 Vector DB key: {vector_db_key}")
    
    try:
        vector_db_key = _normalize_vector_db_key(vector_db_key)
        embedding_model_name = _normalize_embedding_model(embedding_model_name)
        vector_db_path = resolve_vector_db_path(vector_db_key)
        print(f"📁 Vector DB path: {vector_db_path}")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(vector_db_path, exist_ok=True)
        print(f"✅ Đã tạo thư mục: {vector_db_path}")
        
        # Load embedding model
        print(f"\n⏳ Đang load embedding model...")
        embeddings = get_embeddings_model(embedding_model_name)
        print(f"✅ Embedding model loaded thành công")
        
        index_path = os.path.join(vector_db_path, "index.faiss")
        print(f"📄 Index path: {index_path}")
        
        # Tạo hoặc update vector store
        if os.path.exists(index_path):
            print(f"📌 Index đã tồn tại, đang merge thêm chunks...")
            vector_store = FAISS.load_local(
                vector_db_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vector_store.add_texts(chunks)
            print(f"✅ Đã merge {len(chunks)} chunks vào index cũ")
        else:
            print(f"📌 Index mới, đang tạo từ đầu...")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            print(f"✅ Đã tạo index mới với {len(chunks)} chunks")
        
        # Lưu vector store
        print(f"\n💾 Đang lưu vector store vào disk...")
        vector_store.save_local(vector_db_path)
        print(f"✅ Lưu thành công vào: {vector_db_path}")
        
        # Update cache
        cache_key = f"{vector_db_key}::{embedding_model_name}"
        _vector_store_cache[cache_key] = vector_store
        print(f"✅ Update cache: {cache_key}")
        
        print(f"\n{'='*60}")
        print(f"✅ VECTORIZE HOÀN THÀNH!")
        print(f"{'='*60}\n")
        
        return vector_store
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ LỖI VECTORIZE: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        raise


def ask_gemma(
    question,
    chat_history="",
    llm_model_name=DEFAULT_LLM_MODEL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    vector_db_key=DEFAULT_VECTOR_DB_KEY,
):
    print('Đang tìm kiếm thông tin cho câu hỏi...')

    if not embedding_model_name or not vector_db_key:
        return iter([
            "Xin lỗi, tôi chưa xác định được kho dữ liệu cho tài liệu này. "
            "Vui lòng tải lại tài liệu hoặc chọn tài liệu khác."
        ])

    # Dùng DB đã cache trong RAM thay vì đọc ổ cứng
    vector_store = get_cached_vector_store(vector_db_key, embedding_model_name)

    if vector_store is None:
        return iter([
            "Xin lỗi, kho dữ liệu của tài liệu này chưa có nội dung. "
            "Vui lòng tải tài liệu lên trước."
        ])

    # Re-ranking: Lấy ít docs để tiết kiệm context
    relevant_doc = _rerank_documents(vector_store, question, k_initial=4, k_final=1)

    context  = "\n\n".join([doc.page_content for doc in relevant_doc])

    prompt_template =  """Bạn là SmartDoc AI.
Lịch sử:
{chat_history}

Ngữ cảnh:
{context}

Hỏi: {question}
Đáp:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history" ,"context", "question"])

    print('Gemma 4 đang suy nghĩ...')
    llm = get_llm_model(llm_model_name)
    chain = prompt | llm

    return chain.stream({
        "chat_history": _truncate_text(chat_history, 300),
        "context" : _truncate_text(context, 600),
        "question" : question
    })


# ============================================================================
# GENERAL CHAT MODE - DIRECT LLM (NO RAG)
# ============================================================================

def ask_llm_direct(
    question,
    chat_history="",
    llm_model_name=DEFAULT_LLM_MODEL,
):
    """
    General Chat Mode - Hỏi đáp trực tiếp với LLM không cần RAG
    Sử dụng khi không có tài liệu nào được upload
    
    Args:
        question: Câu hỏi của user
        chat_history: Lịch sử chat (optional)
        llm_model_name: Tên LLM model
    
    Returns:
        Stream generator từ LLM
    """
    print('\n💬 [GENERAL CHAT] Đang trả lời không cần RAG...')

    is_small_model = _is_small_cpu_model(llm_model_name)

    if is_small_model:
        compact_history = _truncate_text(chat_history, 240)
        formatted_prompt = _build_small_general_prompt(question, compact_history)
        print(f'🤖 [GENERAL CHAT] Sử dụng model compact: {llm_model_name}')
        fallback_model = _get_llm_fallback_model(llm_model_name)
        return _stream_small_model_with_fallback(
            formatted_prompt,
            llm_model_name,
            fallback_model,
        )

    prompt_template = """Bạn là SmartDoc AI.
Ngữ cảnh gần đây:
{chat_history}

Hỏi: {question}
Đáp:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "question"])
    
    print(f'🤖 [GENERAL CHAT] Sử dụng model: {llm_model_name}')
    formatted_prompt = prompt.format(
        chat_history=_truncate_text(chat_history, 600),
        question=question,
    )

    llm = get_llm_model(llm_model_name)
    chain = prompt | llm
    
    return chain.stream({
        "chat_history": _truncate_text(chat_history, 600),
        "question": question
    })


# ============================================================================
# MEMORY-AUGMENTED RAG FUNCTIONS
# ============================================================================

def get_memory_cache_max_size():
    """Trả về kích thước tối đa của memory cache"""
    return _memory_cache_max_size


def update_memory_cache(session_id, memory_data):
    """
    Cập nhật memory cache với LRU eviction
    Memory-Augmented RAG: Quản lý bộ nhớ đệm cho hội thoại
    """
    global _memory_cache
    
    # Nếu đã tồn tại, move to end (most recently used)
    if session_id in _memory_cache:
        _memory_cache.move_to_end(session_id)
    else:
        # Evict oldest nếu vượt quá giới hạn
        while len(_memory_cache) >= _memory_cache_max_size:
            _memory_cache.popitem(last=False)
    
    _memory_cache[session_id] = memory_data
    return memory_data


def get_recent_conversation_history(session_id, limit=3):
    """
    Lấy lịch sử hội thoại gần đây (Short-term Memory)
    Memory-Augmented RAG: Conversation Buffer Memory
    """
    session_id = _normalize_session_id(session_id)
    if session_id is None:
        return []

    messages = ChatMessage.objects.filter(
        session_id=session_id
    ).order_by('-created_at')[:limit]
    
    # Đảo ngược để theo thứ tự thời gian
    return list(reversed(messages))


def get_or_create_conversation_memory(session_id):
    """
    Lấy hoặc tạo ConversationMemory (Long-term Memory)
    Memory-Augmented RAG: Summary Memory
    """
    session_id = _normalize_session_id(session_id)
    if session_id is None:
        print(f"Lỗi khi lấy memory: session_id không hợp lệ ({session_id})")
        return None

    try:
        memory = ConversationMemory.objects.get(session_id=session_id)
        return memory
    except ConversationMemory.DoesNotExist:
        # Tạo memory mới nếu chưa tồn tại
        try:
            session = ChatSession.objects.get(id=session_id)
        except ChatSession.DoesNotExist:
            print(f"Lỗi khi tạo memory: session {session_id} không tồn tại")
            return None

        memory = ConversationMemory.objects.create(
            session=session,
            memory_type='summary'
        )
        return memory


def compress_conversation_to_summary(messages, llm_model_name="gemma4:e2b"):
    """
    Nén lịch sử hội thoại thành summary bằng LLM
    Memory-Augmented RAG: Memory Compression
    """
    if not messages:
        return ""
    
    # Format conversation
    conversation_text = "\n".join([
        f"{msg.role}: {msg.content}" for msg in messages
    ])
    
    prompt_template = """
Bạn là trợ lý AI. Hãy tóm tắt hội thoại sau thành các ý chính quan trọng.
Chỉ giữ lại: sự kiện, thông tin, con số, tên riêng, khái niệm quan trọng.
Bỏ qua: lời chào hỏi, câu hỏi lặp, thông tin không quan trọng.

Hội thoại:
{conversation}

Tóm tắt (tiếng Việt, ngắn gọn):
"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["conversation"]
    )
    
    try:
        if _is_small_cpu_model(llm_model_name):
            recent_lines = []
            for msg in list(messages)[-4:]:
                recent_lines.append(f"{msg.role}: {msg.content}")
            summary = _truncate_text("\n".join(recent_lines), 600)
        else:
            llm = get_llm_model(llm_model_name)
            chain = prompt | llm
            summary = chain.invoke({"conversation": conversation_text})
        return summary.strip()
    except Exception as e:
        print(f"Lỗi khi nén bộ nhớ: {e}")
        return ""


def _safe_parse_json(text, fallback=None):
    """
    Parse JSON an toàn từ LLM output.
    Tự động loại bỏ markdown code blocks và sửa lỗi JSON nhỏ.
    """
    if not text or not text.strip():
        return fallback
    
    text = text.strip()
    
    # Loại bỏ markdown code blocks bằng regex
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        text = json_match.group(1).strip()
    
    # Thử parse trực tiếp
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Thử sửa lỗi JSON bằng json_repair
    if _HAS_JSON_REPAIR:
        try:
            result = _json_repair.loads(text)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    
    # Fallback: Cố gắng tìm JSON object trong text
    json_obj_match = re.search(r'\{[\s\S]*\}', text)
    if json_obj_match:
        try:
            return json.loads(json_obj_match.group(0))
        except json.JSONDecodeError:
            if _HAS_JSON_REPAIR:
                try:
                    return _json_repair.loads(json_obj_match.group(0))
                except Exception:
                    pass
    
    return fallback


def extract_key_facts_from_conversation(messages, llm_model_name="gemma4:e2b"):
    """
    Trích xuất các sự kiện quan trọng từ hội thoại
    Memory-Augmented RAG: Entity Memory
    """
    _default_facts = {"entities": [], "facts": [], "numbers": []}

    try:
        return _extract_key_facts_quick(messages) if messages else _default_facts
    except Exception as e:
        print(f"Lỗi khi trích xuất sự kiện: {e}")
        return dict(_default_facts)


def update_conversation_memory(session_id, force_update=False):
    """
    Cập nhật ConversationMemory từ lịch sử hội thoại
    Memory-Augmented RAG: Memory Update Strategy
    """
    session_id = _normalize_session_id(session_id)
    if session_id is None:
        print(f"Lỗi khi update memory: session_id không hợp lệ ({session_id})")
        return None

    messages = list(ChatMessage.objects.filter(
        session_id=session_id
    ).order_by('-created_at')[:3])
    messages.reverse()

    if not messages:
        return None
    
    # Lấy memory hiện tại
    memory = get_or_create_conversation_memory(session_id)
    if memory is None:
        return None
    
    summary = _format_recent_messages(messages, max_chars=400)
    if summary:
        memory.summary = summary
    
    # Trích xuất key facts theo cách nhẹ, không gọi thêm LLM
    facts = extract_key_facts_from_conversation(messages)
    if facts:
        memory.key_facts = json.dumps(facts, ensure_ascii=False)
    
    memory.save(update_fields=['summary', 'key_facts', 'last_updated'])
    
    # Update cache
    update_memory_cache(session_id, {
        'summary': memory.summary,
        'key_facts': memory.key_facts,
        'user_preferences': memory.user_preferences
    })
    
    return memory


def _rerank_documents(vector_store, question, k_initial=10, k_final=2):
    """
    Retrieval nhẹ: lấy trực tiếp top-k theo similarity, không rerank hai bước.
    """
    try:
        k = max(1, min(k_initial, k_final))
        docs = vector_store.similarity_search(question, k=k)
        print(f"🔄 [RETRIEVE] top {k} docs")
        return docs
    except Exception as e:
        print(f"⚠️  [RE-RANK] Lỗi, fallback về retriever thường: {e}")
        retriever = vector_store.as_retriever(search_kwargs={'k': max(1, k_final)})
        return retriever.invoke(question)


def retrieve_with_memory_augmentation(
    question,
    session_id,
    vector_store,
    k_chunks=1,
    k_memories=1
):
    """
    Memory-Augmented Retrieval: Kết hợp retrieval từ nhiều nguồn
    1. Short-term: Last N messages (Conversation Buffer)
    2. Semantic: FAISS document chunks (Semantic Memory)
    
    Returns:
        dict: {
            'recent_messages': list of ChatMessage,
            'document_chunks': list of Document chunks,
            'combined_context': str (tất cả context)
        }
    """
    result = {
        'recent_messages': [],
        'document_chunks': [],
        'combined_context': ''
    }

    session_id = _normalize_session_id(session_id)
    if session_id is None:
        return result
    
    # 1. Get short-term memory (recent messages)
    recent_messages = get_recent_conversation_history(session_id, limit=3)
    result['recent_messages'] = recent_messages

    # 2. Get semantic memory (FAISS retrieval with top-k)
    if vector_store:
        relevant_docs = _rerank_documents(vector_store, question, k_initial=k_chunks, k_final=k_chunks)
        result['document_chunks'] = relevant_docs

    # 3. Build combined context
    context_parts = []
    
    # Add document chunks
    if result['document_chunks']:
        doc_context = "\n\n".join([doc.page_content for doc in result['document_chunks']])
        context_parts.append(doc_context)
    
    result['combined_context'] = "\n\n".join(context_parts)
    
    return result


# ============================================================================
# MEMORY-AUGMENTED RAG: Main entry point
# ============================================================================

def ask_gemma_with_memory(
    question,
    session_id,
    llm_model_name=DEFAULT_LLM_MODEL,
    embedding_model_name="",
    vector_db_key="",
    use_memory_augmentation=True,
    is_rag_mode=False
):
    """
    Memory-Augmented RAG: Hỏi đáp với memory augmentation
    Thay thế cho ask_gemma() truyền thống
    
    Args:
        question: Câu hỏi của user
        session_id: ID của chat session
        llm_model_name: Tên LLM model
        embedding_model_name: Tên embedding model
        vector_db_key: Key của vector DB
        use_memory_augmentation: Có sử dụng memory augmentation không
        is_rag_mode: True nếu đang ở chế độ RAG (có document), False nếu general chat
    
    Returns:
        Stream generator từ LLM
    """
    is_small_model = _is_small_cpu_model(llm_model_name)
    simple_question = _is_simple_question(question)

    # Nếu không ở RAG mode, fallback về general chat
    if not is_rag_mode:
        print('\n🔹 [MODE] General Chat - Không dùng RAG')
        chat_history = _build_general_chat_history(
            session_id,
            small_model=is_small_model,
            simple_question=simple_question,
        )
        return ask_llm_direct(
            question=question,
            chat_history=chat_history,
            llm_model_name=llm_model_name
        )

    print('Đang tìm kiếm thông tin cho câu hỏi (Memory-Augmented RAG)...')

    if not embedding_model_name or not vector_db_key:
        # Fallback về general chat nếu không có embedding info
        print('⚠️  [RAG] Không có embedding info, fallback về general chat')
        chat_history = _build_general_chat_history(
            session_id,
            small_model=is_small_model,
            simple_question=simple_question,
        )
        return ask_llm_direct(
            question=question,
            chat_history=chat_history,
            llm_model_name=llm_model_name
        )

    # Load vector store
    vector_store = get_cached_vector_store(vector_db_key, embedding_model_name)

    if vector_store is None:
        # Fallback về general chat nếu không có vector store
        print('⚠️  [RAG] Không có vector store, fallback về general chat')
        chat_history = _build_general_chat_history(
            session_id,
            small_model=is_small_model,
            simple_question=simple_question,
        )
        return ask_llm_direct(
            question=question,
            chat_history=chat_history,
            llm_model_name=llm_model_name
        )

    # Memory-Augmented Retrieval
    if use_memory_augmentation:
        k_chunks = 1
        retrieval_result = retrieve_with_memory_augmentation(
            question=question,
            session_id=session_id,
            vector_store=vector_store,
            k_chunks=k_chunks,
            k_memories=1
        )

        recent_messages = retrieval_result['recent_messages']
        if is_small_model:
            chat_history = _format_recent_messages(recent_messages, max_chars=240)
            context = _build_small_rag_context(retrieval_result, simple_question=simple_question)
        else:
            chat_history = _format_recent_messages(recent_messages, max_chars=600)
            context = _truncate_text(retrieval_result['combined_context'], 1200)

        # Update memory sau khi retrieve (async, không block)
        try:
            update_conversation_memory(session_id)
        except Exception as e:
            print(f"Lỗi khi update memory: {e}")

    else:
        # Fallback: Không dùng memory (như ask_gemma cũ)
        retriever = vector_store.as_retriever(search_kwargs={'k': 2})
        relevant_docs = retriever.invoke(question)
        raw_context = "\n\n".join([doc.page_content for doc in relevant_docs])
        context = _truncate_text(raw_context, 360 if is_small_model else 1200)
        chat_history = ""

    if is_small_model:
        safe_chat_history = _truncate_text(chat_history, 240)
        safe_context = _truncate_text(context, 360)
        formatted_prompt = _build_small_rag_prompt(question, safe_context, safe_chat_history)
        fallback_model = _get_llm_fallback_model(llm_model_name)
        return _stream_small_model_with_fallback(
            formatted_prompt,
            llm_model_name,
            fallback_model,
        )

    # Build prompt với memory context
    prompt_template = """Bạn là SmartDoc AI.
Lịch sử:
{chat_history}

Ngữ cảnh:
{context}

Hỏi: {question}
Đáp:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])

    safe_chat_history = _truncate_text(chat_history, 600)
    safe_context = _truncate_text(context, 1200)

    print('Gemma 4 đang suy nghĩ (với memory context)...')
    llm = get_llm_model(llm_model_name)
    chain = prompt | llm
    
    return chain.stream({
        "chat_history": safe_chat_history,
        "context": safe_context,
        "question": question
    })