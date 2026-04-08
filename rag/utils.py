import os
import pdfplumber
import docx
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

os.environ['HF_HUB_OFFLINE'] = '1'
VECTOR_DB_BASE_PATH = "vector_store"

SUPPORTED_LLM_MODELS = [
    "gemma4:e2b",
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
]

SUPPORTED_EMBEDDING_MODELS = [
    "nomic-embed-text-v2-moe",
    "qwen3-embedding:0.6b",
    "nomic-embed-text",
    "bge-m3:567m",
]

VECTOR_DB_CONFIG = {
    "qwen_db": {
        "path": os.path.join(VECTOR_DB_BASE_PATH, "qwen_db"),
        "embedding_model": "qwen3-embedding:0.6b",
    },
    "bge_db": {
        "path": os.path.join(VECTOR_DB_BASE_PATH, "bge_db"),
        "embedding_model": "bge-m3:567m",
    },
    "nomic_v2_db": {
        "path": os.path.join(VECTOR_DB_BASE_PATH, "nomic_v2_db"),
        "embedding_model": "nomic-embed-text-v2-moe",
    },
    "nomic_v1_db": {
        "path": os.path.join(VECTOR_DB_BASE_PATH, "nomic_v1_db"),
        "embedding_model": "nomic-embed-text",
    },
}

_embedding_model_cache = {}
_llm_model_cache = {}
_vector_store_cache = {}


def get_available_llm_models():
    return list(SUPPORTED_LLM_MODELS)


def get_available_embedding_models():
    return list(SUPPORTED_EMBEDDING_MODELS)


def _validate_llm_model(model_name):
    if model_name not in SUPPORTED_LLM_MODELS:
        raise ValueError(f"LLM model không được hỗ trợ: {model_name}")


def _validate_embedding_model(model_name):
    if model_name not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(f"Embedding model không được hỗ trợ: {model_name}")


def resolve_vector_db_path(vector_db_key):
    config = VECTOR_DB_CONFIG.get(vector_db_key)
    if not config:
        raise ValueError(f"Vector DB key không hợp lệ: {vector_db_key}")
    return config["path"]


def route_embedding_target(file_size_bytes, has_vietnamese):
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Quy tắc route theo yêu cầu hiện tại: tiếng Việt/file lớn ưu tiên Qwen,
    # tiếng Việt/file nhỏ ưu tiên BGE, tiếng Anh ưu tiên họ Nomic.
    if has_vietnamese:
        vector_db_key = "qwen_db" if file_size_mb > 5 else "bge_db"
    else:
        vector_db_key = "nomic_v2_db" if file_size_mb > 5 else "nomic_v1_db"

    config = VECTOR_DB_CONFIG[vector_db_key]
    return {
        "file_size_mb": round(file_size_mb, 2),
        "vector_db_key": vector_db_key,
        "vector_db_path": config["path"],
        "embedding_model": config["embedding_model"],
    }


def get_embeddings_model(model_name):
    _validate_embedding_model(model_name)

    if model_name not in _embedding_model_cache:
        print(f"Đang nạp embedding model vào RAM: {model_name}...")
        _embedding_model_cache[model_name] = OllamaEmbeddings(model=model_name)
    return _embedding_model_cache[model_name]


def get_llm_model(model_name):
    _validate_llm_model(model_name)

    if model_name not in _llm_model_cache:
        print(f"Khởi tạo kết nối tới Ollama với model: {model_name}...")
        _llm_model_cache[model_name] = OllamaLLM(
            model=model_name,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    return _llm_model_cache[model_name]


def get_cached_vector_store(vector_db_key, embedding_model_name):
    """Load FAISS theo từng kho và embedding model, chỉ 1 lần cho mỗi cặp."""
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
    text = ""
    try:
        if file_extension == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif file_extension == "docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text +  "\n"
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1111,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_document(file_path, file_extension):
    print(f'đang xử lý file: {file_path}...')
    raw_text = extract_text(file_path, file_extension)

    if not raw_text.strip():
        print(f"không tim thấy nội dung trong tài liệu")
        return []
    
    chunks = get_text_chunks(raw_text)
    print(f"Đã băm tài liệu thành {len(chunks)}  đoạn nhỏ")
    return chunks

def get_vector_store(chunks, embedding_model_name, vector_db_key):
    print(f"Vectorize document bằng {embedding_model_name} -> {vector_db_key}...")
    vector_db_path = resolve_vector_db_path(vector_db_key)
    os.makedirs(vector_db_path, exist_ok=True)

    embeddings = get_embeddings_model(embedding_model_name)
    index_path = os.path.join(vector_db_path, "index.faiss")

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(
            vector_db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vector_store.add_texts(chunks)
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    vector_store.save_local(vector_db_path)
    cache_key = f"{vector_db_key}::{embedding_model_name}"
    _vector_store_cache[cache_key] = vector_store
    return vector_store


def ask_gemma(
    question,
    chat_history="",
    llm_model_name="gemma4:e2b",
    embedding_model_name="",
    vector_db_key="",
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

    # GIẢM TẢI CHO CPU: Chỉ lấy 2 đoạn văn bản liên quan nhất (thay vì 3) để Prompt ngắn lại
    retriever = vector_store.as_retriever(search_kwargs={'k' : 2})
    relevant_doc = retriever.invoke(question)

    context  = "\n\n".join([doc.page_content for doc in relevant_doc])

    prompt_template =  """Bạn là trợ lý AI tên SmartDoc AI.
Lịch sử chat:
{chat_history}

Thông tin tài liệu:
{context}

Câu hỏi: {question}
Trả lời:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history" ,"context", "question"])

    print('Gemma 4 đang suy nghĩ...')
    llm = get_llm_model(llm_model_name)
    chain = prompt | llm

    return chain.stream({
        "chat_history": chat_history, 
        "context" : context, 
        "question" : question
    })