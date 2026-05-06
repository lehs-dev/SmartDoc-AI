import os
import faiss
import numpy as np
import pickle
import ollama

EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_DIR = "data/vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

def get_embedding(text: str) -> list[float]:
    """Gọi Ollama lấy vector embedding"""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response['embedding']

def create_and_save_index(chunks: list[str], session_id: str):
    """Nhúng toàn bộ chunks và lưu vào FAISS trên ổ cứng"""
    if not chunks:
        return
    
    print(f"Đang nhúng {len(chunks)} chunks... Sẽ tốn vài giây trên CPU!")
    
    # 1. Lấy vectors cho tất cả các chunks
    vectors = [get_embedding(chunk) for chunk in chunks]
    
    # Kích thước vector của nomic-embed-text là 768
    dim = 768 
    vector_array = np.array(vectors).astype('float32')

    # 2. Khởi tạo FAISS Index (Dùng L2 distance)
    index = faiss.IndexFlatL2(dim)
    index.add(vector_array)

    # 3. Lưu xuống ổ cứng (1 file vector, 1 file chứa chữ)
    faiss.write_index(index, f"{VECTOR_DIR}/index_{session_id}.faiss")
    with open(f"{VECTOR_DIR}/chunks_{session_id}.pkl", "wb") as f:
        pickle.dump(chunks, f)
        
    print(f"Đã tạo xong VectorDB cho session {session_id}")


_index_cache = {}  # Dict lưu cache: session_id -> (index, chunks, mtime)

def search_context(query: str, session_id: str, top_k: int = 4) -> str:
    index_path = f"{VECTOR_DIR}/index_{session_id}.faiss"
    chunks_path = f"{VECTOR_DIR}/chunks_{session_id}.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return ""

    # Kiểm tra thời gian sửa file cuối cùng
    mtime = os.path.getmtime(index_path)
    
    # Nếu chưa có trong cache hoặc file bị sửa đổi -> Đọc từ đĩa
    if session_id not in _index_cache or _index_cache[session_id][2] != mtime:
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        _index_cache[session_id] = (index, chunks, mtime)

    # Lấy từ RAM
    index, chunks, _ = _index_cache[session_id]

    query_vector = np.array([get_embedding(query)]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    
    context = ""
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            context += chunks[idx] + "\n\n"
    return context.strip()