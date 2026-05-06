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

def search_context(query: str, session_id: str, top_k: int = 4) -> str:
    """Tìm kiếm 4 đoạn văn bản liên quan nhất đến câu hỏi"""
    index_path = f"{VECTOR_DIR}/index_{session_id}.faiss"
    chunks_path = f"{VECTOR_DIR}/chunks_{session_id}.pkl"

    # Nếu session này chưa up tài liệu nào, trả về rỗng
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return ""

    # Load FAISS và Chunks lên
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    # Lấy vector của câu hỏi
    query_vector = np.array([get_embedding(query)]).astype('float32')

    # Tìm Top_K đoạn giống nhất
    distances, indices = index.search(query_vector, top_k)
    
    # Gom chữ lại thành 1 cục context
    context = ""
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            context += chunks[idx] + "\n\n"
            
    return context.strip()