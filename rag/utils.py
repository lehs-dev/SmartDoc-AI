import os
import pdfplumber
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

os.environ['HF_HUB_OFFLINE'] = '1'
VECTOR_DB_PATH = "vector_store"

_embeddings_model = None
_llm_model = None
_vector_store = None # Thêm biến cache cho Database

def get_embeddings_model():
    global _embeddings_model
    if _embeddings_model is None:
        print("Đang nạp embedding model vào ram...")
        _embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    return _embeddings_model

def get_llm_model():
    global _llm_model
    if _llm_model is None:
        print("Khởi tạo kết nối tới Ollama...")
        _llm_model = OllamaLLM(model="gemma4:e2b")
    return _llm_model

def get_cached_vector_store():
    """Hàm này giúp chỉ load FAISS từ ổ cứng 1 lần duy nhất"""
    global _vector_store
    if _vector_store is None:
        print("Đang nạp Vector Database vào RAM...")
        embeddings = get_embeddings_model()
        if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
            _vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return _vector_store


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

def get_vector_store(chunks):
    global _vector_store
    print("vectorize document...")
    embeddings = get_embeddings_model()
    
    # Logic nạp dồn tài liệu (Không xóa cũ)
    index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if os.path.exists(index_path):
        _vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        _vector_store.add_texts(chunks)
    else:
        _vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
    _vector_store.save_local(VECTOR_DB_PATH)

def ask_gemma(question, chat_history=""):
    print('Đang tìm kiếm thông tin cho câu hỏi...')
    
    # Dùng DB đã cache trong RAM thay vì đọc ổ cứng
    vector_store = get_cached_vector_store()

    if vector_store is None:
        return "Xin lỗi, tôi chưa được nạp tài liệu nào. Vui lòng tải tài liệu lên trước."

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
    llm = get_llm_model()
    chain = prompt | llm

    answer = chain.invoke({
        "chat_history": chat_history, 
        "context" : context, 
        "question" : question
    })

    return answer