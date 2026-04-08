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
        chunk_size=300,
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
    print("vectorize document...")
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_DB_PATH)

def ask_gemma(question, chat_histoty=""):
    print('Đang tìm kiếm thông tin cho câu hỏi...')
    embeddings = get_embeddings_model()
    vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={'k' : 3})
    relevant_doc = retriever.invoke(question)
    context  = "\n\n".join([doc.page_content for doc in relevant_doc])
    prompt_template ="""
    Bạn là một trợ lý AI thông minh tên là SmartDoc AI. 
    Dưới đây là lịch sử cuộc trò chuyện gần đây của bạn với người dùng:
    {chat_history}
    Hãy trả lời câu hỏi của người dùng DỰA VÀO phần thông tin (Context) được trích xuất từ tài liệu bên dưới. 
    Nếu thông tin không có trong Context, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu", TUYỆT ĐỐI KHÔNG tự bịa ra câu trả lời.
    Thông tin tài liệu (Context):
    {context}
    Câu hỏi hiện tại của người dùng: {question}
    Trả lời:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history" ,"context", "question"])
    print('Gemma 4 đang xử lý câu hỏi')
    llm = get_llm_model()
    chain = prompt | llm
    answer = chain.invoke({"chat_history" : chat_histoty ,"context" : context, "question" : question})
    return answer