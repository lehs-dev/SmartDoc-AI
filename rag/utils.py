import os
import pdfplumber
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def extract_text(file_path, file_extension):
    # Đọc file và rút trích toàn bộ văn bản
    text = ""
    try:
        if file_extension == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif file_extension == 'docx':
           doc = docx.Document(file_path)
           for para in doc.paragraphs:
               text += para.text + "\n"
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")

    return text 

def get_text_chunks(text):
    # Cắt nội dung thành nhiều chunk nhỏ để AI tiêu thụ
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_document(file_path, file_extension):
    # Quy tình: Đoc file -> băm nhỏ -> Trả về các chunks
    print(f"Đang xử lý file: {file_path}...")
    raw_text = extract_text(file_path, file_extension)

    if not raw_text.strip():
        print(f"Không tìm thấy chữ nào trong file!")
        return []
    
    chunks = get_text_chunks(raw_text)
    print(f"Đã băm file thành {len(chunks)} đoạn chunk")
    return chunks

VECTOR_DB_PATH = "vector_store"

def get_vector_store(chunks):
    # Biễn chunks thành vector và lưu xuống ổ cứn bằng FAISS
    print("Đang tải mô hình nhúng...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    print("Đang chuyển đổi văn bản thành vector và nạp vào FAISS...")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    vector_store.save_local(VECTOR_DB_PATH)

def ask_gemma(question):
    print('Đang tìm kiếm thông tin cho câu hỏi...')
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_kwargs={'k' : 3})
    relevant_doc = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in relevant_doc])

    prompt_template = """Bạn là một trợ lý AI thông minh tên là SmartDoc AI. 
    Hãy trả lời câu hỏi của người dùng DỰA VÀO phần thông tin (Context) được trích xuất từ tài liệu bên dưới. 
    Nếu thông tin không có trong Context, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu", TUYỆT ĐỐI KHÔNG tự bịa ra câu trả lời.

    Thông tin tài liệu (Context):
    {context}

    Câu hỏi của người dùng: {question}
    
    Trả lời:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    print('Đang chuyển thông tin cho Gemma4 xử lý...')
    llm = OllamaLLM(model="gemma4:e2b")

    chain = prompt | llm

    answer = chain.invoke({"context" : context, "question" : question})

    print('Đã có câu trả lời!')

    return answer

