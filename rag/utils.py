import os
import pdfplumber
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

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



