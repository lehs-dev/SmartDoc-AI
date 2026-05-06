import fitz  # PyMuPDF
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(file_path: str, filename: str) -> str:
    """Trích xuất text từ PDF hoặc DOCX."""
    text = ""
    ext = filename.split('.')[-1].lower()

    try:
        if ext == 'pdf':
            # PyMuPDF đọc PDF siêu nhanh bằng C
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()

        elif ext == 'docx':
            doc = docx.Document(file_path)
            # 1. Rút text từ các đoạn văn bình thường
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            
            # 2. RÚT TEXT TỪ BẢNG (Vũ khí bí mật để hỏi số liệu)
            for table in doc.tables:
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_data:
                        text += " | ".join(row_data) + "\n"
                        
    except Exception as e:
        print(f"Lỗi đọc file {filename}: {e}")
        
    return text

def chunk_text(text: str) -> list[str]:
    """
    Băm văn bản thành các khúc nhỏ.
    Cấu hình Chunk size = 300 để ép FAISS tìm ra đúng chỗ chứa số liệu,
    tránh LLM 7B bị ngợp thông tin (hallucinate).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)