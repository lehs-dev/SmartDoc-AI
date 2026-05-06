import fitz  # PyMuPDF
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(file_path: str, filename: str) -> str:
    """Trích xuất text từ PDF (xử lý đa cột + BẢNG BIỂU) hoặc DOCX."""
    text = ""
    ext = filename.split('.')[-1].lower()

    try:
        if ext == 'pdf':
            doc = fitz.open(file_path)
            for page in doc:
                # 1. ƯU TIÊN RÚT BẢNG BIỂU (TABLES) ĐỂ GIỮ NGUYÊN SỐ LIỆU
                tables = page.find_tables()
                for table in tables:
                    text += "\n[DỮ LIỆU BẢNG]:\n"
                    for row in table.extract():
                        # Làm sạch NoneType và nối các ô bằng dấu |
                        clean_row = [str(cell).strip().replace('\n', ' ') if cell else "" for cell in row]
                        text += " | ".join(clean_row) + "\n"
                    text += "\n"

                # 2. XỬ LÝ TEXT ĐA CỘT (Tránh bị lộn xộn trái phải)
                blocks = page.get_text("blocks")
                text_blocks = [b for b in blocks if b[6] == 0]
                text_blocks.sort(key=lambda b: (round(b[0] / 100), b[1]))
                
                for b in text_blocks:
                    clean_text = b[4].replace('\n', ' ').strip()
                    if clean_text:
                        text += clean_text + "\n\n"
            doc.close()

        elif ext == 'docx':
            # ... (Giữ nguyên logic DOCX của ông) ...
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
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
    Băm văn bản. TĂNG KÍCH THƯỚC CHUNK để chứa trọn vẹn được một cái bảng biểu,
    không làm mất ngữ cảnh của các con số.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # Tăng từ 300 lên 600
        chunk_overlap=150,    # Tăng overlap để nối câu mượt hơn
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)