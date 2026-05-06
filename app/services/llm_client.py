import ollama
import json

# Khai báo model (Dùng đúng bản Q4 ông đã pull về)
LLM_MODEL = "qcwind/qwen2.5-7B-instruct-Q4_K_M"

# Cấu hình "Hack" ép CPU chạy mượt
OLLAMA_OPTIONS = {
    "num_thread": 4,     # Chỉ dùng 4 nhân P-Core
    "num_ctx": 2048,     # Giới hạn context để xử lý lẹ
    "temperature": 0.1   # Giữ cho AI trả lời nghiêm túc, bám sát tài liệu
}

def stream_chat_response(messages: list):
    """
    Hàm này gọi Ollama và dùng 'yield' để nhả từng chữ ra ngay lập tức
    mà không cần chờ AI gen xong cả câu. (Bí quyết của độ mượt)
    """
    try:
        # Gọi API local của Ollama, bật stream=True
        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            options=OLLAMA_OPTIONS,
            stream=True
        )
        
        # Lặp qua từng chunk (từng chữ) AI nhả ra
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                # Trả về theo chuẩn Server-Sent Events (SSE) để Frontend hứng
                yield content
                
    except Exception as e:
        yield f"\n[Lỗi kết nối Ollama]: {str(e)}\n"