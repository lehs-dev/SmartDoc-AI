import ollama
import json

# Khai báo model (Dùng đúng bản Q4 ông đã pull về)
LLM_MODEL = "qcwind/qwen2.5-7B-instruct-Q4_K_M"

OLLAMA_OPTIONS = {
    "num_thread": 8,      # Tăng lên 8 luồng cho i5
    "num_ctx": 1024,      # Giảm xuống 1024 để Warm-up siêu tốc
    "num_batch": 512,     # Giúp xử lý prompt dài nhanh hơn
    "temperature": 0.1,
    "repeat_penalty": 1.1 # Ngăn AI nói lặp từ vô nghĩa
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