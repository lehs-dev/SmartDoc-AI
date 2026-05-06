from sqlalchemy.orm import Session
from app.models import UserProfile, ChatMessage
import ollama

def extract_and_update_long_term_memory(session_id: str, user_message: str, db: Session):
    """
    Chạy ngầm: Nhờ LLM trích xuất sự kiện quan trọng từ câu hỏi của user 
    và lưu vào bảng UserProfile (Long-term memory).
    """
    # Lấy hoặc tạo profile cho session này
    profile = db.query(UserProfile).filter(UserProfile.name == session_id).first()
    if not profile:
        profile = UserProfile(name=session_id, preferences="Chưa có thông tin đáng chú ý.")
        db.add(profile)
        db.commit()

    # Prompt bắt LLM trích xuất sự kiện
    prompt = f"""
    Dựa vào câu nói sau của người dùng: "{user_message}"
    Hãy trích xuất những sự kiện, thông tin cá nhân hoặc yêu cầu quan trọng (nếu có). 
    Nếu không có gì quan trọng, trả về rỗng. Nếu có, hãy tóm tắt thật ngắn gọn trong 1 câu.
    """
    
    try:
        response = ollama.chat(
            model="qcwind/qwen2.5-7B-instruct-Q4_K_M",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 50, "temperature": 0.1} # Sinh nhanh, không bịa
        )
        new_fact = response['message']['content'].strip()
        
        # Nếu LLM tìm thấy fact mới, nối vào preferences cũ
        if new_fact and len(new_fact) > 5 and "rỗng" not in new_fact.lower():
            old_facts = profile.preferences if profile.preferences != "Chưa có thông tin đáng chú ý." else ""
            profile.preferences = f"{old_facts} | {new_fact}".strip(" | ")
            db.commit()
            print(f"[Memory Updated]: {new_fact}")
    except Exception as e:
        print(f"Lỗi cập nhật trí nhớ: {e}")