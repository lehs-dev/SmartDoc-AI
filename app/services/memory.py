from sqlalchemy.orm import Session
from app.models import UserProfile
import ollama

def extract_and_update_long_term_memory(user_id: str, user_message: str, db: Session):
    """
    Chạy ngầm: Nén và cập nhật trí nhớ dài hạn.
    Thay vì cộng dồn, LLM sẽ hợp nhất ký ức cũ và thông tin mới thành một bản tóm tắt ngắn gọn.
    """
    profile = db.query(UserProfile).filter(UserProfile.name == user_id).first()
    if not profile:
        profile = UserProfile(name=user_id, preferences="Chưa có thông tin.")
        db.add(profile)
        db.commit()

    # CHỐT CHẶN 1: Bỏ qua các câu chat quá ngắn hoặc vô nghĩa để tiết kiệm CPU
    if len(user_message.split()) < 3:
        return

    current_memory = profile.preferences

    # PROMPT SIÊU NÉN: Ép AI gộp thông tin và tự động ghi đè nếu có mâu thuẫn
    prompt = f"""
    Bạn là hệ thống quản lý trí nhớ AI. Nhiệm vụ của bạn là cập nhật hồ sơ người dùng một cách NGẮN GỌN.
    
    [HỒ SƠ HIỆN TẠI]: {current_memory}
    [TƯƠNG TÁC MỚI]: "{user_message}"
    
    Yêu cầu thực hiện:
    1. Nếu [TƯƠNG TÁC MỚI] KHÔNG chứa thông tin cá nhân, sở thích, hay dữ kiện thực tế về người dùng -> Trả về y nguyên [HỒ SƠ HIỆN TẠI].
    2. Nếu [TƯƠNG TÁC MỚI] có thông tin quan trọng -> Gộp nó vào hồ sơ. LOẠI BỎ thông tin cũ nếu bị mâu thuẫn (Ví dụ: Đổi tên, đổi ý định).
    3. Tóm tắt lại toàn bộ thành MỘT ĐOẠN VĂN DUY NHẤT (tối đa 40 từ).
    4. TUYỆT ĐỐI KHÔNG mở bài (không dùng câu "Hồ sơ mới là..."), KHÔNG giải thích, CHỈ in ra nội dung hồ sơ.
    """
    
    try:
        response = ollama.chat(
            model="qcwind/qwen2.5-7B-instruct-Q4_K_M",
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": 100, # CHỐT CHẶN 2: Giới hạn token trả về để không bị phình to
                "temperature": 0.1  # Nhiệt độ thấp để tóm tắt chính xác, không sáng tạo thêm
            }
        )
        
        new_memory = response['message']['content'].strip()
        
        # Chỉ cập nhật nếu model trả về một chuỗi hợp lệ, không quá dài
        if new_memory and len(new_memory) > 5 and len(new_memory) < 500:
            profile.preferences = new_memory
            db.commit()
            print(f"🧠 [Memory Compressed]: {new_memory}")
            
    except Exception as e:
        print(f"Lỗi cập nhật trí nhớ: {e}")