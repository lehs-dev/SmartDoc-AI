import json
import re
from sqlalchemy.orm import Session
from app.models import UserProfile

# Khởi tạo spaCy (Load 1 lần duy nhất để tối ưu tốc độ)
nlp = None
def get_nlp():
    global nlp
    if nlp is None:
        import spacy
        try:
            # Load model đa ngôn ngữ (hỗ trợ Tiếng Việt)
            nlp = spacy.load("xx_ent_wiki_sm") 
        except OSError:
            print("⚠️ Chưa tải model spaCy. Hãy chạy: python -m spacy download xx_ent_wiki_sm")
            nlp = spacy.blank("xx") # Fallback an toàn để không crash
    return nlp

def extract_and_update_long_term_memory(session_id: str, user_message: str, db: Session):
    """
    Phiên bản siêu nhẹ: Dùng spaCy NER và Regex thay cho LLM.
    Chạy mất ~0.05 giây, không tốn chút CPU nào của Ollama.
    """
    if len(user_message.split()) < 3:
        return

    profile = db.query(UserProfile).filter(UserProfile.name == session_id).first()
    if not profile:
        profile = UserProfile(name=session_id, preferences="{}")
        db.add(profile)
        db.commit()

    # Parse JSON memory hiện tại (Hoặc tạo mới nếu là dữ liệu text cũ)
    try:
        mem = json.loads(profile.preferences)
    except:
        mem = {}

    _nlp = get_nlp()
    doc = _nlp(user_message)

    # Layer 1: Episodic (Trích xuất Tên, Tổ chức bằng spaCy NER)
    for ent in doc.ents:
        if ent.label_ == "PER" and len(ent.text) > 1:
            mem["name"] = ent.text
        elif ent.label_ == "ORG":
            mem.setdefault("organizations", [])
            if ent.text not in mem["organizations"]:
                mem["organizations"].append(ent.text)

    # Layer 2: Semantic (Trích xuất Sở thích bằng Regex)
    likes = re.findall(r'(?:tôi (?:thích|yêu thích|quan tâm)|I (?:like|love|prefer))\s+(.+?)(?:\.|,|$)', 
                        user_message, re.IGNORECASE)
    if likes:
        mem.setdefault("interests", [])
        for l in likes:
            clean = l.strip()[:50]
            if clean and clean not in mem["interests"]:
                mem["interests"].append(clean)

    # Giới hạn kích thước mảng để DB không phình to (Chỉ nhớ 5-10 cái gần nhất)
    if "organizations" in mem:
        mem["organizations"] = mem["organizations"][-5:]
    if "interests" in mem:
        mem["interests"] = mem["interests"][-10:]

    # Lưu lại vào DB dạng chuẩn JSON
    profile.preferences = json.dumps(mem, ensure_ascii=False)
    db.commit()
    print(f"🧠 [Fast Memory Updated]: {mem}")

def build_memory_context(session_id: str, db: Session) -> str:
    """
    Đọc JSON từ DB và chuyển thành Text ngắn gọn để đưa vào System Prompt.
    """
    profile = db.query(UserProfile).filter(UserProfile.name == session_id).first()
    if not profile:
        return "Chưa có."
    
    try:
        mem = json.loads(profile.preferences)
    except:
        # Tương thích ngược với dữ liệu cũ đang lưu kiểu text
        return profile.preferences 

    parts = []
    if mem.get("name"):
        parts.append(f"Tên user: {mem['name']}")
    if mem.get("interests"):
        parts.append(f"Sở thích: {', '.join(mem['interests'])}")
    if mem.get("organizations"):
        parts.append(f"Tổ chức liên quan: {', '.join(mem['organizations'])}")
    
    return "; ".join(parts) if parts else "Chưa có."