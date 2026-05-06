import os
import shutil
from app.services.doc_parser import extract_text, chunk_text
from app.services.vector_db import create_and_save_index, search_context
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import ChatSession, ChatMessage
from app.services.llm_client import stream_chat_response
from app.services.memory import extract_and_update_long_term_memory
from app.models import UserProfile

router = APIRouter()

# Schema để hứng dữ liệu từ Frontend
class ChatRequest(BaseModel):
    message: str
    session_id: str

@router.get("/ping")
async def ping(db: Session = Depends(get_db)):
    return {"status": "success", "message": "Backend FastAPI đã lên sóng!"}

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    session_id: str = Form(...)
):
    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/{file.filename}"
    
    # 1. Lưu file từ web xuống ổ cứng
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Rút chữ và Băm chữ
    raw_text = extract_text(file_path, file.filename)
    chunks = chunk_text(raw_text)
    
    # 3. Nhúng FAISS
    create_and_save_index(chunks, session_id)
    
    return {
        "status": "success", 
        "message": f"Đã học xong file {file.filename} ({len(chunks)} đoạn).",
        "chunks_count": len(chunks)
    }

@router.post("/chat")
async def chat_with_ai(
    req: ChatRequest, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)):
    
    # 1. KIỂM TRA VÀ TẠO SESSION NẾU CHƯA CÓ
    session = db.query(ChatSession).filter(ChatSession.id == req.session_id).first()
    if not session:
        session = ChatSession(id=req.session_id, title=req.message[:30])
        db.add(session)
        db.commit()

    # 2. LƯU TIN NHẮN CỦA USER VÀO DATABASE
    user_msg = ChatMessage(session_id=req.session_id, role="user", content=req.message)
    db.add(user_msg)
    db.commit()

    # - Kich hoạt hàm chạy ngầm để trích xuất sự kiện quan trọng và cập nhật vào long-term memory -
    background_tasks.add_task(extract_and_update_long_term_memory, req.session_id, req.message, db)

    # 3. Lấy Short-term memory (Lịch sử 6 câu - Giữ nguyên)
    history_records = db.query(ChatMessage).filter(ChatMessage.session_id == req.session_id).order_by(ChatMessage.created_at.desc()).limit(6).all()
    history_records.reverse()

    # 4. Lấy Semantic Memory trong FAISS với kỹ thuật query rewriting
    search_query = req.message  # Có thể cải tiến bằng cách thêm tiền xử lý query
    # Nếu một câu hỏi ngắn < 10 từ và có đủ lịch sử (ít nhất 3 tin nhắn: user cũ, ai cũ, user hiện tại)
    if len(search_query.split()) < 10 and len(history_records) >= 3:
        # Nối câu hỏi cũ của user với câu hỏi hiện tại
        prev_user_msg = history_records[-3].content  # Câu hỏi của user trước đó
        search_query = f"{prev_user_msg} {req.message}"
        print(f"Đã rewrite query từ '{req.message}' thành '{search_query}' để tìm ngữ cảnh tốt hơn.")
    
    context = search_context(search_query, req.session_id)

    # --- 5. LẤY LONG-TERM MEMORY TỪ DB RA ---
    profile = db.query(UserProfile).filter(UserProfile.name == req.session_id).first()
    long_term_facts = profile.preferences if profile else "Chưa có thông tin."

    # 6. XÂY DỰNG MARAG SYSTEM PROMPT
    system_prompt = (
        "Bạn là SmartDoc AI, trợ lý thông minh, nhiệt tình. Luôn giải thích cặn kẽ.\n"
        f"📚 [THÔNG TIN ĐÃ GHI NHỚ VỀ USER]: {long_term_facts}\n"
    )
    
    if context:
        system_prompt += f"\n📄 [NGỮ CẢNH RAG]: Dựa vào tài liệu sau để trả lời:\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history_records:
        messages.append({"role": msg.role, "content": msg.content})

    def stream_and_save():
        # ... (Giữ nguyên logic yield stream và lưu db của ông) ...
        ai_full_response = ""
        for chunk in stream_chat_response(messages):
            ai_full_response += chunk
            yield chunk
            
        ai_msg = ChatMessage(session_id=req.session_id, role="ai", content=ai_full_response)
        db.add(ai_msg)
        db.commit()

    return StreamingResponse(stream_and_save(), media_type="text/plain")