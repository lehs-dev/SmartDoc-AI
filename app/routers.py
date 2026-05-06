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

    # 4. Lấy Semantic Memory trong FAISS với Smart Query Rewriting
    search_query = req.message
    word_count = len(search_query.split())
    
    # CHỈ nối câu hỏi cũ nếu câu hỏi hiện tại QUÁ NGẮN (<= 5 từ) 
    # (Ví dụ: "Vì sao?", "Chi tiết hơn đi", "Chắc chắn chứ?")
    if word_count <= 5 and len(history_records) >= 3:
        prev_user_msg = history_records[-3].content 
        search_query = f"{prev_user_msg} {req.message}"
        print(f"🔄 Đã rewrite query từ '{req.message}' thành '{search_query}'")
    else:
        print(f"🔎 Giữ nguyên query để FAISS tìm kiếm chính xác: '{search_query}'")
    
    # Truyền thêm top_k=6 để gọi hàm
    context = search_context(search_query, req.session_id, top_k=6)

    # --- 5. LẤY LONG-TERM MEMORY TỪ DB RA ---
    profile = db.query(UserProfile).filter(UserProfile.name == req.session_id).first()
    long_term_facts = profile.preferences if profile else "Chưa có thông tin."

    # 6. XÂY DỰNG MARAG SYSTEM PROMPT RÕ RÀNG HƠN
    system_prompt = (
        "Bạn là SmartDoc AI. Hãy trả lời chính xác dựa vào ngữ cảnh.\n"
        "Nếu ngữ cảnh có nhiều thông tin rải rác, hãy tổng hợp lại đầy đủ.\n"
        f"📚 [SỰ KIỆN GHI NHỚ VỀ USER]: {long_term_facts}\n"
    )
    
    if context:
        system_prompt += f"\n📄 [TÀI LIỆU TRÍCH XUẤT]: Dựa vào đây để trả lời (không bịa thông tin):\n{context}\n"

    messages = [{"role": "system", "content": system_prompt}]
    
    # Để tránh AI nhầm lẫn giữa Lịch sử và Câu hỏi hiện tại, ta add lịch sử bình thường
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