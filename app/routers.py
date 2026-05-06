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
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    raw_text = extract_text(file_path, file.filename)
    chunks = chunk_text(raw_text)
    create_and_save_index(chunks, session_id)
    
    return {"status": "success", "message": f"Đã học xong file {file.filename} ({len(chunks)} đoạn)."}

@router.post("/chat")
async def chat_with_ai(
    req: ChatRequest, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)):
    
    # 1. Tạo session & lưu tin nhắn
    session = db.query(ChatSession).filter(ChatSession.id == req.session_id).first()
    if not session:
        session = ChatSession(id=req.session_id, title=req.message[:30])
        db.add(session)
        db.commit()

    user_msg = ChatMessage(session_id=req.session_id, role="user", content=req.message)
    db.add(user_msg)
    db.commit()

    # 2. RÚT GỌN LỊCH SỬ CHAT (Chỉ lấy 4 câu gần nhất để LLM không bị ngộp & lặp từ)
    history_records = db.query(ChatMessage).filter(ChatMessage.session_id == req.session_id).order_by(ChatMessage.created_at.desc()).limit(4).all()
    history_records.reverse()

    # 3. FAISS SEARCH (Lấy top_k = 4 là đủ bao quát mà CPU vẫn chạy nhanh)
    search_query = req.message
    if len(search_query.split()) <= 6 and len(history_records) >= 3:
        # Nối nhẹ câu hỏi cũ nếu câu này quá ngắn
        search_query = f"{history_records[-3].content} {req.message}"
        
    context = search_context(search_query, req.session_id, top_k=4)

    # 4. TỐI ƯU HÓA BACKGROUND TASK (Chỉ chạy cập nhật trí nhớ nếu KHÔNG đọc tài liệu)
    # Điều này cứu CPU i5 của ông khỏi việc bị thắt cổ chai
    if not context:
        background_tasks.add_task(extract_and_update_long_term_memory, req.session_id, req.message, db)

    # 5. LẤY LONG-TERM MEMORY
    profile = db.query(UserProfile).filter(UserProfile.name == req.session_id).first()
    long_term_facts = profile.preferences if profile else "Chưa có."

    # 6. ÉP SYSTEM PROMPT (Thần chú chống lặp lại)
    system_prompt = (
        "Bạn là SmartDoc AI. Trả lời NGẮN GỌN, đi thẳng vào trọng tâm câu hỏi mới nhất.\n"
        "TUYỆT ĐỐI KHÔNG lặp lại những thông tin đã trả lời ở câu trước.\n"
        f"👤 [GHI NHỚ USER]: {long_term_facts}\n"
    )
    
    if context:
        system_prompt += f"\n📄 [TÀI LIỆU (Trả lời bằng tiếng Việt)]: Dựa vào dữ liệu sau để trả lời (KHÔNG BỊA RA THÔNG TIN):\n{context}\n"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history_records:
        messages.append({"role": msg.role, "content": msg.content})

    def stream_and_save():
        ai_full_response = ""
        for chunk in stream_chat_response(messages):
            ai_full_response += chunk
            yield chunk
            
        ai_msg = ChatMessage(session_id=req.session_id, role="ai", content=ai_full_response)
        db.add(ai_msg)
        db.commit()

    return StreamingResponse(stream_and_save(), media_type="text/plain")