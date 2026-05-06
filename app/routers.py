import os
import shutil
from app.services.doc_parser import extract_text, chunk_text
from app.services.vector_db import create_and_save_index, search_context
from fastapi import APIRouter, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import ChatSession, ChatMessage
from app.services.llm_client import stream_chat_response

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
async def chat_with_ai(req: ChatRequest, db: Session = Depends(get_db)):
    
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

    # 3. LẤY LỊCH SỬ CHAT (Đã bao gồm cả câu user vừa hỏi ở bước 2)
    history_records = db.query(ChatMessage)\
        .filter(ChatMessage.session_id == req.session_id)\
        .order_by(ChatMessage.created_at.desc())\
        .limit(6).all()
    history_records.reverse()

    # 4. TÌM NGỮ CẢNH TỪ FAISS (RAG)
    context = search_context(req.message, req.session_id)
    
    # 5. XÂY DỰNG NHÂN CÁCH VÀ NHỒI NGỮ CẢNH RAG
    system_prompt = (
        "Bạn là SmartDoc AI, trợ lý thông minh, nhiệt tình và thân thiện. "
        "Luôn giải thích cặn kẽ, chi tiết, dùng ngữ điệu tự nhiên, lịch sự và sử dụng emoji cho sinh động. "
    )
    if context:
        system_prompt += f"\n\nDựa vào tài liệu sau đây để trả lời câu hỏi chính xác nhất (tuyệt đối không bịa thông tin):\n\n[TÀI LIỆU]:\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    
    # Nhét lịch sử vào
    for msg in history_records:
        messages.append({"role": msg.role, "content": msg.content})

    # 6. HÀM BỌC ĐỂ VỪA STREAM VỪA LƯU DB
    def stream_and_save():
        ai_full_response = ""
        for chunk in stream_chat_response(messages):
            ai_full_response += chunk
            yield chunk
            
        # Lưu câu trả lời của AI vào DB khi stream xong
        ai_msg = ChatMessage(session_id=req.session_id, role="ai", content=ai_full_response)
        db.add(ai_msg)
        db.commit()

    # 7. TRẢ VỀ LUỒNG STREAM
    return StreamingResponse(stream_and_save(), media_type="text/plain")