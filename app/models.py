from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, default="Cuộc trò chuyện mới")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Quan hệ 1-Nhiều với tin nhắn. Xóa session -> xóa sạch tin nhắn cũ
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    role = Column(String)  # 'user' hoặc 'ai'
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")

class UserProfile(Base):
    """
    Bảng này chính là vũ khí bí mật cho kịch bản "Test trí nhớ".
    Background Task sẽ phân tích tin nhắn và lưu Tên/Sở thích của ông vào cột 'preferences'.
    """
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    # Lưu dưới dạng chuỗi JSON hoặc text thô để nhét vào System Prompt
    preferences = Column(Text, default="[]") 
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())