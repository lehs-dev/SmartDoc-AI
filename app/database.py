import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Đảm bảo thư mục data/ tồn tại
os.makedirs("data", exist_ok=True)

# File DB sẽ được lưu tại đây
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/db.sqlite3"

# check_same_thread=False là bắt buộc cho SQLite khi chạy trên FastAPI (đa luồng)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Hàm này dùng để FastAPI "mượn" kết nối DB mỗi khi có request tới, xong việc tự trả lại
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()