import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import từ các file module (chúng ta sẽ tạo sau)
from app.database import engine, Base
from app.routers import router as api_router

# 1. Tự động tạo các bảng trong Database (SQLite) nếu chưa có
# Ngay khi chạy app, SQLAlchemy sẽ quét models.py và tạo file db.sqlite3
Base.metadata.create_all(bind=engine)

# 2. Khởi tạo FastAPI App
app = FastAPI(
    title="SmartDoc AI - RAG System",
    description="Backend siêu nhẹ tối ưu cho 30 phút Demo",
    version="2.0.0"
)

# 3. Cấu hình CORS (Bảo mật chéo miền)
# Cho phép Frontend gọi API thoải mái mà không bị block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên để domain cụ thể, nhưng demo thì "*" cho lẹ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Gắn các API Router (Xử lý logic chat, upload...)
# Toàn bộ API sẽ có tiền tố là /api (VD: http://localhost:8000/api/chat)
app.include_router(api_router, prefix="/api")

# 5. Mount thư mục Frontend (HTML, CSS, JS)
# ĐOẠN NÀY QUAN TRỌNG: Nó giúp ông gõ localhost:8000 là ra luôn giao diện web
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
else:
    print("⚠️ CẢNH BÁO: Chưa tạo thư mục 'frontend'. Hãy tạo thư mục này và bỏ file index.html vào nhé!")