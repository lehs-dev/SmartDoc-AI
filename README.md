# SmartDoc-AI

SmartDoc-AI là một ứng dụng Django thông minh giúp quản lý và xử lý tài liệu tự động bằng AI.

## Yêu cầu hệ thống

- Python 3.8 trở lên
- pip (Python package manager)
- virtualenv (khuyến nghị)

## Cài đặt

### 1. Clone repository (nếu chưa có)

```bash
git clone <repository-url>
cd SmartDoc-AI
```

### 2. Tạo và kích hoạt môi trường ảo

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### 3. Cài đặt các dependencies

```bash
pip install -r requirements.txt
```

### 4. Thiết lập database

```bash
python manage.py migrate
```

### 5. Tạo tài khoản admin

```bash
python manage.py createsuperuser
```

Nhập thông tin theo hướng dẫn (username, email, password).

### 6. Chạy server

```bash
python manage.py runserver
```

Truy cập ứng dụng tại: `http://127.0.0.1:8000/`

Truy cập admin panel tại: `http://127.0.0.1:8000/admin/`

## Cấu trúc dự án

```
SmartDoc-AI/
├── manage.py
├── requirements.txt
├── .venv/
└── [các app Django]
```

## Các lệnh hữu ích

| Lệnh | Mô tả |
|------|-------|
| `python manage.py runserver` | Chạy development server |
| `python manage.py migrate` | Áp dụng migrations |
| `python manage.py makemigrations` | Tạo migrations mới |
| `python manage.py createsuperuser` | Tạo admin user |
| `python manage.py collectstatic` | Thu thập static files |

## Lưu ý

- Đảm bảo đã kích hoạt virtual environment trước khi chạy các lệnh
- Trong production, sử dụng `DEBUG = False` và cấu hình security phù hợp
- Backup database thường xuyên

## License

MIT License