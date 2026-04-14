# 🔧 Debug Guide: Vectorization Issue

## ✅ Các fixes đã apply

### 1. **Fix views.py** (Quan trọng nhất!)
- ✅ Đã thêm gọi `get_vector_store()` sau khi chunking
- ✅ Đã thêm exception handling
- ✅ Đã set `doc.is_embedded = True` sau khi vectorize thành công

### 2. **Enhanced Logging**
- ✅ `get_vector_store()`: Log chi tiết từng bước
- ✅ `get_embeddings_model()`: Log khi load model
- ✅ `extract_text()`: Log số trang, số ký tự
- ✅ `check_ollama_connection()`: Health check

### 3. **Updated Embedding Models**
- ✅ Đã update `SUPPORTED_EMBEDDING_MODELS` với models bạn đã pull
- ✅ Priority: `nomic-embed-text`, `bge-m3:567m`

---

## 🧪 Cách Test

### **Option 1: Chạy test script (Recommended)**

```bash
python test_memory_rag.py
```

Script sẽ test:
1. ✅ Ollama connection
2. ✅ Embedding models loading
3. ✅ LLM models available
4. ✅ Routing logic
5. ✅ Chunking
6. ✅ Text extraction (nếu có file)

### **Option 2: Test qua Django server**

1. Chạy server:
```bash
python manage.py runserver
```

2. Upload file PDF/DOCX qua web interface

3. **Xem console output** - Sẽ thấy log chi tiết:
```
============================================================
🚀 BẮT ĐẦU VECTORIZE DOCUMENT
============================================================
📊 Số chunks: 15
🤖 Embedding model: bge-m3:567m
💾 Vector DB key: bge_db
📁 Vector DB path: vector_store/bge_db
✅ Đã tạo thư mục: vector_store/bge_db

⏳ Đang load embedding model...
🤖 [EMBEDDING] Đang nạp model vào RAM: bge-m3:567m...
✅ [EMBEDDING] Model loaded thành công: bge-m3:567m
✅ Embedding model loaded thành công

📄 Index path: vector_store/bge_db/index.faiss
📌 Index mới, đang tạo từ đầu...
✅ Đã tạo index mới với 15 chunks

💾 Đang lưu vector store vào disk...
✅ Lưu thành công vào: vector_store/bge_db
✅ Update cache: bge_db::bge-m3:567m

============================================================
✅ VECTORIZE HOÀN THÀNH!
============================================================
```

---

## 🔍 Các lỗi thường gặp & Cách fix

### **1. Ollama không chạy**
**Triệu chứng:**
```
❌ [OLLAMA] Không thể kết nối: Connection refused
```

**Giải pháp:**
```bash
# Terminal 1: Chạy Ollama server
ollama serve

# Terminal 2: Chạy Django
python manage.py runserver
```

---

### **2. Embedding model chưa pull**
**Triệu chứng:**
```
❌ [EMBEDDING] Lỗi khi load model bge-m3:567m
```

**Giải pháp:**
```bash
# Pull models
ollama pull nomic-embed-text
ollama pull bge-m3:567m

# Kiểm tra đã pull chưa
ollama list
```

---

### **3. File không extract được text**
**Triệu chứng:**
```
⚠️  [EXTRACT] Cảnh báo: Không extract được text nào!
```

**Nguyên nhân:**
- File PDF scan (image-based, không phải text)
- File bị corrupt
- File password-protected

**Giải pháp:**
- Test với file PDF text-based khác
- Kiểm tra file có mở được bằng PDF reader không

---

### **4. Out of Memory (RAM thấp)**
**Triệu chứng:**
```
Killed
hoặc
MemoryError
```

**Giải pháp:**
- Giảm số chunks bằng cách tăng `chunk_size` trong `get_text_chunks_optimized()`
- Đóng các ứng dụng khác để giải phóng RAM
- Sử dụng `nomic-embed-text` thay vì `bge-m3:567m` (nhẹ hơn)

---

### **5. Vector store không lưu được**
**Triệu chứng:**
```
❌ Lỗi khi lưu vector store: Permission denied
```

**Giải pháp:**
```bash
# Kiểm tra permissions
ls -la vector_store/

# Fix permissions
chmod -R 755 vector_store/
```

---

## 📊 Checklist Debug

Khi upload file không hoạt động, check theo thứ tự:

- [ ] **Ollama đang chạy?** → `ollama list` có output không?
- [ ] **Embedding models đã pull?** → `ollama pull nomic-embed-text`
- [ ] **File có text extractable?** → Mở PDF bằng reader kiểm tra
- [ ] **Console log lỗi gì?** → Xem output khi upload
- [ ] **`is_embedded` có được set?** → Check trong Django admin
- [ ] **Vector store có được tạo?** → `ls -la vector_store/`

---

## 🎯 Expected Flow

```
1. Upload file PDF/DOCX
   ↓
2. Extract text (log: "📄 [EXTRACT] Đang trích xuất...")
   ↓
3. Chunking (log: "✅ Adaptive chunking: 15 chunks")
   ↓
4. Route embedding (log: "→ Embedding: bge-m3:567m")
   ↓
5. Load embedding model (log: "✅ Model loaded thành công")
   ↓
6. Create FAISS index (log: "✅ Đã tạo index mới")
   ↓
7. Save to disk (log: "✅ Lưu thành công vào: vector_store/bge_db")
   ↓
8. Set is_embedded=True (Database update)
   ↓
9. ✅ Success message hiển thị
```

---

## 📝 Test Cases

### **Test Case 1: File nhỏ (<5MB), tiếng Việt**
- Expected: `bge-m3:567m` + `bge_db`
- Check: `vector_store/bge_db/index.faiss` tồn tại

### **Test Case 2: File lớn (>5MB), tiếng Việt**
- Expected: `qwen3-embedding:0.6b` + `qwen_db`
- Check: `vector_store/qwen_db/index.faiss` tồn tại

### **Test Case 3: File tiếng Anh**
- Expected: `nomic-embed-text` + `nomic_v1_db` hoặc `nomic_v2_db`
- Check: `vector_store/nomic_v1_db/` hoặc `nomic_v2_db/`

---

## 💡 Tips

1. **Xem logs kỹ**: Console output sẽ cho biết chính xác bước nào failed
2. **Test với file nhỏ trước**: File <1MB để test nhanh
3. **Dùng test script**: `python test_memory_rag.py` để test không cần Django
4. **Check Django admin**: Xem `is_embedded` có được set không
5. **Xóa vector store cũ**: `rm -rf vector_store/*/index.faiss` để test lại từ đầu

---

## 🆘 Nếu vẫn không hoạt động

Gửi cho tôi:
1. Console output khi upload file
2. Output của `python test_memory_rag.py`
3. Kết quả `ollama list`
4. Kích thước và loại file bạn đang test

Tôi sẽ help debug chi tiết hơn!
