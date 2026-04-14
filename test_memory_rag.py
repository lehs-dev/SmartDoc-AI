"""
Test script cho Memory-Augmented RAG
Kiểm tra không cần Django server
"""

import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from rag.utils import (
    check_ollama_connection,
    get_embeddings_model,
    get_available_llm_models,
    get_installed_ollama_models,
    extract_text,
    get_text_chunks_optimized,
    route_embedding_target,
)

def test_ollama():
    """Test kết nối Ollama"""
    print("\n" + "="*60)
    print("🧪 TEST 1: OLLAMA CONNECTION")
    print("="*60)
    
    connected = check_ollama_connection()
    if not connected:
        print("❌ Ollama không chạy! Hãy chạy: ollama serve")
        return False
    
    print("✅ Ollama đang chạy\n")
    return True

def test_embedding_models():
    """Test embedding models"""
    print("\n" + "="*60)
    print("🧪 TEST 2: EMBEDDING MODELS")
    print("="*60)
    
    models = get_installed_ollama_models()
    print(f"📦 Models đã pull: {models}")
    
    # Test load embedding model
    test_models = ["nomic-embed-text", "bge-m3:567m"]
    
    for model_name in test_models:
        if model_name in models:
            print(f"\n⏳ Testing {model_name}...")
            try:
                embeddings = get_embeddings_model(model_name)
                print(f"✅ {model_name} loaded thành công")
                
                # Test embedding
                test_text = "Hello world"
                embedding = embeddings.embed_query(test_text)
                print(f"✅ Embedding dimension: {len(embedding)}")
                
            except Exception as e:
                print(f"❌ Lỗi với {model_name}: {str(e)}")
        else:
            print(f"⚠️  {model_name} chưa được pull")
    
    print()

def test_llm_models():
    """Test LLM models"""
    print("\n" + "="*60)
    print("🧪 TEST 3: LLM MODELS")
    print("="*60)
    
    available = get_available_llm_models()
    print(f"✅ LLM models available: {available}")
    print()

def test_routing():
    """Test embedding routing logic"""
    print("\n" + "="*60)
    print("🧪 TEST 4: EMBEDDING ROUTING")
    print("="*60)
    
    test_cases = [
        (1024 * 1024 * 3, True),    # 3MB, tiếng Việt
        (1024 * 1024 * 8, True),    # 8MB, tiếng Việt
        (1024 * 1024 * 5, False),   # 5MB, tiếng Anh
        (1024 * 1024 * 15, False),  # 15MB, tiếng Anh
    ]
    
    for size, has_vietnamese in test_cases:
        result = route_embedding_target(size, has_vietnamese)
        print(f"\n📊 File {result['file_size_mb']}MB | Vietnamese: {has_vietnamese}")
        print(f"   → Embedding: {result['embedding_model']}")
        print(f"   → Vector DB: {result['vector_db_key']}")
    
    print()

def test_chunking():
    """Test adaptive chunking"""
    print("\n" + "="*60)
    print("🧪 TEST 5: ADAPTIVE CHUNKING")
    print("="*60)
    
    test_text = """
    Đây là văn bản tiếng Việt để test chunking.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    
    Paragraph 2:
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore.
    
    Paragraph 3:
    Excepteur sint occaecat cupidatat non proident.
    Sunt in culpa qui officia deserunt mollit anim id est laborum.
    """ * 10
    
    chunks = get_text_chunks_optimized(test_text, file_size_mb=3, has_vietnamese=True)
    print(f"✅ Số chunks: {len(chunks)}")
    print(f"✅ Chunk sizes: {[len(c) for c in chunks[:5]]}...")
    print()

def test_text_extraction():
    """Test text extraction (nếu có file test)"""
    print("\n" + "="*60)
    print("🧪 TEST 6: TEXT EXTRACTION")
    print("="*60)
    
    # Tìm file PDF/DOCX trong thư mục media
    media_dir = os.path.join(os.path.dirname(__file__), 'media', 'documents')
    
    if os.path.exists(media_dir):
        files = [f for f in os.listdir(media_dir) if f.endswith(('.pdf', '.docx'))]
        if files:
            test_file = os.path.join(media_dir, files[0])
            ext = 'pdf' if test_file.endswith('.pdf') else 'docx'
            
            print(f"📄 Testing với file: {test_file}")
            text = extract_text(test_file, ext)
            print(f"📊 Extracted {len(text)} characters")
            
            if text:
                print(f"📝 Preview: {text[:200]}...")
        else:
            print("⚠️  Không có file PDF/DOCX trong media/documents")
    else:
        print("⚠️  Thư mục media/documents không tồn tại")
    
    print()

if __name__ == "__main__":
    print("\n" + "🚀 " + "="*58)
    print("🧪 MEMORY-AUGMENTED RAG - TEST SUITE")
    print("="*60 + "\n")
    
    # Test 1: Ollama connection
    if not test_ollama():
        print("\n❌ Dừng test vì Ollama không chạy!")
        sys.exit(1)
    
    # Test 2: Embedding models
    test_embedding_models()
    
    # Test 3: LLM models
    test_llm_models()
    
    # Test 4: Routing
    test_routing()
    
    # Test 5: Chunking
    test_chunking()
    
    # Test 6: Text extraction
    test_text_extraction()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60 + "\n")
