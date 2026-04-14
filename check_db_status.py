"""
Quick check database status
"""

import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from rag.models import Document, ChatSession, ChatMessage, ConversationMemory

def check_documents():
    print("\n" + "="*60)
    print("📄 DOCUMENTS STATUS")
    print("="*60)
    
    docs = Document.objects.all().order_by('-uploaded_at')
    print(f"📊 Total documents: {docs.count()}")
    
    embedded = docs.filter(is_embedded=True)
    print(f"✅ Embedded: {embedded.count()}")
    print(f"⏳ Not embedded: {docs.filter(is_embedded=False).count()}")
    
    if docs.exists():
        print(f"\n📋 Recent documents:")
        for doc in docs[:5]:
            status = "✅" if doc.is_embedded else "❌"
            print(f"  {status} {doc.filename} ({doc.file_type}, {doc.file_size_mb:.2f}MB)")
            print(f"      Embedding: {doc.embedding_model}")
            print(f"      Vector DB: {doc.vector_db_key}")
            print(f"      Uploaded: {doc.uploaded_at}")
            print()

def check_sessions():
    print("\n" + "="*60)
    print("💬 CHAT SESSIONS STATUS")
    print("="*60)
    
    sessions = ChatSession.objects.all().order_by('-created_at')
    print(f"📊 Total sessions: {sessions.count()}")
    
    if sessions.exists():
        print(f"\n📋 Recent sessions:")
        for session in sessions[:5]:
            doc_title = session.document.filename if session.document else "No document"
            print(f"  💬 {session.title}")
            print(f"      Document: {doc_title}")
            print(f"      LLM Model: {session.llm_model}")
            print(f"      Embedding: {session.embedding_model}")
            print(f"      Created: {session.created_at}")
            
            msg_count = session.messages.count()
            print(f"      Messages: {msg_count}")
            print()

def check_memory():
    print("\n" + "="*60)
    print("🧠 CONVERSATION MEMORY STATUS")
    print("="*60)
    
    if ConversationMemory.objects.exists():
        memories = ConversationMemory.objects.all()[:5]
        print(f"📊 Total memories: {ConversationMemory.objects.count()}")
        for mem in memories:
            print(f"  🧠 Session: {mem.session.title}")
            print(f"      Summary: {mem.summary[:100]}...")
            print(f"      Updated: {mem.updated_at}")
            print()
    else:
        print("⏳ No conversation memories yet (will be created after chats)")

def check_vector_stores():
    print("\n" + "="*60)
    print("💾 VECTOR STORES ON DISK")
    print("="*60)
    
    vector_dir = os.path.join(os.path.dirname(__file__), 'vector_store')
    
    if os.path.exists(vector_dir):
        print(f"📁 Vector store directory: {vector_dir}")
        
        for db_name in os.listdir(vector_dir):
            db_path = os.path.join(vector_dir, db_name)
            if os.path.isdir(db_path):
                index_path = os.path.join(db_path, 'index.faiss')
                if os.path.exists(index_path):
                    size_mb = os.path.getsize(index_path) / (1024 * 1024)
                    print(f"  ✅ {db_name}/ ({size_mb:.2f} MB)")
                else:
                    print(f"  ⚠️  {db_name}/ (no index.faiss)")
    else:
        print("⚠️  Vector store directory does not exist yet")

if __name__ == "__main__":
    print("\n🔍 " + "="*58)
    print("DATABASE STATUS CHECK")
    print("="*60 + "\n")
    
    check_documents()
    check_sessions()
    check_memory()
    check_vector_stores()
    
    print("\n" + "="*60)
    print("✅ CHECK COMPLETE!")
    print("="*60 + "\n")
