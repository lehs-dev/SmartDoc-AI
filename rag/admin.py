from django.contrib import admin
from .models import Document, ChatSession, ChatMessage

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('filename', 'file_type', 'file_size_mb', 'has_vietnamese', 'embedding_model', 'vector_db_key', 'uploaded_at', 'is_embedded')
    list_filter  = ('file_type', 'has_vietnamese', 'embedding_model', 'vector_db_key', 'is_embedded')

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('title', 'document', 'llm_model', 'embedding_model', 'vector_db_key', 'created_at')

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('session', 'role', 'created_at', 'content')
    list_filter = ('role', 'session')
