from django.contrib import admin
from .models import Document, ChatSession, ChatMessage

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('filename', 'file_type', 'uploaded_at', 'is_embedded')
    list_filter  = ('file_type', 'is_embedded')

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at')

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('session', 'role', 'created_at', 'content')
    list_filter = ('role', 'session')
