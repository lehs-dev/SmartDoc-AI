from django.db import models
import os

LLM_MODEL_CHOICES = (
    ('gemma4:e2b', 'gemma4:e2b'),
    ('qwen3.5:0.8b', 'qwen3.5:0.8b'),
    ('qwen3.5:2b', 'qwen3.5:2b'),
    ('qwen3.5:4b', 'qwen3.5:4b'),
    ('qwen3.5:9b', 'qwen3.5:9b'),
)

EMBEDDING_MODEL_CHOICES = (
    ('nomic-embed-text-v2-moe', 'nomic-embed-text-v2-moe'),
    ('qwen3-embedding:0.6b', 'qwen3-embedding:0.6b'),
    ('nomic-embed-text', 'nomic-embed-text'),
    ('bge-m3:567m', 'bge-m3:567m'),
)

VECTOR_DB_CHOICES = (
    ('qwen_db', 'qwen_db'),
    ('bge_db', 'bge_db'),
    ('nomic_v2_db', 'nomic_v2_db'),
    ('nomic_v1_db', 'nomic_v1_db'),
)

class Document(models.Model):
    FILE_TYPES = (
        ('pdf', 'PDF'),
        ('docx', 'DOCX'),
    )

    file = models.FileField(upload_to='documents/')
    filename = models.CharField(max_length=255, blank=True)
    file_type = models.CharField(max_length=10, choices=FILE_TYPES)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_embedded = models.BooleanField(default=False)
    has_vietnamese = models.BooleanField(default=False)
    file_size_mb = models.FloatField(default=0)
    embedding_model = models.CharField(max_length=64, choices=EMBEDDING_MODEL_CHOICES, blank=True)
    vector_db_key = models.CharField(max_length=32, choices=VECTOR_DB_CHOICES, blank=True)

    def save(self, *args, **kwargs):
        if self.file and not self.filename:
            self.filename = os.path.basename(self.file.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.filename
    
class ChatSession(models.Model):
    title = models.CharField(max_length=255, default="New chat")
    document = models.ForeignKey(Document, on_delete=models.SET_NULL, null=True, blank=True, related_name='chat_sessions')
    llm_model = models.CharField(max_length=50, choices=LLM_MODEL_CHOICES, default='gemma4:e2b')
    embedding_model = models.CharField(max_length=64, choices=EMBEDDING_MODEL_CHOICES, blank=True)
    vector_db_key = models.CharField(max_length=32, choices=VECTOR_DB_CHOICES, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} - {self.created_at.strftime('%d/%m/%Y %H:%M')}"
    
class ChatMessage(models.Model):
    ROLES_CHOICES = (
        ('user', 'Người dùng'),
        ('ai', 'Gemma 4'),
    )

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLES_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.get_role_display()}] {self.content[:50]}..."
