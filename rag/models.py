from django.db import models
import os

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

    def save(self, *args, **kwargs):
        if self.file and not self.filename:
            self.filename = os.path.basename(self.file.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.filename
    
class ChatSession(models.Model):
    title = models.CharField(max_length=255, default="New chat")
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
