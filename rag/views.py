from django.shortcuts import render
from django.contrib import messages
from .models import Document

def index(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('document')

        if uploaded_file:
            file_extension = 'pdf' if uploaded_file.name.lower().endswith('.pdf') else 'docx'

            doc = Document.objects.create(
                file=uploaded_file,
                file_type=file_extension
            )

            messages.success(request, f'Đã tải lên và lưu thành công: {doc.filename}')
        else: 
            messages.error(request, 'file tải lên chỉ hỗ trợ PDF và DOCX')
    return render(request, 'rag/index.html')
