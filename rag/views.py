from django.shortcuts import render
from django.contrib import messages
from .models import Document
from .utils import process_document, get_vector_store

def index(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('document')

        if uploaded_file:
            file_extension = 'pdf' if uploaded_file.name.lower().endswith('.pdf') else 'docx'

            doc = Document.objects.create(
                file=uploaded_file,
                file_type=file_extension
            )

            #Chunkin' ở chỗ này
            chunks = process_document(doc.file.path, doc.file_type)

            if chunks:
                get_vector_store(chunks)
                doc.is_embedded = True
                doc.save()
                messages.success(request, f'tải thành công: {doc.filename}. Hệ thống chia thành {len(chunks)} đoạn dữ liệu')
            else:
                messages.warning(request, f'Đã tải file nhưng không tìm thấy dữ liệu')

            messages.success(request, f'Đã tải lên và lưu thành công: {doc.filename}')
        else: 
            messages.error(request, 'file tải lên chỉ hỗ trợ PDF và DOCX')
    return render(request, 'rag/index.html')
