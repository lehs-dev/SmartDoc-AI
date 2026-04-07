from django.shortcuts import render
from django.contrib import messages
from .models import Document
from .utils import process_document, get_vector_store, ask_gemma
import json
from django.http import JsonResponse

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

def chat_api(request):
    # Nhận thông tin từ giao diện và trả về câu hỏi của gemma4
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_question = data.get('message')

            if not user_question:
                return JsonResponse({'error' : 'phải nhập câu hỏi'}, status=400)
            
            bot_answer = ask_gemma(user_question)

            return JsonResponse({'response' : bot_answer})
        
        except Exception as e :
            return JsonResponse({'error' : str(e)}, status=500)
        
    return JsonResponse({'error' : 'Câu hỏi phải là câu hỏi'}, status=400)
