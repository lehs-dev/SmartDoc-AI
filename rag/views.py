from django.shortcuts import render
from django.contrib import messages
from .models import Document, ChatSession, ChatMessage
from .utils import process_document, get_vector_store, ask_gemma
import json
from django.http import JsonResponse

def index(request):
    sessions = ChatSession.objects.all().order_by('-created_at')

    current_session_id = request.GET.get('session')
    current_messages = []

    if current_session_id:
        # Lấy toàn bộ tin nhắn của session đó
        current_messages = ChatMessage.objects.filter(session_id=current_session_id).order_by('created_at')

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

    context = {
        'sessions' : sessions,
        'current_session_id' : current_session_id,
        'current_messages' : current_messages,
    }
    return render(request, 'rag/index.html', context)

def chat_api(request):
    # Nhận thông tin từ giao diện và trả về câu hỏi của gemma4
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_question = data.get('message')
            session_id = data.get('session_id')

            if not user_question:
                return JsonResponse({'error' : 'phải nhập câu hỏi'}, status=400)
            
            # ... (phần đầu giữ nguyên)
            if session_id:
                session = ChatSession.objects.get(id=session_id)
            else:
                title = user_question[:30] + "..." if len(user_question) > 30 else user_question
                session = ChatSession.objects.create(title=title)
            
            # 🚀 LẤY TRÍ NHỚ: Rút 6 tin nhắn gần nhất (trước khi lưu tin mới)
            past_messages = ChatMessage.objects.filter(session=session).order_by('-created_at')[:6]
            # Lật ngược lại để chat cũ nằm trên, chat mới nằm dưới
            past_messages = reversed(list(past_messages))
            
            chat_history_text = ""
            for msg in past_messages:
                role_name = "Người dùng" if msg.role == 'user' else "SmartDoc AI"
                chat_history_text += f"{role_name}: {msg.content}\n"

            # Lưu câu hỏi mới của User
            ChatMessage.objects.create(session=session, role='user', content=user_question)
            
            # 🚀 Truyền thêm chat_history_text vào
            bot_answer = ask_gemma(user_question, chat_history_text)

            # Lưu câu trả lời của AI
            ChatMessage.objects.create(session=session, role='ai', content=bot_answer)

            return JsonResponse({'response' : bot_answer, 'session_id' : session.id})
        
        except Exception as e :
            return JsonResponse({'error' : str(e)}, status=500)
        
    return JsonResponse({'error' : 'Câu hỏi phải là câu hỏi'}, status=400)
