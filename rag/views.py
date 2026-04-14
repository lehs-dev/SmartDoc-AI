from django.shortcuts import render
from django.contrib import messages
from .models import Document, ChatSession, ChatMessage
from .utils import (
    process_document,
    process_document_optimized,
    get_vector_store,
    ask_gemma,
    ask_gemma_with_memory,
    route_embedding_target,
    get_available_llm_models,
    update_conversation_memory,
)
import json
from django.http import JsonResponse, StreamingHttpResponse


def _resolve_document_for_chat(document_id, session=None):
    if document_id:
        selected_doc = Document.objects.filter(id=document_id, is_embedded=True).first()
        if selected_doc:
            return selected_doc

    if session and session.document and session.document.is_embedded:
        return session.document

    return Document.objects.filter(is_embedded=True).order_by('-uploaded_at').first()


def index(request):
    sessions = ChatSession.objects.select_related('document').all().order_by('-created_at')
    embedded_documents = Document.objects.filter(is_embedded=True).order_by('-uploaded_at')
    llm_models = get_available_llm_models()

    current_session_id = request.GET.get('session')
    current_session = None
    current_messages = []

    if current_session_id:
        current_session = ChatSession.objects.select_related('document').filter(id=current_session_id).first()
        if current_session:
            # Lấy toàn bộ tin nhắn của session đó
            current_messages = ChatMessage.objects.filter(session=current_session).order_by('created_at')

    if request.method == 'POST':
        uploaded_file = request.FILES.get('document')
        has_vietnamese = request.POST.get('has_vietnamese') == 'on'

        if uploaded_file:
            lower_name = uploaded_file.name.lower()
            if lower_name.endswith('.pdf'):
                file_extension = 'pdf'
            elif lower_name.endswith('.docx'):
                file_extension = 'docx'
            else:
                messages.error(request, 'File tải lên chỉ hỗ trợ PDF và DOCX')
                file_extension = ''

            if file_extension:
                route_info = route_embedding_target(uploaded_file.size, has_vietnamese)

                doc = Document.objects.create(
                    file=uploaded_file,
                    file_type=file_extension,
                    has_vietnamese=has_vietnamese,
                    file_size_mb=route_info['file_size_mb'],
                    embedding_model=route_info['embedding_model'],
                    vector_db_key=route_info['vector_db_key'],
                )

                # Chunking với adaptive optimization
                chunks = process_document_optimized(
                        doc.file.path, 
                        doc.file_type,
                        route_info['file_size_mb'],
                        has_vietnamese
                    )

                if chunks:
                    # Vectorize và lưu vào FAISS
                    try:
                        get_vector_store(chunks, doc.embedding_model, doc.vector_db_key)
                        doc.is_embedded = True
                        doc.save(update_fields=['is_embedded'])

                        # Nếu đang trong 1 session thì tự gắn tài liệu vừa upload vào session đó.
                        if current_session:
                            current_session.document = doc
                            current_session.embedding_model = doc.embedding_model
                            current_session.vector_db_key = doc.vector_db_key
                            current_session.save(update_fields=['document', 'embedding_model', 'vector_db_key'])

                        messages.success(
                            request,
                            (
                                f'Tải thành công: {doc.filename}. '
                                f'Đã chia {len(chunks)} đoạn | '
                                f'Embedding: {doc.embedding_model} | '
                                f'Kho vector: {doc.vector_db_key}'
                            )
                        )
                    except Exception as e:
                        messages.error(request, f'Lỗi khi vector hóa: {str(e)}')
                        print(f"Vectorization error: {e}")
                else:
                    messages.warning(request, 'Đã tải file nhưng không tìm thấy dữ liệu để nhúng')

                embedded_documents = Document.objects.filter(is_embedded=True).order_by('-uploaded_at')
        else:
            messages.error(request, 'Bạn chưa chọn file để tải lên')

    current_document_id = ''
    if current_session and current_session.document_id:
        current_document_id = str(current_session.document_id)
    elif embedded_documents.exists():
        current_document_id = str(embedded_documents.first().id)

    default_llm_model = current_session.llm_model if current_session else llm_models[0]

    context = {
        'sessions': sessions,
        'embedded_documents': embedded_documents,
        'llm_models': llm_models,
        'default_llm_model': default_llm_model,
        'current_document_id': current_document_id,
        'current_session_id': current_session_id,
        'current_messages': current_messages,
    }
    return render(request, 'rag/index.html', context)


def chat_api(request):
    # Nhận thông tin từ giao diện và trả về câu hỏi của gemma4
    if request.method == 'POST':
        try:
            data = json.loads(request.body or '{}')
            user_question = (data.get('message') or '').strip()
            session_id = data.get('session_id')
            llm_model_name = (data.get('llm_model') or 'gemma4:e4b').strip()
            document_id = data.get('document_id')

            if llm_model_name not in get_available_llm_models():
                return JsonResponse({'error': f'LLM model không hợp lệ: {llm_model_name}'}, status=400)

            if not user_question:
                return JsonResponse({'error': 'Phải nhập câu hỏi'}, status=400)

            if session_id:
                session = ChatSession.objects.select_related('document').filter(id=session_id).first()
                if session is None:
                    return JsonResponse({'error': 'Session không tồn tại'}, status=404)
            else:
                title = user_question[:30] + "..." if len(user_question) > 30 else user_question
                # Tạo session mới với mode='general' mặc định
                session = ChatSession.objects.create(
                    title=title, 
                    llm_model=llm_model_name,
                    mode='general'  # Mặc định là general chat
                )

            # Kiểm tra xem có document không để xác định mode
            selected_doc = _resolve_document_for_chat(document_id, session=session)
            
            if selected_doc and selected_doc.is_embedded:
                # Có document → RAG mode
                session.document = selected_doc
                session.llm_model = llm_model_name
                session.embedding_model = selected_doc.embedding_model
                session.vector_db_key = selected_doc.vector_db_key
                session.mode = 'rag'
                session.save(update_fields=['document', 'llm_model', 'embedding_model', 'vector_db_key', 'mode'])
                
                is_rag_mode = True
                print(f"✅ [CHAT] RAG Mode - Document: {selected_doc.filename}")
            else:
                # Không có document → General Chat mode
                session.llm_model = llm_model_name
                session.mode = 'general'
                session.save(update_fields=['llm_model', 'mode'])
                
                is_rag_mode = False
                print(f"💬 [CHAT] General Mode - Không có document")

            # Lưu câu hỏi mới của User trước khi gọi AI
            ChatMessage.objects.create(session=session, role='user', content=user_question)

            # Gọi AI với mode phù hợp
            # is_rag_mode=True: RAG với memory augmentation
            # is_rag_mode=False: General chat với LLM trực tiếp
            stream_response = ask_gemma_with_memory(
                question=user_question,
                session_id=session.id,
                llm_model_name=llm_model_name,
                embedding_model_name=selected_doc.embedding_model if selected_doc else "",
                vector_db_key=selected_doc.vector_db_key if selected_doc else "",
                use_memory_augmentation=True,  # Luôn dùng memory
                is_rag_mode=is_rag_mode  # Truyền mode vào
            )

            def generate_stream():
                full_answer = ""
                try:
                    for chunk in stream_response:
                        full_answer += chunk
                        yield chunk
                except Exception as stream_error:
                    error_text = (
                        "Loi khi goi mo hinh LLM. "
                        f"Chi tiet: {stream_error}"
                    )
                    full_answer += error_text
                    yield error_text

                # Lưu câu trả lời của AI
                if full_answer.strip():
                    ChatMessage.objects.create(session=session, role='ai', content=full_answer)
                
                # Update memory sau khi hoàn thành câu trả lời (non-blocking)
                try:
                    update_conversation_memory(session.id)
                except Exception as e:
                    print(f"Lỗi khi update memory: {e}")

            response = StreamingHttpResponse(generate_stream(), content_type="text/plain; charset=utf-8")
            response['X-Session-Id'] = str(session.id)
            response['X-Mode'] = session.mode  # Thêm header để frontend biết mode
            if selected_doc:
                response['X-Document-Id'] = str(selected_doc.id)
            return response

        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON gửi lên không hợp lệ'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Phương thức không hợp lệ'}, status=405)


def delete_session_api(request, session_id):
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ'}, status=405)

    session = ChatSession.objects.filter(id=session_id).first()
    if session is None:
        return JsonResponse({'error': 'Hội thoại không tồn tại'}, status=404)

    session.delete()
    return JsonResponse({'ok': True, 'deleted_session_id': session_id})
