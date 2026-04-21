/* ===================================================
   SmartDoc AI – Chat JS
   =================================================== */
document.addEventListener('DOMContentLoaded', function () {

    /* ── DOM refs ───────────────────────────────── */
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesWrap = document.getElementById('messages-wrap');
    const messagesInner = document.getElementById('messages-inner');
    const chatArea = document.getElementById('chat-area');
    const welcomeSection = document.getElementById('welcome-section');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadModal = document.getElementById('upload-modal');
    const cancelUpload = document.getElementById('cancel-upload');
    const uploadForm = document.getElementById('upload-form');
    const fileDropZone = document.querySelector('.file-drop-zone');
    const documentSelect = document.getElementById('document-select');
    const documentInput = document.getElementById('document-input');
    const uploadFilename = document.getElementById('upload-filename');
    const uploadStatus = document.getElementById('upload-status');
    const modeBadge = document.getElementById('mode-badge');
    const chatStatus = document.getElementById('chat-status');
    const toastContainer = document.getElementById('toast-container');
    const confirmOverlay = document.getElementById('delete-confirm-dialog');
    const confirmCancelBtn = document.getElementById('delete-confirm-cancel');
    const confirmOkBtn = document.getElementById('delete-confirm-ok');
    const confirmTarget = document.getElementById('delete-confirm-target');

    let currentSessionId = chatArea?.dataset.sessionId || '';
    let currentDocumentId = chatArea?.dataset.documentId || '';
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';

    if (documentSelect && documentSelect.value) {
        currentDocumentId = documentSelect.value;
    }

    if (documentSelect) {
        documentSelect.addEventListener('change', () => {
            currentDocumentId = documentSelect.value || '';
            if (chatArea) {
                chatArea.dataset.documentId = currentDocumentId;
            }
        });
    }

    /* ── Helpers ────────────────────────────────── */
    function scrollToBottom() {
        if (messagesWrap) {
            messagesWrap.scrollTop = messagesWrap.scrollHeight;
        }
    }

    function showToast(text, level = 'info') {
        if (!toastContainer || !text) return;

        const iconMap = {
            success: 'bi-check-circle-fill',
            danger: 'bi-exclamation-circle-fill',
            warning: 'bi-exclamation-triangle-fill',
            info: 'bi-info-circle-fill'
        };

        const toast = document.createElement('div');
        toast.className = `toast toast-${level}`;
        toast.innerHTML = `<i class="bi ${iconMap[level] || iconMap.info}"></i><span>${text}</span>`;
        toastContainer.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => {
            requestAnimationFrame(() => toast.classList.add('is-visible'));
        });

        setTimeout(() => {
            toast.classList.remove('is-visible');
            setTimeout(() => toast.remove(), 180);
        }, 3200);
    }

    function updateSelectedFilename() {
        if (!uploadFilename) return;
        const selectedFile = documentInput && documentInput.files && documentInput.files[0];
        uploadFilename.textContent = selectedFile ? selectedFile.name : 'Chưa chọn file';
    }

    function switchToConversation() {
        if (!chatArea) return;
        chatArea.classList.remove('is-empty');
        if (welcomeSection) welcomeSection.style.display = 'none';
        // Show messages-wrap if hidden
        if (messagesWrap) messagesWrap.style.display = '';
    }

    function setChatStatus(text, variant = 'thinking') {
        if (!chatStatus) return;

        if (!text) {
            chatStatus.className = 'chat-status';
            chatStatus.style.display = 'none';
            chatStatus.textContent = '';
            return;
        }

        chatStatus.className = `chat-status is-visible ${variant}`;
        chatStatus.textContent = text;
        chatStatus.style.display = '';
    }

    /* Auto-resize textarea */
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    });

    /* ── Suggestion cards ───────────────────────── */
    document.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            chatInput.value = card.dataset.suggestion;
            chatInput.dispatchEvent(new Event('input'));
            chatInput.focus();
        });
    });

    /* ── Upload modal ───────────────────────────── */
    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            if (!uploadModal) return;
            uploadModal.classList.add('is-open');
            updateSelectedFilename();
            if (uploadStatus) {
                uploadStatus.textContent = '';
                uploadStatus.classList.remove('is-busy');
            }
        });
    }

    if (cancelUpload) {
        cancelUpload.addEventListener('click', () => {
            uploadModal.classList.remove('is-open');
            if (uploadStatus) {
                uploadStatus.textContent = '';
                uploadStatus.classList.remove('is-busy');
            }
        });
    }

    if (uploadModal) {
        uploadModal.addEventListener('click', (e) => {
            if (e.target === uploadModal) uploadModal.classList.remove('is-open');
        });
    }

    if (fileDropZone) {
        fileDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileDropZone.classList.add('is-dragover');
        });

        fileDropZone.addEventListener('dragleave', () => {
            fileDropZone.classList.remove('is-dragover');
        });

        fileDropZone.addEventListener('drop', () => {
            fileDropZone.classList.remove('is-dragover');
        });
    }

    if (documentInput) {
        documentInput.addEventListener('change', updateSelectedFilename);
        documentInput.addEventListener('input', updateSelectedFilename);
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!documentInput || !documentInput.files || !documentInput.files.length) {
                showToast('Bạn chưa chọn file để tải lên', 'warning');
                return;
            }

            const submitBtn = uploadForm.querySelector('button[type="submit"]');
            const submitLabel = submitBtn ? submitBtn.textContent : '';
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Đang xử lý...';
            }
            if (cancelUpload) {
                cancelUpload.disabled = true;
            }
            if (uploadStatus) {
                uploadStatus.textContent = 'Đang tải và xử lý tài liệu...';
                uploadStatus.classList.add('is-busy');
            }
            const formData = new FormData(uploadForm);

            try {
                const response = await fetch(window.location.href, {
                    method: 'POST',
                    body: formData,
                    credentials: 'same-origin'
                });

                if (response.ok) {
                    uploadModal.classList.remove('is-open');
                    showToast('Tải tài liệu thành công! Đang chuyển sang chế độ RAG…', 'success');
                    setTimeout(() => window.location.reload(), 1500);
                } else {
                    const err = await response.text();
                    if (uploadStatus) {
                        uploadStatus.textContent = 'Tải tài liệu thất bại. Vui lòng thử lại.';
                        uploadStatus.classList.remove('is-busy');
                    }
                    showToast('Lỗi: ' + err, 'danger');
                }
            } catch {
                if (uploadStatus) {
                    uploadStatus.textContent = 'Không thể kết nối server. Vui lòng thử lại.';
                    uploadStatus.classList.remove('is-busy');
                }
                showToast('Không thể kết nối server', 'danger');
            } finally {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.textContent = submitLabel || 'Tải lên';
                }
                if (cancelUpload) {
                    cancelUpload.disabled = false;
                }
            }
        });
    }

    /* ── Send message ───────────────────────────── */
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        let responseStarted = false;

        // Clear + reset height
        chatInput.value = '';
        chatInput.style.height = '';

        setChatStatus('Đang gửi câu hỏi...', 'thinking');

        // Switch layout if first message
        if (chatArea.classList.contains('is-empty')) {
            switchToConversation();
        }

        // Append user bubble
        const userMsg = document.createElement('div');
        userMsg.className = 'msg msg-user';
        userMsg.innerHTML = `
            <div class="msg-body">
                <div class="msg-bubble">${escapeHtml(message)}</div>
            </div>`;
        messagesInner.appendChild(userMsg);
        scrollToBottom();

        // Typing indicator
        const typingEl = document.createElement('div');
        typingEl.className = 'msg msg-ai';
        typingEl.innerHTML = `
            <div class="msg-avatar ai-avatar">AI</div>
            <div class="msg-body">
                <div class="msg-bubble">
                    <div class="typing-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>`;
        messagesInner.appendChild(typingEl);
        scrollToBottom();

        try {
            const response = await fetch('/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({
                    message,
                    session_id: currentSessionId || null,
                    document_id: currentDocumentId || null,
                    force_general: !currentDocumentId
                })
            });

            if (!response.ok) {
                const err = await response.text();
                throw new Error(err);
            }

            const newSessionId = response.headers.get('X-Session-Id');
            const mode = response.headers.get('X-Mode');
            const newDocumentId = response.headers.get('X-Document-Id');

            if (newSessionId && !currentSessionId) {
                currentSessionId = newSessionId;
                window.history.pushState({}, '', `/?session=${newSessionId}`);
            }

            if (newDocumentId) {
                currentDocumentId = newDocumentId;
                if (documentSelect) {
                    documentSelect.value = newDocumentId;
                }
                if (chatArea) {
                    chatArea.dataset.documentId = newDocumentId;
                }
            }

            // Mode badge update
            if (modeBadge && mode) {
                if (mode === 'rag') {
                    modeBadge.className = 'mode-badge is-rag';
                    modeBadge.innerHTML = '<i class="bi bi-journal-text"></i><span>Chế độ RAG – Có tài liệu ngữ cảnh</span>';
                    setChatStatus('Đang truy xuất tài liệu và ngữ cảnh...', 'rag');
                } else {
                    modeBadge.className = 'mode-badge';
                    modeBadge.innerHTML = '<i class="bi bi-chat-dots"></i><span>Chat thường</span>';
                    setChatStatus('Đang chuẩn bị câu trả lời...', 'thinking');
                }
                modeBadge.style.display = '';
            }

            // Remove typing, start streaming
            typingEl.remove();

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let aiEl = null;
            let aiContent = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                if (!responseStarted) {
                    responseStarted = true;
                    setChatStatus('Đang sinh câu trả lời...', 'thinking');
                }

                aiContent += decoder.decode(value, { stream: true });

                if (!aiEl) {
                    aiEl = document.createElement('div');
                    aiEl.className = 'msg msg-ai';
                    aiEl.innerHTML = `
                        <div class="msg-avatar ai-avatar">AI</div>
                        <div class="msg-body">
                            <div class="msg-bubble"></div>
                        </div>`;
                    messagesInner.appendChild(aiEl);
                }

                aiEl.querySelector('.msg-bubble').textContent = aiContent;
                scrollToBottom();
            }

            setChatStatus('Đã hoàn tất', 'thinking');
            setTimeout(() => setChatStatus(''), 900);

        } catch (err) {
            typingEl.remove();
            setChatStatus('');
            showToast('Lỗi: ' + err.message, 'danger');
        }
    }

    /* Escape HTML for user messages */
    function escapeHtml(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    sendBtn.addEventListener('click', sendMessage);

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    /* ── Delete session ─────────────────────────── */
    let pendingDeleteId = null;
    let pendingDeleteItem = null;

    document.querySelectorAll('.history-item-delete').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            pendingDeleteId = btn.dataset.sessionId;
            pendingDeleteItem = btn.closest('.history-item');

            const title = btn.dataset.sessionTitle || 'hội thoại này';
            if (confirmTarget) confirmTarget.textContent = `"${title}"`;
            if (confirmOverlay) confirmOverlay.classList.add('is-open');
        });
    });

    if (confirmCancelBtn) {
        confirmCancelBtn.addEventListener('click', () => {
            confirmOverlay.classList.remove('is-open');
            pendingDeleteId = null;
            pendingDeleteItem = null;
        });
    }

    if (confirmOverlay) {
        confirmOverlay.addEventListener('click', (e) => {
            if (e.target === confirmOverlay) {
                confirmOverlay.classList.remove('is-open');
                pendingDeleteId = null;
                pendingDeleteItem = null;
            }
        });
    }

    if (confirmOkBtn) {
        confirmOkBtn.addEventListener('click', async () => {
            if (!pendingDeleteId) return;
            confirmOverlay.classList.remove('is-open');

            try {
                const response = await fetch(`/sessions/${pendingDeleteId}/delete/`, {
                    method: 'POST',
                    headers: { 'X-CSRFToken': csrfToken }
                });

                if (response.ok) {
                    pendingDeleteItem?.remove();
                    showToast('Đã xóa hội thoại', 'success');
                    if (currentSessionId === pendingDeleteId) {
                        setTimeout(() => { window.location.href = '/'; }, 900);
                    }
                } else {
                    showToast('Không thể xóa hội thoại', 'danger');
                }
            } catch {
                showToast('Lỗi khi xóa', 'danger');
            }

            pendingDeleteId = null;
            pendingDeleteItem = null;
        });
    }

    /* ── Server messages (Django messages) ─────── */
    document.querySelectorAll('.message-seed-item').forEach(item => {
        showToast(item.dataset.text, item.dataset.level);
    });

    /* ── Init ───────────────────────────────────── */
    chatInput.focus();
    scrollToBottom();
});