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
    const llmModelSelect = document.getElementById('llm-model');
    const modeBadge = document.getElementById('mode-badge');
    const toastContainer = document.getElementById('toast-container');
    const confirmOverlay = document.getElementById('delete-confirm-dialog');
    const confirmCancelBtn = document.getElementById('delete-confirm-cancel');
    const confirmOkBtn = document.getElementById('delete-confirm-ok');
    const confirmTarget = document.getElementById('delete-confirm-target');

    let currentSessionId = chatArea?.dataset.sessionId || '';
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';

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

    function switchToConversation() {
        if (!chatArea) return;
        chatArea.classList.remove('is-empty');
        if (welcomeSection) welcomeSection.style.display = 'none';
        // Show messages-wrap if hidden
        if (messagesWrap) messagesWrap.style.display = '';
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
            uploadModal.classList.add('is-open');
        });
    }

    if (cancelUpload) {
        cancelUpload.addEventListener('click', () => uploadModal.classList.remove('is-open'));
    }

    if (uploadModal) {
        uploadModal.addEventListener('click', (e) => {
            if (e.target === uploadModal) uploadModal.classList.remove('is-open');
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(uploadForm);

            try {
                const response = await fetch(window.location.href, {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    uploadModal.classList.remove('is-open');
                    showToast('Tải tài liệu thành công! Đang chuyển sang chế độ RAG…', 'success');
                    setTimeout(() => window.location.reload(), 1500);
                } else {
                    const err = await response.text();
                    showToast('Lỗi: ' + err, 'danger');
                }
            } catch {
                showToast('Không thể kết nối server', 'danger');
            }
        });
    }

    /* ── Send message ───────────────────────────── */
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        const selectedModel = llmModelSelect?.value || '';

        // Clear + reset height
        chatInput.value = '';
        chatInput.style.height = '';

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
                    llm_model: selectedModel,
                    document_id: null
                })
            });

            if (!response.ok) {
                const err = await response.text();
                throw new Error(err);
            }

            const newSessionId = response.headers.get('X-Session-Id');
            const mode = response.headers.get('X-Mode');

            if (newSessionId && !currentSessionId) {
                currentSessionId = newSessionId;
                window.history.pushState({}, '', `/?session=${newSessionId}`);
            }

            // Mode badge update
            if (modeBadge && mode) {
                if (mode === 'rag') {
                    modeBadge.className = 'mode-badge is-rag';
                    modeBadge.innerHTML = '<i class="bi bi-journal-text"></i><span>Chế độ RAG – Có tài liệu ngữ cảnh</span>';
                } else {
                    modeBadge.className = 'mode-badge';
                    modeBadge.innerHTML = '<i class="bi bi-chat-dots"></i><span>Chat thường</span>';
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

                aiContent += decoder.decode(value);

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

        } catch (err) {
            typingEl.remove();
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