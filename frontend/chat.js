// Markdown renderer
const md = window.markdownit({
    highlight: function (str, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try { return hljs.highlight(str, { language: lang }).value; } catch (__) { }
        }
        return '';
    }
});

const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatHistory = document.getElementById('chat-history');
const attachBtn = document.getElementById('attach-btn');
const fileUploadInput = document.getElementById('file-upload');
const filePreviewArea = document.getElementById('file-preview-area');
const fileNameDisplay = document.getElementById('file-name');
const fileExtDisplay = document.getElementById('file-ext');
const removeFileBtn = document.getElementById('remove-file-btn');

let currentFile = null;

// Tự động co giãn Textarea & Sáng nút gửi
chatInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight < 200 ? this.scrollHeight : 200) + 'px';

    if (this.value.trim().length > 0 || currentFile) {
        sendBtn.classList.add('active');
        sendBtn.disabled = false;
    } else {
        sendBtn.classList.remove('active');
        sendBtn.disabled = true;
    }
});

// Nút đính kèm File
attachBtn.addEventListener('click', () => {
    fileUploadInput.click();
});

// Xử lý khi chọn file
fileUploadInput.addEventListener('change', function () {
    if (this.files && this.files[0]) {
        currentFile = this.files[0];

        // Cập nhật UI Thẻ File
        fileNameDisplay.textContent = currentFile.name.split('.')[0]; // Tên file
        fileExtDisplay.textContent = currentFile.name.split('.').pop().toUpperCase(); // Đuôi file

        filePreviewArea.style.display = 'block'; // Hiện thẻ file trong input box

        sendBtn.classList.add('active');
        sendBtn.disabled = false;
    }
});

// Xóa file đã chọn
removeFileBtn.addEventListener('click', () => {
    currentFile = null;
    fileUploadInput.value = '';
    filePreviewArea.style.display = 'none';
    chatInput.dispatchEvent(new Event('input')); // Check lại nút gửi
});

// Cuộn xuống cuối chat
function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Render bong bóng chat
function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;

    const htmlContent = role === 'ai' ? md.render(text) : text.replace(/\n/g, '<br>');

    // Nếu là user và có file đính kèm, thêm cái badge báo file vào chat
    let fileBadge = '';
    if (role === 'user' && currentFile) {
        fileBadge = `<div style="background: rgba(0,0,0,0.05); padding: 4px 8px; border-radius: 8px; font-size: 0.8rem; margin-bottom: 8px; display: inline-block;">
            📎 ${currentFile.name}
        </div><br>`;
    }

    if (role === 'user') {
        msgDiv.innerHTML = `<div class="content">${fileBadge}${htmlContent}</div>`;
    } else {
        msgDiv.innerHTML = `
            <div class="avatar"></div>
            <div class="content">${htmlContent}</div>
        `;
    }

    chatHistory.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv.querySelector('.content');
}

// Gửi tin nhắn
async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text && !currentFile) return;

    // KÍCH HOẠT CHẾ ĐỘ CHAT (Dịch chuyển input xuống dưới, hiện lịch sử)
    document.body.classList.add('is-chatting');

    // Hiển thị chat của user
    appendMessage('user', text);

    // Reset khung nhập
    chatInput.value = '';
    chatInput.style.height = 'auto';
    filePreviewArea.style.display = 'none';
    sendBtn.classList.remove('active');
    sendBtn.disabled = true;

    // Giữ biến file để mốt gọi API (hiện tại demo stream text trước)
    const fileToSend = currentFile;
    currentFile = null;

    // Hiện ô chờ của AI
    const aiContentDiv = appendMessage('ai', '...');

    try {
        // --- BƯỚC 1: NẾU CÓ FILE THÌ UPLOAD LÊN FAISS TRƯỚC ---
        if (fileToSend) {
            aiContentDiv.innerHTML = "<em>Đang đọc và học tài liệu của bạn... (Xin chờ vài giây) ⏳</em>";

            const formData = new FormData();
            formData.append("file", fileToSend);
            formData.append("session_id", "demo-session-1");

            const uploadRes = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadRes.ok) throw new Error("Lỗi tải tài liệu lên hệ thống!");
            // Đã upload xong, đổi text để báo AI bắt đầu nghĩ câu trả lời
            aiContentDiv.innerHTML = "<em>Đang suy nghĩ... 🤔</em>";
        }

        // --- BƯỚC 2: GỌI API CHAT ĐỂ TRẢ LỜI ---
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                session_id: "demo-session-1"
            })
        });

        if (!response.ok) throw new Error("Lỗi kết nối tới Server Chat");

        // --- BƯỚC 3: HỨNG LUỒNG CHỮ TRẢ VỀ ---
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let aiFullText = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            aiFullText += decoder.decode(value, { stream: true });
            aiContentDiv.innerHTML = md.render(aiFullText);
            scrollToBottom();
        }
    } catch (error) {
        aiContentDiv.innerHTML = `<span style="color: red;">[Lỗi]: ${error.message}</span>`;
    }
}

// Bắt phím Enter
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (chatInput.value.trim() || currentFile) sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);