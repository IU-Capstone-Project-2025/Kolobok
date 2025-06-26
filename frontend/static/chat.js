const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const fileInput = document.querySelector('input[type="file"]');

const API_URL = 'http://localhost:5000';

async function loadMessages() {
    const res = await fetch(`${API_URL}/api/messages`);
    const messages = await res.json();
    messagesContainer.innerHTML = '';
    messages.forEach(msg => {
        const div = document.createElement('div');
        div.className = 'message ' + (msg.role === 'user' ? 'user' : 'bot');
        if (msg.text) {
            const textDiv = document.createElement('div');
            textDiv.textContent = msg.text;
            div.appendChild(textDiv);
        }
        if (msg.image) {
            const img = document.createElement('img');
            img.src = `${API_URL}${msg.image}`;
            img.alt = 'image';
            img.style.maxWidth = '200px';
            img.style.display = 'block';
            img.style.marginTop = '10px';
            div.appendChild(img);
        }
        messagesContainer.appendChild(div);
    });
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function sendMessage(text, imageUrl = null) {
    // Получаем текущие сообщения
    const res = await fetch(`${API_URL}/api/messages`);
    const messages = await res.json();
    messages.push({ role: 'user', text, image: imageUrl });
    // Сохраняем на сервере
    await fetch(`${API_URL}/api/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(messages)
    });
    // Ответ GOIDA
    messages.push({ role: 'bot', text: 'GOIDA' });
    await fetch(`${API_URL}/api/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(messages)
    });
    loadMessages();
}

chatForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    const text = userInput.value.trim();
    let imageUrl = null;
    if (fileInput.files.length > 0) {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        const res = await fetch(`${API_URL}/api/upload`, { method: 'POST', body: formData });
        const data = await res.json();
        imageUrl = data.url;
        fileInput.value = '';
    }
    if (!text && !imageUrl) return;
    await sendMessage(text, imageUrl);
    userInput.value = '';
});

loadMessages(); 