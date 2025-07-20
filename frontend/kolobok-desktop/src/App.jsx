import { useState, useRef, useEffect } from "react";
import "./App.css";
import tireImg from "./assets/tire.png";

// Состояния приложения (как в logic.py + chat)
const STATES = {
  MENU: "MENU",
  SIDE_PHOTO: "SIDE_PHOTO",
  SIDE_RESULT: "SIDE_RESULT",
  SIDE_CUSTOM: "SIDE_CUSTOM",
  TREAD_PHOTO: "TREAD_PHOTO",
  TREAD_RESULT: "TREAD_RESULT",
  TREAD_CUSTOM: "TREAD_CUSTOM",
  CHAT: "CHAT",
};

function App() {
  const [state, setState] = useState(STATES.MENU);
  const [sideCustom, setSideCustom] = useState("");
  const [treadCustom, setTreadCustom] = useState("");
  
  // Chat functionality
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load chat history from localStorage
  useEffect(() => {
    const savedMessages = localStorage.getItem('chatHistory');
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages));
    }
  }, []);

  // Save messages to localStorage
  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(messages));
  }, [messages]);

  const sendMessage = async (text, imageUrl = null) => {
    const newMessage = { role: 'user', text, image: imageUrl };
    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    
    // Simulate bot response (like in the original chat.js)
    setTimeout(() => {
      const botResponse = { role: 'bot', text: 'GOIDA' };
      setMessages(prev => [...prev, botResponse]);
    }, 1000);
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() && !selectedFile) return;

    let imageUrl = null;
    if (selectedFile) {
      // In a real app, you'd upload to server
      // For now, create a local URL
      imageUrl = URL.createObjectURL(selectedFile);
    }

    await sendMessage(chatInput, imageUrl);
    setChatInput("");
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
  };

  // Главное меню с новым стилем
  if (state === STATES.MENU) {
    return (
      <>
        <header>
          <span className="logo">Kolobok</span>
        </header>
        <main>
          <div className="left-block">
            <div className="blob">
              <img src={tireImg} alt="tire" className="tire-img" />
            </div>
          </div>
          <div className="right-block">
            <h1>
              Чем я могу помочь?
              <span className="custom-bar" />
            </h1>
            <div className="buttons">
              <button onClick={() => setState(STATES.SIDE_PHOTO)}>
                <span>Марка и модель шины</span>
              </button>
              <button onClick={() => setState(STATES.TREAD_PHOTO)}>
                <span>Глубина протектора</span>
              </button>
              <button onClick={() => setState(STATES.CHAT)}>
                <span>Чат с ботом</span>
              </button>
            </div>
          </div>
        </main>
      </>
    );
  }

  // CHAT: Chat interface with original class names
  if (state === STATES.CHAT) {
    return (
      <>
        <header>
          <span className="logo" style={{ cursor: 'pointer' }} onClick={() => setState(STATES.MENU)}>Kolobok</span>
        </header>
        <main>
          <div className="chat-container">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
              <h2>Чат с ботом</h2>
              <button onClick={() => setState(STATES.MENU)}>Назад в меню</button>
            </div>
            <div className="messages">
              {messages.map((msg, index) => (
                <div key={index} className={`message ${msg.role}`}>
                  {msg.text && <div>{msg.text}</div>}
                  {msg.image && (
                    <img
                      src={msg.image}
                      alt="uploaded"
                    />
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
            <form className="input-area" onSubmit={handleChatSubmit}>
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Введите сообщение..."
              />
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                style={{ display: 'none' }}
              />
              <span className="attach-icon" onClick={() => fileInputRef.current?.click()} title="Прикрепить изображение">
                📷
              </span>
              <button type="submit" className="send-btn" title="Отправить">
                <svg width="28" height="28" viewBox="0 0 24 24"><path fill="#a259e6" d="M2 21l21-9-21-9v7l15 2-15 2z"/></svg>
              </button>
            </form>
            {selectedFile && (
              <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
                Выбран файл: {selectedFile.name}
              </div>
            )}
          </div>
        </main>
      </>
    );
  }

  // SIDE_PHOTO: Загрузка фото (имитация)
  if (state === STATES.SIDE_PHOTO) {
    return (
      <main>
        <div className="right-block">
          <h2>Загрузите фотографию боковой стороны шины</h2>
          <div className="buttons">
            <button onClick={() => setState(STATES.SIDE_RESULT)}>
              Имитация загрузки (далее)
            </button>
            <button onClick={() => setState(STATES.MENU)}>Назад</button>
          </div>
        </div>
      </main>
    );
  }

  // SIDE_RESULT: Результат обработки фото
  if (state === STATES.SIDE_RESULT) {
    return (
      <main>
        <div className="right-block">
          <h2>Марка: …<br />Модель: …<br />Диаметр: …</h2>
          <div className="buttons">
            <button onClick={() => setState(STATES.MENU)}>OK</button>
            <button onClick={() => setState(STATES.SIDE_CUSTOM)}>Свой вариант</button>
          </div>
        </div>
      </main>
    );
  }

  // SIDE_CUSTOM: Ввод своего варианта
  if (state === STATES.SIDE_CUSTOM) {
    return (
      <main>
        <div className="right-block">
          <h2>Введите свой вариант марки, модели и диаметра шины:</h2>
          <input
            type="text"
            value={sideCustom}
            onChange={e => setSideCustom(e.target.value)}
            placeholder="Ваш вариант..."
            style={{ fontSize: 22, padding: 8, borderRadius: 8, border: '1px solid #a259e6', marginBottom: 16 }}
          />
          <div className="buttons">
            <button onClick={() => setState(STATES.MENU)}>
              Отправить
            </button>
          </div>
        </div>
      </main>
    );
  }

  // TREAD_PHOTO: Загрузка фото протектора (имитация)
  if (state === STATES.TREAD_PHOTO) {
    return (
      <main>
        <div className="right-block">
          <h2>Загрузите фотографию протектора шины.<br />Убедитесь, что шина полностью в кадре</h2>
          <div className="buttons">
            <button onClick={() => setState(STATES.TREAD_RESULT)}>
              Имитация загрузки (далее)
            </button>
            <button onClick={() => setState(STATES.MENU)}>Назад</button>
          </div>
        </div>
      </main>
    );
  }

  // TREAD_RESULT: Результат обработки протектора
  if (state === STATES.TREAD_RESULT) {
    return (
      <main>
        <div className="right-block">
          <h2>Глубина протектора: …<br />Количество шин: …</h2>
          <div className="buttons">
            <button onClick={() => setState(STATES.MENU)}>OK</button>
            <button onClick={() => setState(STATES.TREAD_CUSTOM)}>Свой вариант</button>
          </div>
        </div>
      </main>
    );
  }

  // TREAD_CUSTOM: Ввод своего варианта
  if (state === STATES.TREAD_CUSTOM) {
    return (
      <main>
        <div className="right-block">
          <h2>Введите свой вариант глубины протектора и количества шин:</h2>
          <input
            type="text"
            value={treadCustom}
            onChange={e => setTreadCustom(e.target.value)}
            placeholder="Ваш вариант..."
            style={{ fontSize: 22, padding: 8, borderRadius: 8, border: '1px solid #a259e6', marginBottom: 16 }}
          />
          <div className="buttons">
            <button onClick={() => setState(STATES.MENU)}>
              Отправить
            </button>
          </div>
        </div>
      </main>
    );
  }

  return null;
}

export default App;
