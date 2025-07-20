import React, { useState } from "react";
import "@fontsource/roboto-condensed";
import "../styles/kolochat.css";
import Logo from "../static/KOLOBOK.svg";
import { useNavigate } from "react-router-dom";
import { analyzeThread, extractInformation } from "../api/api";

interface Message {
  text: string;
  sender: "user" | "bot";
  image?: string;
}

type Mode = 'analyze' | 'identify' | null;

export default function KolobokChat() {
  const [messages, setMessages] = useState<Message[]>([
    { text: "Привет! Напиши 'Анализ шипов' или 'Марка и модель'", sender: "bot" },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [mode, setMode] = useState<Mode>(null);
  const navigate = useNavigate();

  const appendMessage = (msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  };

  const handleGoToMain = () => {
    navigate("/");
  };

  const handleSendMessage = () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;

    appendMessage({ text: trimmed, sender: "user" });

    if (trimmed.toLowerCase() === 'анализ шипов') {
      setMode('analyze');
      appendMessage({ text: "Пожалуйста, отправь фото для анализа протектора.", sender: "bot" });
    } else if (trimmed.toLowerCase() === 'марка и модель') {
      setMode('identify');
      appendMessage({ text: "Пожалуйста, отправь фото для определения марки и модели.", sender: "bot" });
    } else {
      appendMessage({ text: "Команда не распознана. Напиши 'Анализ шипов' или 'Марка и модель'.", sender: "bot" });
    }

    setInputValue("");
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const input = e.target;
    const file = input.files?.[0];
    if (!file) return;

    if (file.size > 5 * 1024 * 1024) {
      alert("Размер изображения слишком большой. Пожалуйста, ≤5 Мб.");
      input.value = '';
      return;
    }
    if (!["image/jpeg", "image/png"].includes(file.type)) {
      alert("Только JPG/PNG, пожалуйста.");
      input.value = '';
      return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
      const dataUrl = reader.result as string;
      appendMessage({ text: dataUrl, sender: "user", image: dataUrl });
      setInputValue("");

      if (!mode) {
        appendMessage({ text: "Сначала выбери команду: 'Анализ шипов' или 'Марка и модель'.", sender: "bot" });
        input.value = '';
        return;
      }

      const base64 = dataUrl.split(",")[1];

      if (mode === 'analyze') {
        try {
          const threadRes = await analyzeThread(base64);
          // Выводим текстовые данные
          appendMessage({ sender: "bot", text: `Уровень успеха: ${threadRes.success}` });
          appendMessage({ sender: "bot", text: `Глубина протектора: ${threadRes.thread_depth.toFixed(1)} мм.` });
          appendMessage({ sender: "bot", text: `Обнаружено ${threadRes.spikes.length} шипов.` });
          appendMessage({ sender: "bot", text: `Классы шипов: ${threadRes.spikes.map(s => s.class).join(', ')}.` });
          // Показываем аннотированное изображение
          appendMessage({ sender: "bot", image: `data:image/png;base64,${threadRes.image}`, text: "Аннотированное изображение:" });
        } catch (err: any) {
          appendMessage({ sender: "bot", text: `Ошибка анализа: ${err.message}` });
        }
      } else {
        // mode === 'identify'
        try {
          const infoRes = await extractInformation(base64);
          if (infoRes.index_results && infoRes.index_results.length > 0) {
            // Выводим все варианты
            infoRes.index_results
              .sort((a, b) => b.combined_score - a.combined_score)
              .forEach((item, idx) => {
                appendMessage({
                  sender: "bot",
                  text: `Вариант ${idx + 1}: ${item.brand_name} ${item.model_name} (оценка ${item.combined_score.toFixed(2)})`,
                });
              });
            // И отдельно размер шины
            appendMessage({ sender: "bot", text: `Размер шины: ${infoRes.tire_size}` });
          } else {
            appendMessage({ sender: "bot", text: "Не удалось определить марку и модель шины." });
          }
        } catch (err: any) {
          appendMessage({ sender: "bot", text: `Ошибка извлечения информации: ${err.message}` });
        }
      }

      setMode(null);
      input.value = '';
    };

    reader.onerror = () => {
      alert("Не удалось прочитать файл.");
      input.value = '';
    };
    reader.readAsDataURL(file);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSendMessage();
  };

  return (
    <div className="kolobok-container">
      <div className="kolobok-header">
        <div className="kolobok-logo" onClick={handleGoToMain}>
          <img src={Logo} alt="Kolobok Logo" />
        </div>
        <button className="kolobok-telegram">Telegram-bot</button>
      </div>

      <div className="chat-window">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.sender}`}>
            {!msg.image && <p>{msg.text}</p>}
            {msg.image && (
              <div className="chat-image">
                <img
                  src={msg.image}
                  alt={msg.sender === "user" ? "Ваше фото" : "Ответ бота"}
                  style={{
                    maxWidth: "100%",
                    maxHeight: "300px",
                    objectFit: "contain",
                    borderRadius: "10px",
                    boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
                  }}
                />
                {msg.sender === "bot" && <p style={{ textAlign: "center" }}>Аннотированное изображение</p>}
              </div>
            )}
          </div>
        ))}
         <div className="chat-input">
        <button className="attach-button" onClick={() => document.getElementById("file-input")?.click()}>
          📎
        </button>
        <input
          type="file"
          accept="image/jpeg, image/png"
          style={{ display: "none" }}
          id="file-input"
          onChange={handleImageUpload}
        />
        <input
          type="text"
          className="message-input"
          placeholder="Написать сообщение..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <button className="send-button" onClick={handleSendMessage}>
          ➤
        </button>
      </div>
      </div>

     
    </div>
  );
}
