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

export default function KolobokChat() {
  const [messages, setMessages] = useState<Message[]>([
    { text: "Привет, чем могу помочь?", sender: "bot" },
  ]);
  const [inputValue, setInputValue] = useState("");
  const navigate = useNavigate();

  const appendMessage = (msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  };

  const handleGoToMain = () => {
    navigate("/");
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 5 * 1024 * 1024) {
      alert("Размер изображения слишком большой. Пожалуйста, ≤5 Мб.");
      return;
    }
    if (!["image/jpeg", "image/png"].includes(file.type)) {
      alert("Только JPG/PNG, пожалуйста.");
      return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
      const dataUrl = reader.result as string;
      // Добавляем фото пользователя
      appendMessage({ text: dataUrl, sender: "user", image: dataUrl });

      const base64 = dataUrl.split(",")[1];

      // Анализ протектора
      try {
        const threadRes = await analyzeThread(base64);
        appendMessage({
          sender: "bot",
          text: `Глубина протектора: ${threadRes.thread_depth.toFixed(1)} мм. Спайков: ${threadRes.spikes.length}.`,
          // <-- вот тут оборачиваем:
          image: `data:image/png;base64,${threadRes.image}`,
        });
      } catch (err: any) {
        appendMessage({
          sender: "bot",
          text: `Ошибка при анализе протектора: ${err.message}`,
        });
      }

      // Извлечение информации о шине
      try {
        const infoRes = await extractInformation(base64);
        const best = infoRes.index_results.sort((a, b) => b.combined_score - a.combined_score)[0];
        appendMessage({
          sender: "bot",
          text: `Возможная шина: ${best.brand_name} ${best.model_name} (оценка ${best.combined_score.toFixed(2)}), размер ${infoRes.tire_size}".`,
        });
      } catch (err: any) {
        appendMessage({
          sender: "bot",
          text: `Ошибка при извлечении информации: ${err.message}`,
        });
      }
    };

    reader.onerror = () => {
      alert("Не удалось прочитать файл.");
    };
    reader.readAsDataURL(file);
  };

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;
    appendMessage({ text: inputValue.trim(), sender: "user" });
    appendMessage({ text: "Отправь, пожалуйста, фото 😊", sender: "bot" });
    setInputValue("");
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSendMessage();
  };

  return (
    <div className="kolobok-container">
      {/* Header */}
      <div className="kolobok-header">
        <div className="kolobok-logo" onClick={handleGoToMain}>
          <img src={Logo} alt="Kolobok Logo" />
        </div>
        <button className="kolobok-telegram">Telegram-bot</button>
      </div>

      {/* Chat Section */}
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
                {msg.sender === "bot" && (
                  <p style={{ textAlign: "center" }}>Аннотированное изображение</p>
                )}
              </div>
            )}
          </div>
        ))}

         {/* Input Section */}
      <div className="chat-input">
        <button
          className="attach-button"
          onClick={() => document.getElementById("file-input")?.click()}
        >
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
