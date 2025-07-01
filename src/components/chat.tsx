import React from "react";
import "@fontsource/roboto-condensed";
import "../styles/kolochat.css";
import Logo from "../static/KOLOBOK.svg";
import Photo from "../static/tire.png"
import { useNavigate } from "react-router-dom";

export default function KolobokChat() {
    const navigate = useNavigate();

    const handleGoToMain = () => {
    navigate("/");
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
        <div className="chat-message bot">Привет, чем могу помочь?</div>

        <div className="chat-message user">
          Привет! Мне нужно разметить шину!
          <div className="chat-image">
            <img src={Photo} alt="Загруженное изображение" />
          </div>
        </div>

        <div className="chat-message bot">Обрабатываю...</div>

        <div className="chat-input">
          <button className="attach-button">📎</button>
          <input
            type="text"
            className="message-input"
            placeholder="Написать сообщение..."
          />
          <button className="send-button">➤</button>
        </div>
      </div>
    </div>
  );
}
