import React from "react";
import "@fontsource/roboto-condensed";
import Logo from "../static/KOLOBOK.svg"
import Photo from "../static/tire.png"
import "../styles/main.css"
import { useNavigate } from "react-router-dom";

export default function KolobokHero() {
  const navigate = useNavigate();

  const handleGoToChat = () => {
    navigate("/chat");
  };

  return (
    <div className="kolobok-container">
      {/* Header */}
      <div className="kolobok-header">
        <div className="kolobok-logo">
          <img src={Logo} alt="Kolobok Logo" />
        </div>
        <button className="kolobok-telegram">Telegram-bot</button>
      </div>

      {/* Hero Section */}
      <div className="kolobok-hero">
        <img
          src={Photo}
          alt="Tire"
          className="kolobok-photo"
        />

        <div className="kolobok-text left-text">
          <h1 className="kolobok-title">
            Привет, чем могу помочь? <span className="animate-blink">|</span>
          </h1>

          <div className="kolobok-buttons">
            <button className="kolobok-btn" onClick={handleGoToChat}>Разметить шину ➔</button>
          </div>
        </div>
      </div>

      {/* Info Section */}
      <div className="kolobok-info left-text">
        <h2 className="kolobok-info-title">Привет, я — Kolobok</h2>
        <p className="kolobok-info-text">
          Я — искусственный интеллект,  со мной можно найти ответы на вопросы о шинах: о
          количестве шипов, о проблемах с ними или о еще чем-либо. Общение ведется в формате
          бота: просто и понятно! Только выберите тему диалога, загрузите фото и я с радостью
          помогу вам! Дополнительно я могу помочь с анализом состояния шин, выявлением износа и даже подсказать рекомендации по уходу и сезонной замене. Просто начните диалог!
        </p>
        <button className="kolobok-info-button" onClick={handleGoToChat}>Перейти к диалогу →</button>
      </div>
    </div>
  );
}

