import React, { useState } from "react";
import "@fontsource/roboto-condensed";
import "../styles/kolochat.css";
import Logo from "../static/KOLOBOK.svg";
import { useNavigate } from "react-router-dom";

// Тип для сообщения, которое может быть либо текстом, либо изображением
interface Message {
  text: string; 
  sender: "user" | "bot"; 
  image?: string; // Если сообщение содержит изображение, оно будет храниться в этом поле
}

export default function KolobokChat() {
  const [messages, setMessages] = useState<Message[]>([
    { text: "Привет, чем могу помочь?", sender: "bot" }
  ]);
  const [inputValue, setInputValue] = useState(""); // Для хранения значения инпута
  const [imageFile, setImageFile] = useState<File | null>(null); // Хранение самого файла изображения
  const navigate = useNavigate();

  const handleGoToMain = () => {
    navigate("/");
  };

  const handleSendMessage = () => {
    if (inputValue.trim() === "") return; // Не отправлять пустое сообщение

    // Добавление нового текстового сообщения в чат
    setMessages((prevMessages) => [
      ...prevMessages,
      { text: inputValue, sender: "user" },
      { text: "Обрабатываю...", sender: "bot" }, // Для имитации ответа бота
    ]);

    setInputValue(""); // Очистить инпут после отправки
  };

  // Обработчик нажатия клавиши
  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSendMessage(); // Отправить сообщение по нажатию Enter
    }
  };

  // Обработчик для загрузки изображения
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];

    if (file) {
      // Проверяем размер файла (ограничение на размер 5 Мб)
      if (file.size > 5 * 1024 * 1024) {
        alert("Размер изображения слишком большой. Пожалуйста, выберите изображение меньшего размера.");
        return;
      }

      // Проверяем расширение файла (jpg, jpeg, png)
      if (file.type === "image/jpeg" || file.type === "image/png") {
        const reader = new FileReader();
        reader.onloadend = () => {
          // Добавляем новое изображение в список сообщений как самостоятельное сообщение
          setMessages((prevMessages) => [
            ...prevMessages,
            {
              text: reader.result as string,
              sender: "user",
              image: reader.result as string, // Сохраняем Data URL изображения
            },
          ]);
          setImageFile(file); // Сохраняем сам файл изображения
        };

        reader.onerror = (error) => {
          console.error("Ошибка при чтении файла:", error);
        };

        reader.readAsDataURL(file); // Читаем файл как Data URL
      } else {
        alert("Загрузите изображение формата JPG, JPEG или PNG.");
      }
    } else {
      console.log("Файл не выбран.");
    }
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
        {messages.map((message, index) => (
          <div
            key={index}
            className={`chat-message ${message.sender}`}
          >
            {/* Отображаем текстовое сообщение */}
            {message.text && !message.image && <p>{message.text}</p>}

            {/* Отображаем изображение как самостоятельное сообщение */}
            {message.image && (
              <div className="chat-image">
                <img
                  src={message.image}
                  alt="Загруженное изображение"
                  style={{
                    maxWidth: "100%",
                    maxHeight: "300px", // Ограничиваем размер
                    objectFit: "contain", // Подгоняем изображение по контейнеру
                    borderRadius: "10px",
                    boxShadow: "0 4px 8px rgba(0,0,0,0.2)"
                  }}
                />
              </div>
            )}
          </div>
        ))}

        <div className="chat-input">
          {/* Кнопка для загрузки изображения */}
          <button
            className="attach-button"
            onClick={() => document.getElementById("file-input")?.click()}
          >
            📎
          </button>

          {/* Скрытый input для загрузки изображений */}
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
            onKeyPress={handleKeyPress} // Обработка нажатия клавиши Enter
          />
          <button className="send-button" onClick={handleSendMessage}>
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}
