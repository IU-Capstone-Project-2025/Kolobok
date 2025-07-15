import React, { useState } from "react";
import "@fontsource/roboto-condensed";
import "../styles/header.css";
import { useNavigate } from "react-router-dom";
import Logo from "../static/KOLOBOK.svg";


export const Header = () => {
    const navigate = useNavigate();

    const handleGoToMain = () => {
    navigate("/");
  };
    return (<div className="kolobok-header">
        <div className="kolobok-logo" onClick={handleGoToMain}>
          <img src={Logo} alt="Kolobok Logo" />
        </div>
        <button className="kolobok-telegram"> <a href="https://t.me/kolobok_tire_bot">Telegram-bot </a></button>
      </div>)
}
