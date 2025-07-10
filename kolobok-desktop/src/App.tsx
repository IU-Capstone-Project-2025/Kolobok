import React from 'react';
import './App.css';
import KolobokHero from './components/main';
import KolobokChat from './components/chat';
import {Route, Routes} from "react-router-dom"

function App() {
  return (
    <div className='App'>
        <Routes>
          <Route path="/" element={<KolobokHero />} />
          <Route path="/chat" element={<KolobokChat />} />
        </Routes>
      </div>
     
  );
}

export default App;
