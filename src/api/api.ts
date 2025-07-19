const API_URL = "http://158.160.163.245:8000";
const API_TOKEN = "kolobok_token"; // должен совпадать с API_TOKEN на бэке

export interface AnalyzeThreadResponse {
  success: number;
  thread_depth: number;
  spikes: { class: number }[];
  image: string; // base64-encoded annotated image
}

// Анализ шипов -> фото от юзера -> analse_thread
// Марка и модель -> фото от юзера + токен -> identify_tir

export interface ExtractInformationResponse {
  index_results: {
    brand_name: string;
    model_name: string;
    combined_score: number;
  }[];
  strings: string[];    // теперь массив
  tire_size: string;    // теперь строка
}


// Общая функция для POST-запросов
async function post<T>(endpoint: string, base64Image: string): Promise<T> {
  const response = await fetch(`${API_URL}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_TOKEN}`,
    },
    body: JSON.stringify({ image: base64Image }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Ошибка ${response.status}: ${errText}`);
  }
  
  // const data: T = await response.json();
  // console.log(`Response from ${endpoint}:`, data);
  // return response.json();
  // после проверки response.ok
  const data = await response.json();        // читаем тело один раз
  console.log(`Response from ${endpoint}:`, data);
  return data;                               // возвращаем распарсенный объект

}

// Функция для анализа протектора
export const analyzeThread = (imageBase64: string) =>
  post<AnalyzeThreadResponse>("/analyze_thread", imageBase64);

// Функция для извлечения информации о шине
export const extractInformation = (imageBase64: string) =>
  post<ExtractInformationResponse>("/extract_information", imageBase64);
