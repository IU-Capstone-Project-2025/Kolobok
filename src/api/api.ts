const API_URL = "http://localhost:8000/api/v1";
const API_TOKEN = "darriyano"; // должен совпадать с API_TOKEN на бэке

export interface AnalyzeThreadResponse {
  thread_depth: number;
  spikes: { class: number }[];
  image: string; // base64-encoded annotated image
}

export interface ExtractInformationResponse {
  index_results: {
    brand_name: string;
    model_name: string;
    combined_score: number;
  }[];
  strings: string;
  tire_size: number;
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

  return response.json();
}

// Функция для анализа протектора
export const analyzeThread = (imageBase64: string) =>
  post<AnalyzeThreadResponse>("/analyze_thread", imageBase64);

// Функция для извлечения информации о шине
export const extractInformation = (imageBase64: string) =>
  post<ExtractInformationResponse>("/extract_information", imageBase64);
