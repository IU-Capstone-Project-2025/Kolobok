const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";
const API_TOKEN = import.meta.env.VITE_API_TOKEN || "";

export interface AnalyzeThreadResponse {
  thread_depth: number;
  spikes: number;
  image: string; // base64-encoded annotated image
}

export interface ExtractInformationResponse {
  tire_mark: string;
  tire_manufacturer: string;
  tire_diameter: number;
}

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

export const analyzeThread = (imageBase64: string) =>
  post<AnalyzeThreadResponse>("/analyze_thread", imageBase64);

export const extractInformation = (imageBase64: string) =>
  post<ExtractInformationResponse>("/extract_information", imageBase64);
