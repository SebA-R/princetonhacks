import type { ModelInsights, PredictionResponse } from "@/types/insights"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}))
    const message = detail.detail ?? `Request failed with status ${response.status}`
    throw new Error(message)
  }
  return (await response.json()) as T
}

export async function fetchInsights(): Promise<ModelInsights> {
  const response = await fetch(`${API_BASE_URL}/insights`, { cache: "no-store" })
  return handleResponse<ModelInsights>(response)
}

export async function predictVoice(file: File): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append("file", file)
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  })
  return handleResponse<PredictionResponse>(response)
}
