import * as FileSystem from "expo-file-system/legacy";

import {
  ClinicalMetadata,
  ConsultationResponse,
  ModelCurrentResponse,
  PatientConsultationHistoryResponse,
  PredictionResponse,
  SelectedImage,
  TokenResponse
} from "./types";

export const API_BASE_URL =
  process.env.EXPO_PUBLIC_TELEDERM_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

type ApiOptions = RequestInit & {
  token?: string | null;
};

async function apiRequest<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const { token, headers, body, ...rest } = options;
  const isFormData = typeof FormData !== "undefined" && body instanceof FormData;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...rest,
    body,
    headers: {
      Accept: "application/json",
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers
    }
  });
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      const text = await response.text();
      detail = text || detail;
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  return apiRequest<TokenResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
}

export async function getCurrentModel(token: string): Promise<ModelCurrentResponse> {
  return apiRequest<ModelCurrentResponse>("/models/current", { token });
}

export async function createConsultation(
  token: string,
  symptomsNotes: string,
  clinicalMetadata: ClinicalMetadata
): Promise<ConsultationResponse> {
  return apiRequest<ConsultationResponse>("/patient/consultations", {
    method: "POST",
    token,
    body: JSON.stringify({
      symptoms_notes: symptomsNotes,
      clinical_metadata: clinicalMetadata
    })
  });
}

export async function uploadConsultationImage(
  token: string,
  consultationId: number,
  image: SelectedImage
): Promise<void> {
  const result = await FileSystem.uploadAsync(
    `${API_BASE_URL}/patient/consultations/${consultationId}/image`,
    image.uri,
    {
      fieldName: "image",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${token}`
      },
      httpMethod: "POST",
      mimeType: image.mimeType,
      uploadType: FileSystem.FileSystemUploadType.MULTIPART
    }
  );
  if (result.status < 200 || result.status >= 300) {
    let detail = `Upload failed with status ${result.status}`;
    try {
      const payload = JSON.parse(result.body) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      detail = result.body || detail;
    }
    throw new Error(detail);
  }
}

export async function predictConsultation(
  token: string,
  consultationId: number
): Promise<PredictionResponse> {
  return apiRequest<PredictionResponse>(`/patient/consultations/${consultationId}/predict`, {
    method: "POST",
    token
  });
}

export async function listPatientHistory(
  token: string
): Promise<PatientConsultationHistoryResponse[]> {
  return apiRequest<PatientConsultationHistoryResponse[]>("/patient/consultations/history", {
    token
  });
}
