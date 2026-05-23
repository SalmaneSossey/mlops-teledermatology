export type TokenResponse = {
  access_token: string;
  token_type: string;
  role: "patient" | "doctor" | "admin";
};

export type ConsultationResponse = {
  id: number;
  status: string;
  symptoms_notes: string | null;
  clinical_metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type PredictionResponse = {
  id: number;
  consultation_id: number;
  predicted_label: string;
  risk_level: string;
  probabilities: Record<string, number>;
  model_run_id: string;
  warning: string;
};

export type PatientConsultationHistoryResponse = {
  consultation: ConsultationResponse;
  latest_prediction: PredictionResponse | null;
};

export type ModelCurrentResponse = {
  available: boolean;
  model_run_id: string | null;
  bundle_dir: string;
  labels: string[];
  metrics: Record<string, unknown>;
  warning: string;
};

export type ClinicalMetadata = {
  age: number;
  region: string;
  itch: string;
  grew: string;
  hurt: string;
  changed: string;
  bleed: string;
  elevation: string;
  gender: string | null;
  fitspatrick: number;
  diameter_1: number;
  diameter_2: number;
  skin_cancer_history: string;
  cancer_history: string;
  smoke: string;
  drink: string;
  pesticide: string;
};

export type SelectedImage = {
  uri: string;
  fileName: string;
  mimeType: string;
};
