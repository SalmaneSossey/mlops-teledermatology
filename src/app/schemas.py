"""Pydantic schemas for the telemedicine API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TokenResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    access_token: str
    token_type: str = "bearer"
    role: str


class LoginRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    email: str
    password: str


class ConsultationCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    symptoms_notes: str | None = None
    clinical_metadata: dict[str, Any] = Field(default_factory=dict)


class ConsultationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    status: str
    symptoms_notes: str | None
    clinical_metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    consultation_id: int
    stored_path: str
    original_filename: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    consultation_id: int
    predicted_label: str
    risk_level: str
    probabilities: dict[str, float]
    model_run_id: str
    warning: str


class ReviewCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    final_diagnosis: str
    triage_decision: str
    notes: str | None = None


class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    consultation_id: int
    final_diagnosis: str
    triage_decision: str
    notes: str | None
    created_at: datetime


class ModelCurrentResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    available: bool
    model_run_id: str | None = None
    bundle_dir: str
    labels: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warning: str = "Decision support only; doctor review required."


class DoctorConsultationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    consultation: ConsultationResponse
    patient_email: str
    image_count: int
    latest_prediction: PredictionResponse | None = None
    latest_review: ReviewResponse | None = None


class UserSummaryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    email: str
    role: str
    created_at: datetime


class AdminSummaryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    counts: dict[str, int]
    status_counts: dict[str, int]
    prediction_label_counts: dict[str, int]
    high_risk_predictions: int
    active_model: ModelCurrentResponse
    users: list[UserSummaryResponse]


class RetrainingCaseResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    consultation_id: int
    patient_email: str
    image_path: str | None
    original_filename: str | None
    symptoms_notes: str | None
    clinical_metadata: dict[str, Any]
    predicted_label: str | None
    prediction_risk_level: str | None
    probabilities: dict[str, float] | None
    model_run_id: str | None
    final_diagnosis: str
    triage_decision: str
    doctor_email: str
    review_notes: str | None
    reviewed_at: datetime
    disagreement: bool | None


class MonitoringRecentPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    consultation_id: int
    predicted_label: str
    risk_level: str
    latency_ms: float
    model_run_id: str
    created_at: datetime


class MonitoringSummaryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction_count: int
    review_count: int
    retraining_candidate_count: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    label_counts: dict[str, int]
    risk_counts: dict[str, int]
    review_agreement_rate: float | None
    disagreement_count: int
    alerts: list[str]
    recent_predictions: list[MonitoringRecentPredictionResponse]
