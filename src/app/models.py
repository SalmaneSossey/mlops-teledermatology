"""SQLAlchemy models for the telemedicine consultation workflow."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    patient_profile: Mapped["PatientProfile | None"] = relationship(back_populates="user")


class PatientProfile(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    display_name: Mapped[str] = mapped_column(String(255))
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user: Mapped[User] = relationship(back_populates="patient_profile")
    consultations: Mapped[list["Consultation"]] = relationship(back_populates="patient")


class Consultation(Base):
    __tablename__ = "consultations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    symptoms_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    clinical_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    patient: Mapped[PatientProfile] = relationship(back_populates="consultations")
    images: Mapped[list["LesionImage"]] = relationship(back_populates="consultation")
    predictions: Mapped[list["ModelPrediction"]] = relationship(back_populates="consultation")
    reviews: Mapped[list["DoctorReview"]] = relationship(back_populates="consultation")


class LesionImage(Base):
    __tablename__ = "lesion_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    consultation_id: Mapped[int] = mapped_column(ForeignKey("consultations.id"), index=True)
    stored_path: Mapped[str] = mapped_column(String(1024))
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    consultation: Mapped[Consultation] = relationship(back_populates="images")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    checkpoint_path: Mapped[str] = mapped_column(String(1024))
    metrics: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    predictions: Mapped[list["ModelPrediction"]] = relationship(back_populates="model_version")


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    consultation_id: Mapped[int] = mapped_column(ForeignKey("consultations.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    predicted_label: Mapped[str] = mapped_column(String(32))
    risk_level: Mapped[str] = mapped_column(String(32))
    probabilities: Mapped[dict[str, float]] = mapped_column(JSON)
    warning: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    consultation: Mapped[Consultation] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion] = relationship(back_populates="predictions")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    consultation_id: Mapped[int] = mapped_column(ForeignKey("consultations.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    predicted_label: Mapped[str] = mapped_column(String(32))
    risk_level: Mapped[str] = mapped_column(String(32), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class DoctorReview(Base):
    __tablename__ = "doctor_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    consultation_id: Mapped[int] = mapped_column(ForeignKey("consultations.id"), index=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    final_diagnosis: Mapped[str] = mapped_column(String(32))
    triage_decision: Mapped[str] = mapped_column(String(64))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    consultation: Mapped[Consultation] = relationship(back_populates="reviews")
