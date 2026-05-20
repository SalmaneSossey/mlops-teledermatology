"""FastAPI backend for the telemedicine inference demo."""

from __future__ import annotations

import csv
import io
import json
import shutil
import time
from math import ceil
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.app.config import Settings, get_settings
from src.app.database import SessionLocal, get_db, init_db
from src.app.models import (
    Consultation,
    DoctorReview,
    LesionImage,
    ModelPrediction,
    PredictionLog,
    ModelVersion,
    PatientProfile,
    User,
)
from src.app.schemas import (
    AdminSummaryResponse,
    ConsultationCreate,
    ConsultationResponse,
    DoctorConsultationResponse,
    ImageResponse,
    LoginRequest,
    ModelCurrentResponse,
    MonitoringRecentPredictionResponse,
    MonitoringSummaryResponse,
    PredictionResponse,
    RetrainingCaseResponse,
    ReviewCreate,
    ReviewResponse,
    TokenResponse,
    UserSummaryResponse,
)
from src.app.security import create_access_token, require_role, verify_password
from src.app.seed import seed_demo_users, seed_model_version
from src.inference.build_multimodal_bundle import build_multimodal_bundle
from src.inference.predict_multimodal import MultimodalPredictor

app = FastAPI(title="MLOps Teledermatology Demo API")


@app.on_event("startup")
def startup() -> None:
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.model_bundle_dir.mkdir(parents=True, exist_ok=True)
    maybe_build_model_bundle(settings)
    init_db()
    with SessionLocal() as db:
        if settings.seed_demo_users:
            seed_demo_users(db)
        seed_model_version(db, settings.model_bundle_dir)


def maybe_build_model_bundle(settings: Settings) -> None:
    manifest_path = settings.model_bundle_dir / "manifest.json"
    if manifest_path.exists() or not settings.auto_build_model_bundle:
        return
    if settings.metadata_path is None or settings.splits_dir is None:
        return
    if not settings.metadata_path.exists() or not settings.splits_dir.exists():
        return
    build_multimodal_bundle(
        output_dir=settings.model_bundle_dir,
        metadata_path=settings.metadata_path,
        splits_dir=settings.splits_dir,
        mlflow_run_id=settings.model_mlflow_run_id,
    )


def model_version_to_response(settings: Settings) -> ModelCurrentResponse:
    manifest_path = settings.model_bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return ModelCurrentResponse(available=False, bundle_dir=str(settings.model_bundle_dir))
    manifest = json.loads(manifest_path.read_text())
    return ModelCurrentResponse(
        available=True,
        model_run_id=str(manifest.get("model_run_id")),
        bundle_dir=str(settings.model_bundle_dir),
        labels=list(manifest.get("labels", [])),
        metrics=dict(manifest.get("metrics", {})),
        warning=str(manifest.get("warning", "Decision support only; doctor review required.")),
    )


def get_predictor(settings: Settings) -> MultimodalPredictor:
    cache_key = "_multimodal_predictor"
    cached = getattr(app.state, cache_key, None)
    if cached is not None:
        return cached
    manifest_path = settings.model_bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not available. Build it before running predictions.",
        )
    predictor = MultimodalPredictor.from_bundle(settings.model_bundle_dir)
    setattr(app.state, cache_key, predictor)
    return predictor


def get_patient_profile(db: Session, user: User) -> PatientProfile:
    if user.patient_profile is None:
        raise HTTPException(status_code=400, detail="Patient profile is missing")
    return user.patient_profile


def get_patient_consultation(db: Session, consultation_id: int, user: User) -> Consultation:
    patient = get_patient_profile(db, user)
    consultation = (
        db.query(Consultation)
        .filter(Consultation.id == consultation_id, Consultation.patient_id == patient.id)
        .one_or_none()
    )
    if consultation is None:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return consultation


def latest_prediction_response(prediction: ModelPrediction | None) -> PredictionResponse | None:
    if prediction is None:
        return None
    return PredictionResponse(
        id=prediction.id,
        consultation_id=prediction.consultation_id,
        predicted_label=prediction.predicted_label,
        risk_level=prediction.risk_level,
        probabilities=prediction.probabilities,
        model_run_id=prediction.model_version.run_id,
        warning=prediction.warning,
    )


def build_retraining_cases(db: Session) -> list[RetrainingCaseResponse]:
    consultations = db.query(Consultation).order_by(Consultation.created_at.desc()).all()
    candidates: list[RetrainingCaseResponse] = []
    for consultation in consultations:
        latest_review = (
            db.query(DoctorReview)
            .filter(DoctorReview.consultation_id == consultation.id)
            .order_by(DoctorReview.created_at.desc())
            .first()
        )
        if latest_review is None:
            continue
        latest_prediction = (
            db.query(ModelPrediction)
            .filter(ModelPrediction.consultation_id == consultation.id)
            .order_by(ModelPrediction.created_at.desc())
            .first()
        )
        latest_image = (
            db.query(LesionImage)
            .filter(LesionImage.consultation_id == consultation.id)
            .order_by(LesionImage.uploaded_at.desc())
            .first()
        )
        doctor = db.query(User).filter(User.id == latest_review.doctor_id).one_or_none()
        predicted_label = latest_prediction.predicted_label if latest_prediction else None
        candidates.append(
            RetrainingCaseResponse(
                consultation_id=consultation.id,
                patient_email=consultation.patient.user.email,
                image_path=latest_image.stored_path if latest_image else None,
                original_filename=latest_image.original_filename if latest_image else None,
                symptoms_notes=consultation.symptoms_notes,
                clinical_metadata=consultation.clinical_metadata,
                predicted_label=predicted_label,
                prediction_risk_level=latest_prediction.risk_level if latest_prediction else None,
                probabilities=latest_prediction.probabilities if latest_prediction else None,
                model_run_id=latest_prediction.model_version.run_id if latest_prediction else None,
                final_diagnosis=latest_review.final_diagnosis,
                triage_decision=latest_review.triage_decision,
                doctor_email=doctor.email if doctor else "unknown",
                review_notes=latest_review.notes,
                reviewed_at=latest_review.created_at,
                disagreement=(
                    predicted_label != latest_review.final_diagnosis
                    if predicted_label is not None
                    else None
                ),
            )
        )
    return candidates


def retraining_cases_csv(cases: list[RetrainingCaseResponse]) -> str:
    output = io.StringIO()
    fieldnames = [
        "consultation_id",
        "patient_email",
        "image_path",
        "original_filename",
        "symptoms_notes",
        "clinical_metadata",
        "predicted_label",
        "prediction_risk_level",
        "probabilities",
        "model_run_id",
        "final_diagnosis",
        "triage_decision",
        "doctor_email",
        "review_notes",
        "reviewed_at",
        "disagreement",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for case in cases:
        row = case.model_dump(mode="json")
        row["clinical_metadata"] = json.dumps(row["clinical_metadata"], sort_keys=True)
        row["probabilities"] = json.dumps(row["probabilities"], sort_keys=True)
        writer.writerow(row)
    return output.getvalue()


def count_rows(rows) -> dict[str, int]:
    return {str(key): int(count) for key, count in rows}


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, ceil(percentile_value * len(sorted_values)) - 1))
    return sorted_values[index]


def build_monitoring_summary(db: Session) -> MonitoringSummaryResponse:
    prediction_count = db.query(ModelPrediction).count()
    review_count = db.query(DoctorReview).count()
    retraining_cases = build_retraining_cases(db)
    label_counts = count_rows(
        db.query(ModelPrediction.predicted_label, func.count(ModelPrediction.id))
        .group_by(ModelPrediction.predicted_label)
        .all()
    )
    risk_counts = count_rows(
        db.query(ModelPrediction.risk_level, func.count(ModelPrediction.id))
        .group_by(ModelPrediction.risk_level)
        .all()
    )
    latencies = [float(row[0]) for row in db.query(PredictionLog.latency_ms).all()]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else None
    p95_latency_ms = percentile(latencies, 0.95)

    reviewed_with_prediction = [
        case
        for case in retraining_cases
        if case.disagreement is not None
    ]
    disagreement_count = sum(1 for case in reviewed_with_prediction if case.disagreement)
    review_agreement_rate = None
    if reviewed_with_prediction:
        review_agreement_rate = 1.0 - (disagreement_count / len(reviewed_with_prediction))

    alerts = []
    if avg_latency_ms is not None and avg_latency_ms > 5000:
        alerts.append("Average prediction latency is above 5 seconds.")
    if review_agreement_rate is not None and review_agreement_rate < 0.75:
        alerts.append("Doctor/model agreement is below 75% on reviewed cases.")
    if risk_counts.get("high", 0) > max(5, prediction_count * 0.5):
        alerts.append("High-risk predictions are dominating recent activity.")
    if not alerts:
        alerts.append("No monitoring alerts.")

    recent_logs = (
        db.query(PredictionLog, ModelVersion)
        .join(ModelVersion, PredictionLog.model_version_id == ModelVersion.id)
        .order_by(PredictionLog.created_at.desc())
        .limit(10)
        .all()
    )
    return MonitoringSummaryResponse(
        prediction_count=prediction_count,
        review_count=review_count,
        retraining_candidate_count=len(retraining_cases),
        avg_latency_ms=avg_latency_ms,
        p95_latency_ms=p95_latency_ms,
        label_counts=label_counts,
        risk_counts=risk_counts,
        review_agreement_rate=review_agreement_rate,
        disagreement_count=disagreement_count,
        alerts=alerts,
        recent_predictions=[
            MonitoringRecentPredictionResponse(
                consultation_id=log.consultation_id,
                predicted_label=log.predicted_label,
                risk_level=log.risk_level,
                latency_ms=log.latency_ms,
                model_run_id=model_version.run_id,
                created_at=log.created_at,
            )
            for log, model_version in recent_logs
        ],
    )


@app.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "model_bundle_available": (settings.model_bundle_dir / "manifest.json").exists(),
    }


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email).one_or_none()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return TokenResponse(
        access_token=create_access_token(user.email, user.role),
        role=user.role,
    )


@app.get("/models/current", response_model=ModelCurrentResponse)
def current_model() -> ModelCurrentResponse:
    return model_version_to_response(get_settings())


@app.get("/admin/summary", response_model=AdminSummaryResponse)
def admin_summary(
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
) -> AdminSummaryResponse:
    status_rows = (
        db.query(Consultation.status, func.count(Consultation.id))
        .group_by(Consultation.status)
        .all()
    )
    label_rows = (
        db.query(ModelPrediction.predicted_label, func.count(ModelPrediction.id))
        .group_by(ModelPrediction.predicted_label)
        .all()
    )
    high_risk_predictions = (
        db.query(ModelPrediction)
        .filter(ModelPrediction.risk_level == "high")
        .count()
    )
    users = db.query(User).order_by(User.role, User.email).all()
    counts = {
        "users": db.query(User).count(),
        "patients": db.query(PatientProfile).count(),
        "consultations": db.query(Consultation).count(),
        "images": db.query(LesionImage).count(),
        "predictions": db.query(ModelPrediction).count(),
        "doctor_reviews": db.query(DoctorReview).count(),
        "model_versions": db.query(ModelVersion).count(),
    }
    return AdminSummaryResponse(
        counts=counts,
        status_counts={str(status): int(count) for status, count in status_rows},
        prediction_label_counts={str(label): int(count) for label, count in label_rows},
        high_risk_predictions=high_risk_predictions,
        active_model=model_version_to_response(get_settings()),
        users=[
            UserSummaryResponse(
                id=row.id,
                email=row.email,
                role=row.role,
                created_at=row.created_at,
            )
            for row in users
        ],
    )


@app.get("/admin/retraining-cases", response_model=list[RetrainingCaseResponse])
def list_retraining_cases(
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
) -> list[RetrainingCaseResponse]:
    return build_retraining_cases(db)


@app.get("/admin/retraining-cases.csv")
def download_retraining_cases_csv(
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
) -> Response:
    csv_text = retraining_cases_csv(build_retraining_cases(db))
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="retraining_cases.csv"'},
    )


@app.get("/admin/monitoring", response_model=MonitoringSummaryResponse)
def monitoring_summary(
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
) -> MonitoringSummaryResponse:
    return build_monitoring_summary(db)


@app.post("/patient/consultations", response_model=ConsultationResponse)
def create_consultation(
    payload: ConsultationCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("patient")),
) -> Consultation:
    patient = get_patient_profile(db, user)
    consultation = Consultation(
        patient_id=patient.id,
        status="draft",
        symptoms_notes=payload.symptoms_notes,
        clinical_metadata=payload.clinical_metadata,
    )
    db.add(consultation)
    db.commit()
    db.refresh(consultation)
    return consultation


@app.get("/patient/consultations", response_model=list[ConsultationResponse])
def list_patient_consultations(
    db: Session = Depends(get_db),
    user: User = Depends(require_role("patient")),
) -> list[Consultation]:
    patient = get_patient_profile(db, user)
    return (
        db.query(Consultation)
        .filter(Consultation.patient_id == patient.id)
        .order_by(Consultation.created_at.desc())
        .all()
    )


@app.post("/patient/consultations/{consultation_id}/image", response_model=ImageResponse)
def upload_consultation_image(
    consultation_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("patient")),
) -> LesionImage:
    settings = get_settings()
    consultation = get_patient_consultation(db, consultation_id, user)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(image.filename or "lesion.jpg").suffix or ".jpg"
    stored_name = f"{consultation.id}_{uuid4().hex}{extension}"
    stored_path = settings.upload_dir / stored_name
    with stored_path.open("wb") as output:
        shutil.copyfileobj(image.file, output)

    record = LesionImage(
        consultation_id=consultation.id,
        stored_path=str(stored_path),
        original_filename=image.filename or stored_name,
        content_type=image.content_type,
    )
    consultation.status = "submitted"
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.post("/patient/consultations/{consultation_id}/predict", response_model=PredictionResponse)
def predict_consultation(
    consultation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("patient")),
) -> PredictionResponse:
    settings = get_settings()
    consultation = get_patient_consultation(db, consultation_id, user)
    image = (
        db.query(LesionImage)
        .filter(LesionImage.consultation_id == consultation.id)
        .order_by(LesionImage.uploaded_at.desc())
        .first()
    )
    if image is None:
        raise HTTPException(status_code=400, detail="Upload an image before prediction")

    predictor = get_predictor(settings)
    started_at = time.perf_counter()
    try:
        result = predictor.predict(Path(image.stored_path), consultation.clinical_metadata)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - started_at) * 1000
    model_version = db.query(ModelVersion).filter(ModelVersion.active.is_(True)).first()
    if model_version is None:
        seed_model_version(db, settings.model_bundle_dir)
        model_version = db.query(ModelVersion).filter(ModelVersion.active.is_(True)).first()
    if model_version is None:
        raise HTTPException(status_code=503, detail="Active model version is unavailable")

    prediction = ModelPrediction(
        consultation_id=consultation.id,
        model_version_id=model_version.id,
        predicted_label=str(result["predicted_label"]),
        risk_level=str(result["risk_level"]),
        probabilities=dict(result["probabilities"]),
        warning=str(result["warning"]),
    )
    prediction_log = PredictionLog(
        consultation_id=consultation.id,
        model_version_id=model_version.id,
        predicted_label=str(result["predicted_label"]),
        risk_level=str(result["risk_level"]),
        latency_ms=latency_ms,
    )
    consultation.status = "predicted"
    db.add(prediction)
    db.add(prediction_log)
    db.commit()
    db.refresh(prediction)
    return latest_prediction_response(prediction)


@app.get("/doctor/consultations", response_model=list[DoctorConsultationResponse])
def list_doctor_consultations(
    db: Session = Depends(get_db),
    user: User = Depends(require_role("doctor", "admin")),
) -> list[DoctorConsultationResponse]:
    consultations = db.query(Consultation).order_by(Consultation.created_at.desc()).all()
    responses = []
    for consultation in consultations:
        latest_prediction = (
            db.query(ModelPrediction)
            .filter(ModelPrediction.consultation_id == consultation.id)
            .order_by(ModelPrediction.created_at.desc())
            .first()
        )
        latest_review = (
            db.query(DoctorReview)
            .filter(DoctorReview.consultation_id == consultation.id)
            .order_by(DoctorReview.created_at.desc())
            .first()
        )
        responses.append(
            DoctorConsultationResponse(
                consultation=consultation,
                patient_email=consultation.patient.user.email,
                image_count=len(consultation.images),
                latest_prediction=latest_prediction_response(latest_prediction),
                latest_review=latest_review,
            )
        )
    return responses


@app.post("/doctor/consultations/{consultation_id}/review", response_model=ReviewResponse)
def review_consultation(
    consultation_id: int,
    payload: ReviewCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("doctor", "admin")),
) -> DoctorReview:
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).one_or_none()
    if consultation is None:
        raise HTTPException(status_code=404, detail="Consultation not found")
    review = DoctorReview(
        consultation_id=consultation.id,
        doctor_id=user.id,
        final_diagnosis=payload.final_diagnosis,
        triage_decision=payload.triage_decision,
        notes=payload.notes,
    )
    consultation.status = "reviewed"
    db.add(review)
    db.commit()
    db.refresh(review)
    return review
