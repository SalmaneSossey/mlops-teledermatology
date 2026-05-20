# Telemedicine Demo App

This demo turns the best tracked model into a database-backed telemedicine
workflow: patient case submission, model-assisted prediction, and doctor review.

## Build The Model Bundle

Create patient-safe PAD splits first if they do not already exist:

```bash
python -m src.data.make_image_splits \
  --metadata-path data/raw/pad_ufes_20/metadata.csv \
  --images-dir data/raw/pad_ufes_20/all_images \
  --output-dir data/processed/splits
```

Build an inference bundle from the best DagsHub MLflow run:

```bash
python -m src.inference.build_multimodal_bundle \
  --metadata-path data/raw/pad_ufes_20/metadata.csv \
  --splits-dir data/processed/splits \
  --output-dir storage/model_bundle
```

The default run is `ef084927bef741f996894b8a0fdd63e3`, the current
ISIC-initialized multimodal candidate.

## Run Locally

For a quick API run without Docker, the backend uses SQLite by default:

```bash
uvicorn src.app.main:app --reload
streamlit run src/app/streamlit_client.py
```

For the project demo with PostgreSQL:

```bash
cp .env.example .env
docker compose up --build
```

Open:

```text
FastAPI docs: http://localhost:8000/docs
Streamlit app: http://localhost:8501
```

Demo users:

```text
patient@example.com / patient123
doctor@example.com / doctor123
admin@example.com / admin123
```

## Clinical Boundary

The prediction is stored as decision support only. The final clinical decision is
stored separately in `doctor_reviews`, after a doctor reviews the case.

## Retraining Feedback

The admin dashboard exposes reviewed consultations as retraining candidates.
Each row links the latest uploaded image path, clinical metadata, model
prediction, doctor-confirmed final diagnosis, triage decision, and a disagreement
flag. This is the handoff point from the telemedicine workflow back into the
future retraining pipeline.

API exports:

```text
GET /admin/retraining-cases
GET /admin/retraining-cases.csv
```

Only admin users can access these exports.

## Monitoring

The v1 monitoring layer records every prediction event in `prediction_logs`.
The admin dashboard and API summarize:

- prediction volume
- label and risk distribution
- average and p95 prediction latency
- doctor/model agreement after reviews
- recent prediction events
- simple alerts for high latency, low agreement, or high-risk skew

API:

```text
GET /admin/monitoring
```

This is intentionally lightweight for the final-year demo. It shows the MLOps
feedback and monitoring loop without requiring a separate Prometheus/Grafana
stack.
