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

## Patient Mobile App

The Expo patient app lives in `apps/mobile`. Use a LAN-reachable API URL for
physical-phone demos:

```bash
cd apps/mobile
npm install
EXPO_PUBLIC_TELEDERM_API_URL=http://<computer-lan-ip>:8000 npm start
```

For Android emulator-only testing, `http://10.0.2.2:8000` can usually reach the
host FastAPI service. The mobile app uses the seeded patient demo account and
supports case creation, camera/gallery upload, prediction, and patient history.
For WSL2 plus a physical Android phone, use the repository-level
`Run_mobile_app.md` runbook.

## Clinical Boundary

The prediction is stored as decision support only. The final clinical decision is
stored separately in `doctor_reviews`, after a doctor reviews the case.

## Retraining Feedback

The admin dashboard exposes reviewed consultations as retraining candidates.
Each row links the latest uploaded image path, clinical metadata, model
prediction, doctor-confirmed final diagnosis, triage decision, and a disagreement
flag. This is the handoff point from the telemedicine workflow back into the
automated retraining pipeline.

API exports:

```text
GET /admin/retraining-cases
GET /admin/retraining-cases.csv
```

Only admin users can access these exports.

## Automated Retraining

The feedback retraining MVP is intentionally review-gated. It can export
doctor-reviewed cases, validate them, append valid feedback cases to the
training split, train a new multimodal candidate, compare that candidate against
the current model metrics, and optionally build a candidate inference bundle.
It does not replace `storage/model_bundle` automatically.

Export reviewed cases from the configured telemedicine database:

```bash
PYTHONPATH=. python scripts/export_retraining_manifest.py \
  --output-path data/feedback/retraining_candidates.csv
```

Preview the retraining inputs without launching GPU training:

```bash
PYTHONPATH=. python scripts/run_retraining_pipeline.py \
  --feedback-path data/feedback/retraining_candidates.csv \
  --images-dir data/raw/pad_ufes_20/all_images \
  --metadata-path data/raw/pad_ufes_20/metadata.csv \
  --splits-dir data/processed/splits \
  --output-dir runs/feedback_retraining \
  --dry-run
```

Run the candidate training job when enough reviewed cases exist:

```bash
PYTHONPATH=. python scripts/run_retraining_pipeline.py \
  --feedback-path data/feedback/retraining_candidates.csv \
  --images-dir data/raw/pad_ufes_20/all_images \
  --metadata-path data/raw/pad_ufes_20/metadata.csv \
  --splits-dir data/processed/splits \
  --output-dir runs/feedback_retraining \
  --current-metrics-path storage/model_bundle/manifest.json \
  --build-candidate-bundle
```

Outputs:

- `runs/feedback_retraining/prepared/metadata.csv`
- `runs/feedback_retraining/prepared/splits/`
- `runs/feedback_retraining/training/`
- `runs/feedback_retraining/retraining_report.json`
- `storage/model_bundle_candidate/` when `--build-candidate-bundle` is used

The promotion gate requires no macro-F1 drop, no selection-score drop, and at
most a small tolerated drop in high-risk recall or balanced accuracy. Review
`retraining_report.json` before promoting a candidate bundle.

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
