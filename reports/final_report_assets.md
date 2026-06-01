# Final Report And Demo Assets

This file gathers the diagrams, screenshot checklist, and report sections needed for the final presentation.

## Model Decision

The final selected model remains the PAD-UFES-20 multimodal EfficientNet-B0 model initialized from ISIC pretraining:

- Run ID: `ef084927bef741f996894b8a0fdd63e3`
- Macro F1: `0.6902`
- Balanced accuracy: `0.6804`
- High-risk recall: `0.8902`
- SCC recall limitation: `0.2069`

Class-aware augmentation failed the promotion gates. Derm8 improved SCC recall, but hurt macro F1, balanced accuracy, and high-risk recall enough that it should be reported as an ablation only.

## Architecture Diagram

```mermaid
flowchart LR
    Patient[Patient Mobile App] -->|JWT + image upload| API[FastAPI Backend]
    Doctor[Doctor Review UI] -->|Review decisions| API
    Admin[Streamlit Admin UI] -->|Monitoring and retraining export| API
    API --> DB[(PostgreSQL)]
    API --> Storage[(Uploaded Images)]
    API --> Model[Active Model Bundle]
    Training[Colab Training Notebooks] -->|metrics/artifacts| MLflow[DagsHub MLflow]
    MLflow -->|selected run metadata| Model
    DB --> Feedback[Reviewed Cases CSV]
    Feedback --> Training
```

## MLOps Lifecycle Diagram

```mermaid
flowchart TD
    Data[PAD-UFES-20 + ISIC/Derm8 data] --> Validate[Validation and split checks]
    Validate --> Train[Training experiments in Colab]
    Train --> Track[MLflow tracking on DagsHub]
    Track --> Select[Promotion gates]
    Select -->|passes| Register[Model bundle selected]
    Select -->|fails| Ablation[Ablation documented]
    Register --> Deploy[FastAPI model endpoint]
    Deploy --> Monitor[Prediction logs and admin monitoring]
    Monitor --> Review[Doctor feedback]
    Review --> Retrain[Automated retraining candidate export]
    Retrain --> Train
```

## Database Schema Diagram

```mermaid
erDiagram
    USERS ||--o| PATIENT_PROFILES : owns
    USERS ||--o| DOCTOR_PROFILES : owns
    PATIENT_PROFILES ||--o{ CONSULTATIONS : creates
    CONSULTATIONS ||--o{ LESION_IMAGES : contains
    CONSULTATIONS ||--o{ MODEL_PREDICTIONS : receives
    MODEL_VERSIONS ||--o{ MODEL_PREDICTIONS : produces
    CONSULTATIONS ||--o{ DOCTOR_REVIEWS : reviewed_by
    DOCTOR_PROFILES ||--o{ DOCTOR_REVIEWS : writes
    MODEL_PREDICTIONS ||--o{ PREDICTION_LOGS : logs
```

## Screenshot Checklist

Screenshot folder:

- Windows: `C:\Users\lione\Downloads\determa-pics`
- WSL: `/mnt/c/Users/lione/Downloads/determa-pics`
- Repo copies: `docs/assets/demo-screenshots/`

Available screenshots:

- `mobile-login.jpg`
- `mobile-new-case-form.jpg`
- `mobile-image-preview.jpg`
- `mobile-prediction-result.jpg`
- `mobile-history.jpg`
- `streamlit-doctor-review.png`
- `streamlit-bcc-review.png`
- `streamlit-admin-dashboard.png`

Needed or captured views:

- DagsHub MLflow experiment page showing the final run and ablation runs.
- FastAPI `/docs` with auth, patient, doctor, admin, monitoring, and model endpoints.
- Streamlit admin monitoring with metrics, alerts, recent predictions, and retraining candidates.
- Mobile patient flow: login, new case form, image preview, loading state, prediction result with probability bars.
- Doctor review flow: queue, lesion image preview, prediction probabilities, review submission.

## Mobile Demo Verification

Use [Run_mobile_app.md](../Run_mobile_app.md) for the physical-phone validation. The preferred professor-demo path is USB debugging with `adb reverse`.

- Expo starts in localhost mode with `adb reverse tcp:8081 tcp:8081`.
- The backend is reachable from the phone through `adb reverse tcp:8000 tcp:8000`.
- Login works with `patient@example.com`.
- Gallery/camera upload succeeds through `expo-file-system/legacy`.
- The prediction screen shows the image preview, loading progress, risk badge, label, and probability bars.
- The submitted case appears in patient history and doctor review.

Physical-phone validation succeeded on June 1, 2026. Captured screenshots should include the mobile prediction result screen, mobile history screen, Streamlit doctor review, and Streamlit admin monitoring.

## Final Report Sections

1. Dataset and preprocessing: PAD-UFES-20 structure, metadata normalization, split policy, ISIC/Derm8 mapping, leakage prevention.
2. Model experiments: image baseline, metadata baseline, multimodal fusion, ISIC pretraining, class-aware augmentation, Derm8 ablation, model selection gates.
3. MLOps pipeline: DVC/data versioning, Colab training notebooks, MLflow tracking on DagsHub, artifact handling, automated retraining workflow.
4. Deployment and backend: FastAPI, PostgreSQL schema, authentication, model bundle loading, patient/doctor/admin endpoints.
5. Monitoring and feedback loop: prediction logs, latency, label/risk counts, doctor reviews, retraining candidate export.
6. Limitations and ethics: decision support only, SCC weakness, dataset bias, skin tone coverage, clinician oversight, privacy/security.
7. Future work: more validated data, stronger SCC-focused training, focal loss/sampler ablations, calibration, deployment hardening.
