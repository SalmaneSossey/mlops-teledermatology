"""Streamlit client for the telemedicine demo API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("TELEDERM_API_URL", "http://localhost:8000").rstrip("/")
LABELS = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
TRIAGE_DECISIONS = ["routine", "urgent", "refer", "monitor"]


def probability_frame(probabilities: dict[str, float] | None) -> pd.DataFrame:
    rows = [
        {"label": label, "probability": float(probability)}
        for label, probability in (probabilities or {}).items()
    ]
    rows.sort(key=lambda row: row["probability"], reverse=True)
    return pd.DataFrame(rows)


def show_probability_panel(probabilities: dict[str, float] | None) -> None:
    frame = probability_frame(probabilities)
    if frame.empty:
        st.info("No prediction probabilities available yet.")
        return
    display_frame = frame.copy()
    display_frame["probability"] = display_frame["probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(display_frame, use_container_width=True, hide_index=True)
    st.bar_chart(frame.set_index("label")["probability"])


def local_image_path(image_path: str | None) -> Path | None:
    if not image_path:
        return None
    path = Path(image_path)
    if path.exists():
        return path
    container_prefix = Path("/app/storage")
    try:
        relative_path = path.relative_to(container_prefix)
    except ValueError:
        return None
    host_path = Path("storage") / relative_path
    return host_path if host_path.exists() else None


def show_image_preview(
    image: dict[str, Any] | None = None,
    image_path: str | None = None,
    caption: str | None = None,
) -> None:
    if image and image.get("id"):
        response = api_request("GET", f"/doctor/images/{image['id']}")
        if response.ok:
            st.image(
                response.content,
                caption=caption or image.get("original_filename"),
                use_container_width=True,
            )
            return

    resolved_path = local_image_path(image_path or (image.get("stored_path") if image else None))
    if resolved_path is None:
        st.info("No image preview is available for this case.")
        return
    st.image(str(resolved_path), caption=caption or resolved_path.name, use_container_width=True)


def api_headers() -> dict[str, str]:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_request(method: str, path: str, **kwargs) -> requests.Response:
    response = requests.request(method, f"{API_URL}{path}", headers=api_headers(), timeout=120, **kwargs)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        st.error(detail)
    return response


def login_page() -> None:
    st.title("Teledermatology")
    with st.form("login"):
        email = st.text_input("Email", value="patient@example.com")
        password = st.text_input("Password", type="password", value="patient123")
        submitted = st.form_submit_button("Login")
    if submitted:
        response = requests.post(
            f"{API_URL}/auth/login",
            json={"email": email, "password": password},
            timeout=30,
        )
        if response.ok:
            payload = response.json()
            st.session_state["access_token"] = payload["access_token"]
            st.session_state["role"] = payload["role"]
            st.rerun()
        else:
            st.error("Invalid credentials")


def metadata_form() -> dict[str, Any]:
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=0, max_value=120, value=55)
        region = st.selectbox(
            "Region",
            ["FACE", "ARM", "BACK", "CHEST", "FOREARM", "HAND", "LEG", "NECK", "SCALP", "THIGH"],
        )
        gender = st.selectbox("Gender", ["", "FEMALE", "MALE"])
        fitspatrick = st.number_input("Fitzpatrick", min_value=0, max_value=6, value=3)
        diameter_1 = st.number_input("Diameter 1", min_value=0.0, value=6.0)
        diameter_2 = st.number_input("Diameter 2", min_value=0.0, value=5.0)
    with right:
        itch = st.selectbox("Itch", ["False", "True", "UNK"])
        grew = st.selectbox("Grew", ["False", "True", "UNK"])
        hurt = st.selectbox("Hurt", ["False", "True", "UNK"])
        changed = st.selectbox("Changed", ["False", "True", "UNK"])
        bleed = st.selectbox("Bleed", ["False", "True", "UNK"])
        elevation = st.selectbox("Elevation", ["False", "True", "UNK"])

    return {
        "age": age,
        "region": region,
        "itch": itch,
        "grew": grew,
        "hurt": hurt,
        "changed": changed,
        "bleed": bleed,
        "elevation": elevation,
        "gender": gender or None,
        "fitspatrick": fitspatrick,
        "diameter_1": diameter_1,
        "diameter_2": diameter_2,
        "skin_cancer_history": "UNK",
        "cancer_history": "UNK",
        "smoke": "UNK",
        "drink": "UNK",
        "pesticide": "UNK",
    }


def patient_page() -> None:
    st.title("Patient Case")
    model = api_request("GET", "/models/current")
    if model.ok:
        payload = model.json()
        st.caption(f"Model: {payload.get('model_run_id') or 'not available'}")

    with st.form("case"):
        notes = st.text_area("Notes")
        metadata = metadata_form()
        image = st.file_uploader("Lesion image", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Submit And Predict")

    if submitted:
        consultation = api_request(
            "POST",
            "/patient/consultations",
            json={"symptoms_notes": notes, "clinical_metadata": metadata},
        )
        if not consultation.ok:
            return
        consultation_id = consultation.json()["id"]
        if image is not None:
            files = {"image": (image.name, image.getvalue(), image.type)}
            upload = api_request("POST", f"/patient/consultations/{consultation_id}/image", files=files)
            if not upload.ok:
                return
        prediction = api_request("POST", f"/patient/consultations/{consultation_id}/predict")
        if prediction.ok:
            payload = prediction.json()
            st.success("Prediction saved")
            result_columns = st.columns([1, 2])
            result_columns[0].metric("Predicted label", payload["predicted_label"])
            result_columns[0].metric("Risk level", payload["risk_level"])
            with result_columns[1]:
                st.subheader("Prediction probabilities")
                show_probability_panel(payload.get("probabilities"))
            with st.expander("Raw prediction payload"):
                st.json(payload)

    response = api_request("GET", "/patient/consultations")
    if response.ok:
        st.subheader("My Cases")
        st.dataframe(response.json(), use_container_width=True)


def doctor_page() -> None:
    st.title("Doctor Review")
    response = api_request("GET", "/doctor/consultations")
    if not response.ok:
        return
    consultations = response.json()
    if not consultations:
        st.info("No consultations yet")
        return

    options = {
        f"Case {item['consultation']['id']} - {item['patient_email']} - {item['consultation']['status']}": item
        for item in consultations
    }
    selected_label = st.selectbox("Case", list(options))
    selected = options[selected_label]
    consultation = selected["consultation"]
    latest_prediction = selected.get("latest_prediction")
    latest_image = selected.get("latest_image")

    overview_columns = st.columns([1, 1, 2])
    overview_columns[0].metric("Case", consultation["id"])
    overview_columns[1].metric("Images", selected["image_count"])
    overview_columns[2].write(f"Patient: `{selected['patient_email']}`")
    overview_columns[2].write(f"Status: `{consultation['status']}`")

    preview_columns = st.columns([1, 1])
    with preview_columns[0]:
        st.subheader("Lesion Image")
        show_image_preview(
            image=latest_image,
            caption=latest_image.get("original_filename") if latest_image else None,
        )
    with preview_columns[1]:
        st.subheader("Latest Prediction")
        if latest_prediction:
            st.metric("Predicted label", latest_prediction["predicted_label"])
            st.metric("Risk level", latest_prediction["risk_level"])
            show_probability_panel(latest_prediction.get("probabilities"))
        else:
            st.info("No prediction has been generated yet.")

    with st.expander("Case payload"):
        st.json(selected)

    with st.form("review"):
        final_diagnosis = st.selectbox("Final diagnosis", LABELS)
        triage_decision = st.selectbox("Triage decision", TRIAGE_DECISIONS)
        notes = st.text_area("Review notes")
        submitted = st.form_submit_button("Save Review")

    if submitted:
        consultation_id = selected["consultation"]["id"]
        review = api_request(
            "POST",
            f"/doctor/consultations/{consultation_id}/review",
            json={
                "final_diagnosis": final_diagnosis,
                "triage_decision": triage_decision,
                "notes": notes,
            },
        )
        if review.ok:
            st.success("Review saved")
            st.rerun()


def admin_page() -> None:
    st.title("Admin")
    response = api_request("GET", "/admin/summary")
    if not response.ok:
        return
    summary = response.json()

    counts = summary["counts"]
    columns = st.columns(4)
    columns[0].metric("Consultations", counts.get("consultations", 0))
    columns[1].metric("Predictions", counts.get("predictions", 0))
    columns[2].metric("Reviews", counts.get("doctor_reviews", 0))
    columns[3].metric("High-risk predictions", summary.get("high_risk_predictions", 0))

    st.subheader("Active Model")
    active_model = summary["active_model"]
    st.write(active_model.get("model_run_id") or "No active model bundle")
    metrics = active_model.get("metrics", {})
    if metrics:
        metric_columns = st.columns(3)
        metric_columns[0].metric("Macro F1", f"{metrics.get('test_macro_f1', 0):.4f}")
        metric_columns[1].metric(
            "Balanced Acc",
            f"{metrics.get('test_balanced_accuracy', 0):.4f}",
        )
        metric_columns[2].metric(
            "High-risk Recall",
            f"{metrics.get('test_high_risk_recall', 0):.4f}",
        )

    st.subheader("Monitoring")
    monitoring = api_request("GET", "/admin/monitoring")
    if monitoring.ok:
        payload = monitoring.json()
        monitoring_columns = st.columns(4)
        monitoring_columns[0].metric("Predictions", payload["prediction_count"])
        monitoring_columns[1].metric(
            "Avg latency",
            (
                f"{payload['avg_latency_ms']:.0f} ms"
                if payload["avg_latency_ms"] is not None
                else "n/a"
            ),
        )
        monitoring_columns[2].metric(
            "P95 latency",
            (
                f"{payload['p95_latency_ms']:.0f} ms"
                if payload["p95_latency_ms"] is not None
                else "n/a"
            ),
        )
        monitoring_columns[3].metric(
            "Agreement",
            (
                f"{payload['review_agreement_rate']:.0%}"
                if payload["review_agreement_rate"] is not None
                else "n/a"
            ),
        )
        for alert in payload["alerts"]:
            if alert == "No monitoring alerts.":
                st.success(alert)
            else:
                st.warning(alert)
        monitor_left, monitor_right = st.columns(2)
        with monitor_left:
            st.dataframe(
                [{"risk": key, "count": value} for key, value in payload["risk_counts"].items()],
                use_container_width=True,
            )
        with monitor_right:
            st.dataframe(payload["recent_predictions"], use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Consultation Status")
        st.dataframe(
            [{"status": key, "count": value} for key, value in summary["status_counts"].items()],
            use_container_width=True,
        )
    with right:
        st.subheader("Predicted Labels")
        st.dataframe(
            [
                {"label": key, "count": value}
                for key, value in summary["prediction_label_counts"].items()
            ],
            use_container_width=True,
        )

    st.subheader("Users")
    st.dataframe(summary["users"], use_container_width=True)

    st.subheader("Retraining Candidates")
    retraining = api_request("GET", "/admin/retraining-cases")
    if retraining.ok:
        cases = retraining.json()
        if cases:
            options = {
                f"Case {item['consultation_id']} - {item['patient_email']} - {item['final_diagnosis']}": item
                for item in cases
            }
            selected_candidate = options[st.selectbox("Preview retraining case", list(options))]
            candidate_left, candidate_right = st.columns([1, 1])
            with candidate_left:
                show_image_preview(
                    image_path=selected_candidate.get("image_path"),
                    caption=selected_candidate.get("original_filename"),
                )
            with candidate_right:
                st.metric("Final diagnosis", selected_candidate["final_diagnosis"])
                st.metric("Model prediction", selected_candidate.get("predicted_label") or "n/a")
                st.metric(
                    "Disagreement",
                    "yes" if selected_candidate.get("disagreement") else "no",
                )
                show_probability_panel(selected_candidate.get("probabilities"))
            st.dataframe(cases, use_container_width=True)
        else:
            st.info("No doctor-reviewed cases are ready for retraining yet")
        csv_response = api_request("GET", "/admin/retraining-cases.csv")
        if csv_response.ok:
            st.download_button(
                "Download CSV",
                data=csv_response.text,
                file_name="retraining_cases.csv",
                mime="text/csv",
            )

    st.subheader("Doctor Queue")
    doctor_page()


def main() -> None:
    st.set_page_config(page_title="Teledermatology Demo", layout="wide")
    if "access_token" not in st.session_state:
        login_page()
        return

    with st.sidebar:
        st.write(st.session_state.get("role", ""))
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    role = st.session_state.get("role")
    if role == "patient":
        patient_page()
    elif role == "admin":
        admin_page()
    else:
        doctor_page()


if __name__ == "__main__":
    main()
