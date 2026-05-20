import io
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import sessionmaker

from src.app.database import Base, get_db, make_engine
from src.app.main import app
from src.app.models import ModelVersion
from src.app.seed import seed_demo_users


class DummyPredictor:
    def predict(self, image_path: Path, metadata: dict[str, object]) -> dict[str, object]:
        return {
            "predicted_label": "BCC",
            "risk_level": "high",
            "probabilities": {
                "ACK": 0.05,
                "BCC": 0.70,
                "MEL": 0.05,
                "NEV": 0.10,
                "SCC": 0.05,
                "SEK": 0.05,
            },
            "model_run_id": "test-run",
            "warning": "Decision support only; doctor review required.",
        }


class TelemedicineApiTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.old_upload_dir = os.environ.get("TELEDERM_UPLOAD_DIR")
        self.old_model_bundle_dir = os.environ.get("TELEDERM_MODEL_BUNDLE_DIR")
        os.environ["TELEDERM_UPLOAD_DIR"] = str(self.root / "uploads")
        os.environ["TELEDERM_MODEL_BUNDLE_DIR"] = str(self.root / "model_bundle")
        self.engine = make_engine(f"sqlite:///{self.root / 'test.db'}")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        with self.Session() as db:
            seed_demo_users(db)
            db.add(
                ModelVersion(
                    run_id="test-run",
                    checkpoint_path=str(self.root / "fake.pt"),
                    metrics={"test_macro_f1": 0.69},
                    active=True,
                )
            )
            db.commit()

        def override_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        app.state._multimodal_predictor = DummyPredictor()
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        if hasattr(app.state, "_multimodal_predictor"):
            delattr(app.state, "_multimodal_predictor")
        if self.old_upload_dir is None:
            os.environ.pop("TELEDERM_UPLOAD_DIR", None)
        else:
            os.environ["TELEDERM_UPLOAD_DIR"] = self.old_upload_dir
        if self.old_model_bundle_dir is None:
            os.environ.pop("TELEDERM_MODEL_BUNDLE_DIR", None)
        else:
            os.environ["TELEDERM_MODEL_BUNDLE_DIR"] = self.old_model_bundle_dir
        self.tmpdir.cleanup()

    def login(self, email: str, password: str) -> dict[str, str]:
        response = self.client.post("/auth/login", json={"email": email, "password": password})
        self.assertEqual(response.status_code, 200, response.text)
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_login_accepts_seeded_user_and_rejects_bad_password(self):
        ok = self.client.post(
            "/auth/login",
            json={"email": "patient@example.com", "password": "patient123"},
        )
        bad = self.client.post(
            "/auth/login",
            json={"email": "patient@example.com", "password": "wrong"},
        )

        self.assertEqual(ok.status_code, 200)
        self.assertEqual(bad.status_code, 401)

    def test_patient_cannot_access_doctor_queue(self):
        headers = self.login("patient@example.com", "patient123")

        response = self.client.get("/doctor/consultations", headers=headers)

        self.assertEqual(response.status_code, 403)

    def test_admin_summary_is_admin_only(self):
        patient_headers = self.login("patient@example.com", "patient123")
        admin_headers = self.login("admin@example.com", "admin123")

        forbidden = self.client.get("/admin/summary", headers=patient_headers)
        retraining_forbidden = self.client.get("/admin/retraining-cases", headers=patient_headers)
        monitoring_forbidden = self.client.get("/admin/monitoring", headers=patient_headers)
        allowed = self.client.get("/admin/summary", headers=admin_headers)

        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(retraining_forbidden.status_code, 403)
        self.assertEqual(monitoring_forbidden.status_code, 403)
        self.assertEqual(allowed.status_code, 200, allowed.text)
        payload = allowed.json()
        self.assertEqual(payload["counts"]["users"], 3)
        self.assertIn("active_model", payload)
        self.assertEqual(
            sorted(user["role"] for user in payload["users"]),
            ["admin", "doctor", "patient"],
        )

    def test_patient_prediction_and_doctor_review_flow(self):
        patient_headers = self.login("patient@example.com", "patient123")
        doctor_headers = self.login("doctor@example.com", "doctor123")
        admin_headers = self.login("admin@example.com", "admin123")
        metadata = {
            "age": 55,
            "region": "FACE",
            "itch": "False",
            "grew": "True",
            "hurt": "False",
            "changed": "True",
            "bleed": "False",
            "elevation": "True",
        }
        consultation = self.client.post(
            "/patient/consultations",
            headers=patient_headers,
            json={"symptoms_notes": "Raised lesion", "clinical_metadata": metadata},
        )
        self.assertEqual(consultation.status_code, 200, consultation.text)
        consultation_id = consultation.json()["id"]
        image = Image.new("RGB", (8, 8), color="white")
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        upload = self.client.post(
            f"/patient/consultations/{consultation_id}/image",
            headers=patient_headers,
            files={"image": ("lesion.png", image_bytes.getvalue(), "image/png")},
        )
        self.assertEqual(upload.status_code, 200, upload.text)

        prediction = self.client.post(
            f"/patient/consultations/{consultation_id}/predict",
            headers=patient_headers,
        )
        self.assertEqual(prediction.status_code, 200, prediction.text)
        self.assertEqual(prediction.json()["predicted_label"], "BCC")
        self.assertEqual(prediction.json()["risk_level"], "high")

        queue = self.client.get("/doctor/consultations", headers=doctor_headers)
        self.assertEqual(queue.status_code, 200, queue.text)
        self.assertEqual(queue.json()[0]["latest_prediction"]["predicted_label"], "BCC")

        review = self.client.post(
            f"/doctor/consultations/{consultation_id}/review",
            headers=doctor_headers,
            json={
                "final_diagnosis": "BCC",
                "triage_decision": "urgent",
                "notes": "Needs in-person review.",
            },
        )
        self.assertEqual(review.status_code, 200, review.text)
        self.assertEqual(review.json()["final_diagnosis"], "BCC")

        retraining = self.client.get("/admin/retraining-cases", headers=admin_headers)
        self.assertEqual(retraining.status_code, 200, retraining.text)
        self.assertEqual(len(retraining.json()), 1)
        candidate = retraining.json()[0]
        self.assertEqual(candidate["consultation_id"], consultation_id)
        self.assertEqual(candidate["patient_email"], "patient@example.com")
        self.assertEqual(candidate["predicted_label"], "BCC")
        self.assertEqual(candidate["prediction_risk_level"], "high")
        self.assertEqual(candidate["final_diagnosis"], "BCC")
        self.assertEqual(candidate["triage_decision"], "urgent")
        self.assertFalse(candidate["disagreement"])
        self.assertEqual(candidate["clinical_metadata"]["region"], "FACE")

        csv_export = self.client.get("/admin/retraining-cases.csv", headers=admin_headers)
        self.assertEqual(csv_export.status_code, 200, csv_export.text)
        self.assertIn("text/csv", csv_export.headers["content-type"])
        self.assertIn("consultation_id,patient_email,image_path", csv_export.text)
        self.assertIn("patient@example.com", csv_export.text)

        monitoring = self.client.get("/admin/monitoring", headers=admin_headers)
        self.assertEqual(monitoring.status_code, 200, monitoring.text)
        monitoring_payload = monitoring.json()
        self.assertEqual(monitoring_payload["prediction_count"], 1)
        self.assertEqual(monitoring_payload["review_count"], 1)
        self.assertEqual(monitoring_payload["retraining_candidate_count"], 1)
        self.assertGreaterEqual(monitoring_payload["avg_latency_ms"], 0)
        self.assertGreaterEqual(monitoring_payload["p95_latency_ms"], 0)
        self.assertEqual(monitoring_payload["label_counts"]["BCC"], 1)
        self.assertEqual(monitoring_payload["risk_counts"]["high"], 1)
        self.assertEqual(monitoring_payload["review_agreement_rate"], 1.0)
        self.assertEqual(monitoring_payload["disagreement_count"], 0)
        self.assertEqual(monitoring_payload["recent_predictions"][0]["model_run_id"], "test-run")


if __name__ == "__main__":
    unittest.main()
