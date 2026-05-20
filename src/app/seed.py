"""Seed demo users and the active model version."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from src.app.models import ModelVersion, PatientProfile, User
from src.app.security import hash_password

DEMO_USERS = (
    ("patient@example.com", "patient123", "patient", "Demo Patient"),
    ("doctor@example.com", "doctor123", "doctor", "Demo Doctor"),
    ("admin@example.com", "admin123", "admin", "Demo Admin"),
)


def seed_demo_users(db: Session) -> None:
    for email, password, role, display_name in DEMO_USERS:
        user = db.query(User).filter(User.email == email).one_or_none()
        if user is None:
            user = User(email=email, hashed_password=hash_password(password), role=role)
            db.add(user)
            db.flush()
        if role == "patient" and user.patient_profile is None:
            db.add(PatientProfile(user_id=user.id, display_name=display_name))
    db.commit()


def seed_model_version(db: Session, bundle_dir: Path) -> None:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    run_id = str(manifest["model_run_id"])
    model_version = db.query(ModelVersion).filter(ModelVersion.run_id == run_id).one_or_none()
    if model_version is None:
        model_version = ModelVersion(
            run_id=run_id,
            checkpoint_path=str(bundle_dir / manifest["checkpoint_filename"]),
            metrics=manifest.get("metrics", {}),
            active=True,
        )
        db.add(model_version)
    else:
        model_version.checkpoint_path = str(bundle_dir / manifest["checkpoint_filename"])
        model_version.metrics = manifest.get("metrics", {})
        model_version.active = True
    db.commit()
