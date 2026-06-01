"""Utilities for feedback-driven multimodal retraining."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.make_image_splits import LABELS, MANIFEST_COLUMNS, TRIAGE_PRIORITY, compute_class_weights
from src.features.clinical_metadata import clinical_feature_fields
from src.inference.build_multimodal_bundle import build_multimodal_bundle
from src.training.train_image_baseline import load_json
from src.training.train_multimodal_baseline import MultimodalTrainingConfig, train_multimodal_baseline

COMPLETE_METADATA_FIELDS = clinical_feature_fields(include_optional=False)
ALL_METADATA_FIELDS = clinical_feature_fields(include_optional=True)
JSON_COLUMNS = ("clinical_metadata", "probabilities")


@dataclass(frozen=True)
class RetrainingPipelineConfig:
    feedback_path: Path
    images_dir: Path
    metadata_path: Path
    splits_dir: Path
    output_dir: Path
    current_metrics_path: Path | None = None
    candidate_bundle_dir: Path | None = None
    min_feedback_cases: int = 1
    epochs: int = 8
    batch_size: int = 32
    image_size: int = 224
    seed: int = 42
    num_workers: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    experiment_name: str = "pad-ufes-20-feedback-retraining"
    tracking_uri: str | None = None
    require_gpu: bool = True
    loss_type: str = "weighted_cross_entropy"
    sampler: str = "weighted_random"
    augment_strength: str = "current"
    focal_gamma: float = 2.0
    include_optional_metadata: bool = True
    metadata_hidden_dim: int = 64
    fusion_hidden_dim: int = 256
    metadata_dropout: float = 0.1
    initial_image_checkpoint: Path | None = None
    build_candidate_bundle: bool = False
    dry_run: bool = False


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def parse_jsonish(value: object) -> object:
    if isinstance(value, (dict, list)):
        return value
    if is_missing(value):
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return value


def load_feedback_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Feedback manifest not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else payload.get("cases", [])
    else:
        rows = pd.read_csv(path).to_dict(orient="records")

    normalized = []
    for row in rows:
        item = dict(row)
        for column in JSON_COLUMNS:
            if column in item:
                item[column] = parse_jsonish(item[column])
        normalized.append(item)
    return normalized


def validate_feedback_cases(
    rows: list[dict[str, Any]],
    min_cases: int = 1,
    allowed_labels: tuple[str, ...] = tuple(LABELS),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_consultations: set[str] = set()

    for index, row in enumerate(rows):
        reasons = []
        consultation_id = str(row.get("consultation_id", f"row-{index}"))
        if consultation_id in seen_consultations:
            reasons.append("duplicate consultation_id")
        seen_consultations.add(consultation_id)

        image_path = row.get("image_path")
        if is_missing(image_path) or not Path(str(image_path)).exists():
            reasons.append("image_path is missing or does not exist")

        final_diagnosis = str(row.get("final_diagnosis", "")).strip().upper()
        if final_diagnosis not in allowed_labels:
            reasons.append(f"final_diagnosis must be one of {list(allowed_labels)}")

        metadata = row.get("clinical_metadata")
        if not isinstance(metadata, dict):
            reasons.append("clinical_metadata must be a JSON object")
            metadata = {}

        missing_metadata = [
            field
            for field in COMPLETE_METADATA_FIELDS
            if field not in metadata or is_missing(metadata[field])
        ]
        if missing_metadata:
            reasons.append(f"missing complete metadata fields: {missing_metadata}")

        if reasons:
            skipped.append(
                {
                    "consultation_id": consultation_id,
                    "row_index": index,
                    "reasons": reasons,
                }
            )
            continue

        normalized_row = dict(row)
        normalized_row["consultation_id"] = consultation_id
        normalized_row["final_diagnosis"] = final_diagnosis
        normalized_row["clinical_metadata"] = metadata
        valid_rows.append(normalized_row)

    label_counts = pd.Series([row["final_diagnosis"] for row in valid_rows]).value_counts().to_dict()
    report = {
        "candidate_count": len(rows),
        "valid_count": len(valid_rows),
        "min_feedback_cases": min_cases,
        "ready": len(valid_rows) >= min_cases,
        "label_counts": {str(label): int(count) for label, count in label_counts.items()},
        "skipped": skipped,
    }
    return valid_rows, report


def metadata_value(metadata: dict[str, Any], field: str) -> Any:
    if field == "fitspatrick" and "fitspatrick" not in metadata:
        return metadata.get("fitzpatrick")
    return metadata.get(field)


def feedback_img_id(row: dict[str, Any]) -> str:
    image_name = Path(str(row["image_path"])).name
    return f"feedback_{row['consultation_id']}_{image_name}"


def build_augmented_training_inputs(
    valid_rows: list[dict[str, Any]],
    metadata_path: Path,
    splits_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_splits_dir = output_dir / "splits"
    output_splits_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_path = output_dir / "metadata.csv"

    original_train = pd.read_csv(splits_dir / "train.csv")
    original_metadata = pd.read_csv(metadata_path)
    label_mapping = load_json(splits_dir / "label_mapping.json")
    preprocessing_summary = load_json(splits_dir / "preprocessing_summary.json")
    labels = [
        label
        for _, label in sorted(
            (int(index), label)
            for index, label in label_mapping["index_to_label"].items()
        )
    ]

    feedback_split_rows = []
    feedback_metadata_rows = []
    for row in valid_rows:
        label = row["final_diagnosis"]
        img_id = feedback_img_id(row)
        feedback_split_rows.append(
            {
                "patient_id": f"feedback_patient_{row['consultation_id']}",
                "lesion_id": f"feedback_consultation_{row['consultation_id']}",
                "img_id": img_id,
                "image_path": str(Path(str(row["image_path"])).resolve()),
                "image_rel_path": f"feedback/{Path(str(row['image_path'])).name}",
                "diagnostic": label,
                "label_idx": labels.index(label),
                "triage_priority": TRIAGE_PRIORITY[label],
                "split": "train",
            }
        )

        metadata_row = {
            "patient_id": f"feedback_patient_{row['consultation_id']}",
            "lesion_id": f"feedback_consultation_{row['consultation_id']}",
            "img_id": img_id,
            "diagnostic": label,
        }
        metadata = row["clinical_metadata"]
        for field in ALL_METADATA_FIELDS:
            value = metadata_value(metadata, field)
            if not is_missing(value):
                metadata_row[field] = value
        feedback_metadata_rows.append(metadata_row)

    feedback_frame = pd.DataFrame(feedback_split_rows)
    train_columns = list(dict.fromkeys([*original_train.columns, *MANIFEST_COLUMNS]))
    train_augmented = pd.concat(
        [
            original_train.reindex(columns=train_columns),
            feedback_frame.reindex(columns=train_columns),
        ],
        ignore_index=True,
    )
    train_augmented.to_csv(output_splits_dir / "train.csv", index=False)
    for split_name in ("val", "test"):
        shutil.copyfile(splits_dir / f"{split_name}.csv", output_splits_dir / f"{split_name}.csv")
    shutil.copyfile(splits_dir / "label_mapping.json", output_splits_dir / "label_mapping.json")

    class_weights = compute_class_weights(train_augmented)
    class_weight_payload = {
        "objective": "weighted_cross_entropy",
        "class_weights": class_weights,
        "train_class_counts": {
            label: int(count)
            for label, count in train_augmented["diagnostic"]
            .value_counts()
            .reindex(LABELS, fill_value=0)
            .items()
        },
    }
    (output_splits_dir / "class_weights.json").write_text(
        json.dumps(class_weight_payload, indent=2) + "\n"
    )

    preprocessing_summary["feedback_retraining"] = {
        "feedback_cases_added_to_train": len(feedback_split_rows),
        "source_metadata_path": str(metadata_path),
        "source_splits_dir": str(splits_dir),
    }
    (output_splits_dir / "preprocessing_summary.json").write_text(
        json.dumps(preprocessing_summary, indent=2) + "\n"
    )

    metadata_augmented = pd.concat(
        [original_metadata, pd.DataFrame(feedback_metadata_rows)],
        ignore_index=True,
    )
    duplicates = metadata_augmented.loc[metadata_augmented["img_id"].duplicated(), "img_id"]
    if not duplicates.empty:
        raise ValueError(f"Augmented metadata has duplicate img_id values: {duplicates.head().tolist()}")
    metadata_augmented.to_csv(output_metadata_path, index=False)

    return {
        "metadata_path": str(output_metadata_path),
        "splits_dir": str(output_splits_dir),
        "feedback_cases_added_to_train": len(feedback_split_rows),
        "train_rows": int(len(train_augmented)),
    }


def load_current_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", payload)
    return {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def evaluate_promotion_gate(
    candidate_metrics: dict[str, Any],
    current_metrics: dict[str, float],
    min_macro_f1_delta: float = 0.0,
    high_risk_recall_tolerance: float = 0.02,
    balanced_accuracy_tolerance: float = 0.02,
    selection_score_tolerance: float = 0.0,
) -> dict[str, Any]:
    checks = []
    rules = [
        ("test_macro_f1", min_macro_f1_delta, "at_least_delta"),
        ("test_high_risk_recall", -high_risk_recall_tolerance, "no_more_than_tolerance_drop"),
        ("test_balanced_accuracy", -balanced_accuracy_tolerance, "no_more_than_tolerance_drop"),
        ("test_selection_score", -selection_score_tolerance, "no_drop"),
    ]

    for metric, threshold_delta, rule in rules:
        candidate = candidate_metrics.get(metric)
        current = current_metrics.get(metric)
        if candidate is None or current is None:
            checks.append({"metric": metric, "status": "skipped", "reason": "missing metric"})
            continue
        required = current + threshold_delta
        passed = float(candidate) >= required
        checks.append(
            {
                "metric": metric,
                "candidate": float(candidate),
                "current": float(current),
                "required": float(required),
                "rule": rule,
                "status": "pass" if passed else "fail",
            }
        )

    compared = any(check["status"] != "skipped" for check in checks)
    passed = compared and all(check["status"] in {"pass", "skipped"} for check in checks)
    return {
        "passed": passed,
        "review_required": not passed,
        "checks": checks,
    }


def run_retraining_pipeline(config: RetrainingPipelineConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    feedback_rows = load_feedback_cases(config.feedback_path)
    valid_rows, validation = validate_feedback_cases(
        feedback_rows,
        min_cases=config.min_feedback_cases,
    )
    report: dict[str, Any] = {
        "feedback_path": str(config.feedback_path),
        "validation": validation,
        "dry_run": config.dry_run,
    }
    if not validation["ready"]:
        report["status"] = "not_ready"
        (config.output_dir / "retraining_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    prepared_inputs = build_augmented_training_inputs(
        valid_rows,
        metadata_path=config.metadata_path,
        splits_dir=config.splits_dir,
        output_dir=config.output_dir / "prepared",
    )
    report["prepared_inputs"] = prepared_inputs
    if config.dry_run:
        report["status"] = "dry_run_ready"
        (config.output_dir / "retraining_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    training_config = MultimodalTrainingConfig(
        images_dir=config.images_dir,
        metadata_path=Path(prepared_inputs["metadata_path"]),
        splits_dir=Path(prepared_inputs["splits_dir"]),
        output_dir=config.output_dir / "training",
        epochs=config.epochs,
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed=config.seed,
        num_workers=config.num_workers,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        experiment_name=config.experiment_name,
        tracking_uri=config.tracking_uri,
        require_gpu=config.require_gpu,
        loss_type=config.loss_type,
        sampler=config.sampler,
        augment_strength=config.augment_strength,
        focal_gamma=config.focal_gamma,
        include_optional_metadata=config.include_optional_metadata,
        metadata_hidden_dim=config.metadata_hidden_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        metadata_dropout=config.metadata_dropout,
        initial_image_checkpoint=config.initial_image_checkpoint,
    )
    candidate_metrics = train_multimodal_baseline(training_config)
    current_metrics = load_current_metrics(config.current_metrics_path)
    promotion_gate = evaluate_promotion_gate(candidate_metrics, current_metrics)
    report.update(
        {
            "status": "trained",
            "candidate_metrics": candidate_metrics,
            "current_metrics": current_metrics,
            "promotion_gate": promotion_gate,
        }
    )

    if config.build_candidate_bundle and config.candidate_bundle_dir is not None:
        run_id = str(candidate_metrics.get("mlflow_run_id", "local-candidate"))
        bundle_dir = build_multimodal_bundle(
            output_dir=config.candidate_bundle_dir,
            metadata_path=Path(prepared_inputs["metadata_path"]),
            splits_dir=Path(prepared_inputs["splits_dir"]),
            checkpoint_path=config.output_dir / "training" / "efficientnet_b0_multimodal_best.pt",
            mlflow_run_id=run_id,
            include_optional_metadata=config.include_optional_metadata,
        )
        manifest_path = bundle_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metrics"] = candidate_metrics
        manifest["promotion_gate"] = promotion_gate
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        report["candidate_bundle_dir"] = str(bundle_dir)

    (config.output_dir / "retraining_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
