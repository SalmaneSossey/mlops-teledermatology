"""Train a metadata-only baseline for PAD-UFES-20 clinical features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.features.clinical_metadata import (
    encode_clinical_metadata,
    merge_clinical_metadata,
    read_metadata,
)
from src.training.train_image_baseline import (
    DEFAULT_EXPERIMENT_NAME,
    high_risk_label_indices,
    high_risk_recall,
    load_split_inputs,
    selection_score,
)


@dataclass(frozen=True)
class MetadataArtifactPaths:
    output_dir: Path
    metrics_json: Path
    classification_report_csv: Path
    confusion_matrix_csv: Path
    feature_names_json: Path


def build_metadata_artifact_paths(output_dir: Path) -> MetadataArtifactPaths:
    return MetadataArtifactPaths(
        output_dir=output_dir,
        metrics_json=output_dir / "metadata_test_metrics.json",
        classification_report_csv=output_dir / "metadata_classification_report.csv",
        confusion_matrix_csv=output_dir / "metadata_confusion_matrix.csv",
        feature_names_json=output_dir / "metadata_feature_names.json",
    )


def build_metadata_model(max_iter: int = 1000, c_value: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42,
    )


def evaluate_predictions(
    targets: Sequence[int],
    predictions: Sequence[int],
    labels: Sequence[str],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    high_risk_indices = high_risk_label_indices(labels)
    metrics = {
        "test_macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "test_high_risk_recall": high_risk_recall(targets, predictions, high_risk_indices),
    }
    report = pd.DataFrame(
        classification_report(
            targets,
            predictions,
            target_names=list(labels),
            zero_division=0,
            output_dict=True,
        )
    ).T
    matrix = pd.DataFrame(
        confusion_matrix(targets, predictions, labels=list(range(len(labels)))),
        index=labels,
        columns=labels,
    )
    return metrics, report, matrix


def train_metadata_baseline(
    metadata_path: Path,
    splits_dir: Path,
    output_dir: Path,
    include_optional: bool = True,
    max_iter: int = 1000,
    c_value: float = 1.0,
) -> dict[str, object]:
    splits = load_split_inputs(splits_dir)
    metadata = read_metadata(metadata_path)
    train_frame = merge_clinical_metadata(splits.train, metadata, include_optional=include_optional)
    val_frame = merge_clinical_metadata(splits.val, metadata, include_optional=include_optional)
    test_frame = merge_clinical_metadata(splits.test, metadata, include_optional=include_optional)

    encoder, [train_features, val_features, test_features] = encode_clinical_metadata(
        train_frame,
        [train_frame, val_frame, test_frame],
        include_optional=include_optional,
    )
    model = build_metadata_model(max_iter=max_iter, c_value=c_value)
    model.fit(train_features, train_frame["label_idx"].astype(int))

    val_predictions = model.predict(val_features)
    val_metrics, _, _ = evaluate_predictions(
        val_frame["label_idx"].astype(int).tolist(),
        val_predictions.tolist(),
        splits.labels,
    )
    test_predictions = model.predict(test_features)
    test_metrics, report, matrix = evaluate_predictions(
        test_frame["label_idx"].astype(int).tolist(),
        test_predictions.tolist(),
        splits.labels,
    )
    metrics = {
        **test_metrics,
        "val_selection_score": selection_score(
            {
                "macro_f1": val_metrics["test_macro_f1"],
                "high_risk_recall": val_metrics["test_high_risk_recall"],
            }
        ),
        "test_selection_score": selection_score(
            {
                "macro_f1": test_metrics["test_macro_f1"],
                "high_risk_recall": test_metrics["test_high_risk_recall"],
            }
        ),
        "include_optional_metadata": include_optional,
        "feature_count": len(encoder.feature_names),
        "model": "balanced_logistic_regression",
    }

    paths = build_metadata_artifact_paths(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    paths.feature_names_json.write_text(json.dumps(encoder.feature_names, indent=2) + "\n")
    report.to_csv(paths.classification_report_csv)
    matrix.to_csv(paths.confusion_matrix_csv)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument(
        "--complete-fields-only",
        action="store_true",
        help="Use only complete metadata fields and skip optional fields with missingness.",
    )
    parser.add_argument(
        "--experiment-name",
        default=f"{DEFAULT_EXPERIMENT_NAME}-metadata-baseline",
        help="Reserved for naming consistency with MLflow-tracked image experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_metadata_baseline(
        metadata_path=args.metadata_path,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        include_optional=not args.complete_fields_only,
        max_iter=args.max_iter,
        c_value=args.c_value,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
