"""Build an inference-ready bundle for the best multimodal teledermatology model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.features.clinical_metadata import (
    encode_clinical_metadata,
    merge_clinical_metadata,
    read_metadata,
)
from src.training.train_image_baseline import load_split_inputs, resolve_tracking_uri

DEFAULT_BEST_RUN_ID = "ef084927bef741f996894b8a0fdd63e3"
DEFAULT_CHECKPOINT_ARTIFACT = "efficientnet_b0_multimodal_best.pt"
DEFAULT_METRICS_ARTIFACT = "multimodal_test_metrics.json"


def download_mlflow_artifact(run_id: str, artifact_path: str, output_dir: Path) -> Path:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    downloaded_path = client.download_artifacts(run_id, artifact_path, str(output_dir))
    return Path(downloaded_path)


def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def load_checkpoint_metadata(checkpoint_path: Path) -> dict[str, object]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    return {
        "labels": list(checkpoint.get("labels", [])),
        "metadata_feature_names": list(checkpoint.get("metadata_feature_names", [])),
        "metadata_hidden_dim": int(config.get("metadata_hidden_dim", 64)),
        "fusion_hidden_dim": int(config.get("fusion_hidden_dim", 256)),
        "metadata_dropout": float(config.get("metadata_dropout", 0.1)),
        "image_size": int(config.get("image_size", 224)),
        "checkpoint_config": make_json_safe(config),
    }


def build_encoder(metadata_path: Path, splits_dir: Path, include_optional_metadata: bool = True):
    splits = load_split_inputs(splits_dir)
    metadata = read_metadata(metadata_path)
    train_frame = merge_clinical_metadata(
        splits.train,
        metadata,
        include_optional=include_optional_metadata,
    )
    encoder, [_] = encode_clinical_metadata(
        train_frame,
        [train_frame],
        include_optional=include_optional_metadata,
    )
    return encoder


def build_multimodal_bundle(
    output_dir: Path,
    metadata_path: Path,
    splits_dir: Path,
    checkpoint_path: Path | None = None,
    mlflow_run_id: str = DEFAULT_BEST_RUN_ID,
    include_optional_metadata: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = output_dir / "_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint = checkpoint_path
    if source_checkpoint is None:
        source_checkpoint = download_mlflow_artifact(
            mlflow_run_id,
            DEFAULT_CHECKPOINT_ARTIFACT,
            downloads_dir,
        )

    checkpoint_metadata = load_checkpoint_metadata(source_checkpoint)
    encoder = build_encoder(
        metadata_path,
        splits_dir,
        include_optional_metadata=include_optional_metadata,
    )
    checkpoint_feature_names = checkpoint_metadata["metadata_feature_names"]
    if checkpoint_feature_names and checkpoint_feature_names != encoder.feature_names:
        raise ValueError(
            "The rebuilt clinical metadata encoder does not match the checkpoint feature order."
        )

    bundle_checkpoint = output_dir / DEFAULT_CHECKPOINT_ARTIFACT
    if source_checkpoint.resolve() != bundle_checkpoint.resolve():
        shutil.copyfile(source_checkpoint, bundle_checkpoint)

    encoder_path = output_dir / "clinical_metadata_encoder.json"
    encoder.save(encoder_path)

    metrics = {}
    try:
        if checkpoint_path is None:
            metrics_path = download_mlflow_artifact(
                mlflow_run_id,
                DEFAULT_METRICS_ARTIFACT,
                downloads_dir,
            )
            metrics = json.loads(Path(metrics_path).read_text())
    except Exception:
        metrics = {}

    manifest = {
        "model_run_id": mlflow_run_id,
        "checkpoint_filename": bundle_checkpoint.name,
        "metadata_encoder_filename": encoder_path.name,
        "labels": checkpoint_metadata["labels"],
        "metadata_feature_count": len(encoder.feature_names),
        "metadata_hidden_dim": checkpoint_metadata["metadata_hidden_dim"],
        "fusion_hidden_dim": checkpoint_metadata["fusion_hidden_dim"],
        "metadata_dropout": checkpoint_metadata["metadata_dropout"],
        "image_size": checkpoint_metadata["image_size"],
        "metrics": metrics,
        "checkpoint_config": checkpoint_metadata["checkpoint_config"],
        "warning": "Decision support only; doctor review required.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("storage/model_bundle"))
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--mlflow-run-id", default=DEFAULT_BEST_RUN_ID)
    parser.add_argument(
        "--complete-fields-only",
        action="store_true",
        help="Use only complete PAD-UFES-20 metadata fields, matching training when optional metadata was disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_dir = build_multimodal_bundle(
        output_dir=args.output_dir,
        metadata_path=args.metadata_path,
        splits_dir=args.splits_dir,
        checkpoint_path=args.checkpoint_path,
        mlflow_run_id=args.mlflow_run_id,
        include_optional_metadata=not args.complete_fields_only,
    )
    print(f"Wrote multimodal inference bundle to {bundle_dir}")


if __name__ == "__main__":
    main()
