"""Multimodal single-case inference for the PAD-UFES-20 telemedicine demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
from PIL import Image

from src.features.clinical_metadata import ClinicalMetadataEncoder
from src.training.train_image_baseline import HIGH_RISK_LABELS, make_transforms
from src.training.train_multimodal_baseline import build_multimodal_model, require_torch

WARNING = "Decision support only; doctor review required."


@dataclass(frozen=True)
class MultimodalBundle:
    bundle_dir: Path
    checkpoint_path: Path
    encoder_path: Path
    labels: list[str]
    model_run_id: str
    image_size: int = 224
    metadata_hidden_dim: int = 64
    fusion_hidden_dim: int = 256
    metadata_dropout: float = 0.1

    @classmethod
    def load(cls, bundle_dir: Path) -> "MultimodalBundle":
        bundle_dir = Path(bundle_dir)
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing inference bundle manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        checkpoint_path = bundle_dir / manifest["checkpoint_filename"]
        encoder_path = bundle_dir / manifest["metadata_encoder_filename"]
        return cls(
            bundle_dir=bundle_dir,
            checkpoint_path=checkpoint_path,
            encoder_path=encoder_path,
            labels=list(manifest["labels"]),
            model_run_id=str(manifest["model_run_id"]),
            image_size=int(manifest.get("image_size", 224)),
            metadata_hidden_dim=int(manifest.get("metadata_hidden_dim", 64)),
            fusion_hidden_dim=int(manifest.get("fusion_hidden_dim", 256)),
            metadata_dropout=float(manifest.get("metadata_dropout", 0.1)),
        )


class MultimodalPredictor:
    def __init__(self, bundle: MultimodalBundle, device=None):
        torch_module = require_torch()
        self.bundle = bundle
        self.device = device or torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
        self.encoder = ClinicalMetadataEncoder.load(bundle.encoder_path)
        checkpoint = torch_module.load(bundle.checkpoint_path, map_location=self.device, weights_only=False)
        self.labels = list(checkpoint.get("labels", bundle.labels))
        self.model = build_multimodal_model(
            metadata_feature_count=len(self.encoder.feature_names),
            num_classes=len(self.labels),
            metadata_hidden_dim=bundle.metadata_hidden_dim,
            fusion_hidden_dim=bundle.fusion_hidden_dim,
            metadata_dropout=bundle.metadata_dropout,
            pretrained_image_weights=False,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        _, self.transform = make_transforms(bundle.image_size)

    @classmethod
    def from_bundle(cls, bundle_dir: Path, device=None) -> "MultimodalPredictor":
        return cls(MultimodalBundle.load(bundle_dir), device=device)

    def predict(self, image_path: Path, metadata: Mapping[str, object]) -> dict[str, object]:
        torch_module = require_torch()
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        metadata_frame = pd.DataFrame([self._metadata_row(metadata)])
        metadata_features = self.encoder.transform(metadata_frame).astype("float32")
        metadata_tensor = torch_module.as_tensor(
            metadata_features.to_numpy(dtype="float32", copy=True),
            dtype=torch_module.float32,
            device=self.device,
        )

        with torch_module.no_grad():
            logits = self.model(image_tensor, metadata_tensor)
            scores = torch_module.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        probabilities = {
            label: float(score)
            for label, score in zip(self.labels, scores)
        }
        predicted_label = max(probabilities, key=probabilities.get)
        return {
            "predicted_label": predicted_label,
            "risk_level": risk_level_for_label(predicted_label),
            "probabilities": probabilities,
            "model_run_id": self.bundle.model_run_id,
            "warning": WARNING,
        }

    def _metadata_row(self, metadata: Mapping[str, object]) -> dict[str, object]:
        expected_fields = list(self.encoder.numeric_fields) + list(self.encoder.categorical_levels)
        return {
            field: metadata.get(field)
            for field in expected_fields
        }


def risk_level_for_label(label: str) -> str:
    return "high" if label in HIGH_RISK_LABELS else "routine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument(
        "--metadata-json",
        required=True,
        help="Clinical metadata as a JSON object, for example '{\"age\": 62, \"region\": \"FACE\"}'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = MultimodalPredictor.from_bundle(args.bundle_dir)
    print(json.dumps(predictor.predict(args.image_path, json.loads(args.metadata_json)), indent=2))


if __name__ == "__main__":
    main()
