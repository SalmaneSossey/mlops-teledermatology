import unittest
import json
import tempfile
from importlib.util import find_spec
from pathlib import Path

from PIL import Image

from src.features.clinical_metadata import ClinicalMetadataEncoder
from src.training.train_image_baseline import DEFAULT_LABELS
from src.training.train_multimodal_baseline import build_multimodal_model

from src.inference.predict_multimodal import MultimodalPredictor, risk_level_for_label


class PredictMultimodalTest(unittest.TestCase):
    def test_high_risk_labels_map_to_high_triage(self):
        for label in ("BCC", "MEL", "SCC"):
            self.assertEqual(risk_level_for_label(label), "high")

    def test_non_high_risk_labels_map_to_routine(self):
        for label in ("ACK", "NEV", "SEK"):
            self.assertEqual(risk_level_for_label(label), "routine")

    def test_single_case_metadata_fills_missing_encoder_fields(self):
        encoder = ClinicalMetadataEncoder(
            numeric_fields=("age", "diameter_1"),
            optional_numeric_fields=("diameter_1",),
            categorical_levels={"region": ("FACE",), "gender": ("MALE", "FEMALE")},
            optional_categorical_fields=("gender",),
            means={"age": 50.0, "diameter_1": 5.0},
            stds={"age": 10.0, "diameter_1": 2.0},
        )
        predictor = object.__new__(MultimodalPredictor)
        predictor.encoder = encoder

        row = predictor._metadata_row({"age": 60, "region": "FACE"})

        self.assertEqual(row["age"], 60)
        self.assertEqual(row["region"], "FACE")
        self.assertIsNone(row["diameter_1"])
        self.assertIsNone(row["gender"])

    def test_predictor_loads_bundle_and_returns_probability_distribution(self):
        if find_spec("torch") is None or find_spec("torchvision") is None:
            self.skipTest("PyTorch and torchvision are required for full predictor loading")
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            encoder = ClinicalMetadataEncoder(
                numeric_fields=("age",),
                optional_numeric_fields=(),
                categorical_levels={},
                optional_categorical_fields=(),
                means={"age": 50.0},
                stds={"age": 10.0},
            )
            encoder.save(root / "clinical_metadata_encoder.json")
            model = build_multimodal_model(
                metadata_feature_count=1,
                num_classes=len(DEFAULT_LABELS),
                metadata_dropout=0.0,
                pretrained_image_weights=False,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": DEFAULT_LABELS,
                    "metadata_feature_names": encoder.feature_names,
                },
                root / "efficientnet_b0_multimodal_best.pt",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "model_run_id": "test-run",
                        "checkpoint_filename": "efficientnet_b0_multimodal_best.pt",
                        "metadata_encoder_filename": "clinical_metadata_encoder.json",
                        "labels": DEFAULT_LABELS,
                        "metadata_hidden_dim": 64,
                        "fusion_hidden_dim": 256,
                        "metadata_dropout": 0.0,
                        "image_size": 224,
                    }
                )
            )
            image_path = root / "image.jpg"
            Image.new("RGB", (8, 8), color="white").save(image_path)

            predictor = MultimodalPredictor.from_bundle(root, device=torch.device("cpu"))
            result = predictor.predict(image_path, {"age": 60})

        self.assertEqual(result["model_run_id"], "test-run")
        self.assertIn(result["predicted_label"], DEFAULT_LABELS)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
