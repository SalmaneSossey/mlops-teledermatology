import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
from PIL import Image

from src.training.train_image_baseline import DEFAULT_LABELS
from src.training.train_multimodal_baseline import (
    MultimodalTrainingConfig,
    PadUfesMultimodalDataset,
    build_multimodal_artifact_paths,
    evaluate_multimodal_predictions,
    prepare_multimodal_inputs,
    validate_multimodal_training_options,
)


class TrainMultimodalBaselineTest(unittest.TestCase):
    def make_config(self) -> MultimodalTrainingConfig:
        return MultimodalTrainingConfig(
            images_dir=Path("/images"),
            metadata_path=Path("/metadata.csv"),
            splits_dir=Path("/splits"),
            output_dir=Path("/out"),
        )

    def make_split_rows(self) -> dict[str, list[dict[str, object]]]:
        return {
            "train": [
                {
                    "patient_id": "P2",
                    "lesion_id": "L2",
                    "img_id": "b.png",
                    "image_path": "/data/b.png",
                    "image_rel_path": "imgs_part_1/b.png",
                    "diagnostic": "BCC",
                    "label_idx": 1,
                    "triage_priority": "high",
                    "split": "train",
                },
                {
                    "patient_id": "P1",
                    "lesion_id": "L1",
                    "img_id": "a.png",
                    "image_path": "/data/a.png",
                    "image_rel_path": "imgs_part_1/a.png",
                    "diagnostic": "ACK",
                    "label_idx": 0,
                    "triage_priority": "medium",
                    "split": "train",
                },
            ],
            "val": [
                {
                    "patient_id": "P3",
                    "lesion_id": "L3",
                    "img_id": "c.png",
                    "image_path": "/data/c.png",
                    "image_rel_path": "imgs_part_1/c.png",
                    "diagnostic": "MEL",
                    "label_idx": 2,
                    "triage_priority": "high",
                    "split": "val",
                }
            ],
            "test": [
                {
                    "patient_id": "P4",
                    "lesion_id": "L4",
                    "img_id": "d.png",
                    "image_path": "/data/d.png",
                    "image_rel_path": "imgs_part_1/d.png",
                    "diagnostic": "NEV",
                    "label_idx": 3,
                    "triage_priority": "low",
                    "split": "test",
                }
            ],
        }

    def write_split_inputs(self, root: Path) -> None:
        for split, rows in self.make_split_rows().items():
            pd.DataFrame(rows).to_csv(root / f"{split}.csv", index=False)

        (root / "label_mapping.json").write_text(
            json.dumps(
                {
                    "label_to_index": {label: index for index, label in enumerate(DEFAULT_LABELS)},
                    "index_to_label": {
                        str(index): label for index, label in enumerate(DEFAULT_LABELS)
                    },
                }
            )
        )
        (root / "class_weights.json").write_text(
            json.dumps(
                {
                    "class_weights": {
                        label: float(index + 1) for index, label in enumerate(DEFAULT_LABELS)
                    }
                }
            )
        )
        (root / "preprocessing_summary.json").write_text(json.dumps({"seed": 42}))

    def write_metadata(self, path: Path) -> None:
        pd.DataFrame(
            {
                "img_id": ["a.png", "b.png", "c.png", "d.png"],
                "age": [50, 70, 60, 40],
                "region": ["FACE", "ARM", "BACK", "CHEST"],
                "itch": ["True", "False", "UNK", "False"],
                "grew": ["True", "False", "False", "True"],
                "hurt": ["False", "False", "True", "False"],
                "changed": ["True", "False", "UNK", "False"],
                "bleed": ["False", "True", "False", "False"],
                "elevation": ["True", "False", "UNK", "True"],
            }
        ).to_csv(path, index=False)

    def test_artifact_paths_use_expected_multimodal_names(self):
        paths = build_multimodal_artifact_paths(Path("/tmp/run"))

        self.assertEqual(
            paths.best_checkpoint,
            Path("/tmp/run/efficientnet_b0_multimodal_best.pt"),
        )
        self.assertEqual(paths.history_csv, Path("/tmp/run/multimodal_history.csv"))
        self.assertEqual(paths.test_metrics_json, Path("/tmp/run/multimodal_test_metrics.json"))
        self.assertEqual(
            paths.classification_report_csv,
            Path("/tmp/run/multimodal_classification_report.csv"),
        )
        self.assertEqual(
            paths.metadata_feature_names_json,
            Path("/tmp/run/multimodal_metadata_feature_names.json"),
        )

    def test_prepare_multimodal_inputs_joins_metadata_by_img_id_and_keeps_split_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            splits_dir = root / "splits"
            splits_dir.mkdir()
            metadata_path = root / "metadata.csv"
            self.write_split_inputs(splits_dir)
            self.write_metadata(metadata_path)

            inputs = prepare_multimodal_inputs(
                metadata_path,
                splits_dir,
                include_optional_metadata=False,
            )

        self.assertEqual(inputs.train["img_id"].tolist(), ["b.png", "a.png"])
        self.assertEqual(inputs.labels, DEFAULT_LABELS)
        self.assertEqual(inputs.train_metadata_features.shape[0], 2)
        self.assertIn("age__z", inputs.metadata_feature_names)

    def test_dataset_returns_metadata_features_aligned_with_image_rows(self):
        if find_spec("torch") is None:
            self.skipTest("PyTorch is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "all_images" / "imgs_part_1"
            images_dir.mkdir(parents=True)
            for image_name in ["a.png", "b.png"]:
                Image.new("RGB", (2, 2), color="white").save(images_dir / image_name)

            frame = pd.DataFrame(
                {
                    "image_rel_path": ["imgs_part_1/a.png", "imgs_part_1/b.png"],
                    "label_idx": [0, 1],
                }
            )
            metadata_features = pd.DataFrame(
                {
                    "age__z": [10.0, 20.0],
                    "region__FACE": [1.0, 0.0],
                }
            )
            dataset = PadUfesMultimodalDataset(frame, metadata_features, root / "all_images")

            _, first_features, first_label = dataset[0]
            _, second_features, second_label = dataset[1]

        self.assertEqual(first_label, 0)
        self.assertEqual(second_label, 1)
        self.assertEqual(first_features.tolist(), [10.0, 1.0])
        self.assertEqual(second_features.tolist(), [20.0, 0.0])

    def test_validate_multimodal_training_options_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            validate_multimodal_training_options(
                MultimodalTrainingConfig(
                    images_dir=Path("/images"),
                    metadata_path=Path("/metadata.csv"),
                    splits_dir=Path("/splits"),
                    output_dir=Path("/out"),
                    metadata_dropout=1.0,
                )
            )

        with self.assertRaises(ValueError):
            validate_multimodal_training_options(
                MultimodalTrainingConfig(
                    images_dir=Path("/images"),
                    metadata_path=Path("/metadata.csv"),
                    splits_dir=Path("/splits"),
                    output_dir=Path("/out"),
                    metadata_hidden_dim=0,
                )
            )

        with self.assertRaises(ValueError):
            validate_multimodal_training_options(
                MultimodalTrainingConfig(
                    images_dir=Path("/images"),
                    metadata_path=Path("/metadata.csv"),
                    splits_dir=Path("/splits"),
                    output_dir=Path("/out"),
                    sampler="mystery",
                )
            )

    def test_evaluate_multimodal_predictions_adds_selection_score(self):
        if find_spec("sklearn") is None:
            self.skipTest("scikit-learn is not installed")

        metrics, report, matrix = evaluate_multimodal_predictions(
            targets=[0, 1, 2, 3, 4],
            predictions=[0, 1, 2, 0, 3],
            labels=DEFAULT_LABELS,
        )

        self.assertAlmostEqual(metrics["test_high_risk_recall"], 2 / 3)
        self.assertIn("test_selection_score", metrics)
        self.assertIn("SEK", report.index)
        self.assertEqual(matrix.loc["BCC", "BCC"], 1)


if __name__ == "__main__":
    unittest.main()
