import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.training.train_image_baseline import (
    DEFAULT_LABELS,
    PadUfesImageDataset,
    TrainingConfig,
    WorkerSeeder,
    build_artifact_paths,
    high_risk_label_indices,
    high_risk_recall,
    load_initial_checkpoint,
    load_split_inputs,
    sample_weights_for_training,
    selection_score,
    validate_training_options,
)


class TrainImageBaselineHelpersTest(unittest.TestCase):
    def write_split_inputs(self, root: Path) -> None:
        rows = [
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
            {
                "patient_id": "P2",
                "lesion_id": "L2",
                "img_id": "b.png",
                "image_path": "/data/b.png",
                "image_rel_path": "imgs_part_1/b.png",
                "diagnostic": "BCC",
                "label_idx": 1,
                "triage_priority": "high",
                "split": "val",
            },
        ]
        for split in ["train", "val", "test"]:
            pd.DataFrame(rows).to_csv(root / f"{split}.csv", index=False)

        (root / "label_mapping.json").write_text(
            json.dumps(
                {
                    "label_to_index": {label: index for index, label in enumerate(DEFAULT_LABELS)},
                    "index_to_label": {str(index): label for index, label in enumerate(DEFAULT_LABELS)},
                }
            )
        )
        (root / "class_weights.json").write_text(
            json.dumps({"class_weights": {label: float(index + 1) for index, label in enumerate(DEFAULT_LABELS)}})
        )
        (root / "preprocessing_summary.json").write_text(json.dumps({"seed": 42}))

    def test_load_split_inputs_reads_labels_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_split_inputs(root)

            inputs = load_split_inputs(root)

        self.assertEqual(inputs.labels, DEFAULT_LABELS)
        self.assertEqual(len(inputs.train), 2)
        self.assertEqual(inputs.class_weight_payload["class_weights"]["MEL"], 3.0)
        self.assertEqual(inputs.preprocessing_summary["seed"], 42)

    def test_high_risk_recall_counts_high_risk_predictions(self):
        labels = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
        high_risk_indices = high_risk_label_indices(labels)

        recall = high_risk_recall(
            targets=[0, 1, 2, 3, 4],
            predictions=[0, 1, 2, 1, 3],
            high_risk_indices=high_risk_indices,
        )

        self.assertAlmostEqual(recall, 2 / 3)

    def test_selection_score_averages_macro_f1_and_high_risk_recall(self):
        score = selection_score({"macro_f1": 0.5, "high_risk_recall": 0.9})

        self.assertAlmostEqual(score, 0.7)

    def test_sample_weights_prioritize_rare_training_classes(self):
        frame = pd.DataFrame({"diagnostic": ["ACK", "ACK", "ACK", "MEL"]})

        weights = sample_weights_for_training(frame, ["ACK", "MEL"])

        self.assertLess(weights[0], weights[-1])

    def test_high_risk_indices_accept_derm8_label_space(self):
        labels = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK", "DF", "VASC"]

        self.assertEqual(high_risk_label_indices(labels), [1, 2, 4])

    def test_validate_training_options_rejects_unknown_sampler(self):
        config = TrainingConfig(
            images_dir=Path("/images"),
            splits_dir=Path("/splits"),
            output_dir=Path("/out"),
            sampler="mystery",
        )

        with self.assertRaises(ValueError):
            validate_training_options(config)

    def test_validate_training_options_accepts_class_aware_augmentation(self):
        config = TrainingConfig(
            images_dir=Path("/images"),
            splits_dir=Path("/splits"),
            output_dir=Path("/out"),
            augment_strength="class_aware",
        )

        validate_training_options(config)

    def test_dataset_passes_diagnostic_to_label_aware_transform(self):
        class Recorder:
            requires_label = True

            def __init__(self):
                self.labels = []

            def __call__(self, image, label):
                self.labels.append(label)
                return image

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "all_images" / "imgs_part_1"
            images_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), color="white").save(images_dir / "scc.png")
            frame = pd.DataFrame(
                {
                    "image_rel_path": ["imgs_part_1/scc.png"],
                    "diagnostic": ["SCC"],
                    "label_idx": [4],
                }
            )
            transform = Recorder()
            dataset = PadUfesImageDataset(frame, root / "all_images", transform=transform)

            _, label = dataset[0]

        self.assertEqual(label, 4)
        self.assertEqual(transform.labels, ["SCC"])

    def test_dataset_keeps_normal_transform_signature(self):
        class NormalTransform:
            def __init__(self):
                self.calls = 0

            def __call__(self, image):
                self.calls += 1
                return image

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "all_images" / "imgs_part_1"
            images_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), color="white").save(images_dir / "ack.png")
            frame = pd.DataFrame(
                {
                    "image_rel_path": ["imgs_part_1/ack.png"],
                    "diagnostic": ["ACK"],
                    "label_idx": [0],
                }
            )
            transform = NormalTransform()
            dataset = PadUfesImageDataset(frame, root / "all_images", transform=transform)

            _, label = dataset[0]

        self.assertEqual(label, 0)
        self.assertEqual(transform.calls, 1)

    def test_worker_seeder_is_repeatable(self):
        seeder = WorkerSeeder(base_seed=123)

        seeder(worker_id=5)
        first = (random.random(), np.random.random())
        seeder(worker_id=5)
        second = (random.random(), np.random.random())

        self.assertEqual(first, second)

    def test_artifact_paths_use_expected_names(self):
        paths = build_artifact_paths(Path("/tmp/run"), backbone="efficientnet_b0")

        self.assertEqual(paths.best_checkpoint, Path("/tmp/run/efficientnet_b0_best.pt"))
        self.assertEqual(paths.history_csv, Path("/tmp/run/history.csv"))
        self.assertEqual(paths.test_metrics_json, Path("/tmp/run/test_metrics.json"))
        self.assertEqual(paths.classification_report_csv, Path("/tmp/run/classification_report.csv"))
        self.assertEqual(paths.confusion_matrix_csv, Path("/tmp/run/confusion_matrix.csv"))

    def test_initial_checkpoint_skips_incompatible_classifier_tensors(self):
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skipTest("PyTorch is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            model = nn.Sequential(nn.Linear(4, 2))
            compatible_weight = torch.full_like(model[0].weight, 0.25)
            incompatible_bias = torch.ones(3)
            torch.save(
                {
                    "model_state_dict": {
                        "0.weight": compatible_weight,
                        "0.bias": incompatible_bias,
                    }
                },
                checkpoint_path,
            )

            checkpoint = load_initial_checkpoint(model, checkpoint_path, "cpu")

        self.assertTrue(torch.equal(model[0].weight, compatible_weight))
        self.assertEqual(checkpoint["skipped_incompatible_keys"], ["0.bias"])


if __name__ == "__main__":
    unittest.main()
