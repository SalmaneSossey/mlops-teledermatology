import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.train_image_baseline import (
    DEFAULT_LABELS,
    WorkerSeeder,
    build_artifact_paths,
    high_risk_label_indices,
    high_risk_recall,
    load_split_inputs,
    selection_score,
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


if __name__ == "__main__":
    unittest.main()
