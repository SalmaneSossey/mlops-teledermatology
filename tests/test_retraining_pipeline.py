import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.training.retraining_pipeline import (
    build_augmented_training_inputs,
    evaluate_promotion_gate,
    load_feedback_cases,
    validate_feedback_cases,
)
from src.training.train_image_baseline import DEFAULT_LABELS
from src.training.train_multimodal_baseline import PadUfesMultimodalDataset


class RetrainingPipelineTest(unittest.TestCase):
    def write_base_inputs(self, root: Path) -> tuple[Path, Path]:
        splits_dir = root / "splits"
        splits_dir.mkdir()
        rows = []
        metadata_rows = []
        for index, label in enumerate(DEFAULT_LABELS):
            img_id = f"{label.lower()}.png"
            rows.append(
                {
                    "patient_id": f"P{index}",
                    "lesion_id": f"L{index}",
                    "img_id": img_id,
                    "image_path": f"/data/{img_id}",
                    "image_rel_path": f"imgs_part_1/{img_id}",
                    "diagnostic": label,
                    "label_idx": index,
                    "triage_priority": "high" if label in {"BCC", "MEL", "SCC"} else "low",
                    "split": "train",
                }
            )
            metadata_rows.append(
                {
                    "patient_id": f"P{index}",
                    "lesion_id": f"L{index}",
                    "img_id": img_id,
                    "diagnostic": label,
                    "age": 60,
                    "region": "FACE",
                    "itch": "False",
                    "grew": "False",
                    "hurt": "False",
                    "changed": "False",
                    "bleed": "False",
                    "elevation": "False",
                    "fitspatrick": 3,
                    "diameter_1": 5.0,
                    "diameter_2": 4.0,
                    "gender": "FEMALE",
                    "skin_cancer_history": "False",
                    "cancer_history": "False",
                    "smoke": "False",
                    "drink": "False",
                    "pesticide": "False",
                }
            )

        pd.DataFrame(rows).to_csv(splits_dir / "train.csv", index=False)
        pd.DataFrame(rows[:1]).assign(split="val").to_csv(splits_dir / "val.csv", index=False)
        pd.DataFrame(rows[1:2]).assign(split="test").to_csv(splits_dir / "test.csv", index=False)
        (splits_dir / "label_mapping.json").write_text(
            json.dumps(
                {
                    "label_to_index": {label: index for index, label in enumerate(DEFAULT_LABELS)},
                    "index_to_label": {
                        str(index): label for index, label in enumerate(DEFAULT_LABELS)
                    },
                }
            )
        )
        (splits_dir / "class_weights.json").write_text(
            json.dumps({"class_weights": {label: 1.0 for label in DEFAULT_LABELS}})
        )
        (splits_dir / "preprocessing_summary.json").write_text(json.dumps({"seed": 42}))
        metadata_path = root / "metadata.csv"
        pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
        return metadata_path, splits_dir

    def feedback_row(self, image_path: Path) -> dict[str, object]:
        return {
            "consultation_id": 101,
            "patient_email": "patient@example.com",
            "image_path": str(image_path),
            "original_filename": image_path.name,
            "symptoms_notes": "Reviewed case",
            "clinical_metadata": {
                "age": 51,
                "region": "ARM",
                "itch": "False",
                "grew": "True",
                "hurt": "False",
                "changed": "True",
                "bleed": "False",
                "elevation": "True",
                "fitzpatrick": 4,
            },
            "predicted_label": "ACK",
            "prediction_risk_level": "medium",
            "probabilities": {"ACK": 0.8},
            "model_run_id": "old-run",
            "final_diagnosis": "BCC",
            "triage_decision": "urgent",
            "doctor_email": "doctor@example.com",
            "review_notes": "Use for feedback.",
            "disagreement": True,
        }

    def test_load_and_validate_feedback_cases_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "feedback.png"
            Image.new("RGB", (2, 2), color="white").save(image_path)
            csv_path = root / "feedback.csv"
            row = self.feedback_row(image_path)
            csv_row = dict(row)
            csv_row["clinical_metadata"] = json.dumps(csv_row["clinical_metadata"])
            csv_row["probabilities"] = json.dumps(csv_row["probabilities"])
            pd.DataFrame([csv_row]).to_csv(csv_path, index=False)

            rows = load_feedback_cases(csv_path)
            valid_rows, report = validate_feedback_cases(rows)

        self.assertEqual(len(valid_rows), 1)
        self.assertTrue(report["ready"])
        self.assertEqual(report["label_counts"]["BCC"], 1)
        self.assertEqual(valid_rows[0]["clinical_metadata"]["region"], "ARM")

    def test_validate_feedback_cases_rejects_missing_image_and_metadata(self):
        rows = [
            {
                "consultation_id": 5,
                "image_path": "/missing/image.png",
                "final_diagnosis": "MEL",
                "clinical_metadata": {"age": 50},
            }
        ]

        valid_rows, report = validate_feedback_cases(rows)

        self.assertEqual(valid_rows, [])
        self.assertFalse(report["ready"])
        self.assertIn("image_path is missing or does not exist", report["skipped"][0]["reasons"])
        self.assertIn("missing complete metadata fields", report["skipped"][0]["reasons"][1])

    def test_build_augmented_training_inputs_adds_feedback_to_train_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path, splits_dir = self.write_base_inputs(root)
            image_path = root / "feedback.png"
            Image.new("RGB", (2, 2), color="white").save(image_path)
            valid_rows, _ = validate_feedback_cases([self.feedback_row(image_path)])

            summary = build_augmented_training_inputs(
                valid_rows,
                metadata_path=metadata_path,
                splits_dir=splits_dir,
                output_dir=root / "prepared",
            )

            train = pd.read_csv(Path(summary["splits_dir"]) / "train.csv")
            val = pd.read_csv(Path(summary["splits_dir"]) / "val.csv")
            metadata = pd.read_csv(summary["metadata_path"])

        self.assertEqual(summary["feedback_cases_added_to_train"], 1)
        self.assertEqual(len(train), len(DEFAULT_LABELS) + 1)
        self.assertEqual(len(val), 1)
        feedback_train_row = train.loc[train["img_id"].str.startswith("feedback_")].iloc[0]
        self.assertEqual(feedback_train_row["diagnostic"], "BCC")
        self.assertEqual(Path(feedback_train_row["image_path"]), image_path.resolve())
        feedback_metadata_row = metadata.loc[metadata["img_id"] == feedback_train_row["img_id"]].iloc[0]
        self.assertEqual(feedback_metadata_row["region"], "ARM")
        self.assertEqual(feedback_metadata_row["fitspatrick"], 4)

    def test_multimodal_dataset_can_load_feedback_absolute_image_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "feedback.png"
            Image.new("RGB", (2, 2), color="white").save(image_path)
            frame = pd.DataFrame(
                {
                    "image_path": [str(image_path)],
                    "image_rel_path": ["does/not/exist.png"],
                    "label_idx": [1],
                }
            )
            features = pd.DataFrame({"age__z": [0.0]})
            dataset = PadUfesMultimodalDataset(frame, features, root / "all_images")

            _, feature_tensor, label = dataset[0]

        self.assertEqual(label, 1)
        self.assertEqual(feature_tensor.tolist(), [0.0])

    def test_evaluate_promotion_gate_requires_no_metric_regression(self):
        gate = evaluate_promotion_gate(
            candidate_metrics={
                "test_macro_f1": 0.70,
                "test_high_risk_recall": 0.88,
                "test_balanced_accuracy": 0.67,
                "test_selection_score": 0.79,
            },
            current_metrics={
                "test_macro_f1": 0.69,
                "test_high_risk_recall": 0.89,
                "test_balanced_accuracy": 0.68,
                "test_selection_score": 0.79,
            },
        )

        self.assertTrue(gate["passed"])

        failed = evaluate_promotion_gate(
            candidate_metrics={
                "test_macro_f1": 0.68,
                "test_high_risk_recall": 0.80,
                "test_balanced_accuracy": 0.68,
            },
            current_metrics={
                "test_macro_f1": 0.69,
                "test_high_risk_recall": 0.89,
                "test_balanced_accuracy": 0.68,
            },
        )

        self.assertFalse(failed["passed"])
        self.assertTrue(any(check["status"] == "fail" for check in failed["checks"]))


if __name__ == "__main__":
    unittest.main()
