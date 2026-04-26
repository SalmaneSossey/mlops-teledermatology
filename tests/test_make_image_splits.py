import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.make_image_splits import (
    LABELS,
    LABEL_TO_INDEX,
    SplitConfig,
    add_image_paths,
    apply_splits,
    build_image_index,
    compute_class_weights,
    validate_split_manifest,
)


class MakeImageSplitsTest(unittest.TestCase):
    def make_dataset(self, root: Path) -> pd.DataFrame:
        images_dir = root / "all_images" / "imgs_part_1"
        images_dir.mkdir(parents=True)
        rows = []

        for label in LABELS:
            for patient_number in range(6):
                patient_id = f"PAT_{label}_{patient_number}"
                for image_number in range(2):
                    img_id = f"{patient_id}_{image_number}.png"
                    (images_dir / img_id).touch()
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "lesion_id": patient_number,
                            "diagnostic": label,
                            "img_id": img_id,
                        }
                    )

        return pd.DataFrame(rows)

    def test_patient_safe_deterministic_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = self.make_dataset(root)
            image_index = build_image_index(root / "all_images")
            manifest = add_image_paths(metadata, image_index)
            config = SplitConfig(seed=123)

            first = apply_splits(manifest, config)
            second = apply_splits(manifest, config)

            pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
            validate_split_manifest(first)

            patient_split_counts = first.groupby("patient_id")["split"].nunique()
            self.assertEqual(patient_split_counts.max(), 1)

    def test_derived_fields_are_added_without_clinical_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = self.make_dataset(root)
            metadata["biopsed"] = True
            metadata["age"] = 50

            image_index = build_image_index(root / "all_images")
            manifest = add_image_paths(metadata, image_index)

            self.assertIn("image_path", manifest.columns)
            self.assertIn("label_idx", manifest.columns)
            self.assertIn("triage_priority", manifest.columns)
            self.assertNotIn("biopsed", manifest.columns)
            self.assertNotIn("age", manifest.columns)
            self.assertEqual(manifest.loc[manifest["diagnostic"] == "ACK", "label_idx"].iloc[0], 0)
            self.assertEqual(manifest.loc[manifest["diagnostic"] == "MEL", "triage_priority"].iloc[0], "high")

    def test_missing_images_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = self.make_dataset(root)
            missing_img = metadata.loc[0, "img_id"]
            (root / "all_images" / "imgs_part_1" / missing_img).unlink()

            image_index = build_image_index(root / "all_images")

            with self.assertRaises(FileNotFoundError):
                add_image_paths(metadata, image_index)

    def test_class_weights_use_training_distribution(self):
        train_manifest = pd.DataFrame(
            {
                "diagnostic": ["ACK", "ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
            }
        )

        weights = compute_class_weights(train_manifest)

        self.assertEqual(set(weights), set(LABEL_TO_INDEX))
        self.assertLess(weights["ACK"], weights["MEL"])


if __name__ == "__main__":
    unittest.main()
