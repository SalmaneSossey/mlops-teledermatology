import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.prepare_isic_2019 import (
    build_image_index,
    first_non_empty,
    infer_source_label,
    metadata_to_manifest,
    normalize_isic_label,
)


class PrepareIsic2019Test(unittest.TestCase):
    def test_normalize_isic_labels_to_pad_space(self):
        self.assertEqual(normalize_isic_label("AK"), "ACK")
        self.assertEqual(normalize_isic_label("AKIEC"), "ACK")
        self.assertEqual(normalize_isic_label("BCC"), "BCC")
        self.assertEqual(normalize_isic_label("MEL"), "MEL")
        self.assertEqual(normalize_isic_label("NV"), "NEV")
        self.assertEqual(normalize_isic_label("SCC"), "SCC")
        self.assertEqual(normalize_isic_label("BKL"), "SEK")
        self.assertIsNone(normalize_isic_label("DF"))

    def test_infer_source_label_from_one_hot_columns(self):
        row = pd.Series({"MEL": 0.0, "BCC": 1.0, "SCC": 0.0})

        self.assertEqual(infer_source_label(row), "BCC")

    def test_build_manifest_maps_images_and_drops_unmapped_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir()
            for name in ["ISIC_0001.jpg", "ISIC_0002.jpg", "ISIC_0003.jpg"]:
                Image.new("RGB", (4, 4), color="white").save(images_dir / name)

            metadata = pd.DataFrame(
                [
                    {"image": "ISIC_0001", "MEL": 1, "NV": 0, "BCC": 0, "DF": 0},
                    {"image": "ISIC_0002", "MEL": 0, "NV": 0, "BCC": 1, "DF": 0},
                    {"image": "ISIC_0003", "MEL": 0, "NV": 0, "BCC": 0, "DF": 1},
                ]
            )

            manifest = metadata_to_manifest(metadata, images_dir)

        self.assertEqual(manifest["diagnostic"].tolist(), ["MEL", "BCC"])
        self.assertEqual(manifest["image_rel_path"].tolist(), ["ISIC_0001.jpg", "ISIC_0002.jpg"])
        self.assertEqual(manifest["triage_priority"].tolist(), ["high", "high"])

    def test_build_image_index_allows_lookup_by_stem_and_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir)
            Image.new("RGB", (4, 4), color="white").save(images_dir / "ISIC_0001.jpg")

            index = build_image_index(images_dir)

        self.assertIn("ISIC_0001", index)
        self.assertIn("ISIC_0001.jpg", index)

    def test_first_non_empty_skips_blank_and_nan_values(self):
        row = pd.Series({"patient_id": float("nan"), "lesion_id": "  "})

        self.assertEqual(first_non_empty(row, ("patient_id", "lesion_id"), "ISIC_0001"), "ISIC_0001")


if __name__ == "__main__":
    unittest.main()
