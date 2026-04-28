import tempfile
import unittest
import zipfile
from pathlib import Path

from src.data.download_pad_ufes_20 import (
    find_dataset_layout,
    has_image_parts,
    materialize_dataset,
)


class DownloadPadUfes20Test(unittest.TestCase):
    def test_finds_root_layout_with_all_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir)
            images_dir = snapshot / "all_images" / "imgs_part_1"
            images_dir.mkdir(parents=True)
            (snapshot / "metadata.csv").write_text("img_id,diagnostic\n")
            (images_dir / "PAT_1_1_1.png").write_bytes(b"fake")

            layout = find_dataset_layout(snapshot)

        self.assertEqual(layout.root_dir, snapshot)
        self.assertEqual(layout.metadata_path, snapshot / "metadata.csv")
        self.assertEqual(layout.images_dir, snapshot / "all_images")

    def test_finds_nested_layout_with_images_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir)
            root = snapshot / "nested" / "pad_ufes_20"
            images_dir = root / "images" / "imgs_part_2"
            images_dir.mkdir(parents=True)
            (root / "metadata.csv").write_text("img_id,diagnostic\n")
            (images_dir / "PAT_2_2_2.png").write_bytes(b"fake")

            layout = find_dataset_layout(snapshot)

        self.assertEqual(layout.root_dir, root)
        self.assertEqual(layout.images_dir, root / "images")

    def test_materializes_zip_layout_to_expected_all_images_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "snapshot"
            images_dir = root / "images"
            images_dir.mkdir(parents=True)
            (root / "metadata.csv").write_text("img_id,diagnostic\n")
            with zipfile.ZipFile(images_dir / "imgs_part_1.zip", "w") as archive:
                archive.writestr("imgs_part_1/PAT_3_3_3.png", b"fake")

            layout = find_dataset_layout(root)
            output_dir = Path(tmpdir) / "output"
            materialize_dataset(layout, output_dir)

            self.assertTrue((output_dir / "metadata.csv").exists())
            self.assertTrue(has_image_parts(output_dir / "all_images"))
            self.assertTrue((output_dir / "all_images" / "imgs_part_1" / "PAT_3_3_3.png").exists())


if __name__ == "__main__":
    unittest.main()
