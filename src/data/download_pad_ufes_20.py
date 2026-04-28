"""Download PAD-UFES-20 from a Hugging Face dataset repo.

The project split pipeline expects this local layout:

data/raw/pad_ufes_20/
  metadata.csv
  all_images/
    imgs_part_1/*.png
    imgs_part_2/*.png
    imgs_part_3/*.png
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO_ID = "SalmaneExploring/pad-ufes-20"
DEFAULT_OUTPUT_DIR = Path("data/raw/pad_ufes_20")


@dataclass(frozen=True)
class DatasetLayout:
    root_dir: Path
    metadata_path: Path
    images_dir: Path


def has_image_parts(path: Path) -> bool:
    return any(path.glob("imgs_part_*/*.png"))


def image_zip_paths(path: Path) -> list[Path]:
    return sorted(path.glob("imgs_part_*.zip"))


def find_images_dir(dataset_root: Path) -> Path | None:
    for candidate in [
        dataset_root / "all_images",
        dataset_root / "images",
        dataset_root,
    ]:
        if candidate.exists() and (has_image_parts(candidate) or image_zip_paths(candidate)):
            return candidate

    for part_dir in sorted(dataset_root.rglob("imgs_part_*")):
        if part_dir.is_dir() and any(part_dir.glob("*.png")):
            return part_dir.parent

    return None


def find_dataset_layout(snapshot_dir: Path) -> DatasetLayout:
    metadata_paths = sorted(snapshot_dir.rglob("metadata.csv"))
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata.csv found under {snapshot_dir}")

    for metadata_path in metadata_paths:
        root_dir = metadata_path.parent
        images_dir = find_images_dir(root_dir)
        if images_dir is not None:
            return DatasetLayout(root_dir=root_dir, metadata_path=metadata_path, images_dir=images_dir)

    metadata_preview = ", ".join(str(path) for path in metadata_paths[:5])
    raise FileNotFoundError(
        "Found metadata.csv but no PAD-UFES-20 image folders or ZIP files. "
        f"Checked metadata candidates: {metadata_preview}"
    )


def reset_output_dir(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        raise FileExistsError(
            f"{output_dir} already exists. Pass --force to replace it, "
            "or choose a different --output-dir."
        )
    shutil.rmtree(output_dir)


def copy_metadata(metadata_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_path, output_dir / "metadata.csv")


def copy_image_tree(images_dir: Path, output_images_dir: Path) -> None:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    for part_dir in sorted(images_dir.glob("imgs_part_*")):
        if part_dir.is_dir():
            shutil.copytree(part_dir, output_images_dir / part_dir.name, dirs_exist_ok=True)


def extract_image_zips(images_dir: Path, output_images_dir: Path) -> None:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in image_zip_paths(images_dir):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(output_images_dir)


def materialize_dataset(layout: DatasetLayout, output_dir: Path, force: bool = False) -> None:
    reset_output_dir(output_dir, force=force)
    output_images_dir = output_dir / "all_images"

    copy_metadata(layout.metadata_path, output_dir)
    copy_image_tree(layout.images_dir, output_images_dir)
    extract_image_zips(layout.images_dir, output_images_dir)

    if not has_image_parts(output_images_dir):
        raise FileNotFoundError(
            f"No imgs_part_*/*.png files were materialized under {output_images_dir}"
        )


def download_snapshot(repo_id: str, revision: str | None, cache_dir: Path | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "huggingface_hub is required. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
    )
    return Path(snapshot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("PAD_UFES20_HF_REPO_ID", DEFAULT_REPO_ID),
        help=(
            "Hugging Face dataset repo id. Defaults to PAD_UFES20_HF_REPO_ID "
            f"or {DEFAULT_REPO_ID}."
        ),
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("PAD_UFES20_HF_REVISION"),
        help="Optional Hugging Face git revision, tag, or commit SHA.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local PAD-UFES-20 output directory expected by the split pipeline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace output-dir if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = download_snapshot(args.repo_id, args.revision, args.cache_dir)
    layout = find_dataset_layout(snapshot_dir)
    materialize_dataset(layout, args.output_dir, force=args.force)
    print(f"Downloaded {args.repo_id} to {args.output_dir}")
    print(f"Metadata: {args.output_dir / 'metadata.csv'}")
    print(f"Images: {args.output_dir / 'all_images'}")


if __name__ == "__main__":
    main()
