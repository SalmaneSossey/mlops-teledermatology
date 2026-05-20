"""Prepare ISIC 2019 images as PAD-compatible external pretraining splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.data.make_image_splits import (
    LABEL_TO_INDEX,
    LABELS,
    MANIFEST_COLUMNS,
    TRIAGE_PRIORITY,
    SplitConfig,
    apply_splits,
    compute_class_weights,
    print_summary,
    split_summary,
)


ISIC_TO_PAD_LABEL = {
    "AK": "ACK",
    "AKIEC": "ACK",
    "ACTINIC KERATOSIS": "ACK",
    "BCC": "BCC",
    "BASAL CELL CARCINOMA": "BCC",
    "MEL": "MEL",
    "MELANOMA": "MEL",
    "NV": "NEV",
    "NEV": "NEV",
    "NEVUS": "NEV",
    "SCC": "SCC",
    "SQUAMOUS CELL CARCINOMA": "SCC",
    "BKL": "SEK",
    "BENIGN KERATOSIS": "SEK",
    "SEBORRHEIC KERATOSIS": "SEK",
}
UNSUPPORTED_ISIC_LABELS = {"DF", "VASC", "UNK"}
ONE_HOT_LABEL_COLUMNS = ("MEL", "NV", "BCC", "AK", "AKIEC", "BKL", "DF", "VASC", "SCC", "UNK")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class IsicPrepareConfig:
    metadata_path: Path
    images_dir: Path
    output_dir: Path
    seed: int = 42
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    image_size: int = 224
    keep_unmapped: bool = False


def normalize_isic_label(label: object) -> str | None:
    normalized = str(label).strip().upper().replace("-", "_")
    if normalized in UNSUPPORTED_ISIC_LABELS:
        return None
    return ISIC_TO_PAD_LABEL.get(normalized)


def infer_source_label(row: pd.Series) -> str:
    one_hot_columns = [column for column in ONE_HOT_LABEL_COLUMNS if column in row.index]
    active_columns = [
        column for column in one_hot_columns
        if pd.notna(row[column]) and float(row[column]) == 1.0
    ]
    if active_columns:
        if len(active_columns) > 1:
            raise ValueError(f"Multiple active ISIC labels for image row: {active_columns}")
        return active_columns[0]

    for column in ("diagnostic", "diagnosis", "dx", "label"):
        if column in row.index and pd.notna(row[column]):
            return str(row[column])

    raise ValueError(
        "Could not infer ISIC label. Expected one-hot columns such as MEL/BCC/SCC "
        "or a diagnosis/dx/label column."
    )


def infer_image_id(row: pd.Series) -> str:
    for column in ("image", "image_id", "isic_id", "img_id"):
        if column in row.index and pd.notna(row[column]):
            return str(row[column]).strip()
    raise ValueError("Could not infer image id. Expected image, image_id, isic_id, or img_id.")


def first_non_empty(row: pd.Series, columns: tuple[str, ...], fallback: str) -> str:
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value:
                return value
    return fallback


def build_image_index(images_dir: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    duplicates: list[str] = []
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        keys = {image_path.name, image_path.stem}
        for key in keys:
            if key in image_index and image_index[key] != image_path:
                duplicates.append(key)
            image_index[key] = image_path

    if duplicates:
        raise ValueError(f"Duplicate image identifiers found: {duplicates[:5]}")
    return image_index


def metadata_to_manifest(metadata: pd.DataFrame, images_dir: Path, keep_unmapped: bool = False) -> pd.DataFrame:
    image_index = build_image_index(images_dir)
    rows: list[dict[str, object]] = []
    skipped: dict[str, int] = {}

    for _, row in metadata.iterrows():
        source_label = infer_source_label(row)
        diagnostic = normalize_isic_label(source_label)
        if diagnostic is None:
            skipped[source_label] = skipped.get(source_label, 0) + 1
            if keep_unmapped:
                continue
            continue

        image_id = infer_image_id(row)
        image_path = image_index.get(image_id) or image_index.get(Path(image_id).stem)
        if image_path is None:
            raise FileNotFoundError(f"Missing image file for ISIC id {image_id!r}")

        patient_id = first_non_empty(row, ("patient_id", "lesion_id"), image_id)
        lesion_id = first_non_empty(row, ("lesion_id",), image_id)
        img_id = image_path.name
        rows.append(
            {
                "patient_id": patient_id,
                "lesion_id": lesion_id,
                "img_id": img_id,
                "image_path": str(image_path),
                "image_rel_path": image_path.resolve().relative_to(images_dir.resolve()).as_posix(),
                "diagnostic": diagnostic,
                "label_idx": LABEL_TO_INDEX[diagnostic],
                "triage_priority": TRIAGE_PRIORITY[diagnostic],
            }
        )

    if not rows:
        raise ValueError("No ISIC rows mapped to PAD-compatible labels.")

    manifest = pd.DataFrame(rows)
    manifest.attrs["skipped_source_labels"] = skipped
    return manifest


def write_external_outputs(
    split_manifest: pd.DataFrame,
    output_dir: Path,
    config: IsicPrepareConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_manifest.loc[split_manifest["split"] == split].to_csv(
            output_dir / f"{split}.csv",
            index=False,
        )

    class_weights = compute_class_weights(split_manifest[split_manifest["split"] == "train"])
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    summary = {
        "dataset": "ISIC 2019 external image pretraining",
        "source_metadata_path": str(config.metadata_path),
        "source_images_dir": str(config.images_dir),
        "label_mapping_note": "ISIC labels are mapped into the PAD-UFES-20 six-class label space.",
        "isic_to_pad_label": ISIC_TO_PAD_LABEL,
        "dropped_source_labels": sorted(UNSUPPORTED_ISIC_LABELS),
        "split_config": config_payload,
        "splits": split_summary(split_manifest),
    }
    label_mapping = {
        "label_to_index": LABEL_TO_INDEX,
        "index_to_label": {str(index): label for label, index in LABEL_TO_INDEX.items()},
        "triage_priority": TRIAGE_PRIORITY,
    }

    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2) + "\n")
    (output_dir / "class_weights.json").write_text(
        json.dumps(
            {
                "objective": "weighted_cross_entropy",
                "class_weights": class_weights,
                "train_class_counts": summary["splits"]["train"]["class_counts"],
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def prepare_isic_2019(config: IsicPrepareConfig) -> pd.DataFrame:
    metadata = pd.read_csv(config.metadata_path)
    manifest = metadata_to_manifest(metadata, config.images_dir, keep_unmapped=config.keep_unmapped)
    split_config = SplitConfig(
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        image_size=config.image_size,
    )
    split_manifest = apply_splits(manifest, split_config)
    write_external_outputs(split_manifest[MANIFEST_COLUMNS], config.output_dir, config)
    return split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_manifest = prepare_isic_2019(
        IsicPrepareConfig(
            metadata_path=args.metadata_path,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            image_size=args.image_size,
        )
    )
    print_summary(split_manifest)
    print(f"Wrote ISIC pretraining manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
