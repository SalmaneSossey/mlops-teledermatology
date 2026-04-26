"""Create patient-safe image-only manifests for PAD-UFES-20."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABELS = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
TRIAGE_PRIORITY = {
    "MEL": "high",
    "SCC": "high",
    "BCC": "high",
    "ACK": "medium",
    "NEV": "low",
    "SEK": "low",
}
MANIFEST_COLUMNS = [
    "patient_id",
    "lesion_id",
    "img_id",
    "image_path",
    "diagnostic",
    "label_idx",
    "triage_priority",
    "split",
]


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    image_size: int = 224

    @property
    def ratios(self) -> dict[str, float]:
        return {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    return current


def validate_ratios(config: SplitConfig) -> None:
    total = config.train_ratio + config.val_ratio + config.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
    if any(ratio <= 0 for ratio in config.ratios.values()):
        raise ValueError("Split ratios must be positive.")


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    required_columns = {"patient_id", "lesion_id", "diagnostic", "img_id"}
    data = pd.read_csv(metadata_path)
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Metadata is missing required columns: {missing}")

    unknown_labels = sorted(set(data["diagnostic"]) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected diagnostic labels: {unknown_labels}")

    duplicated_images = data.loc[data["img_id"].duplicated(), "img_id"].tolist()
    if duplicated_images:
        raise ValueError(f"Duplicate img_id values in metadata: {duplicated_images[:5]}")

    return data


def build_image_index(images_dir: Path) -> dict[str, Path]:
    image_paths = sorted(images_dir.glob("imgs_part_*/*.png"))
    image_index: dict[str, Path] = {}
    duplicates: list[str] = []
    for image_path in image_paths:
        if image_path.name in image_index:
            duplicates.append(image_path.name)
        image_index[image_path.name] = image_path

    if duplicates:
        raise ValueError(f"Duplicate image filenames found: {duplicates[:5]}")
    return image_index


def add_image_paths(data: pd.DataFrame, image_index: dict[str, Path]) -> pd.DataFrame:
    missing_images = sorted(set(data["img_id"]) - set(image_index))
    if missing_images:
        preview = ", ".join(missing_images[:10])
        raise FileNotFoundError(f"Missing image files for metadata rows: {preview}")

    manifest = data[["patient_id", "lesion_id", "img_id", "diagnostic"]].copy()
    manifest["image_path"] = manifest["img_id"].map(lambda img_id: str(image_index[img_id]))
    manifest["label_idx"] = manifest["diagnostic"].map(LABEL_TO_INDEX)
    manifest["triage_priority"] = manifest["diagnostic"].map(TRIAGE_PRIORITY)
    return manifest


def make_patient_split_assignments(
    manifest: pd.DataFrame,
    config: SplitConfig,
    labels: Iterable[str] = LABELS,
) -> dict[str, str]:
    """Greedily assign patient groups while balancing per-class image counts."""

    validate_ratios(config)
    labels = list(labels)
    rng = np.random.default_rng(config.seed)
    split_names = list(config.ratios)

    total_class_counts = manifest["diagnostic"].value_counts().reindex(labels, fill_value=0)
    target_class_counts = {
        split: total_class_counts.to_numpy(dtype=float) * ratio
        for split, ratio in config.ratios.items()
    }
    total_rows = len(manifest)
    target_rows = {split: total_rows * ratio for split, ratio in config.ratios.items()}

    patient_groups = []
    for patient_id, group in manifest.groupby("patient_id", sort=True):
        class_counts = group["diagnostic"].value_counts().reindex(labels, fill_value=0)
        rarity_score = float(
            sum(
                class_counts[label] / max(total_class_counts[label], 1)
                for label in labels
            )
        )
        patient_groups.append(
            {
                "patient_id": patient_id,
                "row_count": len(group),
                "class_counts": class_counts.to_numpy(dtype=float),
                "rarity_score": rarity_score,
                "tie_breaker": float(rng.random()),
            }
        )

    patient_groups.sort(
        key=lambda item: (
            -item["rarity_score"],
            -item["row_count"],
            item["tie_breaker"],
            item["patient_id"],
        )
    )

    split_class_counts = {split: np.zeros(len(labels), dtype=float) for split in split_names}
    split_row_counts = {split: 0 for split in split_names}
    assignments: dict[str, str] = {}

    # Seed each split with every class when enough patient groups exist. This is
    # especially important for rare but clinically important classes like MEL.
    for label_index, label in enumerate(labels):
        eligible_groups = [
            group
            for group in patient_groups
            if group["patient_id"] not in assignments and group["class_counts"][label_index] > 0
        ]
        if len(eligible_groups) < len(split_names):
            continue

        eligible_groups.sort(
            key=lambda item: (
                -item["class_counts"][label_index],
                -item["rarity_score"],
                item["tie_breaker"],
                item["patient_id"],
            )
        )
        for split, group in zip(split_names, eligible_groups):
            assignments[group["patient_id"]] = split
            split_class_counts[split] += group["class_counts"]
            split_row_counts[split] += group["row_count"]

    for group in patient_groups:
        if group["patient_id"] in assignments:
            continue

        scores = []
        for split in split_names:
            total_error = 0.0
            rows_after = split_row_counts[split] + group["row_count"]
            for candidate_split in split_names:
                class_after = split_class_counts[candidate_split]
                row_count_after = split_row_counts[candidate_split]
                if candidate_split == split:
                    class_after = class_after + group["class_counts"]
                    row_count_after = row_count_after + group["row_count"]

                class_error = np.sum(
                    ((class_after - target_class_counts[candidate_split]) ** 2)
                    / np.maximum(target_class_counts[candidate_split], 1.0)
                )
                row_error = (
                    (row_count_after - target_rows[candidate_split]) ** 2
                ) / max(target_rows[candidate_split], 1.0)
                total_error += class_error + 0.25 * row_error
            scores.append((total_error, rows_after, split))

        _, _, chosen_split = min(scores, key=lambda item: (item[0], item[1], item[2]))
        assignments[group["patient_id"]] = chosen_split
        split_class_counts[chosen_split] += group["class_counts"]
        split_row_counts[chosen_split] += group["row_count"]

    return assignments


def apply_splits(manifest: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    assignments = make_patient_split_assignments(manifest, config)
    split_manifest = manifest.copy()
    split_manifest["split"] = split_manifest["patient_id"].map(assignments)
    return split_manifest[MANIFEST_COLUMNS].sort_values(["split", "patient_id", "img_id"])


def compute_class_weights(train_manifest: pd.DataFrame) -> dict[str, float]:
    counts = train_manifest["diagnostic"].value_counts().reindex(LABELS, fill_value=0)
    if (counts == 0).any():
        missing = counts[counts == 0].index.tolist()
        raise ValueError(f"Training split has no examples for labels: {missing}")

    total = int(counts.sum())
    weights = total / (len(LABELS) * counts)
    return {label: round(float(weights[label]), 6) for label in LABELS}


def validate_split_manifest(split_manifest: pd.DataFrame) -> None:
    patient_split_counts = split_manifest.groupby("patient_id")["split"].nunique()
    leaking_patients = patient_split_counts[patient_split_counts > 1]
    if not leaking_patients.empty:
        raise ValueError(f"Patients found in multiple splits: {leaking_patients.index[:5].tolist()}")

    missing_split = split_manifest["split"].isna().sum()
    if missing_split:
        raise ValueError(f"{missing_split} rows do not have a split assignment.")

    for split in ["train", "val", "test"]:
        labels_in_split = set(split_manifest.loc[split_manifest["split"] == split, "diagnostic"])
        missing_labels = sorted(set(LABELS) - labels_in_split)
        if missing_labels:
            raise ValueError(f"{split} split is missing labels: {missing_labels}")


def split_summary(split_manifest: pd.DataFrame) -> dict[str, object]:
    distributions = {}
    for split, group in split_manifest.groupby("split"):
        distributions[split] = {
            "images": int(len(group)),
            "patients": int(group["patient_id"].nunique()),
            "class_counts": {
                label: int(count)
                for label, count in group["diagnostic"]
                .value_counts()
                .reindex(LABELS, fill_value=0)
                .items()
            },
            "triage_priority_counts": {
                label: int(count)
                for label, count in group["triage_priority"].value_counts().sort_index().items()
            },
        }
    return distributions


def write_outputs(split_manifest: pd.DataFrame, output_dir: Path, config: SplitConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_manifest.loc[split_manifest["split"] == split].to_csv(
            output_dir / f"{split}.csv", index=False
        )

    class_weights = compute_class_weights(split_manifest[split_manifest["split"] == "train"])
    label_mapping = {
        "label_to_index": LABEL_TO_INDEX,
        "index_to_label": {str(index): label for label, index in LABEL_TO_INDEX.items()},
        "triage_priority": TRIAGE_PRIORITY,
    }
    summary = {
        "split_ratio": config.ratios,
        "seed": config.seed,
        "image_transform_defaults": {
            "mode": "RGB",
            "resize": [config.image_size, config.image_size],
            "normalize_pixel_values": True,
            "augmentation": "train split only",
        },
        "splits": split_summary(split_manifest),
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


def print_summary(split_manifest: pd.DataFrame) -> None:
    print("Split distribution by diagnosis:")
    print(pd.crosstab(split_manifest["diagnostic"], split_manifest["split"]).reindex(LABELS))
    print()
    print("Images per split:")
    print(split_manifest["split"].value_counts().reindex(["train", "val", "test"]))
    print()
    print("Patients per split:")
    print(split_manifest.groupby("split")["patient_id"].nunique().reindex(["train", "val", "test"]))


def parse_args() -> argparse.Namespace:
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=project_root / "data/raw/pad_ufes_20/metadata.csv",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=project_root / "data/raw/pad_ufes_20/all_images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/processed/splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        image_size=args.image_size,
    )
    data = load_metadata(args.metadata_path)
    image_index = build_image_index(args.images_dir)
    manifest = add_image_paths(data, image_index)
    split_manifest = apply_splits(manifest, config)
    validate_split_manifest(split_manifest)
    write_outputs(split_manifest, args.output_dir, config)
    print_summary(split_manifest)
    print()
    print(f"Wrote split manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
