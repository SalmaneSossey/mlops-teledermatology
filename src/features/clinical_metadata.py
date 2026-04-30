"""Encode PAD-UFES-20 clinical metadata for tabular and multimodal baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


JOIN_KEY = "img_id"
MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER__"

COMPLETE_NUMERIC_FIELDS = ("age",)
COMPLETE_CATEGORICAL_FIELDS = (
    "region",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
)
OPTIONAL_NUMERIC_FIELDS = ("fitspatrick", "diameter_1", "diameter_2")
OPTIONAL_CATEGORICAL_FIELDS = (
    "gender",
    "skin_cancer_history",
    "cancer_history",
    "smoke",
    "drink",
    "pesticide",
)
LEAKAGE_FIELDS = ("biopsed", "diagnostic")


def clinical_feature_fields(include_optional: bool = True) -> tuple[str, ...]:
    fields: list[str] = [
        *COMPLETE_NUMERIC_FIELDS,
        *COMPLETE_CATEGORICAL_FIELDS,
    ]
    if include_optional:
        fields.extend(OPTIONAL_NUMERIC_FIELDS)
        fields.extend(OPTIONAL_CATEGORICAL_FIELDS)
    return tuple(fields)


def read_metadata(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path)


def merge_clinical_metadata(
    split_frame: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    include_optional: bool = True,
) -> pd.DataFrame:
    if JOIN_KEY not in split_frame.columns:
        raise ValueError(f"Split frame must include {JOIN_KEY!r}")
    if JOIN_KEY not in metadata_frame.columns:
        raise ValueError(f"Metadata frame must include {JOIN_KEY!r}")
    if metadata_frame[JOIN_KEY].duplicated().any():
        duplicates = metadata_frame.loc[metadata_frame[JOIN_KEY].duplicated(), JOIN_KEY].head().tolist()
        raise ValueError(f"Metadata contains duplicate {JOIN_KEY} values: {duplicates}")

    fields = clinical_feature_fields(include_optional=include_optional)
    missing = sorted(set(fields) - set(metadata_frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing clinical feature columns: {missing}")

    metadata_columns = [JOIN_KEY, *fields]
    merged = split_frame.merge(
        metadata_frame[metadata_columns],
        on=JOIN_KEY,
        how="left",
        validate="one_to_one",
    )
    missing_rows = merged[list(fields)].isna().all(axis=1).sum()
    if missing_rows:
        raise ValueError(f"{missing_rows} split rows did not match metadata by {JOIN_KEY}")
    return merged


def normalize_category_values(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().fillna(MISSING_TOKEN)


@dataclass(frozen=True)
class ClinicalMetadataEncoder:
    numeric_fields: tuple[str, ...]
    optional_numeric_fields: tuple[str, ...]
    categorical_levels: dict[str, tuple[str, ...]]
    optional_categorical_fields: tuple[str, ...]
    means: dict[str, float]
    stds: dict[str, float]

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for field in self.numeric_fields:
            names.append(f"{field}__z")
            if field in self.optional_numeric_fields:
                names.append(f"{field}__missing")
        for field, levels in self.categorical_levels.items():
            if field in self.optional_categorical_fields:
                names.append(f"{field}__missing")
            names.extend(f"{field}__{level}" for level in levels)
            names.append(f"{field}__{OTHER_TOKEN}")
        return names

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing_columns = sorted(
            (set(self.numeric_fields) | set(self.categorical_levels)) - set(frame.columns)
        )
        if missing_columns:
            raise ValueError(f"Frame is missing clinical feature columns: {missing_columns}")

        encoded = pd.DataFrame(index=frame.index)
        for field in self.numeric_fields:
            values = pd.to_numeric(frame[field], errors="coerce")
            if field in self.optional_numeric_fields:
                encoded[f"{field}__missing"] = values.isna().astype(float)
            filled = values.fillna(self.means[field])
            encoded[f"{field}__z"] = (filled - self.means[field]) / self.stds[field]

        for field, levels in self.categorical_levels.items():
            values = normalize_category_values(frame[field])
            if field in self.optional_categorical_fields:
                encoded[f"{field}__missing"] = (values == MISSING_TOKEN).astype(float)
            for level in levels:
                encoded[f"{field}__{level}"] = (values == level).astype(float)
            known = set(levels)
            if field in self.optional_categorical_fields:
                known.add(MISSING_TOKEN)
            encoded[f"{field}__{OTHER_TOKEN}"] = (~values.isin(known)).astype(float)

        return encoded[self.feature_names]


def fit_clinical_metadata_encoder(
    frame: pd.DataFrame,
    include_optional: bool = True,
) -> ClinicalMetadataEncoder:
    numeric_fields = list(COMPLETE_NUMERIC_FIELDS)
    optional_numeric_fields: list[str] = []
    categorical_fields = list(COMPLETE_CATEGORICAL_FIELDS)
    optional_categorical_fields: list[str] = []

    if include_optional:
        numeric_fields.extend(OPTIONAL_NUMERIC_FIELDS)
        optional_numeric_fields.extend(OPTIONAL_NUMERIC_FIELDS)
        categorical_fields.extend(OPTIONAL_CATEGORICAL_FIELDS)
        optional_categorical_fields.extend(OPTIONAL_CATEGORICAL_FIELDS)

    missing_columns = sorted((set(numeric_fields) | set(categorical_fields)) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Frame is missing clinical feature columns: {missing_columns}")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for field in numeric_fields:
        values = pd.to_numeric(frame[field], errors="coerce")
        if field not in optional_numeric_fields and values.isna().any():
            raise ValueError(f"Complete numeric field {field!r} contains missing values")
        mean = float(values.mean()) if not values.dropna().empty else 0.0
        std = float(values.std(ddof=0)) if not values.dropna().empty else 1.0
        means[field] = mean
        stds[field] = std if std > 0 else 1.0

    categorical_levels = {}
    for field in categorical_fields:
        values = normalize_category_values(frame[field])
        if field not in optional_categorical_fields and (values == MISSING_TOKEN).any():
            raise ValueError(f"Complete categorical field {field!r} contains missing values")
        levels = sorted(value for value in values.unique().tolist() if value != MISSING_TOKEN)
        categorical_levels[field] = tuple(levels)

    return ClinicalMetadataEncoder(
        numeric_fields=tuple(numeric_fields),
        optional_numeric_fields=tuple(optional_numeric_fields),
        categorical_levels=categorical_levels,
        optional_categorical_fields=tuple(optional_categorical_fields),
        means=means,
        stds=stds,
    )


def encode_clinical_metadata(
    train_frame: pd.DataFrame,
    frames: Sequence[pd.DataFrame],
    include_optional: bool = True,
) -> tuple[ClinicalMetadataEncoder, list[pd.DataFrame]]:
    encoder = fit_clinical_metadata_encoder(train_frame, include_optional=include_optional)
    return encoder, [encoder.transform(frame) for frame in frames]


def metadata_missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    fields = clinical_feature_fields(include_optional=True)
    missing = []
    for field in fields:
        if field not in frame.columns:
            missing.append({"feature": field, "missing_rate": np.nan, "present": False})
        else:
            missing.append(
                {
                    "feature": field,
                    "missing_rate": float(frame[field].isna().mean()),
                    "present": True,
                }
            )
    return pd.DataFrame(missing)
