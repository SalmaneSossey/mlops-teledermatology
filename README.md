# MLOps Teledermatology

Exploratory analysis and preprocessing work for a teledermatology image-classification pipeline using the PAD-UFES-20 dataset.

## Current State

- EDA notebook for PAD-UFES-20 metadata and images
- Class balance, biopsy-rate, missingness, patient/lesion, and image-quality checks
- Generated figures under `figures/`

## Data

Raw data is not committed to this repository. Place the PAD-UFES-20 files under:

```text
data/raw/pad_ufes_20/
```

Expected local files include:

```text
data/raw/pad_ufes_20/metadata.csv
data/raw/pad_ufes_20/all_images/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

- `notebooks/pad-ufes-20-analysis.ipynb`: exploratory data analysis before preprocessing.

## Preprocessing

Create image-only, patient-safe split manifests:

```bash
python -m src.data.make_image_splits
```

The command verifies image files, keeps all images from each patient in one split,
and writes local manifests under:

```text
data/processed/splits/
```

Generated files include `train.csv`, `val.csv`, `test.csv`,
`label_mapping.json`, `class_weights.json`, and `preprocessing_summary.json`.
The generated processed data is intentionally ignored by git.
