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
- `notebooks/colab-image-baseline.ipynb`: Google Colab notebook for training
  an image-only EfficientNet baseline on GPU.

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

Each split manifest includes both `image_path` for local runs and
`image_rel_path` for portable runs in environments such as Google Colab.

## Google Colab Training

Open `notebooks/colab-image-baseline.ipynb` from GitHub in Google Colab and use
a GPU runtime. The notebook clones this repository, regenerates patient-safe
splits, trains an image-only baseline, and writes checkpoints and metrics to
Google Drive.

The notebook expects the raw PAD-UFES-20 data in Drive at:

```text
MyDrive/teledermatology-data/zr7vgbcyr2-1/
```

with:

```text
metadata.csv
images/imgs_part_*.zip
```

The Colab notebook extracts the image ZIP files to the Colab runtime disk before
training.
