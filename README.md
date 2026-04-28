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

Recommended workflow:

```bash
python -m src.data.download_pad_ufes_20 \
  --repo-id SalmaneExploring/pad-ufes-20 \
  --force
```

The repo id can also be supplied with `PAD_UFES20_HF_REPO_ID`. The Hugging Face
dataset mirror is public at:

```text
https://huggingface.co/datasets/SalmaneExploring/pad-ufes-20
```

The mirror should preserve the original PAD-UFES-20 license and attribution. A
dataset card template is in:

```text
docs/huggingface_pad_ufes_20_dataset_card.md
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

## DVC Data Versioning

DVC is initialized for local dataset and split reproducibility. Git tracks DVC
metadata and pipeline hashes, while the heavy raw images and generated split
files stay out of Git.

Tracked data:

```text
data/raw/pad_ufes_20.dvc          # extracted PAD-UFES-20 metadata + images
dvc.yaml                          # split-generation pipeline
dvc.lock                          # locked dependency/output hashes
```

Recreate the split manifests from the tracked raw data:

```bash
dvc repro
```

Fresh clone workflow:

```bash
python -m src.data.download_pad_ufes_20 \
  --repo-id SalmaneExploring/pad-ufes-20 \
  --force
dvc repro
```

DVC remotes are still useful for private caches or team storage, but Hugging
Face is the preferred dataset access layer for this project. If a DVC remote is
configured later, standard DVC sync still works:

```bash
dvc pull
dvc push
```

The downloaded ZIP, raw images, generated split files, MLflow tracking
directory, and model checkpoints are intentionally not tracked directly by Git.
Hugging Face local cache metadata under `data/raw/pad_ufes_20/.cache/` is also
ignored by DVC so upload/download bookkeeping does not alter dataset hashes.

## Hugging Face Dataset Publishing

PAD-UFES-20 is already publicly available online, and the Kaggle mirror lists it
under Creative Commons Attribution 4.0 International. Before publishing a mirror,
keep the dataset card attribution and citation intact.

The public dataset repo has been created and the raw dataset has been uploaded.
To update the dataset card, raw data, or split manifests, log in and run:

```bash
export PAD_UFES20_HF_REPO_ID=SalmaneExploring/pad-ufes-20

hf auth login

hf upload "$PAD_UFES20_HF_REPO_ID" \
  docs/huggingface_pad_ufes_20_dataset_card.md README.md \
  --repo-type dataset

hf upload-large-folder "$PAD_UFES20_HF_REPO_ID" \
  data/raw/pad_ufes_20 \
  --repo-type dataset

hf upload "$PAD_UFES20_HF_REPO_ID" \
  data/processed/splits splits \
  --repo-type dataset
```

The upload step intentionally requires an authenticated user action.

After updates, verify the remote layout without downloading 3.6 GB:

```bash
hf download "$PAD_UFES20_HF_REPO_ID" --repo-type dataset --dry-run
```

## Google Colab Training

Open `notebooks/colab-image-baseline.ipynb` from GitHub in Google Colab and use
a GPU runtime. The notebook clones this repository, regenerates patient-safe
splits, trains an image-only baseline, and writes checkpoints and metrics to
Google Drive.

The completed baseline notebook used the raw PAD-UFES-20 data in Drive at:

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

For the next Colab run, prefer the Hugging Face mirror:

```bash
python -m src.data.download_pad_ufes_20 \
  --repo-id SalmaneExploring/pad-ufes-20 \
  --output-dir /content/pad_ufes_20 \
  --force

python -m src.data.make_image_splits \
  --metadata-path /content/pad_ufes_20/metadata.csv \
  --images-dir /content/pad_ufes_20/all_images
```

The notebook also installs and uses MLflow with a Drive-backed tracking URI:

```text
MyDrive/mlops-teledermatology/mlruns/
```

It logs hyperparameters, split metadata, class weights, per-epoch validation
metrics, final test metrics, reports, the best checkpoint, and a PyTorch model
artifact.

## Image Baseline Results

The first Colab EfficientNet-B0 run is summarized in:

```text
reports/image_baseline_experiment.md
```

The current test metrics are:

```text
macro_f1: 0.6636
balanced_accuracy: 0.6803
high_risk_recall: 0.8293
```

To inspect high-risk mistakes after a Colab run, execute this in Colab after
mounting Drive:

```bash
python -m src.evaluation.summarize_image_baseline \
  --run-dir /content/drive/MyDrive/mlops-teledermatology/runs/image_baseline
```

## Single-Image Inference

After training in Colab, use the saved checkpoint to score one image:

```bash
python -m src.inference.predict_image \
  --checkpoint-path /content/drive/MyDrive/mlops-teledermatology/runs/image_baseline/efficientnet_b0_best.pt \
  --image-path /content/pad_ufes_20_images/imgs_part_1/example.png
```

The checkpoint is intentionally not committed to GitHub. Keep it in Google
Drive, MLflow artifact storage, or a model registry.
