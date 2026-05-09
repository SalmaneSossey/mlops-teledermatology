# Colab Hyperparameter Sweep Runbook

Use this runbook only after the image-only baseline and GitHub Actions checks are
green. Keep GPU runs small, MLflow-tracked, and backed up to Google Drive. These
steps are also integrated into `notebooks/colab-image-baseline.ipynb` behind the
`RUN_HPARAM_SWEEP` and `RUN_MULTIMODAL_BASELINE` flags.

## Preconditions

- Use a Colab GPU runtime.
- Mount Google Drive before writing run artifacts.
- Set `DAGSHUB_TOKEN` in Colab Secrets or the environment.
- Keep `DAGSHUB_REPO_OWNER` as `SalmaneSossey`, or override it if the DagsHub
  repo is under a different owner.
- Optionally set `DAGSHUB_USERNAME` when using username/password-style MLflow
  auth.

Check credentials before starting a long run:

```bash
python - <<'PY'
import os

required = ["DAGSHUB_TOKEN"]
missing = [name for name in required if not os.environ.get(name)]
if missing:
    raise SystemExit(f"Missing required DagsHub environment values: {missing}")

print("DagsHub token is present.")
print("Tracking URI:", os.environ.get("DAGSHUB_MLFLOW_TRACKING_URI", "repo defaults"))
PY
```

## Data Setup

```bash
python -m src.data.download_pad_ufes_20 \
  --repo-id SalmaneExploring/pad-ufes-20 \
  --output-dir /content/pad_ufes_20 \
  --force

python -m src.data.make_image_splits \
  --metadata-path /content/pad_ufes_20/metadata.csv \
  --images-dir /content/pad_ufes_20/all_images \
  --output-dir data/processed/splits
```

## Sweep

Preview the trial list first:

```bash
python -m src.training.run_hparam_sweep \
  --images-dir /content/pad_ufes_20/all_images \
  --splits-dir data/processed/splits \
  --output-dir /content/drive/MyDrive/mlops-teledermatology/runs/image_baseline \
  --max-trials 8 \
  --dry-run
```

Run the bounded sweep:

```bash
python -m src.training.run_hparam_sweep \
  --images-dir /content/pad_ufes_20/all_images \
  --splits-dir data/processed/splits \
  --output-dir /content/drive/MyDrive/mlops-teledermatology/runs/image_baseline \
  --max-trials 8
```

Only add `--retrain-best` after reviewing `hparam_sweep_results.csv` and deciding
the extra GPU run is worth it.

## First Multimodal Baseline

After comparing the image sweep with the metadata-only baseline, run the first
late-fusion image plus metadata baseline on the same splits:

```bash
python -m src.training.train_multimodal_baseline \
  --images-dir /content/pad_ufes_20/all_images \
  --metadata-path /content/pad_ufes_20/metadata.csv \
  --splits-dir data/processed/splits \
  --output-dir /content/drive/MyDrive/mlops-teledermatology/runs/multimodal_baseline
```

Compare `test_macro_f1`, `test_balanced_accuracy`, `test_high_risk_recall`, and
`test_selection_score` against the image-only and metadata-only reports.
