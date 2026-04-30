# PAD-UFES-20 Image Baseline Experiment

## Summary

This run trained an image-only EfficientNet-B0 baseline in Google Colab using a
GPU runtime, patient-safe PAD-UFES-20 splits, and DagsHub-hosted MLflow tracking.

- Notebook launcher: `notebooks/colab-image-baseline.ipynb`
- Training CLI: `python -m src.training.train_image_baseline`
- DagsHub run: `4f185f85c76e497f94e429a568f03a04`
- Run URL: `https://dagshub.com/SalmaneSossey/mlops-teledermatology.mlflow/#/experiments/0/runs/4f185f85c76e497f94e429a568f03a04`
- Model: EfficientNet-B0 with ImageNet pretrained weights
- Task: six-class lesion diagnosis classification
- Labels: `ACK`, `BCC`, `MEL`, `NEV`, `SCC`, `SEK`
- High-risk labels for triage: `BCC`, `MEL`, `SCC`
- Split strategy: patient-safe train/validation/test split
- Epochs: 8
- Batch size: 32
- Image size: 224 x 224
- Loss: weighted cross entropy
- Optimizer: AdamW
- Tracking: DagsHub MLflow, with Google Drive checkpoint/report backups

## Data And Artifacts

The current Colab workflow uses the public Hugging Face dataset mirror:

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

The run artifacts are saved in Google Drive and logged to MLflow:

```text
MyDrive/mlops-teledermatology/runs/image_baseline/
  history.csv
  test_metrics.json
  classification_report.csv
  confusion_matrix.csv
  efficientnet_b0_best.pt
```

Do not commit `efficientnet_b0_best.pt` to GitHub. Keep model checkpoints in
Drive, MLflow artifact storage, or a model registry.

## Split

The run regenerated the patient-safe split manifests before training.

| Split | Images | Patients |
| --- | ---: | ---: |
| Train | 1608 | 959 |
| Validation | 346 | 208 |
| Test | 344 | 206 |

Class counts by split:

| Class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| ACK | 511 | 110 | 109 |
| BCC | 591 | 127 | 127 |
| MEL | 36 | 8 | 8 |
| NEV | 171 | 37 | 36 |
| SCC | 134 | 29 | 29 |
| SEK | 165 | 35 | 35 |

## Test Metrics

| Metric | Value |
| --- | ---: |
| Macro F1 | 0.6597 |
| Balanced accuracy | 0.6746 |
| High-risk recall | 0.8232 |

For the triage use case, high-risk recall is the most important baseline metric:
the model caught about 82% of `BCC`, `MEL`, and `SCC` test examples as high-risk.

## Validation History

Best validation selection score occurred at epoch 7, based on the combined
validation macro-F1 and high-risk recall criterion.

| Epoch | Val macro F1 | Val balanced accuracy | Val high-risk recall |
| ---: | ---: | ---: | ---: |
| 1 | 0.5194 | 0.5569 | 0.8598 |
| 2 | 0.4900 | 0.5770 | 0.7683 |
| 3 | 0.5358 | 0.6049 | 0.7866 |
| 4 | 0.5977 | 0.6104 | 0.8171 |
| 5 | 0.5902 | 0.6083 | 0.8293 |
| 6 | 0.5835 | 0.6160 | 0.7744 |
| 7 | 0.5924 | 0.6148 | 0.8354 |
| 8 | 0.5808 | 0.6086 | 0.8049 |

## High-Risk Error Review

The latest per-class report and confusion matrix are logged as DagsHub MLflow
artifacts for run `4f185f85c76e497f94e429a568f03a04` and backed up in Drive.
After downloading or producing a run directory, inspect high-risk mistakes with:

```bash
python -m src.evaluation.summarize_image_baseline \
  --run-dir /content/drive/MyDrive/mlops-teledermatology/runs/image_baseline
```

## Known Limitations

- The rare `MEL` class has only 36 train images and 8 test images, so per-class
  metrics are likely unstable.
- This is image-only. It does not use clinical metadata such as age, lesion
  location, or symptoms.
- The checkpoint was trained in Colab and stored in Drive/MLflow; local
  inference needs the checkpoint path passed explicitly.
- This is a baseline model, not a clinical diagnostic system.

## Recommended Next Experiments

1. Calibrate high-risk triage probabilities and evaluate threshold-based
   recall/precision tradeoffs.
2. Compare weighted cross entropy with focal loss.
3. Try `WeightedRandomSampler` for stronger rare-class exposure.
4. Compare EfficientNet-B0 with ResNet50 or EfficientNet-B1.
5. Add clinical metadata after the image-only baseline is reproducible.
6. Run the single-image inference helper on representative held-out examples and
   log qualitative prediction notes.
