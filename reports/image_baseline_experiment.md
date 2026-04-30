# PAD-UFES-20 Image Baseline Experiment

## Summary

This run trained an image-only EfficientNet-B0 baseline in Google Colab using a
GPU runtime and PAD-UFES-20 data mounted from Google Drive.

- Notebook: `notebooks/colab-image-baseline.ipynb`
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
- Tracking: MLflow file store on Google Drive for this historical run

## Data And Artifacts

This historical baseline used raw data mounted from Google Drive:

```text
MyDrive/teledermatology-data/zr7vgbcyr2-1/
  metadata.csv
  images/imgs_part_*.zip
```

Future runs should use the public Hugging Face dataset mirror as the dataset
access layer, materialized locally with:

```bash
python -m src.data.download_pad_ufes_20 --repo-id SalmaneExploring/pad-ufes-20 --force
```

The run artifacts are saved in Google Drive:

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

Future Colab runs should log MLflow metrics and artifacts directly to the
project's DagsHub MLflow tracking server, with Google Drive kept as a checkpoint
and report backup location.

## Split

The notebook regenerated the patient-safe split manifests before training.

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
| Macro F1 | 0.6636 |
| Balanced accuracy | 0.6803 |
| High-risk recall | 0.8293 |

For the triage use case, high-risk recall is the most important baseline metric:
the model caught about 83% of `BCC`, `MEL`, and `SCC` test examples as high-risk.

## High-Risk Error Review

The local summary helper was run against the downloaded MLflow artifacts for
run `4e62419a0fdf40b8a503ab893ee2d1f3`.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| BCC | 0.7864 | 0.6378 | 0.7043 | 127 |
| MEL | 0.7500 | 0.7500 | 0.7500 | 8 |
| SCC | 0.2034 | 0.4138 | 0.2727 | 29 |

High-risk diagnostic mistakes:

- `BCC`: 81/127 correct; confused as `SCC` 28 times, `ACK` 15 times, and `NEV`
  3 times.
- `MEL`: 6/8 correct; confused as `BCC` once and `NEV` once.
- `SCC`: 12/29 correct; confused as `BCC` 8 times, `ACK` 6 times, and `NEV`
  3 times.

The largest diagnostic confusion is `BCC` predicted as `SCC`, which remains a
high-priority triage prediction. The more clinically relevant misses for this
baseline are high-risk examples predicted as lower-priority `ACK` or `NEV`.

## Validation History

Best validation selection score occurred at epoch 7, based on the notebook's
combined validation macro-F1 and high-risk recall criterion.

| Epoch | Val macro F1 | Val balanced accuracy | Val high-risk recall |
| ---: | ---: | ---: | ---: |
| 1 | 0.5194 | 0.5569 | 0.8598 |
| 2 | 0.4945 | 0.5814 | 0.7683 |
| 3 | 0.5335 | 0.6059 | 0.7927 |
| 4 | 0.6010 | 0.6161 | 0.8171 |
| 5 | 0.5857 | 0.6036 | 0.8293 |
| 6 | 0.5911 | 0.6245 | 0.7805 |
| 7 | 0.5879 | 0.6100 | 0.8354 |
| 8 | 0.5809 | 0.6086 | 0.8049 |

## Known Limitations

- The rare `MEL` class has only 36 train images and 8 test images, so per-class
  metrics are likely unstable.
- This is image-only. It does not use clinical metadata such as age, lesion
  location, or symptoms.
- The checkpoint was trained in Colab and stored in Drive; local inference needs
  the checkpoint path passed explicitly.
- This is a baseline model, not a clinical diagnostic system.

## Recommended Next Experiments

1. Add early stopping and train for more epochs.
2. Compare weighted cross entropy with focal loss.
3. Try `WeightedRandomSampler` for stronger rare-class exposure.
4. Compare EfficientNet-B0 with ResNet50 or EfficientNet-B1.
5. Calibrate the high-risk triage probabilities and evaluate threshold-based
   recall/precision tradeoffs.
6. Run the single-image inference helper on representative held-out examples and
   log qualitative prediction notes.
