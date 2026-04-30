# PAD-UFES-20 Metadata Baseline Experiment

## Summary

This smoke run trained a metadata-only balanced logistic regression baseline on
the existing patient-safe split manifests. It uses `src.features.clinical_metadata`
to encode complete clinical fields and optional fields with missing indicators.

- Training CLI: `python -m src.training.train_metadata_baseline`
- Model: balanced logistic regression
- Feature count: 69
- Leakage exclusions: `diagnostic`, `biopsed`
- Output location for the smoke run: `/tmp/mlops_metadata_baseline_smoke`

## Test Metrics

| Metric | Value |
| --- | ---: |
| Macro F1 | 0.5776 |
| Balanced accuracy | 0.6075 |
| High-risk recall | 0.8598 |
| Selection score | 0.7187 |

Compared with the current image-only baseline, this metadata-only model has
higher high-risk recall but lower macro F1 and balanced accuracy. That is a good
signal for a future multimodal ablation: metadata may help triage sensitivity,
but it is not strong enough to replace image features.

## Reproduce

```bash
python -m src.training.train_metadata_baseline \
  --metadata-path data/raw/pad_ufes_20/metadata.csv \
  --splits-dir data/processed/splits \
  --output-dir /tmp/mlops_metadata_baseline_smoke
```

## Next Step

Use the same split and compare:

```text
image-only baseline
metadata-only baseline
image + metadata multimodal baseline
```
