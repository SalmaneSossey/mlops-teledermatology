# Changelog

## Unreleased

- Added a reusable image-baseline training CLI and converted the Colab notebook
  into a thin launcher.
- Added GitHub Actions CI, Docker packaging, and a notebook hygiene check.
- Added a small MLflow-tracked hyperparameter sweep runner.
- Added clinical metadata encoding utilities and a metadata-only baseline CLI
  for multimodal ablation planning.
- Added a first late-fusion image + clinical metadata baseline CLI and tests.
- Added a Colab hyperparameter sweep runbook with DagsHub credential preflight
  and multimodal baseline commands.
- Integrated optional hyperparameter sweep and multimodal baseline cells into
  the Colab launcher notebook.
- Added a metadata-only smoke report to compare against the image-only baseline.
- Documented AWS cost guardrails and deferred Kubernetes/EKS until it has clear
  project value.
