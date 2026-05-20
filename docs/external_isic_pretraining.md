# External ISIC 2019 Image Pretraining

Use this path when PAD-UFES-20 class coverage is too small for rare/high-risk
lesions. The goal is not to replace PAD-UFES-20, but to initialize the image
encoder with more dermatology images and then keep PAD-UFES-20 as the final
evaluation dataset.

## Why ISIC 2019

ISIC 2019 is useful here because its training labels include `MEL`, `BCC`, and
`SCC`, which are the high-risk labels used by this project. It also includes
`AK`, `BKL`, and `NV`, which can be mapped into the PAD-UFES-20 label space.

Use the official ISIC Challenge data page and citation guidance when downloading
or reporting this dataset:

```text
https://challenge.isic-archive.com/data/
https://challenge.isic-archive.com/landing/2019/
```

Check the dataset license before redistribution. Keep large image archives out of
Git.

The Colab launcher downloads from the current public ISIC challenge S3 paths:

```text
https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv
```

## Label Mapping

The preparer maps ISIC labels into the PAD-UFES-20 six-class space:

```text
AK / AKIEC -> ACK
BCC        -> BCC
MEL        -> MEL
NV         -> NEV
SCC        -> SCC
BKL        -> SEK
DF/VASC    -> dropped
```

This is a pragmatic pretraining mapping, not a claim that the datasets are
clinically identical. The final model should still be judged on PAD-UFES-20.

## Colab Flow

After downloading/extracting ISIC 2019 into Colab, prepare PAD-compatible splits:

```bash
python -m src.data.prepare_isic_2019 \
  --metadata-path /content/isic_2019/ISIC_2019_Training_GroundTruth.csv \
  --images-dir /content/isic_2019/ISIC_2019_Training_Input \
  --output-dir data/processed/isic_2019_splits
```

Pretrain the image encoder on the mapped ISIC data:

```bash
python -m src.training.train_image_baseline \
  --images-dir /content/isic_2019/ISIC_2019_Training_Input \
  --splits-dir data/processed/isic_2019_splits \
  --output-dir /content/drive/MyDrive/mlops-teledermatology/runs/isic_2019_pretrain \
  --experiment-name pad-ufes-20-isic-2019-pretrain \
  --hf-dataset-repo ISIC_2019 \
  --epochs 8 \
  --batch-size 32 \
  --sampler weighted_random
```

Then fine-tune/evaluate PAD-UFES-20 image-only with that checkpoint:

```bash
python -m src.training.train_image_baseline \
  --images-dir /content/pad_ufes_20/all_images \
  --splits-dir data/processed/splits \
  --output-dir /content/drive/MyDrive/mlops-teledermatology/runs/image_baseline/isic_init \
  --experiment-name pad-ufes-20-image-baseline-isic-init \
  --hf-dataset-repo SalmaneExploring/pad-ufes-20 \
  --initial-checkpoint /content/drive/MyDrive/mlops-teledermatology/runs/isic_2019_pretrain/efficientnet_b0_best.pt \
  --epochs 8 \
  --batch-size 32
```

Finally, fine-tune/evaluate the multimodal model with the same initialized image
backbone:

```bash
python -m src.training.train_multimodal_baseline \
  --images-dir /content/pad_ufes_20/all_images \
  --metadata-path /content/pad_ufes_20/metadata.csv \
  --splits-dir data/processed/splits \
  --output-dir /content/drive/MyDrive/mlops-teledermatology/runs/multimodal_baseline/isic_init \
  --experiment-name pad-ufes-20-image-baseline-multimodal-isic-init \
  --hf-dataset-repo SalmaneExploring/pad-ufes-20 \
  --initial-image-checkpoint /content/drive/MyDrive/mlops-teledermatology/runs/isic_2019_pretrain/efficientnet_b0_best.pt \
  --epochs 8 \
  --batch-size 32
```

## Decision Rule

Keep the ISIC-initialized model only if it improves PAD-UFES-20 test behavior,
especially:

```text
macro F1
balanced accuracy
high-risk recall
SCC and MEL recall/F1
```

If it only improves one class by damaging another high-risk class, report it as
an ablation rather than the final model.
