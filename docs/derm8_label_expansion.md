# Dermatology 8-Class Label Expansion

This experiment broadens external pretraining from the PAD-UFES-20 six-class
label space to the ISIC 2019 eight-class dermatology lesion label space.

## Label Spaces

PAD-UFES-20 remains the final target evaluation dataset and keeps six classes:

```text
ACK, BCC, MEL, NEV, SCC, SEK
```

The external ISIC 2019 pretraining path can now use `--label-space derm8`:

```text
ACK, BCC, MEL, NEV, SCC, SEK, DF, VASC
```

ISIC mappings:

```text
AK / AKIEC -> ACK
BCC        -> BCC
MEL        -> MEL
NV         -> NEV
SCC        -> SCC
BKL        -> SEK
DF         -> DF
VASC       -> VASC
UNK        -> dropped
```

## Colab Flow

Use the dedicated notebook:

```text
notebooks/colab-derm8-isic-pretraining.ipynb
```

The notebook downloads ISIC 2019 from Kaggle, prepares `derm8` splits, trains an
eight-class EfficientNet-B0 image encoder, then fine-tunes the PAD-UFES-20
six-class multimodal model from that encoder checkpoint.

## Manual Commands

Prepare ISIC 2019 as derm8:

```bash
python -m src.data.prepare_isic_2019 \
  --metadata-path /content/isic_2019/ISIC_2019_Training_GroundTruth.csv \
  --images-dir /content/isic_2019/ISIC_2019_Training_Input \
  --output-dir data/processed/isic_2019_derm8_splits \
  --label-space derm8
```

Pretrain the eight-class image encoder:

```bash
python -m src.training.train_image_baseline \
  --images-dir /content/isic_2019/ISIC_2019_Training_Input \
  --splits-dir data/processed/isic_2019_derm8_splits \
  --output-dir /content/drive/MyDrive/mlops-teledermatology/runs/isic_2019_derm8_pretrain \
  --experiment-name dermatology-8class-isic2019-pretrain \
  --hf-dataset-repo agsam23/isic-2019-challenge \
  --epochs 8 \
  --batch-size 32 \
  --sampler weighted_random
```

Fine-tune the PAD multimodal model from the derm8 encoder:

```bash
python -m src.training.train_multimodal_baseline \
  --images-dir /content/pad_ufes_20/all_images \
  --metadata-path /content/pad_ufes_20/metadata.csv \
  --splits-dir data/processed/splits \
  --output-dir /content/drive/MyDrive/mlops-teledermatology/runs/multimodal_derm8_isic_init \
  --experiment-name pad-ufes-20-multimodal-derm8-isic-init \
  --hf-dataset-repo SalmaneExploring/pad-ufes-20 \
  --initial-image-checkpoint /content/drive/MyDrive/mlops-teledermatology/runs/isic_2019_derm8_pretrain/efficientnet_b0_best.pt \
  --epochs 8 \
  --batch-size 32
```

The checkpoint loader transfers compatible EfficientNet encoder tensors and
skips the incompatible eight-class classifier head when fine-tuning on PAD's
six-class output head.

## Decision Rule

Keep this approach only if PAD-UFES-20 evaluation improves clinically important
behavior:

```text
SCC recall improves over 0.2069
high-risk recall stays within 0.02 of 0.8902
macro F1 and balanced accuracy do not collapse
```

If the model improves only the external ISIC/derm8 task but not PAD-UFES-20, it
should be reported as an external pretraining ablation.
