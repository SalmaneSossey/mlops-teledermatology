# Project Problem-Solution Log

This file records the main technical and experimental problems encountered
during the MLOps teledermatology project, how they were solved, and what should
be reported as lessons learned.

## Dataset And Modeling Problems

### PAD-UFES-20 Class Imbalance

**Problem:** PAD-UFES-20 has limited samples for rare/high-risk lesions,
especially `SCC` and `MEL`. The model could reach useful high-risk recall, but
class-level metrics showed `SCC` remained weak.

**Evidence:** The best PAD multimodal ISIC-initialized run reached macro F1
`0.6902`, balanced accuracy `0.6804`, and high-risk recall `0.8902`, but `SCC`
recall was only `0.2069` and SCC F1 was `0.1935`.

**Solution:** We treated SCC performance as a model limitation and designed
several controlled ablations instead of blindly promoting any run. The final
promotion rule required SCC improvement without unacceptable drops in macro F1,
balanced accuracy, or high-risk recall.

**Report takeaway:** The model is useful as decision support, but SCC remains a
known weakness. Reporting class-level metrics was essential because aggregate
metrics alone hid this failure mode.

### External ISIC Pretraining

**Problem:** PAD-UFES-20 alone was too small for robust rare-class learning.

**Solution:** We added an ISIC 2019 external image-pretraining workflow. ISIC
labels were mapped into the PAD six-class space:

```text
AK / AKIEC -> ACK
BCC        -> BCC
MEL        -> MEL
NV         -> NEV
SCC        -> SCC
BKL        -> SEK
DF/VASC    -> dropped in pad6
```

The EfficientNet image encoder was pretrained on ISIC, then transferred into the
PAD multimodal model.

**Result:** ISIC pretraining helped most when combined with clinical metadata.
The best final model became the PAD multimodal ISIC-initialized run:

```text
MLflow run: ef084927bef741f996894b8a0fdd63e3
Macro F1: 0.6902
Balanced accuracy: 0.6804
High-risk recall: 0.8902
```

**Report takeaway:** External dermatology pretraining improved the final PAD
multimodal model, but did not fully solve SCC.

### Class-Aware Augmentation Ablation

**Problem:** The first hypothesis for improving SCC was that rare/high-risk
classes needed stronger augmentation.

**Solution:** We added `--augment-strength class_aware`, which dynamically
applies stronger train-only augmentation to `SCC` and `MEL`, current-level
augmentation to `BCC`, and milder augmentation to lower-risk classes. Validation
and test transforms stayed unchanged.

**Result:** The class-aware run made performance worse:

```text
Macro F1: 0.612654
Balanced accuracy: 0.611471
High-risk recall: 0.823171
SCC recall: 0.137931
```

**Decision:** Do not promote. Keep as an ablation.

**Report takeaway:** More augmentation was not automatically better. For
clinical images, aggressive transformations can hurt visual signal quality and
reduce generalization.

### Eight-Class Dermatology Pretraining

**Problem:** The project was originally a six-class lesion classifier, but a
broader dermatology project should consider additional lesion classes.

**Solution:** We added a `derm8` ISIC 2019 label space:

```text
ACK, BCC, MEL, NEV, SCC, SEK, DF, VASC
```

This retained `DF` and `VASC` during external ISIC pretraining while keeping the
final PAD-UFES-20 model six-class. The checkpoint loader was updated to transfer
compatible encoder tensors and skip the incompatible eight-class classifier
head.

**Result:** The derm8 experiment improved SCC recall but reduced global metrics:

```text
Macro F1: 0.614515
Balanced accuracy: 0.629140
High-risk recall: 0.865854
SCC recall: 0.310345
```

**Decision:** Do not promote. Report as an ablation because SCC improved, but
macro F1, balanced accuracy, and high-risk recall dropped too much.

**Report takeaway:** Broader external pretraining can improve a weak class, but
it may shift model behavior in ways that hurt the target dataset. This is a good
example of why promotion gates are needed in MLOps.

## MLOps And Pipeline Problems

### Model Artifact Handling

**Problem:** Model checkpoints and raw data are too large for Git, but the API
needs a reproducible inference bundle.

**Solution:** We kept raw data, uploads, and model checkpoints out of Git. The
best MLflow/DagsHub run is downloaded or copied into a local inference bundle
under:

```text
storage/model_bundle
```

The bundle includes the checkpoint, labels, metrics, and clinical metadata
encoder.

**Report takeaway:** Git tracks code and metadata; DagsHub/MLflow tracks
experiments and artifacts; local storage holds runtime-only model bundles.

### Automated Retraining Workflow

**Problem:** Doctor feedback was available in the app, but there was no
structured path from reviewed cases back into training.

**Solution:** We added a feedback-driven retraining MVP:

- export doctor-reviewed cases as retraining candidates
- validate image paths, labels, and metadata
- append valid feedback cases to the training manifest only
- train a candidate model
- compare against current metrics with a promotion gate
- optionally build a candidate model bundle

**Report takeaway:** The retraining loop is semi-automated and review-gated. It
supports continuous improvement without silently replacing the active model.

### Promotion Gates

**Problem:** Some experiments improved one metric while damaging others.

**Solution:** We used explicit promotion gates instead of selecting models by a
single score. For SCC-focused experiments, the candidate had to improve SCC
recall while preserving macro F1, balanced accuracy, and high-risk recall within
an acceptable tolerance.

**Report takeaway:** A model can be scientifically interesting but not safe to
promote. The class-aware and derm8 experiments are examples of ablations that
should not replace the active model.

## Backend And App Problems

### Mobile Networking From WSL2

**Problem:** Expo LAN mode did not reliably load on the physical Android phone.
The QR code pointed to a WSL2/Windows LAN address that the phone could not reach.
The FastAPI backend was reachable locally at `http://localhost:8000/docs`, but
not from the phone through `http://192.168.0.105:8000`.

**Solution:** Use tunnel mode for both pieces:

```bash
docker compose up -d api postgres
npx --yes localtunnel --port 8000

cd apps/mobile
EXPO_PUBLIC_TELEDERM_API_URL=<fresh-localtunnel-url> npx expo start --tunnel --clear
```

**Report takeaway:** Local mobile demos from WSL2 need network tunneling or
Windows firewall/network configuration. The tunnel approach was the fastest and
most reliable for demonstration.

### Incorrect API URL On Mobile

**Problem:** At one point the URL was typed as `https://https://...`, causing
Android to fail with an `UnknownHostException` for host `"https"`.

**Solution:** Always copy the fresh localtunnel URL exactly once, and keep the
localtunnel terminal open during the demo.

**Report takeaway:** Demo reliability depends on explicit runbooks and small
operational checks.

### React Native Multipart Upload

**Problem:** The first React Native upload approach failed with:

```text
Unsupported FormDataPart implementation
```

Reading the picked image with `fetch(image.uri)` also failed with:

```text
Could not read the selected image
```

**Solution:** The mobile app now uses `expo-file-system/legacy` multipart upload
instead of manually building a problematic FormData object.

**Report takeaway:** Native mobile file upload differs from browser FormData.
Using Expo's file-system upload API gave a reliable path for physical-phone
image submission.

## Infrastructure And Reproducibility Problems

### Runtime Inference Downloading Weights

**Problem:** Runtime inference should not download EfficientNet weights, because
that makes Docker/API startup slower and less reproducible.

**Solution:** The Docker inference path uses the packaged local checkpoint and
CPU PyTorch wheels. Runtime inference no longer downloads EfficientNet weights.

**Report takeaway:** Production-like inference should be artifact-driven and not
depend on external weight downloads at startup.

### Kaggle Credentials In Colab

**Problem:** The derm8 notebook could not download ISIC 2019 from Kaggle until
Kaggle API credentials were available.

**Solution:** The notebook supports either:

```text
KAGGLE_USERNAME / KAGGLE_KEY in Colab Secrets
```

or:

```text
/content/drive/MyDrive/kaggle.json
```

**Report takeaway:** External data dependencies need documented authentication
steps for reproducibility.

### Notebook Output Hygiene

**Problem:** Notebooks can accidentally store execution outputs and secrets-like
logs.

**Solution:** A notebook hygiene checker detects saved outputs and execution
counts. Generated run notebooks should be cleaned before committing unless the
saved outputs are intentionally being inspected.

**Report takeaway:** Notebook hygiene is part of reproducible MLOps practice.

## Final Model Decision

The active final model remains:

```text
PAD multimodal ISIC-initialized run
MLflow run ID: ef084927bef741f996894b8a0fdd63e3
Macro F1: 0.6902
Balanced accuracy: 0.6804
High-risk recall: 0.8902
```

Why it remains active:

- best overall PAD-UFES-20 multimodal performance
- strong high-risk recall
- tracked in DagsHub MLflow
- compatible with the FastAPI inference bundle
- later class-aware and derm8 experiments did not pass promotion gates

Known limitation:

- SCC recall remains weak and should be discussed clearly in the final report.
