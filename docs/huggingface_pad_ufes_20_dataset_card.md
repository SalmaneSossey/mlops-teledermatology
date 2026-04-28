---
license: cc-by-4.0
task_categories:
  - image-classification
language:
  - en
tags:
  - dermatology
  - skin-lesion
  - clinical-images
  - teledermatology
  - medical
pretty_name: PAD-UFES-20
size_categories:
  - 1K<n<10K
---

# PAD-UFES-20

This dataset repository mirrors PAD-UFES-20 for reproducible teledermatology
experiments in `mlops-teledermatology`.

PAD-UFES-20 contains smartphone clinical images of skin lesions plus tabular
metadata. The dataset includes 2,298 images from 1,373 patients and 1,641 skin
lesions. The labels used by this project are:

- `ACK`: Actinic keratosis
- `BCC`: Basal cell carcinoma
- `MEL`: Melanoma
- `NEV`: Nevus
- `SCC`: Squamous cell carcinoma, including Bowen's disease/SCC in situ
- `SEK`: Seborrheic keratosis

## Source And License

Original dataset:

- Pacheco, A. G. C. et al. PAD-UFES-20: a skin lesion dataset composed of
  patient data and clinical images collected from smartphones. Data in Brief,
  32, 106221, 2020.
- Mendeley Data: https://data.mendeley.com/datasets/zr7vgbcyr2
- Kaggle mirror: https://www.kaggle.com/datasets/mahdavi1202/skin-cancer

The Kaggle mirror lists the license as Creative Commons Attribution 4.0
International (CC BY 4.0). Keep this attribution and cite the original paper
when using the data.

## Intended Use

This mirror supports research and education around image-based skin lesion
classification, model evaluation, and MLOps reproducibility.

This dataset and any models trained from it are not medical devices and should
not be used for autonomous diagnosis. In this project, predictions are framed as
triage support for clinician review.

## Repository Layout

The project downloader accepts either extracted images or ZIP archives. The
preferred Hugging Face layout is:

```text
metadata.csv
all_images/
  imgs_part_1/*.png
  imgs_part_2/*.png
  imgs_part_3/*.png
splits/
  train.csv
  val.csv
  test.csv
  label_mapping.json
  class_weights.json
  preprocessing_summary.json
```

If you upload the original archives instead, this layout also works:

```text
metadata.csv
images/
  imgs_part_1.zip
  imgs_part_2.zip
  imgs_part_3.zip
```

## Project Split Protocol

The `mlops-teledermatology` project regenerates patient-safe train,
validation, and test manifests with:

```bash
python -m src.data.make_image_splits
```

The split algorithm groups by `patient_id` to avoid patient leakage and keeps
portable `image_rel_path` values for Colab/Hugging Face workflows.

## Limitations

- Labels are imbalanced, with melanoma especially rare.
- Several clinical metadata columns contain missing or unknown values.
- Some classes are biopsy-proven for all samples while others include clinical
  diagnoses, so `biopsed` should not be used as a model feature in this project.
- The images come from smartphone acquisition and vary substantially in
  resolution, lighting, and focus.

## Citation

```bibtex
@article{pacheco2020padufes20,
  title = {PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones},
  author = {Pacheco, Andre G. C. and Lima, Gustavo R. and Salomao, Amanda S. and Krohling, Breno and Biral, Igor P. and de Angelo, Gabriel G. and Alves Jr., Fabio C. R. and Esgario, Jose G. M. and Simora, Alana C. and Castro, Pedro B. C. and Rodrigues, Filipe B. and Frasson, Paulo H. L. and Krohling, Renato A.},
  journal = {Data in Brief},
  volume = {32},
  pages = {106221},
  year = {2020}
}
```
