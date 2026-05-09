"""Train the PAD-UFES-20 image plus clinical metadata baseline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

    class _MissingTorchModule:
        Module = object

    nn = _MissingTorchModule()

from src.features.clinical_metadata import (
    encode_clinical_metadata,
    merge_clinical_metadata,
    read_metadata,
)
from src.training.train_image_baseline import (
    AUGMENT_STRENGTHS,
    DEFAULT_EXPERIMENT_NAME,
    LOSS_TYPES,
    SAMPLERS,
    WorkerSeeder,
    autocast_context,
    build_model,
    configure_mlflow_auth,
    high_risk_label_indices,
    load_split_inputs,
    log_model_compat,
    make_criterion,
    make_grad_scaler,
    make_torch_generator,
    make_transforms,
    resolve_tracking_uri,
    sample_weights_for_training,
    seed_everything,
    selection_score,
)
from src.training.train_metadata_baseline import evaluate_predictions


def require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for multimodal training. Install project requirements first."
        )
    return torch


@dataclass(frozen=True)
class MultimodalInputs:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_metadata_features: pd.DataFrame
    val_metadata_features: pd.DataFrame
    test_metadata_features: pd.DataFrame
    metadata_feature_names: list[str]
    labels: list[str]
    label_mapping: dict[str, object]
    class_weight_payload: dict[str, object]
    preprocessing_summary: dict[str, object]


@dataclass(frozen=True)
class MultimodalArtifactPaths:
    output_dir: Path
    best_checkpoint: Path
    history_csv: Path
    test_metrics_json: Path
    classification_report_csv: Path
    confusion_matrix_csv: Path
    metadata_feature_names_json: Path


@dataclass(frozen=True)
class MultimodalTrainingConfig:
    images_dir: Path
    metadata_path: Path
    splits_dir: Path
    output_dir: Path
    epochs: int = 8
    batch_size: int = 32
    image_size: int = 224
    seed: int = 42
    num_workers: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    experiment_name: str = f"{DEFAULT_EXPERIMENT_NAME}-multimodal"
    tracking_uri: str | None = None
    hf_dataset_repo: str | None = None
    require_gpu: bool = True
    backbone: str = "efficientnet_b0"
    loss_type: str = "weighted_cross_entropy"
    sampler: str = "shuffle"
    augment_strength: str = "current"
    focal_gamma: float = 2.0
    include_optional_metadata: bool = True
    metadata_hidden_dim: int = 64
    fusion_hidden_dim: int = 256
    metadata_dropout: float = 0.1


class EfficientNetMetadataFusion(nn.Module):
    def __init__(
        self,
        image_model: nn.Module,
        metadata_feature_count: int,
        num_classes: int,
        metadata_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
        metadata_dropout: float = 0.1,
    ):
        super().__init__()
        self.image_features = image_model.features
        self.image_pool = image_model.avgpool
        image_feature_dim = image_model.classifier[1].in_features
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_feature_count, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(metadata_dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(metadata_dropout),
            nn.Linear(image_feature_dim + metadata_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(metadata_dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, images, metadata_features):
        torch_module = require_torch()
        image_features = self.image_features(images)
        image_features = self.image_pool(image_features)
        image_features = torch_module.flatten(image_features, 1)
        metadata_features = self.metadata_encoder(metadata_features)
        return self.classifier(torch_module.cat([image_features, metadata_features], dim=1))


class PadUfesMultimodalDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        metadata_features: pd.DataFrame,
        images_dir: Path,
        transform=None,
    ):
        if len(frame) != len(metadata_features):
            raise ValueError("frame and metadata_features must have the same row count")
        self.frame = frame.reset_index(drop=True)
        self.metadata_features = metadata_features.reset_index(drop=True).astype("float32")
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        from PIL import Image

        torch_module = require_torch()
        row = self.frame.iloc[index]
        image_path = self.images_dir / row["image_rel_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        metadata_features = torch_module.as_tensor(
            self.metadata_features.iloc[index].to_numpy(dtype=np.float32),
            dtype=torch_module.float32,
        )
        label = int(row["label_idx"])
        return image, metadata_features, label


def build_multimodal_artifact_paths(
    output_dir: Path,
    backbone: str = "efficientnet_b0",
) -> MultimodalArtifactPaths:
    prefix = f"{backbone}_multimodal"
    return MultimodalArtifactPaths(
        output_dir=output_dir,
        best_checkpoint=output_dir / f"{prefix}_best.pt",
        history_csv=output_dir / "multimodal_history.csv",
        test_metrics_json=output_dir / "multimodal_test_metrics.json",
        classification_report_csv=output_dir / "multimodal_classification_report.csv",
        confusion_matrix_csv=output_dir / "multimodal_confusion_matrix.csv",
        metadata_feature_names_json=output_dir / "multimodal_metadata_feature_names.json",
    )


def validate_multimodal_training_options(config: MultimodalTrainingConfig) -> None:
    if config.backbone != "efficientnet_b0":
        raise ValueError("Only efficientnet_b0 is supported for the first multimodal baseline")
    if config.loss_type not in LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {config.loss_type!r}")
    if config.sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}, got {config.sampler!r}")
    if config.augment_strength not in AUGMENT_STRENGTHS:
        raise ValueError(
            f"augment_strength must be one of {AUGMENT_STRENGTHS}, "
            f"got {config.augment_strength!r}"
        )
    if config.focal_gamma <= 0:
        raise ValueError("focal_gamma must be positive")
    if config.metadata_hidden_dim < 1:
        raise ValueError("metadata_hidden_dim must be at least 1")
    if config.fusion_hidden_dim < 1:
        raise ValueError("fusion_hidden_dim must be at least 1")
    if not 0 <= config.metadata_dropout < 1:
        raise ValueError("metadata_dropout must be in the range [0, 1)")


def prepare_multimodal_inputs(
    metadata_path: Path,
    splits_dir: Path,
    include_optional_metadata: bool = True,
) -> MultimodalInputs:
    splits = load_split_inputs(splits_dir)
    metadata = read_metadata(metadata_path)
    train_frame = merge_clinical_metadata(
        splits.train,
        metadata,
        include_optional=include_optional_metadata,
    )
    val_frame = merge_clinical_metadata(
        splits.val,
        metadata,
        include_optional=include_optional_metadata,
    )
    test_frame = merge_clinical_metadata(
        splits.test,
        metadata,
        include_optional=include_optional_metadata,
    )

    encoder, [train_features, val_features, test_features] = encode_clinical_metadata(
        train_frame,
        [train_frame, val_frame, test_frame],
        include_optional=include_optional_metadata,
    )
    return MultimodalInputs(
        train=train_frame,
        val=val_frame,
        test=test_frame,
        train_metadata_features=train_features,
        val_metadata_features=val_features,
        test_metadata_features=test_features,
        metadata_feature_names=encoder.feature_names,
        labels=splits.labels,
        label_mapping=splits.label_mapping,
        class_weight_payload=splits.class_weight_payload,
        preprocessing_summary=splits.preprocessing_summary,
    )


def build_multimodal_dataloaders(inputs: MultimodalInputs, config: MultimodalTrainingConfig, device):
    torch_module = require_torch()
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_transform, eval_transform = make_transforms(config.image_size, config.augment_strength)
    pin_memory = device.type == "cuda"
    worker_init_fn = WorkerSeeder(config.seed)
    generator = make_torch_generator(config.seed)
    sampler = None
    shuffle = True
    if config.sampler == "weighted_random":
        sampler = WeightedRandomSampler(
            weights=torch_module.DoubleTensor(sample_weights_for_training(inputs.train, inputs.labels)),
            num_samples=len(inputs.train),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    train_loader = DataLoader(
        PadUfesMultimodalDataset(
            inputs.train,
            inputs.train_metadata_features,
            config.images_dir,
            train_transform,
        ),
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        PadUfesMultimodalDataset(
            inputs.val,
            inputs.val_metadata_features,
            config.images_dir,
            eval_transform,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        PadUfesMultimodalDataset(
            inputs.test,
            inputs.test_metadata_features,
            config.images_dir,
            eval_transform,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, test_loader


def build_multimodal_model(
    metadata_feature_count: int,
    num_classes: int,
    metadata_hidden_dim: int = 64,
    fusion_hidden_dim: int = 256,
    metadata_dropout: float = 0.1,
) -> EfficientNetMetadataFusion:
    require_torch()
    image_model = build_model(num_classes=num_classes)
    return EfficientNetMetadataFusion(
        image_model=image_model,
        metadata_feature_count=metadata_feature_count,
        num_classes=num_classes,
        metadata_hidden_dim=metadata_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        metadata_dropout=metadata_dropout,
    )


def multimodal_epoch_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    loss: float,
    labels: Sequence[str],
) -> dict[str, float]:
    metrics, _, _ = evaluate_predictions(targets, predictions, labels)
    return {
        "loss": float(loss),
        "macro_f1": metrics["test_macro_f1"],
        "balanced_acc": metrics["test_balanced_accuracy"],
        "high_risk_recall": metrics["test_high_risk_recall"],
    }


def run_multimodal_epoch(
    model,
    loader,
    device,
    criterion,
    labels: Sequence[str],
    optimizer=None,
    scaler=None,
    amp_enabled: bool = False,
) -> dict[str, float]:
    torch_module = require_torch()
    train = optimizer is not None
    model.train(train)
    running_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, metadata_features, targets in loader:
        images = images.to(device, non_blocking=True)
        metadata_features = metadata_features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch_module.set_grad_enabled(train):
            with autocast_context(torch_module, device.type, amp_enabled):
                logits = model(images, metadata_features)
                loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        running_loss += loss.item() * images.size(0)
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    average_loss = running_loss / len(loader.dataset)
    return multimodal_epoch_metrics(all_targets, all_predictions, average_loss, labels)


def evaluate_multimodal_predictions(
    targets: Sequence[int],
    predictions: Sequence[int],
    labels: Sequence[str],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    metrics, report, matrix = evaluate_predictions(targets, predictions, labels)
    metrics["test_selection_score"] = selection_score(
        {
            "macro_f1": metrics["test_macro_f1"],
            "high_risk_recall": metrics["test_high_risk_recall"],
        }
    )
    return metrics, report, matrix


def evaluate_multimodal_and_log(
    model,
    test_loader,
    device,
    inputs: MultimodalInputs,
    paths: MultimodalArtifactPaths,
) -> dict[str, object]:
    import mlflow

    torch_module = require_torch()
    model.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch_module.no_grad():
        for images, metadata_features, targets in test_loader:
            images = images.to(device, non_blocking=True)
            metadata_features = metadata_features.to(device, non_blocking=True)
            logits = model(images, metadata_features)
            probabilities = torch_module.softmax(logits, dim=1).cpu().numpy()
            all_targets.extend(targets.numpy().tolist())
            all_predictions.extend(probabilities.argmax(axis=1).tolist())

    test_metric_values, report, matrix = evaluate_multimodal_predictions(
        all_targets,
        all_predictions,
        inputs.labels,
    )
    metrics = {
        **test_metric_values,
        "best_checkpoint": str(paths.best_checkpoint),
        "labels": list(inputs.labels),
        "metadata_feature_count": len(inputs.metadata_feature_names),
        "model": "efficientnet_b0_metadata_fusion",
    }

    paths.test_metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    paths.metadata_feature_names_json.write_text(
        json.dumps(inputs.metadata_feature_names, indent=2) + "\n"
    )
    report.to_csv(paths.classification_report_csv)
    matrix.to_csv(paths.confusion_matrix_csv)

    mlflow.log_metrics(test_metric_values)
    mlflow.log_artifact(paths.test_metrics_json)
    mlflow.log_artifact(paths.metadata_feature_names_json)
    mlflow.log_artifact(paths.classification_report_csv)
    mlflow.log_artifact(paths.confusion_matrix_csv)
    mlflow.log_artifact(paths.best_checkpoint)
    log_model_compat(mlflow, model)
    return metrics


def train_multimodal_baseline(config: MultimodalTrainingConfig) -> dict[str, object]:
    import mlflow
    import mlflow.pytorch

    torch_module = require_torch()
    validate_multimodal_training_options(config)
    seed_everything(config.seed)
    configure_mlflow_auth()
    tracking_uri = resolve_tracking_uri(config.tracking_uri)
    paths = build_multimodal_artifact_paths(config.output_dir, config.backbone)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = prepare_multimodal_inputs(
        config.metadata_path,
        config.splits_dir,
        include_optional_metadata=config.include_optional_metadata,
    )
    high_risk_label_indices(inputs.labels)
    device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if config.require_gpu and device.type != "cuda":
        raise RuntimeError(
            "No GPU detected. In Colab, choose Runtime > Change runtime type > T4 GPU, "
            "then rerun the notebook."
        )

    train_loader, val_loader, test_loader = build_multimodal_dataloaders(inputs, config, device)
    model = build_multimodal_model(
        metadata_feature_count=len(inputs.metadata_feature_names),
        num_classes=len(inputs.labels),
        metadata_hidden_dim=config.metadata_hidden_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        metadata_dropout=config.metadata_dropout,
    ).to(device)
    class_weights = torch_module.tensor(
        [
            inputs.class_weight_payload["class_weights"][label]
            for label in inputs.labels
        ],
        dtype=torch_module.float32,
        device=device,
    )
    criterion = make_criterion(config.loss_type, class_weights, config.focal_gamma)
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch_module.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    amp_enabled = device.type == "cuda"
    scaler = make_grad_scaler(torch_module, amp_enabled)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    if mlflow.active_run() is not None:
        mlflow.end_run()

    history: list[dict[str, float]] = []
    best_metric = -1.0
    run_name = f"{config.backbone}_multimodal_{time.strftime('%Y%m%d_%H%M%S')}"

    try:
        mlflow.start_run(run_name=run_name)
        mlflow.set_tags(
            {
                "model_family": "cnn_plus_metadata",
                "backbone": config.backbone,
                "task": "pad_ufes_20_multimodal_classification",
                "triage_goal": "prioritize_high_risk_dermatology_cases",
                "tracking_backend": "dagshub_mlflow",
            }
        )
        mlflow.log_params(
            {
                "seed": config.seed,
                "image_size": config.image_size,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "num_workers": config.num_workers,
                "optimizer": "AdamW",
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "scheduler": "CosineAnnealingLR",
                "loss": config.loss_type,
                "sampler": config.sampler,
                "augment_strength": config.augment_strength,
                "focal_gamma": config.focal_gamma,
                "device": str(device),
                "hf_dataset_repo": config.hf_dataset_repo or "",
                "include_optional_metadata": config.include_optional_metadata,
                "metadata_feature_count": len(inputs.metadata_feature_names),
                "metadata_hidden_dim": config.metadata_hidden_dim,
                "fusion_hidden_dim": config.fusion_hidden_dim,
                "metadata_dropout": config.metadata_dropout,
                "train_rows": len(inputs.train),
                "val_rows": len(inputs.val),
                "test_rows": len(inputs.test),
            }
        )
        mlflow.log_dict(inputs.label_mapping, "label_mapping.json")
        mlflow.log_dict(inputs.class_weight_payload, "class_weights.json")
        mlflow.log_dict(inputs.preprocessing_summary, "preprocessing_summary.json")
        mlflow.log_dict({"feature_names": inputs.metadata_feature_names}, "metadata_feature_names.json")

        for epoch in range(1, config.epochs + 1):
            start = time.time()
            train_metrics = run_multimodal_epoch(
                model,
                train_loader,
                device,
                criterion,
                inputs.labels,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            val_metrics = run_multimodal_epoch(
                model,
                val_loader,
                device,
                criterion,
                inputs.labels,
                amp_enabled=amp_enabled,
            )
            scheduler.step()

            score = selection_score(val_metrics)
            if score > best_metric:
                best_metric = score
                torch_module.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "labels": inputs.labels,
                        "metadata_feature_names": inputs.metadata_feature_names,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "config": asdict(config),
                    },
                    paths.best_checkpoint,
                )

            row = {
                "epoch": epoch,
                "seconds": round(time.time() - start, 1),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)
            mlflow.log_metrics(
                {
                    **row,
                    "selection_score": score,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
            print(row)

        pd.DataFrame(history).to_csv(paths.history_csv, index=False)
        paths.metadata_feature_names_json.write_text(
            json.dumps(inputs.metadata_feature_names, indent=2) + "\n"
        )
        mlflow.log_artifact(paths.history_csv)

        checkpoint = torch_module.load(paths.best_checkpoint, map_location=device, weights_only=False)
        model = build_multimodal_model(
            metadata_feature_count=len(inputs.metadata_feature_names),
            num_classes=len(inputs.labels),
            metadata_hidden_dim=config.metadata_hidden_dim,
            fusion_hidden_dim=config.fusion_hidden_dim,
            metadata_dropout=config.metadata_dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return evaluate_multimodal_and_log(model, test_loader, device, inputs, paths)
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--experiment-name", default=f"{DEFAULT_EXPERIMENT_NAME}-multimodal")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--hf-dataset-repo")
    parser.add_argument("--loss-type", choices=LOSS_TYPES, default="weighted_cross_entropy")
    parser.add_argument("--sampler", choices=SAMPLERS, default="shuffle")
    parser.add_argument("--augment-strength", choices=AUGMENT_STRENGTHS, default="current")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--metadata-hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--metadata-dropout", type=float, default=0.1)
    parser.add_argument(
        "--complete-fields-only",
        action="store_true",
        help="Use only complete metadata fields and skip optional fields with missingness.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training without CUDA. Intended for smoke tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultimodalTrainingConfig(
        images_dir=args.images_dir,
        metadata_path=args.metadata_path,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        hf_dataset_repo=args.hf_dataset_repo,
        require_gpu=not args.allow_cpu,
        loss_type=args.loss_type,
        sampler=args.sampler,
        augment_strength=args.augment_strength,
        focal_gamma=args.focal_gamma,
        include_optional_metadata=not args.complete_fields_only,
        metadata_hidden_dim=args.metadata_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        metadata_dropout=args.metadata_dropout,
    )
    metrics = train_multimodal_baseline(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
