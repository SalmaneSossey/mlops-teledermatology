"""Train the PAD-UFES-20 image-only EfficientNet baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


DEFAULT_LABELS = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
HIGH_RISK_LABELS = ["BCC", "MEL", "SCC"]
DEFAULT_EXPERIMENT_NAME = "pad-ufes-20-image-baseline"
LOSS_TYPES = ["weighted_cross_entropy", "focal_loss"]
SAMPLERS = ["shuffle", "weighted_random"]
AUGMENT_STRENGTHS = ["current", "mild"]


@dataclass(frozen=True)
class SplitInputs:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]
    label_mapping: dict[str, object]
    class_weight_payload: dict[str, object]
    preprocessing_summary: dict[str, object]


@dataclass(frozen=True)
class ArtifactPaths:
    output_dir: Path
    best_checkpoint: Path
    history_csv: Path
    test_metrics_json: Path
    classification_report_csv: Path
    confusion_matrix_csv: Path


@dataclass(frozen=True)
class TrainingConfig:
    images_dir: Path
    splits_dir: Path
    output_dir: Path
    epochs: int = 8
    batch_size: int = 32
    image_size: int = 224
    seed: int = 42
    num_workers: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    tracking_uri: str | None = None
    hf_dataset_repo: str | None = None
    require_gpu: bool = True
    backbone: str = "efficientnet_b0"
    loss_type: str = "weighted_cross_entropy"
    sampler: str = "shuffle"
    augment_strength: str = "current"
    focal_gamma: float = 2.0


@dataclass(frozen=True)
class WorkerSeeder:
    base_seed: int

    def __call__(self, worker_id: int) -> None:
        seed_worker(worker_id, self.base_seed)


class PadUfesImageDataset:
    def __init__(self, frame: pd.DataFrame, images_dir: Path, transform: Callable | None = None):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.frame.iloc[index]
        image_path = self.images_dir / row["image_rel_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_idx"])
        return image, label


def load_json(path: Path) -> dict[str, object]:
    with path.open() as file:
        return json.load(file)


def load_split_inputs(splits_dir: Path) -> SplitInputs:
    train = pd.read_csv(splits_dir / "train.csv")
    val = pd.read_csv(splits_dir / "val.csv")
    test = pd.read_csv(splits_dir / "test.csv")
    label_mapping = load_json(splits_dir / "label_mapping.json")
    class_weight_payload = load_json(splits_dir / "class_weights.json")
    preprocessing_summary = load_json(splits_dir / "preprocessing_summary.json")

    index_to_label = {
        int(index): label for index, label in label_mapping["index_to_label"].items()
    }
    labels = [index_to_label[index] for index in range(len(index_to_label))]

    return SplitInputs(
        train=train,
        val=val,
        test=test,
        labels=labels,
        label_mapping=label_mapping,
        class_weight_payload=class_weight_payload,
        preprocessing_summary=preprocessing_summary,
    )


def build_artifact_paths(output_dir: Path, backbone: str = "efficientnet_b0") -> ArtifactPaths:
    return ArtifactPaths(
        output_dir=output_dir,
        best_checkpoint=output_dir / f"{backbone}_best.pt",
        history_csv=output_dir / "history.csv",
        test_metrics_json=output_dir / "test_metrics.json",
        classification_report_csv=output_dir / "classification_report.csv",
        confusion_matrix_csv=output_dir / "confusion_matrix.csv",
    )


def high_risk_label_indices(labels: Sequence[str]) -> list[int]:
    missing = [label for label in HIGH_RISK_LABELS if label not in labels]
    if missing:
        raise ValueError(f"Missing high-risk labels from label mapping: {missing}")
    return [labels.index(label) for label in HIGH_RISK_LABELS]


def high_risk_recall(
    targets: Sequence[int],
    predictions: Sequence[int],
    high_risk_indices: Sequence[int],
) -> float:
    target_mask = np.isin(targets, high_risk_indices)
    if int(target_mask.sum()) == 0:
        return 0.0

    prediction_mask = np.isin(predictions, high_risk_indices)
    true_positives = int(np.logical_and(target_mask, prediction_mask).sum())
    return float(true_positives / int(target_mask.sum()))


def selection_score(metrics: dict[str, float]) -> float:
    return float(0.5 * metrics["macro_f1"] + 0.5 * metrics["high_risk_recall"])


def validate_training_options(config: TrainingConfig) -> None:
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


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        import torch

        torch.manual_seed(worker_seed)
    except ImportError:
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_torch_generator(seed: int):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_tracking_uri(explicit_uri: str | None = None) -> str:
    if explicit_uri:
        return explicit_uri
    env_uri = os.environ.get("DAGSHUB_MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    owner = os.environ.get("DAGSHUB_REPO_OWNER", "SalmaneSossey")
    repo_name = os.environ.get("DAGSHUB_REPO_NAME", "mlops-teledermatology")
    return f"https://dagshub.com/{owner}/{repo_name}.mlflow"


def configure_mlflow_auth() -> None:
    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Set DAGSHUB_TOKEN in the environment before training so MLflow can log to DagsHub."
        )

    username = os.environ.get("DAGSHUB_USERNAME")
    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    else:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)


def make_transforms(image_size: int, augment_strength: str = "current"):
    from torchvision import transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if augment_strength == "current":
        train_steps = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        ]
    elif augment_strength == "mild":
        train_steps = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.04, hue=0.01),
        ]
    else:
        raise ValueError(
            f"augment_strength must be one of {AUGMENT_STRENGTHS}, got {augment_strength!r}"
        )

    train_transform = transforms.Compose(
        [
            *train_steps,
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, eval_transform


def sample_weights_for_training(train_frame: pd.DataFrame, labels: Sequence[str]) -> list[float]:
    counts = train_frame["diagnostic"].value_counts().reindex(labels, fill_value=0)
    weights = {label: 1.0 / max(int(counts[label]), 1) for label in labels}
    return [weights[label] for label in train_frame["diagnostic"]]


def build_dataloaders(inputs: SplitInputs, config: TrainingConfig, device):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_transform, eval_transform = make_transforms(config.image_size, config.augment_strength)
    pin_memory = device.type == "cuda"
    worker_init_fn = WorkerSeeder(config.seed)
    generator = make_torch_generator(config.seed)
    sampler = None
    shuffle = True
    if config.sampler == "weighted_random":
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights_for_training(inputs.train, inputs.labels)),
            num_samples=len(inputs.train),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    train_loader = DataLoader(
        PadUfesImageDataset(inputs.train, config.images_dir, train_transform),
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        PadUfesImageDataset(inputs.val, config.images_dir, eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        PadUfesImageDataset(inputs.test, config.images_dir, eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, test_loader


def build_model(num_classes: int):
    from torch import nn
    from torchvision import models

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class FocalLoss:
    def __init__(self, weight=None, gamma: float = 2.0):
        self.weight = weight
        self.gamma = gamma

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as functional

        log_probabilities = functional.log_softmax(logits, dim=1)
        probabilities = torch.exp(log_probabilities)
        target_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        cross_entropy = functional.nll_loss(
            log_probabilities,
            targets,
            weight=self.weight,
            reduction="none",
        )
        return ((1.0 - target_probabilities) ** self.gamma * cross_entropy).mean()


def make_criterion(loss_type: str, class_weights, focal_gamma: float = 2.0):
    from torch import nn

    if loss_type == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal_loss":
        return FocalLoss(weight=class_weights, gamma=focal_gamma)
    raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}")


def make_grad_scaler(torch_module, amp_enabled: bool):
    try:
        return torch_module.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        return torch_module.cuda.amp.GradScaler(enabled=amp_enabled)


def autocast_context(torch_module, device_type: str, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch_module.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch_module.cuda.amp.autocast(enabled=enabled)


def epoch_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    loss: float,
    high_risk_indices: Sequence[int],
) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    return {
        "loss": float(loss),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(targets, predictions)),
        "high_risk_recall": high_risk_recall(targets, predictions, high_risk_indices),
    }


def run_epoch(
    model,
    loader,
    device,
    criterion,
    high_risk_indices: Sequence[int],
    optimizer=None,
    scaler=None,
    amp_enabled: bool = False,
) -> dict[str, float]:
    import torch

    train = optimizer is not None
    model.train(train)
    running_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast_context(torch, device.type, amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item() * images.size(0)
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    average_loss = running_loss / len(loader.dataset)
    return epoch_metrics(all_targets, all_predictions, average_loss, high_risk_indices)


def log_model_compat(mlflow_module, model) -> None:
    try:
        mlflow_module.pytorch.log_model(model, name="model")
    except TypeError:
        mlflow_module.pytorch.log_model(model, artifact_path="model")


def evaluate_and_log(
    model,
    test_loader,
    device,
    labels: Sequence[str],
    high_risk_indices: Sequence[int],
    paths: ArtifactPaths,
) -> dict[str, object]:
    import mlflow
    import torch
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    model.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            all_targets.extend(targets.numpy().tolist())
            all_predictions.extend(probabilities.argmax(axis=1).tolist())

    report = classification_report(
        all_targets,
        all_predictions,
        target_names=list(labels),
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(all_targets, all_predictions, labels=list(range(len(labels))))
    test_metric_values = {
        "test_macro_f1": float(
            f1_score(all_targets, all_predictions, average="macro", zero_division=0)
        ),
        "test_balanced_accuracy": float(balanced_accuracy_score(all_targets, all_predictions)),
        "test_high_risk_recall": high_risk_recall(
            all_targets, all_predictions, high_risk_indices
        ),
    }
    metrics = {
        **test_metric_values,
        "best_checkpoint": str(paths.best_checkpoint),
        "labels": list(labels),
    }

    paths.test_metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(report).T.to_csv(paths.classification_report_csv)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(paths.confusion_matrix_csv)

    mlflow.log_metrics(test_metric_values)
    mlflow.log_artifact(paths.test_metrics_json)
    mlflow.log_artifact(paths.classification_report_csv)
    mlflow.log_artifact(paths.confusion_matrix_csv)
    mlflow.log_artifact(paths.best_checkpoint)
    log_model_compat(mlflow, model)
    return metrics


def train_image_baseline(config: TrainingConfig) -> dict[str, object]:
    import mlflow
    import mlflow.pytorch
    import torch

    validate_training_options(config)
    seed_everything(config.seed)
    configure_mlflow_auth()
    tracking_uri = resolve_tracking_uri(config.tracking_uri)
    paths = build_artifact_paths(config.output_dir, config.backbone)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_split_inputs(config.splits_dir)
    high_risk_indices = high_risk_label_indices(inputs.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.require_gpu and device.type != "cuda":
        raise RuntimeError(
            "No GPU detected. In Colab, choose Runtime > Change runtime type > T4 GPU, "
            "then rerun the notebook."
        )

    train_loader, val_loader, test_loader = build_dataloaders(inputs, config, device)
    model = build_model(num_classes=len(inputs.labels)).to(device)
    class_weights = torch.tensor(
        [
            inputs.class_weight_payload["class_weights"][label]
            for label in inputs.labels
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = make_criterion(config.loss_type, class_weights, config.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    amp_enabled = device.type == "cuda"
    scaler = make_grad_scaler(torch, amp_enabled)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    if mlflow.active_run() is not None:
        mlflow.end_run()

    history: list[dict[str, float]] = []
    best_metric = -1.0
    run_name = f"{config.backbone}_{time.strftime('%Y%m%d_%H%M%S')}"

    try:
        mlflow.start_run(run_name=run_name)
        mlflow.set_tags(
            {
                "model_family": "cnn",
                "backbone": config.backbone,
                "task": "pad_ufes_20_image_classification",
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
                "train_rows": len(inputs.train),
                "val_rows": len(inputs.val),
                "test_rows": len(inputs.test),
            }
        )
        mlflow.log_dict(inputs.label_mapping, "label_mapping.json")
        mlflow.log_dict(inputs.class_weight_payload, "class_weights.json")
        mlflow.log_dict(inputs.preprocessing_summary, "preprocessing_summary.json")

        for epoch in range(1, config.epochs + 1):
            start = time.time()
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                criterion,
                high_risk_indices,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                criterion,
                high_risk_indices,
                amp_enabled=amp_enabled,
            )
            scheduler.step()

            score = selection_score(val_metrics)
            if score > best_metric:
                best_metric = score
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "labels": inputs.labels,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
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
        mlflow.log_artifact(paths.history_csv)

        checkpoint = torch.load(paths.best_checkpoint, map_location=device, weights_only=False)
        model = build_model(num_classes=len(inputs.labels)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return evaluate_and_log(
            model,
            test_loader,
            device,
            inputs.labels,
            high_risk_indices,
            paths,
        )
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--hf-dataset-repo")
    parser.add_argument("--loss-type", choices=LOSS_TYPES, default="weighted_cross_entropy")
    parser.add_argument("--sampler", choices=SAMPLERS, default="shuffle")
    parser.add_argument("--augment-strength", choices=AUGMENT_STRENGTHS, default="current")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training without CUDA. Intended for smoke tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        images_dir=args.images_dir,
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
    )
    metrics = train_image_baseline(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
