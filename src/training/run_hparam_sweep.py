"""Run a small, budget-conscious hyperparameter sweep for the image baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.training.train_image_baseline import (
    DEFAULT_EXPERIMENT_NAME,
    TrainingConfig,
    selection_score,
    train_image_baseline,
)


@dataclass(frozen=True)
class SweepTrial:
    name: str
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    loss_type: str = "weighted_cross_entropy"
    sampler: str = "shuffle"
    augment_strength: str = "current"


DEFAULT_TRIALS = [
    SweepTrial(name="baseline"),
    SweepTrial(name="lr_1e-4", learning_rate=1e-4),
    SweepTrial(name="lr_1e-3", learning_rate=1e-3),
    SweepTrial(name="wd_1e-5", weight_decay=1e-5),
    SweepTrial(name="wd_1e-3", weight_decay=1e-3),
    SweepTrial(name="focal_loss", loss_type="focal_loss"),
    SweepTrial(name="weighted_sampler", sampler="weighted_random"),
    SweepTrial(name="mild_aug", augment_strength="mild"),
    SweepTrial(name="focal_sampler", loss_type="focal_loss", sampler="weighted_random"),
    SweepTrial(name="mild_lr_1e-4", learning_rate=1e-4, augment_strength="mild"),
    SweepTrial(name="focal_mild", loss_type="focal_loss", augment_strength="mild"),
    SweepTrial(name="sampler_mild", sampler="weighted_random", augment_strength="mild"),
]


def select_trials(max_trials: int) -> list[SweepTrial]:
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")
    return DEFAULT_TRIALS[:max_trials]


def trial_output_dir(base_output_dir: Path, trial: SweepTrial) -> Path:
    return base_output_dir / "hparam_sweep" / trial.name


def trial_to_training_config(
    trial: SweepTrial,
    args: argparse.Namespace,
    output_dir: Path | None = None,
    experiment_name: str | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        images_dir=args.images_dir,
        splits_dir=args.splits_dir,
        output_dir=output_dir or trial_output_dir(args.output_dir, trial),
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
        learning_rate=trial.learning_rate,
        weight_decay=trial.weight_decay,
        experiment_name=experiment_name or args.experiment_name,
        tracking_uri=args.tracking_uri,
        hf_dataset_repo=args.hf_dataset_repo,
        require_gpu=not args.allow_cpu,
        loss_type=trial.loss_type,
        sampler=trial.sampler,
        augment_strength=trial.augment_strength,
        focal_gamma=args.focal_gamma,
    )


def summarize_trial_result(trial: SweepTrial, metrics: dict[str, object]) -> dict[str, object]:
    row = asdict(trial)
    row.update(metrics)
    row["selection_score"] = selection_score(
        {
            "macro_f1": float(metrics["test_macro_f1"]),
            "high_risk_recall": float(metrics["test_high_risk_recall"]),
        }
    )
    return row


def write_sweep_results(rows: Iterable[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    frame.to_csv(output_dir / "hparam_sweep_results.csv", index=False)
    (output_dir / "hparam_sweep_results.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2) + "\n"
    )
    return frame


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
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--experiment-name", default=f"{DEFAULT_EXPERIMENT_NAME}-hparam-sweep")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--hf-dataset-repo")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned trials without running training.",
    )
    parser.add_argument(
        "--retrain-best",
        action="store_true",
        help="After the sweep, retrain the best trial once in output-dir/best_retrain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = select_trials(args.max_trials)
    planned = [asdict(trial) for trial in trials]
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return

    rows = []
    for trial in trials:
        print(f"Running sweep trial: {trial.name}")
        metrics = train_image_baseline(trial_to_training_config(trial, args))
        rows.append(summarize_trial_result(trial, metrics))

    results = write_sweep_results(rows, args.output_dir / "hparam_sweep")
    print(results.to_string(index=False))

    if args.retrain_best:
        best = next(trial for trial in trials if trial.name == str(results.iloc[0]["name"]))
        print(f"Retraining best trial: {best.name}")
        best_metrics = train_image_baseline(
            trial_to_training_config(
                best,
                args,
                output_dir=args.output_dir / "best_retrain",
                experiment_name=f"{args.experiment_name}-best-retrain",
            )
        )
        (args.output_dir / "best_retrain" / "selected_trial.json").write_text(
            json.dumps({"trial": asdict(best), "metrics": best_metrics}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
