"""Run feedback-driven multimodal retraining and promotion checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.retraining_pipeline import RetrainingPipelineConfig, run_retraining_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-path", type=Path, default=Path("data/feedback/retraining_candidates.csv"))
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/feedback_retraining"))
    parser.add_argument("--current-metrics-path", type=Path, default=Path("storage/model_bundle/manifest.json"))
    parser.add_argument("--candidate-bundle-dir", type=Path, default=Path("storage/model_bundle_candidate"))
    parser.add_argument("--min-feedback-cases", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--experiment-name", default="pad-ufes-20-feedback-retraining")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--loss-type", default="weighted_cross_entropy")
    parser.add_argument("--sampler", default="weighted_random")
    parser.add_argument("--augment-strength", default="current")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--metadata-hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--metadata-dropout", type=float, default=0.1)
    parser.add_argument("--initial-image-checkpoint", type=Path)
    parser.add_argument("--complete-fields-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--build-candidate-bundle", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate feedback and write augmented manifests without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_retraining_pipeline(
        RetrainingPipelineConfig(
            feedback_path=args.feedback_path,
            images_dir=args.images_dir,
            metadata_path=args.metadata_path,
            splits_dir=args.splits_dir,
            output_dir=args.output_dir,
            current_metrics_path=args.current_metrics_path,
            candidate_bundle_dir=args.candidate_bundle_dir,
            min_feedback_cases=args.min_feedback_cases,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            seed=args.seed,
            num_workers=args.num_workers,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            require_gpu=not args.allow_cpu,
            loss_type=args.loss_type,
            sampler=args.sampler,
            augment_strength=args.augment_strength,
            focal_gamma=args.focal_gamma,
            include_optional_metadata=not args.complete_fields_only,
            metadata_hidden_dim=args.metadata_hidden_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            metadata_dropout=args.metadata_dropout,
            initial_image_checkpoint=args.initial_image_checkpoint,
            build_candidate_bundle=args.build_candidate_bundle,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
