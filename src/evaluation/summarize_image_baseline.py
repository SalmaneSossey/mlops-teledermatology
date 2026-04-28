"""Summarize image-baseline metrics and high-risk mistakes from run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HIGH_RISK_LABELS = ["BCC", "MEL", "SCC"]


def load_json(path: Path) -> dict[str, object]:
    with path.open() as file:
        return json.load(file)


def summarize_confusion_matrix(confusion_matrix_path: Path) -> list[str]:
    matrix = pd.read_csv(confusion_matrix_path, index_col=0)
    lines = ["High-risk class mistakes:"]

    for true_label in HIGH_RISK_LABELS:
        if true_label not in matrix.index:
            lines.append(f"- {true_label}: not present in confusion matrix")
            continue

        row = matrix.loc[true_label].copy()
        correct = int(row.get(true_label, 0))
        total = int(row.sum())
        row = row.drop(labels=[true_label], errors="ignore")
        mistakes = row[row > 0].sort_values(ascending=False)

        if mistakes.empty:
            lines.append(f"- {true_label}: {correct}/{total} correct, no recorded mistakes")
            continue

        mistake_text = ", ".join(f"{label}={int(count)}" for label, count in mistakes.items())
        lines.append(f"- {true_label}: {correct}/{total} correct, confused as {mistake_text}")

    return lines


def summarize_report(classification_report_path: Path) -> list[str]:
    report = pd.read_csv(classification_report_path, index_col=0)
    lines = ["Per-class report for high-risk labels:"]

    for label in HIGH_RISK_LABELS:
        if label not in report.index:
            lines.append(f"- {label}: not present in classification report")
            continue

        row = report.loc[label]
        lines.append(
            "- "
            f"{label}: precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"f1={row['f1-score']:.4f}, "
            f"support={int(row['support'])}"
        )

    return lines


def summarize_run(run_dir: Path) -> str:
    metrics_path = run_dir / "test_metrics.json"
    report_path = run_dir / "classification_report.csv"
    confusion_path = run_dir / "confusion_matrix.csv"

    missing = [path for path in [metrics_path, report_path, confusion_path] if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing expected run artifacts:\n{missing_text}")

    metrics = load_json(metrics_path)
    lines = [
        "Image baseline test metrics:",
        f"- macro_f1: {metrics['test_macro_f1']:.4f}",
        f"- balanced_accuracy: {metrics['test_balanced_accuracy']:.4f}",
        f"- high_risk_recall: {metrics['test_high_risk_recall']:.4f}",
        "",
    ]
    lines.extend(summarize_report(report_path))
    lines.append("")
    lines.extend(summarize_confusion_matrix(confusion_path))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/mlops-teledermatology/runs/image_baseline"),
        help="Directory containing test_metrics.json, classification_report.csv, and confusion_matrix.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(summarize_run(args.run_dir))


if __name__ == "__main__":
    main()
