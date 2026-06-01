"""Export doctor-reviewed consultations for feedback retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.app.database import SessionLocal, init_db
from src.app.main import build_retraining_cases, retraining_cases_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/feedback/retraining_candidates.csv"),
        help="Destination CSV or JSON file.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        help="Defaults to the output file suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_format = args.format or ("json" if args.output_path.suffix == ".json" else "csv")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    init_db()
    with SessionLocal() as db:
        cases = build_retraining_cases(db)

    if output_format == "json":
        payload = [case.model_dump(mode="json") for case in cases]
        args.output_path.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        args.output_path.write_text(retraining_cases_csv(cases))

    print(f"Wrote {len(cases)} retraining candidates to {args.output_path}")


if __name__ == "__main__":
    main()
