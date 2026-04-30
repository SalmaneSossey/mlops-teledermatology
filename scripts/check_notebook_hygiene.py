"""Fail if selected notebooks contain execution counts or saved outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook_has_state(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    problems: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            problems.append(f"{path}: cell {index} has execution_count")
        if cell.get("outputs"):
            problems.append(f"{path}: cell {index} has saved outputs")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("notebooks/colab-image-baseline.ipynb")],
        help="Notebook files or directories to check.",
    )
    return parser.parse_args()


def iter_notebooks(paths: list[Path]) -> list[Path]:
    notebooks: list[Path] = []
    for path in paths:
        if path.is_dir():
            notebooks.extend(sorted(path.rglob("*.ipynb")))
        elif path.suffix == ".ipynb":
            notebooks.append(path)
    return notebooks


def main() -> None:
    args = parse_args()
    problems: list[str] = []
    for notebook_path in iter_notebooks(args.paths):
        problems.extend(notebook_has_state(notebook_path))
    if problems:
        raise SystemExit("\n".join(problems))
    print("Notebook hygiene OK")


if __name__ == "__main__":
    main()
