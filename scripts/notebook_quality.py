"""Quality gates for the paired, executed project notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jupytext


STAMP_NAMESPACE = "book_recommender"
STAMP_KEY = "executed_code_sha256"


class NotebookQualityError(ValueError):
    """Raised when a notebook violates the repository contract."""


def read_json_notebook(path: str | Path) -> dict[str, Any]:
    notebook_path = Path(path)
    try:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NotebookQualityError(f"Cannot read notebook {notebook_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
        raise NotebookQualityError(f"Notebook {notebook_path} has an invalid structure.")
    return payload


def ordered_sources(notebook: Any) -> list[tuple[str, str]]:
    return [(str(cell.cell_type), str(cell.source)) for cell in notebook.cells]


def code_source_digest(notebook: dict[str, Any]) -> str:
    sources = [
        "".join(cell.get("source", []))
        if isinstance(cell.get("source"), list)
        else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and str(cell.get("source", "")).strip()
    ]
    canonical = json.dumps(sources, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_executed(notebook: dict[str, Any]) -> None:
    code_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and (
            "".join(cell.get("source", []))
            if isinstance(cell.get("source"), list)
            else str(cell.get("source", ""))
        ).strip()
    ]
    counts = [cell.get("execution_count") for cell in code_cells]
    expected = list(range(1, len(code_cells) + 1))
    if counts != expected:
        raise NotebookQualityError(
            "Notebook code cells must be saved after Restart & Run All; "
            f"expected execution counts {expected}, found {counts}."
        )

    errors = []
    for index, cell in enumerate(code_cells, start=1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    f"cell {index}: {output.get('ename', 'Error')}: "
                    f"{output.get('evalue', '')}"
                )
    if errors:
        raise NotebookQualityError("Notebook contains error outputs: " + "; ".join(errors))


def check_pair(ipynb_path: str | Path, py_path: str | Path) -> None:
    ipynb = jupytext.read(ipynb_path)
    paired_python = jupytext.read(py_path)
    if ordered_sources(ipynb) != ordered_sources(paired_python):
        raise NotebookQualityError(
            "Jupytext pair is out of sync: ordered cell types or sources differ."
        )


def stamp(ipynb_path: str | Path) -> str:
    path = Path(ipynb_path)
    notebook = read_json_notebook(path)
    validate_executed(notebook)
    digest = code_source_digest(notebook)
    metadata = notebook.setdefault("metadata", {})
    namespace = metadata.setdefault(STAMP_NAMESPACE, {})
    namespace[STAMP_KEY] = digest
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return digest


def check_saved(ipynb_path: str | Path) -> None:
    notebook = read_json_notebook(ipynb_path)
    validate_executed(notebook)
    actual = code_source_digest(notebook)
    expected = (
        notebook.get("metadata", {})
        .get(STAMP_NAMESPACE, {})
        .get(STAMP_KEY)
    )
    if expected != actual:
        raise NotebookQualityError(
            "Saved notebook outputs are stale or unstamped: code source digest does not match."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair = subparsers.add_parser("check-pair")
    pair.add_argument("ipynb", type=Path)
    pair.add_argument("python", type=Path)

    for name in ("stamp", "check-saved", "check-executed"):
        command = subparsers.add_parser(name)
        command.add_argument("ipynb", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "check-pair":
        check_pair(args.ipynb, args.python)
        print("Jupytext pair is synchronized.")
    elif args.command == "stamp":
        digest = stamp(args.ipynb)
        print(f"Stamped executed notebook source: {digest}")
    elif args.command == "check-saved":
        check_saved(args.ipynb)
        print("Saved notebook outputs match the executed source stamp.")
    else:
        validate_executed(read_json_notebook(args.ipynb))
        print("Notebook executed every code cell without an error output.")


if __name__ == "__main__":
    main()
