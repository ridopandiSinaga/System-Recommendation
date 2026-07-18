"""Quality gates for the paired, executed project notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jupytext


STAMP_NAMESPACE = "book_recommender"
STAMP_KEY = "executed_code_sha256"
DEPENDENCIES_STAMP_KEY = "executed_dependencies_sha256"
NEWLINE_NORMALIZED_DEPENDENCY_SUFFIXES = frozenset({".json"})


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


def dependency_digest(dependencies: Iterable[str | Path]) -> str:
    """Hash dependency contents without embedding machine-specific paths."""

    content_digests = []
    seen_paths: set[Path] = set()
    for dependency in dependencies:
        path = Path(dependency)
        resolved = path.resolve()
        if resolved in seen_paths:
            raise NotebookQualityError(f"Notebook dependency is repeated: {path}")
        seen_paths.add(resolved)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise NotebookQualityError(f"Cannot read notebook dependency {path}: {exc}") from exc
        if path.suffix.casefold() in NEWLINE_NORMALIZED_DEPENDENCY_SUFFIXES:
            # Git may check out text with CRLF on Windows and LF on Linux. Those
            # platform-only bytes must not make an otherwise identical JSON
            # dependency look stale; binary artifacts remain byte-exact.
            payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        content_digests.append(hashlib.sha256(payload).hexdigest())

    canonical = json.dumps(sorted(content_digests), separators=(",", ":"))
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


def stamp(
    ipynb_path: str | Path,
    dependencies: Iterable[str | Path] = (),
) -> str:
    path = Path(ipynb_path)
    notebook = read_json_notebook(path)
    validate_executed(notebook)
    digest = code_source_digest(notebook)
    dependency_paths = tuple(dependencies)
    metadata = notebook.setdefault("metadata", {})
    namespace = metadata.setdefault(STAMP_NAMESPACE, {})
    namespace[STAMP_KEY] = digest
    if dependency_paths:
        namespace[DEPENDENCIES_STAMP_KEY] = dependency_digest(dependency_paths)
    else:
        namespace.pop(DEPENDENCIES_STAMP_KEY, None)
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return digest


def check_saved(
    ipynb_path: str | Path,
    dependencies: Iterable[str | Path] = (),
) -> None:
    notebook = read_json_notebook(ipynb_path)
    validate_executed(notebook)
    actual = code_source_digest(notebook)
    namespace = notebook.get("metadata", {}).get(STAMP_NAMESPACE, {})
    expected = namespace.get(STAMP_KEY)
    if expected != actual:
        raise NotebookQualityError(
            "Saved notebook outputs are stale or unstamped: code source digest does not match."
        )

    dependency_paths = tuple(dependencies)
    expected_dependencies = namespace.get(DEPENDENCIES_STAMP_KEY)
    if dependency_paths:
        actual_dependencies = dependency_digest(dependency_paths)
        if expected_dependencies != actual_dependencies:
            raise NotebookQualityError(
                "Saved notebook outputs are stale or unstamped: dependency digest does not match."
            )
    elif expected_dependencies is not None:
        raise NotebookQualityError(
            "Saved notebook outputs have stamped dependencies; pass each --dependency to verify them."
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
        if name in {"stamp", "check-saved"}:
            command.add_argument(
                "--dependency",
                action="append",
                default=[],
                type=Path,
                help="File whose content must match the notebook's saved outputs; repeatable.",
            )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "check-pair":
        check_pair(args.ipynb, args.python)
        print("Jupytext pair is synchronized.")
    elif args.command == "stamp":
        digest = stamp(args.ipynb, dependencies=args.dependency)
        print(f"Stamped executed notebook source: {digest}")
    elif args.command == "check-saved":
        check_saved(args.ipynb, dependencies=args.dependency)
        print("Saved notebook outputs match the executed source stamp.")
    else:
        validate_executed(read_json_notebook(args.ipynb))
        print("Notebook executed every code cell without an error output.")


if __name__ == "__main__":
    main()
