from __future__ import annotations

import json
from pathlib import Path

import jupytext
import nbformat
import pytest

from scripts import notebook_quality


def write_pair(tmp_path: Path) -> tuple[Path, Path]:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Example"),
            nbformat.v4.new_code_cell("value = 1", execution_count=1, outputs=[]),
        ]
    )
    ipynb_path = tmp_path / "example.ipynb"
    py_path = tmp_path / "example.py"
    jupytext.write(notebook, ipynb_path)
    jupytext.write(notebook, py_path, fmt="py:percent")
    return ipynb_path, py_path


def test_pair_and_saved_execution_contract(tmp_path: Path) -> None:
    ipynb_path, py_path = write_pair(tmp_path)

    notebook_quality.check_pair(ipynb_path, py_path)
    notebook_quality.stamp(ipynb_path)
    notebook_quality.check_saved(ipynb_path)


def test_pair_check_detects_source_drift(tmp_path: Path) -> None:
    ipynb_path, py_path = write_pair(tmp_path)
    paired = jupytext.read(py_path)
    paired.cells[1].source = "value = 2"
    jupytext.write(paired, py_path, fmt="py:percent")

    with pytest.raises(notebook_quality.NotebookQualityError, match="out of sync"):
        notebook_quality.check_pair(ipynb_path, py_path)


def test_execution_check_requires_restart_and_run_all(tmp_path: Path) -> None:
    ipynb_path, _ = write_pair(tmp_path)
    payload = json.loads(ipynb_path.read_text(encoding="utf-8"))
    payload["cells"][1]["execution_count"] = None
    ipynb_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(notebook_quality.NotebookQualityError, match="Restart & Run All"):
        notebook_quality.validate_executed(notebook_quality.read_json_notebook(ipynb_path))


def test_execution_check_rejects_error_output(tmp_path: Path) -> None:
    ipynb_path, _ = write_pair(tmp_path)
    payload = json.loads(ipynb_path.read_text(encoding="utf-8"))
    payload["cells"][1]["outputs"] = [
        {
            "output_type": "error",
            "ename": "RuntimeError",
            "evalue": "boom",
            "traceback": [],
        }
    ]
    ipynb_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(notebook_quality.NotebookQualityError, match="boom"):
        notebook_quality.validate_executed(notebook_quality.read_json_notebook(ipynb_path))


def test_saved_check_detects_code_changed_after_stamp(tmp_path: Path) -> None:
    ipynb_path, _ = write_pair(tmp_path)
    notebook_quality.stamp(ipynb_path)
    payload = json.loads(ipynb_path.read_text(encoding="utf-8"))
    payload["cells"][1]["source"] = ["value = 99"]
    ipynb_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(notebook_quality.NotebookQualityError, match="stale or unstamped"):
        notebook_quality.check_saved(ipynb_path)
