"""Generate or verify a compact catalog-size benchmark summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from book_recommender.catalog_benchmark import (
    build_catalog_benchmark_summary,
    read_catalog_benchmark_summary,
)
from book_recommender.cli import read_catalog_benchmark_metrics


def generate_summary(raw_path: Path, output_path: Path) -> dict:
    """Validate the raw artifact, extract its compact summary, and write atomically."""

    if raw_path.resolve() == output_path.resolve():
        raise ValueError("Raw catalog benchmark and summary output paths must be different.")
    verified_raw = read_catalog_benchmark_metrics(raw_path)
    try:
        raw_bytes = raw_path.read_bytes()
        raw_payload = json.loads(raw_bytes.decode("utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read catalog benchmark artifact {raw_path}: {exc}") from exc
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if verified_raw.get("artifact_sha256") != raw_sha256:
        raise ValueError("Catalog benchmark artifact changed after validation.")

    summary = build_catalog_benchmark_summary(raw_payload, raw_sha256=raw_sha256)
    _write_json_atomic(output_path, summary)
    return read_catalog_benchmark_summary(output_path)


def _write_json_atomic(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite JSON constant: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate summary from a raw artifact.")
    generate.add_argument("raw", type=Path)
    generate.add_argument("output", type=Path)

    check = subparsers.add_parser("check", help="Validate an existing summary.")
    check.add_argument("summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        summary = generate_summary(args.raw, args.output)
        print(
            "Generated catalog benchmark summary: "
            f"{args.output.resolve()} ({summary['source']['raw_artifact_sha256']})"
        )
    else:
        summary = read_catalog_benchmark_summary(args.summary)
        print(
            "Catalog benchmark summary is valid: "
            f"{args.summary.resolve()} ({summary['selection']['decision']})"
        )


if __name__ == "__main__":
    main()
