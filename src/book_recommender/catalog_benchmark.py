"""Pure contracts used by the catalog-size benchmark.

The benchmark orchestration deliberately lives elsewhere.  This module contains only
stable identities and JSON-shape validation so configuration selection and artifact
verification can share one implementation without importing the CLI.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION = 1
CATALOG_BENCHMARK_SCHEMA_VERSION = CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION
CATALOG_BENCHMARK_ARTIFACT_TYPE = "catalog_size_benchmark"
CATALOG_BENCHMARK_PROTOCOL = "canonical_work_fixed_test_repeated_validation_paired_catalog_v1"

DEFAULT_CATALOG_GRID = ((1_000, 3_000), (3_000, 10_000), (5_000, 20_000))
DEFAULT_BASELINE_CATALOG = (1_000, 3_000)

CATALOG_CONFIG_FIELDS = ("max_interaction_books", "max_content_books")
HYBRID_WEIGHT_FIELDS = (
    "collaborative_weight",
    "content_weight",
    "popularity_weight",
)

_TIMING_FIELDS = {"trial_count", "seconds", "median_seconds", "p95_seconds"}
_INFERENCE_FIELDS = {
    "warmup_calls",
    "measured_calls",
    "cohort_hash",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "max_ms",
}
_MODEL_PROXY_FIELDS = {
    "serialized_model_bytes",
    "matrix_payload_bytes",
    "content_matrix_bytes",
    "collaborative_matrix_bytes",
    "dense_array_bytes",
}


def parse_catalog_grid(value: str | None) -> list[dict[str, int]]:
    """Parse ``interaction,content`` pairs while preserving the declared order."""

    if value is None or not value.strip():
        pairs = list(DEFAULT_CATALOG_GRID)
    else:
        pairs = []
        for raw_config in value.split(";"):
            config = raw_config.strip()
            if not config:
                raise ValueError("Catalog grid must not contain an empty configuration.")
            parts = [part.strip() for part in config.split(",")]
            if len(parts) != 2 or any(not part for part in parts):
                raise ValueError(
                    "Each catalog config must contain exactly 2 comma-separated integers."
                )
            try:
                interaction_books, content_books = (int(part) for part in parts)
            except ValueError as exc:
                raise ValueError("Catalog sizes must contain only integers.") from exc
            pairs.append((interaction_books, content_books))

    return [
        _catalog_sizes(interaction_books, content_books)
        for interaction_books, content_books in pairs
    ]


def normalize_catalog_config(value: Mapping[str, object]) -> dict[str, int]:
    """Return the validated catalog-size subset of a configuration mapping."""

    if not isinstance(value, Mapping):
        raise ValueError("Catalog configuration must be an object.")
    missing = [field for field in CATALOG_CONFIG_FIELDS if field not in value]
    if missing:
        raise ValueError(f"Catalog configuration is missing fields: {missing}.")
    return _catalog_sizes(
        value["max_interaction_books"],
        value["max_content_books"],
    )


def catalog_config_id(
    config_or_interaction_books: Mapping[str, object] | int,
    max_content_books: int | None = None,
) -> str:
    """Return a stable identity derived solely from the two catalog limits."""

    if isinstance(config_or_interaction_books, Mapping):
        if max_content_books is not None:
            raise ValueError("max_content_books must be omitted when a config object is supplied.")
        config = normalize_catalog_config(config_or_interaction_books)
    else:
        if max_content_books is None:
            raise ValueError("max_content_books is required with an interaction-book limit.")
        config = _catalog_sizes(config_or_interaction_books, max_content_books)
    return f"ib{config['max_interaction_books']}_cb{config['max_content_books']}"


def benchmark_catalog_configs(
    *,
    catalog_grid: str | None,
    baseline_catalog: str | None = None,
) -> tuple[list[dict[str, str | int]], str]:
    """Return a sorted, deduplicated registry that always contains the baseline."""

    baseline_value = baseline_catalog
    if baseline_value is None or not baseline_value.strip():
        baseline_value = f"{DEFAULT_BASELINE_CATALOG[0]},{DEFAULT_BASELINE_CATALOG[1]}"
    baseline_configs = parse_catalog_grid(baseline_value)
    if len(baseline_configs) != 1:
        raise ValueError("--baseline-catalog must contain exactly one catalog configuration.")
    baseline = baseline_configs[0]

    configs_by_sizes: dict[tuple[int, int], dict[str, str | int]] = {}
    for config in [*parse_catalog_grid(catalog_grid), baseline]:
        key = (config["max_interaction_books"], config["max_content_books"])
        configs_by_sizes[key] = {"config_id": catalog_config_id(config), **config}

    configs = [configs_by_sizes[key] for key in sorted(configs_by_sizes)]
    return configs, catalog_config_id(baseline)


def catalog_config_registry(value: object) -> dict[str, dict[str, str | int]]:
    """Validate artifact catalog configs and return them keyed in deterministic order."""

    if not isinstance(value, list) or not value:
        raise ValueError("Catalog configuration registry must be a non-empty list.")
    by_id: dict[str, dict[str, str | int]] = {}
    for index, raw_config in enumerate(value):
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"Catalog configuration {index} must be an object.")
        expected_fields = {"config_id", *CATALOG_CONFIG_FIELDS}
        if set(raw_config) != expected_fields:
            raise ValueError(f"Catalog configuration {index} has an invalid schema.")
        config = normalize_catalog_config(raw_config)
        expected_id = catalog_config_id(config)
        if raw_config.get("config_id") != expected_id:
            raise ValueError(f"Catalog configuration {index} has an inconsistent config_id.")
        if expected_id in by_id:
            raise ValueError("Catalog configuration registry contains duplicate configurations.")
        by_id[expected_id] = {"config_id": expected_id, **config}

    return dict(
        sorted(
            by_id.items(),
            key=lambda item: (
                int(item[1]["max_interaction_books"]),
                int(item[1]["max_content_books"]),
            ),
        )
    )


def normalize_hybrid_weights(
    collaborative_weight: object,
    content_weight: object,
    popularity_weight: object,
) -> dict[str, float]:
    """Validate and normalize the fixed hybrid weights used across catalog configs."""

    values = (collaborative_weight, content_weight, popularity_weight)
    normalized_inputs = [
        _finite_non_negative_number(value, f"hybrid_weights.{field}")
        for field, value in zip(HYBRID_WEIGHT_FIELDS, values)
    ]
    total = sum(normalized_inputs)
    if total <= 0.0:
        raise ValueError("Hybrid weights must sum to a positive value.")
    return {
        field: value / total
        for field, value in zip(HYBRID_WEIGHT_FIELDS, normalized_inputs)
    }


def validate_resource_metrics(value: object) -> dict[str, Any]:
    """Validate the portable resource-measurement schema for one catalog config.

    No absolute timing threshold is imposed: elapsed time is environment-dependent.  The
    contract verifies finite samples, internally consistent counts and percentile ordering.
    """

    if not isinstance(value, Mapping):
        raise ValueError("Resource metrics must be an object.")
    expected_fields = {"preparation", "fit", "warm_inference", "model_proxy"}
    if set(value) != expected_fields:
        raise ValueError("Resource metrics have an invalid schema.")

    result: dict[str, Any] = {}
    for phase in ("preparation", "fit"):
        result[phase] = _validate_timing_trials(value[phase], phase)
    result["warm_inference"] = _validate_warm_inference(value["warm_inference"])
    result["model_proxy"] = _validate_model_proxy(value["model_proxy"])
    return result


def _catalog_sizes(interaction_books: object, content_books: object) -> dict[str, int]:
    interaction = _positive_integer(interaction_books, "max_interaction_books")
    content = _positive_integer(content_books, "max_content_books")
    if content < interaction:
        raise ValueError("max_content_books must be greater than or equal to max_interaction_books.")
    return {"max_interaction_books": interaction, "max_content_books": content}


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _finite_non_negative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return parsed


def _validate_timing_trials(value: object, phase: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TIMING_FIELDS:
        raise ValueError(f"Resource metrics {phase} has an invalid schema.")
    trial_count = _positive_integer(value["trial_count"], f"{phase}.trial_count")
    raw_seconds = value["seconds"]
    if not isinstance(raw_seconds, list) or len(raw_seconds) != trial_count:
        raise ValueError(f"{phase}.seconds must contain exactly trial_count samples.")
    seconds = [
        _finite_non_negative_number(sample, f"{phase}.seconds[{index}]")
        for index, sample in enumerate(raw_seconds)
    ]
    median = _finite_non_negative_number(value["median_seconds"], f"{phase}.median_seconds")
    p95 = _finite_non_negative_number(value["p95_seconds"], f"{phase}.p95_seconds")
    if median > p95:
        raise ValueError(f"{phase} timing percentiles are out of order.")
    if seconds and (median > max(seconds) or p95 > max(seconds)):
        raise ValueError(f"{phase} timing summaries exceed the measured samples.")
    return {
        "trial_count": trial_count,
        "seconds": seconds,
        "median_seconds": median,
        "p95_seconds": p95,
    }


def _validate_warm_inference(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _INFERENCE_FIELDS:
        raise ValueError("Resource metrics warm_inference has an invalid schema.")
    warmup_calls = _non_negative_integer(value["warmup_calls"], "warm_inference.warmup_calls")
    measured_calls = _positive_integer(value["measured_calls"], "warm_inference.measured_calls")
    cohort_hash = value["cohort_hash"]
    if not _is_sha256(cohort_hash):
        raise ValueError("warm_inference.cohort_hash must be a SHA-256 digest.")
    mean = _finite_non_negative_number(value["mean_ms"], "warm_inference.mean_ms")
    p50 = _finite_non_negative_number(value["p50_ms"], "warm_inference.p50_ms")
    p95 = _finite_non_negative_number(value["p95_ms"], "warm_inference.p95_ms")
    maximum = _finite_non_negative_number(value["max_ms"], "warm_inference.max_ms")
    if p50 > p95 or p95 > maximum or mean > maximum:
        raise ValueError("warm_inference latency summaries are out of order.")
    return {
        "warmup_calls": warmup_calls,
        "measured_calls": measured_calls,
        "cohort_hash": str(cohort_hash).lower(),
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": p95,
        "max_ms": maximum,
    }


def _validate_model_proxy(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _MODEL_PROXY_FIELDS:
        raise ValueError("Resource metrics model_proxy has an invalid schema.")
    return {
        field: _non_negative_integer(value[field], f"model_proxy.{field}")
        for field in sorted(_MODEL_PROXY_FIELDS)
    }


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
