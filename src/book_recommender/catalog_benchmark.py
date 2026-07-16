"""Pure contracts used by the catalog-size benchmark.

The benchmark orchestration deliberately lives elsewhere.  This module contains only
stable identities and JSON-shape validation so configuration selection and artifact
verification can share one implementation without importing the CLI.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any


CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION = 1
CATALOG_BENCHMARK_SCHEMA_VERSION = CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION
CATALOG_BENCHMARK_ARTIFACT_TYPE = "catalog_size_benchmark"
CATALOG_BENCHMARK_PROTOCOL = "canonical_work_fixed_test_repeated_validation_paired_catalog_v1"
CATALOG_BENCHMARK_SUMMARY_SCHEMA_VERSION = 1
CATALOG_BENCHMARK_SUMMARY_TYPE = "catalog_size_benchmark_summary"

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

_ALGORITHM_FIELDS = {
    "hybrid_algorithm_id",
    "hybrid_model_format_version",
    "collaborative_algorithm_id",
    "collaborative_model_format_version",
    "content_algorithm_id",
    "content_model_format_version",
}
_SUMMARY_FIELDS = {
    "schema_version",
    "artifact_type",
    "created_at",
    "protocol",
    "relevance_unit",
    "split_unit",
    "algorithms",
    "source",
    "parameters",
    "validation",
    "selection",
    "fixed_test",
    "deployment",
}
_SUMMARY_PARAMETER_FIELDS = {
    "validation_users",
    "test_users",
    "repetitions",
    "seed_stride",
    "random_state",
    "test_random_state",
    "min_interactions",
    "min_relevant_rating",
    "min_user_ratings",
    "min_book_ratings",
    "max_ratings",
    "k",
    "neighbors",
    "hybrid_candidate_pool_size",
    "hybrid_dynamic_weights",
    "positive_rating_threshold",
    "hybrid_weights",
    "catalog_configs",
    "baseline_config_id",
    "selection_metric",
    "min_delta",
    "min_selection_rate",
    "min_content_retention_delta",
    "min_personalization_retention_delta",
    "max_latency_ratio",
    "max_serialized_model_ratio",
    "confidence_level",
    "selection_confidence_level",
    "selection_multiplicity_adjustment",
    "bootstrap_resamples",
}
_SUMMARY_AGGREGATE_FIELD_ORDER = (
    "config_id",
    *HYBRID_WEIGHT_FIELDS,
    "repetitions",
    "sampled_holdouts",
    "conditional_holdouts",
    "end_to_end_metrics",
    "conditional_metrics",
    *CATALOG_CONFIG_FIELDS,
    "availability",
    "resources",
    "config_relative_diagnostics",
    "common_reference_diagnostics",
)
_SUMMARY_AGGREGATE_FIELDS = set(_SUMMARY_AGGREGATE_FIELD_ORDER)
_SUMMARY_AVAILABILITY_FIELDS = {
    "sampled_holdouts",
    "target_in_interaction_catalog",
    "interaction_retention_rate",
    "target_in_content_catalog",
    "content_retention_rate",
    "content_only_targets",
    "content_only_retention_rate",
    "history_available_holdouts",
    "history_retention_rate",
    "positive_seed_available_holdouts",
    "positive_seed_retention_rate",
    "personalization_evaluable_holdouts",
    "personalization_retention_rate",
    "full_recommendation_lists",
    "full_list_rate",
    "empty_recommendation_lists",
    "empty_list_rate",
    "mean_list_fill_rate",
    "status_counts",
}
_SUMMARY_SELECTION_FIELDS = {
    "partition",
    "scope",
    "metric",
    "baseline_config_id",
    "selected_config_id",
    "decision",
    "min_delta",
    "min_selection_rate",
    "min_content_retention_delta",
    "min_personalization_retention_delta",
    "max_latency_ratio",
    "max_serialized_model_ratio",
    "confidence_level",
    "multiplicity_adjustment",
    "ranked_validation_config_ids",
    "candidate_evidence",
    "decision_rule",
}
_SUMMARY_CANDIDATE_FIELDS = {
    "config_id",
    "primary_metric",
    "paired_delta_lower",
    "selection_rate",
    "content_retention_delta",
    "personalization_retention_delta",
    "p95_latency_ratio",
    "serialized_model_ratio",
    "passes_quality",
    "passes_reach",
    "passes_resources",
    "passes",
}
_SUMMARY_DELTA_FIELD_ORDER = (
    "config_id",
    "end_to_end",
    "conditional",
    "primary_metric",
    "selection_rate",
)
_SUMMARY_DELTA_FIELDS = set(_SUMMARY_DELTA_FIELD_ORDER)
_RANKING_METRIC_PREFIXES = ("precision", "recall", "hit_rate", "map", "mrr", "ndcg")


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


def build_catalog_benchmark_summary(
    raw_payload: Mapping[str, object],
    *,
    raw_sha256: str,
) -> dict[str, Any]:
    """Extract a compact, source-controlled summary from a validated raw artifact.

    The caller must validate the raw artifact with ``read_catalog_benchmark_metrics``
    before calling this function.  This extractor intentionally omits validation repeats,
    per-user rows, per-split rows, and model-structure trials.
    """

    if not isinstance(raw_payload, Mapping):
        raise ValueError("Raw catalog benchmark artifact must be an object.")
    if raw_payload.get("artifact_type") != CATALOG_BENCHMARK_ARTIFACT_TYPE:
        raise ValueError("Raw catalog benchmark artifact type is incompatible.")
    if raw_payload.get("schema_version") != CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Raw catalog benchmark schema version is incompatible.")
    if raw_payload.get("protocol") != CATALOG_BENCHMARK_PROTOCOL:
        raise ValueError("Raw catalog benchmark protocol is incompatible.")

    dataset = _mapping(raw_payload.get("dataset"), "raw dataset")
    validation = _mapping(raw_payload.get("validation"), "raw validation")
    fixed_test = _mapping(raw_payload.get("fixed_test"), "raw fixed_test")
    summary = {
        "schema_version": CATALOG_BENCHMARK_SUMMARY_SCHEMA_VERSION,
        "artifact_type": CATALOG_BENCHMARK_SUMMARY_TYPE,
        "created_at": deepcopy(raw_payload.get("created_at")),
        "protocol": CATALOG_BENCHMARK_PROTOCOL,
        "relevance_unit": deepcopy(raw_payload.get("relevance_unit")),
        "split_unit": deepcopy(raw_payload.get("split_unit")),
        "algorithms": deepcopy(raw_payload.get("algorithms")),
        "source": {
            "raw_artifact_sha256": str(raw_sha256).lower(),
            "source_dataset_fingerprint": deepcopy(dataset.get("source_fingerprint")),
        },
        "parameters": deepcopy(raw_payload.get("parameters")),
        "validation": {
            "partition": deepcopy(validation.get("partition")),
            "repetitions": deepcopy(validation.get("repetitions")),
            "aggregate_by_config": _compact_aggregate_records(
                validation.get("aggregate_by_config")
            ),
            "paired_deltas_vs_baseline": _compact_delta_mapping(
                validation.get("paired_deltas_vs_baseline")
            ),
        },
        "selection": deepcopy(raw_payload.get("selection")),
        "fixed_test": {
            "partition": deepcopy(fixed_test.get("partition")),
            "evaluated_after_selection": deepcopy(
                fixed_test.get("evaluated_after_selection")
            ),
            "evaluated_config_ids": deepcopy(fixed_test.get("evaluated_config_ids")),
            "aggregate_by_config": _compact_aggregate_records(
                fixed_test.get("aggregate_by_config")
            ),
            "paired_delta_selected_vs_baseline": _compact_optional_delta(
                fixed_test.get("paired_delta_selected_vs_baseline")
            ),
        },
        "deployment": deepcopy(raw_payload.get("deployment")),
    }
    return validate_catalog_benchmark_summary(summary)


def validate_catalog_benchmark_summary(
    payload: object,
    expected_algorithms: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate summary schema, numeric safety, configuration identity, and decisions."""

    if not isinstance(payload, Mapping):
        raise ValueError("Catalog benchmark summary root must be an object.")
    if set(payload) != _SUMMARY_FIELDS:
        raise ValueError("Catalog benchmark summary root has an invalid schema.")
    _validate_finite_json(payload, "catalog benchmark summary")
    if payload.get("schema_version") != CATALOG_BENCHMARK_SUMMARY_SCHEMA_VERSION:
        raise ValueError("Catalog benchmark summary schema version is incompatible.")
    if payload.get("artifact_type") != CATALOG_BENCHMARK_SUMMARY_TYPE:
        raise ValueError("Catalog benchmark summary artifact type is incompatible.")
    if payload.get("protocol") != CATALOG_BENCHMARK_PROTOCOL:
        raise ValueError("Catalog benchmark summary protocol is incompatible.")
    if payload.get("relevance_unit") != "canonical_work" or payload.get("split_unit") != "user_work":
        raise ValueError("Catalog benchmark summary evaluation units are incompatible.")
    _validate_created_at(payload.get("created_at"))

    algorithms = _validate_summary_algorithms(payload.get("algorithms"))
    if expected_algorithms is not None and algorithms != dict(expected_algorithms):
        raise ValueError("Catalog benchmark summary algorithm contract is incompatible.")
    source = _mapping(payload.get("source"), "summary source")
    if set(source) != {"raw_artifact_sha256", "source_dataset_fingerprint"}:
        raise ValueError("Catalog benchmark summary source has an invalid schema.")
    for field in ("raw_artifact_sha256", "source_dataset_fingerprint"):
        if not _is_sha256(source.get(field)):
            raise ValueError(f"Catalog benchmark summary source.{field} must be a SHA-256 digest.")

    parameters, configs_by_id, weights, selection_metric = _validate_summary_parameters(
        payload.get("parameters")
    )
    validation = _validate_summary_validation(
        payload.get("validation"),
        parameters=parameters,
        configs_by_id=configs_by_id,
        weights=weights,
        selection_metric=selection_metric,
    )
    selection = _validate_summary_selection(
        payload.get("selection"),
        parameters=parameters,
        configs_by_id=configs_by_id,
        validation=validation,
        selection_metric=selection_metric,
    )
    fixed_test = _validate_summary_fixed_test(
        payload.get("fixed_test"),
        parameters=parameters,
        configs_by_id=configs_by_id,
        weights=weights,
        selection=selection,
        selection_metric=selection_metric,
    )
    _validate_summary_deployment(
        payload.get("deployment"),
        parameters=parameters,
        configs_by_id=configs_by_id,
        selection=selection,
        fixed_test=fixed_test,
        selection_metric=selection_metric,
    )
    return deepcopy(dict(payload))


def read_catalog_benchmark_summary(
    path: str | Path,
    expected_algorithms: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Read and validate a compact catalog benchmark summary JSON file."""

    summary_path = Path(path)
    try:
        payload = json.loads(
            summary_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read catalog benchmark summary {summary_path}: {exc}") from exc
    return validate_catalog_benchmark_summary(payload, expected_algorithms=expected_algorithms)


def _compact_aggregate_records(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Raw catalog benchmark aggregate must be a non-empty array.")
    compact = []
    for index, raw_record in enumerate(value):
        record = _mapping(raw_record, f"raw aggregate[{index}]")
        missing = _SUMMARY_AGGREGATE_FIELDS.difference(record)
        if missing:
            raise ValueError(f"Raw catalog benchmark aggregate[{index}] is missing {sorted(missing)}.")
        resources = _mapping(record["resources"], f"raw aggregate[{index}].resources")
        compact_resources = {
            phase: {
                field: deepcopy(_mapping(resources.get(phase), phase).get(field))
                for field in ("trial_count", "median_seconds", "p95_seconds")
            }
            for phase in ("preparation", "fit")
        }
        compact_resources["warm_inference"] = deepcopy(resources.get("warm_inference"))
        compact_resources["model_proxy"] = deepcopy(resources.get("model_proxy"))
        compact.append(
            {
                field: deepcopy(record[field])
                for field in _SUMMARY_AGGREGATE_FIELD_ORDER
                if field != "resources"
            }
            | {"resources": compact_resources}
        )
    return compact


def _compact_delta_mapping(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("Raw catalog benchmark paired deltas must be an object.")
    return {
        str(config_id): _compact_delta(delta, f"paired delta {config_id}")
        for config_id, delta in value.items()
    }


def _compact_optional_delta(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    return _compact_delta(value, "fixed-test paired delta")


def _compact_delta(value: object, name: str) -> dict[str, Any]:
    delta = _mapping(value, name)
    missing = _SUMMARY_DELTA_FIELDS.difference(delta)
    if missing:
        raise ValueError(f"{name} is missing {sorted(missing)}.")
    return {field: deepcopy(delta[field]) for field in _SUMMARY_DELTA_FIELD_ORDER}


def _validate_summary_parameters(
    value: object,
) -> tuple[dict[str, Any], dict[str, dict[str, str | int]], dict[str, float], str]:
    parameters = _mapping(value, "summary parameters")
    if set(parameters) != _SUMMARY_PARAMETER_FIELDS:
        raise ValueError("Catalog benchmark summary parameters do not contain the full schema.")

    for field in (
        "validation_users",
        "test_users",
        "repetitions",
        "seed_stride",
        "min_interactions",
        "min_user_ratings",
        "min_book_ratings",
        "k",
        "neighbors",
        "hybrid_candidate_pool_size",
        "bootstrap_resamples",
    ):
        _positive_integer(parameters[field], f"parameters.{field}")
    if int(parameters["repetitions"]) < 2:
        raise ValueError("Catalog benchmark summary requires at least two validation repetitions.")
    for field in ("random_state", "test_random_state"):
        _integer(parameters[field], f"parameters.{field}")
    max_ratings = parameters["max_ratings"]
    if max_ratings is not None:
        _positive_integer(max_ratings, "parameters.max_ratings")
    for field in ("min_relevant_rating", "positive_rating_threshold"):
        rating = _finite_number(parameters[field], f"parameters.{field}")
        if not 0.0 <= rating <= 10.0:
            raise ValueError(f"parameters.{field} must be between 0 and 10.")
    if not isinstance(parameters["hybrid_dynamic_weights"], bool):
        raise ValueError("parameters.hybrid_dynamic_weights must be boolean.")

    raw_weights = _mapping(parameters["hybrid_weights"], "parameters.hybrid_weights")
    if set(raw_weights) != set(HYBRID_WEIGHT_FIELDS):
        raise ValueError("parameters.hybrid_weights has an invalid schema.")
    weights = normalize_hybrid_weights(*(raw_weights[field] for field in HYBRID_WEIGHT_FIELDS))
    if any(not _numbers_match(raw_weights[field], weights[field]) for field in HYBRID_WEIGHT_FIELDS):
        raise ValueError("parameters.hybrid_weights must already be normalized.")

    configs_by_id = catalog_config_registry(parameters["catalog_configs"])
    if list(parameters["catalog_configs"]) != list(configs_by_id.values()):
        raise ValueError("parameters.catalog_configs must use canonical deterministic order.")
    baseline_config_id = parameters["baseline_config_id"]
    if not isinstance(baseline_config_id, str) or baseline_config_id not in configs_by_id:
        raise ValueError("parameters.baseline_config_id references an invalid configuration.")

    selection_name = parameters["selection_metric"]
    if selection_name not in set(_RANKING_METRIC_PREFIXES):
        raise ValueError("parameters.selection_metric is invalid.")
    selection_metric = f"{selection_name}_at_{int(parameters['k'])}"
    _finite_number(parameters["min_delta"], "parameters.min_delta")
    _fraction(parameters["min_selection_rate"], "parameters.min_selection_rate")
    for field in ("min_content_retention_delta", "min_personalization_retention_delta"):
        delta = _finite_number(parameters[field], f"parameters.{field}")
        if not -1.0 <= delta <= 1.0:
            raise ValueError(f"parameters.{field} must be between -1 and 1.")
    for field in ("max_latency_ratio", "max_serialized_model_ratio"):
        if _finite_number(parameters[field], f"parameters.{field}") <= 0.0:
            raise ValueError(f"parameters.{field} must be positive.")
    for field in ("confidence_level", "selection_confidence_level"):
        confidence = _finite_number(parameters[field], f"parameters.{field}")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"parameters.{field} must be strictly between 0 and 1.")
    multiplicity_adjustment = parameters["selection_multiplicity_adjustment"]
    if multiplicity_adjustment not in {
        "none",
        "bonferroni_familywise",
    }:
        raise ValueError("parameters.selection_multiplicity_adjustment is invalid.")
    challenger_count = len(configs_by_id) - 1
    expected_adjustment = "bonferroni_familywise" if challenger_count > 1 else "none"
    if multiplicity_adjustment != expected_adjustment:
        raise ValueError(
            "parameters.selection_multiplicity_adjustment is inconsistent with challenger count."
        )
    base_confidence = float(parameters["confidence_level"])
    expected_selection_confidence = (
        1.0 - (1.0 - base_confidence) / challenger_count
        if multiplicity_adjustment == "bonferroni_familywise"
        else base_confidence
    )
    if not _numbers_match(
        parameters["selection_confidence_level"], expected_selection_confidence
    ):
        raise ValueError(
            "parameters.selection_confidence_level is inconsistent with multiplicity adjustment."
        )
    return dict(parameters), configs_by_id, weights, selection_metric


def _validate_summary_validation(
    value: object,
    *,
    parameters: Mapping[str, object],
    configs_by_id: Mapping[str, Mapping[str, object]],
    weights: Mapping[str, float],
    selection_metric: str,
) -> dict[str, Any]:
    validation = _mapping(value, "summary validation")
    if set(validation) != {
        "partition",
        "repetitions",
        "aggregate_by_config",
        "paired_deltas_vs_baseline",
    }:
        raise ValueError("Catalog benchmark summary validation has an invalid schema.")
    if validation["partition"] != "validation" or validation["repetitions"] != parameters[
        "repetitions"
    ]:
        raise ValueError("Catalog benchmark summary validation partition is inconsistent.")
    aggregates = _validate_summary_aggregates(
        validation["aggregate_by_config"],
        name="validation.aggregate_by_config",
        expected_ids=set(configs_by_id),
        configs_by_id=configs_by_id,
        weights=weights,
        expected_repetitions=int(parameters["repetitions"]),
        expected_sampled=int(parameters["validation_users"]) * int(parameters["repetitions"]),
        k=int(parameters["k"]),
    )
    challenger_ids = set(configs_by_id).difference({str(parameters["baseline_config_id"])})
    deltas = _validate_summary_delta_mapping(
        validation["paired_deltas_vs_baseline"],
        expected_ids=challenger_ids,
        aggregates=aggregates,
        baseline_config_id=str(parameters["baseline_config_id"]),
        expected_sample_size=int(parameters["validation_users"])
        * int(parameters["repetitions"]),
        selection_metric=selection_metric,
        k=int(parameters["k"]),
        confidence_level=float(parameters["selection_confidence_level"]),
        bootstrap_resamples=int(parameters["bootstrap_resamples"]),
    )
    return {"aggregate_by_id": aggregates, "deltas": deltas}


def _validate_summary_aggregates(
    value: object,
    *,
    name: str,
    expected_ids: set[str],
    configs_by_id: Mapping[str, Mapping[str, object]],
    weights: Mapping[str, float],
    expected_repetitions: int,
    expected_sampled: int,
    k: int,
) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Catalog benchmark summary {name} must be a non-empty array.")
    records: dict[str, Mapping[str, object]] = {}
    cohort_hashes = set()
    for index, raw_record in enumerate(value):
        record = _mapping(raw_record, f"{name}[{index}]")
        if set(record) != _SUMMARY_AGGREGATE_FIELDS:
            raise ValueError(f"Catalog benchmark summary {name}[{index}] has an invalid schema.")
        config_id = record["config_id"]
        if not isinstance(config_id, str) or config_id not in expected_ids or config_id in records:
            raise ValueError(f"Catalog benchmark summary {name} has invalid configuration IDs.")
        config = configs_by_id[config_id]
        for field in CATALOG_CONFIG_FIELDS:
            if record[field] != config[field]:
                raise ValueError(f"Catalog benchmark summary {name} catalog sizes are inconsistent.")
        for field in HYBRID_WEIGHT_FIELDS:
            if not _numbers_match(record[field], weights[field]):
                raise ValueError(f"Catalog benchmark summary {name} hybrid weights are inconsistent.")
        if record["repetitions"] != expected_repetitions:
            raise ValueError(f"Catalog benchmark summary {name} repetition count is inconsistent.")
        sampled = _positive_integer(record["sampled_holdouts"], f"{name}.sampled_holdouts")
        if sampled != expected_sampled:
            raise ValueError(f"Catalog benchmark summary {name} sampled holdouts are inconsistent.")
        conditional = _non_negative_integer(
            record["conditional_holdouts"], f"{name}.conditional_holdouts"
        )
        if conditional > sampled:
            raise ValueError(f"Catalog benchmark summary {name} conditional count exceeds sampled.")
        _validate_summary_metric_mapping(record["end_to_end_metrics"], k=k, name=name)
        _validate_summary_metric_mapping(record["conditional_metrics"], k=k, name=name)
        _validate_summary_availability(record["availability"], sampled=sampled, name=name)
        resources = _validate_summary_resources(
            record["resources"], expected_trials=expected_repetitions, name=name
        )
        if resources["warm_inference"]["measured_calls"] != sampled:
            raise ValueError(f"Catalog benchmark summary {name} latency cohort is inconsistent.")
        cohort_hashes.add(resources["warm_inference"]["cohort_hash"])
        _validate_summary_diagnostics(
            record["config_relative_diagnostics"], k=k, common_reference=False, name=name
        )
        _validate_summary_diagnostics(
            record["common_reference_diagnostics"], k=k, common_reference=True, name=name
        )
        records[config_id] = record
    if set(records) != expected_ids:
        raise ValueError(f"Catalog benchmark summary {name} does not cover all configurations.")
    if len(cohort_hashes) != 1:
        raise ValueError(f"Catalog benchmark summary {name} does not share one latency cohort.")
    return records


def _validate_summary_metric_mapping(value: object, *, k: int, name: str) -> None:
    metrics = _mapping(value, f"{name}.metrics")
    expected = {f"{prefix}_at_{k}" for prefix in _RANKING_METRIC_PREFIXES}
    if set(metrics) != expected:
        raise ValueError(f"Catalog benchmark summary {name} has incompatible ranking metrics.")
    for metric, value in metrics.items():
        number = _finite_number(value, f"{name}.{metric}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"Catalog benchmark summary {name}.{metric} must be between 0 and 1.")


def _validate_summary_availability(value: object, *, sampled: int, name: str) -> None:
    availability = _mapping(value, f"{name}.availability")
    if set(availability) != _SUMMARY_AVAILABILITY_FIELDS:
        raise ValueError(f"Catalog benchmark summary {name}.availability has an invalid schema.")
    if availability["sampled_holdouts"] != sampled:
        raise ValueError(f"Catalog benchmark summary {name}.availability sampled count is inconsistent.")
    count_rate_pairs = (
        ("target_in_interaction_catalog", "interaction_retention_rate"),
        ("target_in_content_catalog", "content_retention_rate"),
        ("content_only_targets", "content_only_retention_rate"),
        ("history_available_holdouts", "history_retention_rate"),
        ("positive_seed_available_holdouts", "positive_seed_retention_rate"),
        ("personalization_evaluable_holdouts", "personalization_retention_rate"),
        ("full_recommendation_lists", "full_list_rate"),
        ("empty_recommendation_lists", "empty_list_rate"),
    )
    counts = {}
    for count_field, rate_field in count_rate_pairs:
        count = _non_negative_integer(availability[count_field], f"{name}.{count_field}")
        rate = _fraction(availability[rate_field], f"{name}.{rate_field}")
        if count > sampled or not _numbers_match(rate, count / sampled):
            raise ValueError(f"Catalog benchmark summary {name}.{rate_field} is inconsistent.")
        counts[count_field] = count
    if counts["target_in_interaction_catalog"] > counts["target_in_content_catalog"]:
        raise ValueError(f"Catalog benchmark summary {name} interaction reach exceeds content reach.")
    if counts["content_only_targets"] != (
        counts["target_in_content_catalog"] - counts["target_in_interaction_catalog"]
    ):
        raise ValueError(f"Catalog benchmark summary {name} content-only reach is inconsistent.")
    if counts["personalization_evaluable_holdouts"] > min(
        counts["target_in_content_catalog"],
        counts["history_available_holdouts"],
        counts["positive_seed_available_holdouts"],
    ):
        raise ValueError(f"Catalog benchmark summary {name} personalization reach is inconsistent.")
    _fraction(availability["mean_list_fill_rate"], f"{name}.mean_list_fill_rate")
    statuses = _mapping(availability["status_counts"], f"{name}.status_counts")
    if not statuses or any(not isinstance(status, str) or not status for status in statuses):
        raise ValueError(f"Catalog benchmark summary {name}.status_counts is invalid.")
    status_total = sum(
        _non_negative_integer(count, f"{name}.status_counts.{status}")
        for status, count in statuses.items()
    )
    if status_total != sampled:
        raise ValueError(f"Catalog benchmark summary {name}.status_counts is inconsistent.")


def _validate_summary_resources(
    value: object,
    *,
    expected_trials: int,
    name: str,
) -> dict[str, Any]:
    resources = _mapping(value, f"{name}.resources")
    if set(resources) != {"preparation", "fit", "warm_inference", "model_proxy"}:
        raise ValueError(f"Catalog benchmark summary {name}.resources has an invalid schema.")
    result: dict[str, Any] = {}
    for phase in ("preparation", "fit"):
        timing = _mapping(resources[phase], f"{name}.resources.{phase}")
        if set(timing) != {"trial_count", "median_seconds", "p95_seconds"}:
            raise ValueError(f"Catalog benchmark summary {name}.{phase} has an invalid schema.")
        trials = _positive_integer(timing["trial_count"], f"{name}.{phase}.trial_count")
        median = _finite_non_negative_number(timing["median_seconds"], f"{name}.{phase}.median")
        p95 = _finite_non_negative_number(timing["p95_seconds"], f"{name}.{phase}.p95")
        if trials != expected_trials or median > p95:
            raise ValueError(f"Catalog benchmark summary {name}.{phase} is inconsistent.")
        result[phase] = dict(timing)
    warm = _validate_warm_inference(resources["warm_inference"])
    if warm["warmup_calls"] != expected_trials:
        raise ValueError(f"Catalog benchmark summary {name} warmup count is inconsistent.")
    result["warm_inference"] = warm
    result["model_proxy"] = _validate_model_proxy(resources["model_proxy"])
    return result


def _validate_summary_diagnostics(
    value: object,
    *,
    k: int,
    common_reference: bool,
    name: str,
) -> None:
    diagnostics = _mapping(value, f"{name}.diagnostics")
    expected = {
        f"catalog_coverage_at_{k}",
        f"intra_list_diversity_at_{k}",
        f"novelty_at_{k}",
        f"top_10pct_popularity_share_at_{k}",
        f"top_1pct_popularity_share_at_{k}",
        f"top_5pct_popularity_share_at_{k}",
    }
    if common_reference:
        expected.update({"coverage_vs_common_reference_at_k", "unique_recommended_works_at_k"})
    if set(diagnostics) != expected:
        raise ValueError(f"Catalog benchmark summary {name}.diagnostics has an invalid schema.")
    for metric, raw_value in diagnostics.items():
        number = _finite_non_negative_number(raw_value, f"{name}.{metric}")
        if "novelty" not in metric and "unique_recommended" not in metric and number > 1.0:
            raise ValueError(f"Catalog benchmark summary {name}.{metric} must be at most 1.")


def _validate_summary_delta_mapping(
    value: object,
    *,
    expected_ids: set[str],
    aggregates: Mapping[str, Mapping[str, object]],
    baseline_config_id: str,
    expected_sample_size: int,
    selection_metric: str,
    k: int,
    confidence_level: float,
    bootstrap_resamples: int,
) -> dict[str, Mapping[str, object]]:
    deltas = _mapping(value, "summary validation paired deltas")
    if set(deltas) != expected_ids:
        raise ValueError("Catalog benchmark summary paired deltas are incomplete.")
    validated = {}
    for config_id, raw_delta in deltas.items():
        delta = _validate_summary_delta(
            raw_delta,
            expected_config_id=config_id,
            baseline_aggregate=aggregates[baseline_config_id],
            candidate_aggregate=aggregates[config_id],
            expected_sample_size=expected_sample_size,
            selection_metric=selection_metric,
            k=k,
            confidence_level=confidence_level,
            bootstrap_resamples=bootstrap_resamples,
        )
        validated[config_id] = delta
    return validated


def _validate_summary_delta(
    value: object,
    *,
    expected_config_id: str,
    baseline_aggregate: Mapping[str, object],
    candidate_aggregate: Mapping[str, object],
    expected_sample_size: int,
    selection_metric: str,
    k: int,
    confidence_level: float,
    bootstrap_resamples: int,
) -> Mapping[str, object]:
    delta = _mapping(value, f"paired delta {expected_config_id}")
    if set(delta) != _SUMMARY_DELTA_FIELDS or delta["config_id"] != expected_config_id:
        raise ValueError(f"Catalog benchmark summary paired delta {expected_config_id} is invalid.")
    if delta["primary_metric"] != selection_metric:
        raise ValueError("Catalog benchmark summary paired delta primary metric is inconsistent.")
    _fraction(delta["selection_rate"], f"paired delta {expected_config_id}.selection_rate")
    expected_metrics = {f"{prefix}_at_{k}" for prefix in _RANKING_METRIC_PREFIXES}
    for scope in ("end_to_end", "conditional"):
        intervals = _mapping(delta[scope], f"paired delta {expected_config_id}.{scope}")
        if set(intervals) != expected_metrics:
            raise ValueError("Catalog benchmark summary paired delta metrics are incomplete.")
        scope_sample_sizes = set()
        for metric, raw_interval in intervals.items():
            interval = _mapping(raw_interval, f"paired delta {expected_config_id}.{metric}")
            if set(interval) != {
                "mean",
                "lower",
                "upper",
                "confidence_level",
                "sample_size",
                "n_resamples",
            }:
                raise ValueError("Catalog benchmark summary paired interval has an invalid schema.")
            mean = _finite_number(interval["mean"], f"paired delta {metric}.mean")
            lower = _finite_number(interval["lower"], f"paired delta {metric}.lower")
            upper = _finite_number(interval["upper"], f"paired delta {metric}.upper")
            if not (-1.0 <= lower <= mean <= upper <= 1.0):
                raise ValueError("Catalog benchmark summary paired interval is out of order.")
            if not _numbers_match(interval["confidence_level"], confidence_level):
                raise ValueError("Catalog benchmark summary paired confidence is inconsistent.")
            sample_size = _positive_integer(
                interval["sample_size"], f"paired delta {metric}.sample_size"
            )
            scope_sample_sizes.add(sample_size)
            if scope == "end_to_end":
                expected_mean = (
                    float(candidate_aggregate["end_to_end_metrics"][metric])
                    - float(baseline_aggregate["end_to_end_metrics"][metric])
                )
                if not _numbers_match(mean, expected_mean):
                    raise ValueError(
                        "Catalog benchmark summary end-to-end paired mean is inconsistent "
                        "with aggregates."
                    )
                if sample_size != expected_sample_size:
                    raise ValueError(
                        "Catalog benchmark summary end-to-end paired sample size is inconsistent."
                    )
            else:
                maximum_conditional = min(
                    int(baseline_aggregate["conditional_holdouts"]),
                    int(candidate_aggregate["conditional_holdouts"]),
                    expected_sample_size,
                )
                if sample_size > maximum_conditional:
                    raise ValueError(
                        "Catalog benchmark summary conditional paired sample size is invalid."
                    )
            if _positive_integer(
                interval["n_resamples"], f"paired delta {metric}.n_resamples"
            ) != bootstrap_resamples:
                raise ValueError("Catalog benchmark summary bootstrap count is inconsistent.")
        if len(scope_sample_sizes) != 1:
            raise ValueError(
                f"Catalog benchmark summary {scope} paired sample sizes are inconsistent."
            )
    return delta


def _validate_summary_selection(
    value: object,
    *,
    parameters: Mapping[str, object],
    configs_by_id: Mapping[str, Mapping[str, object]],
    validation: Mapping[str, object],
    selection_metric: str,
) -> dict[str, Any]:
    selection = _mapping(value, "summary selection")
    if set(selection) != _SUMMARY_SELECTION_FIELDS:
        raise ValueError("Catalog benchmark summary selection has an invalid schema.")
    baseline_id = str(parameters["baseline_config_id"])
    if (
        selection["partition"] != "validation"
        or selection["scope"] != "end_to_end"
        or selection["metric"] != selection_metric
        or selection["baseline_config_id"] != baseline_id
    ):
        raise ValueError("Catalog benchmark summary selection is not validation-only.")
    parameter_links = {
        "min_delta": "min_delta",
        "min_selection_rate": "min_selection_rate",
        "min_content_retention_delta": "min_content_retention_delta",
        "min_personalization_retention_delta": "min_personalization_retention_delta",
        "max_latency_ratio": "max_latency_ratio",
        "max_serialized_model_ratio": "max_serialized_model_ratio",
        "confidence_level": "selection_confidence_level",
    }
    for selection_field, parameter_field in parameter_links.items():
        if not _numbers_match(selection[selection_field], parameters[parameter_field]):
            raise ValueError(
                f"Catalog benchmark summary selection.{selection_field} is inconsistent."
            )
    if selection["multiplicity_adjustment"] != parameters[
        "selection_multiplicity_adjustment"
    ]:
        raise ValueError("Catalog benchmark summary selection multiplicity is inconsistent.")

    aggregates = validation["aggregate_by_id"]
    ranked_ids = sorted(
        configs_by_id,
        key=lambda config_id: (
            -float(aggregates[config_id]["end_to_end_metrics"][selection_metric]),
            int(aggregates[config_id]["max_content_books"]),
            config_id,
        ),
    )
    if selection["ranked_validation_config_ids"] != ranked_ids:
        raise ValueError("Catalog benchmark summary validation ranking is inconsistent.")
    challenger_ids = [config_id for config_id in ranked_ids if config_id != baseline_id]
    evidence_list = selection["candidate_evidence"]
    if not isinstance(evidence_list, list) or len(evidence_list) != len(challenger_ids):
        raise ValueError("Catalog benchmark summary candidate evidence is incomplete.")

    baseline = aggregates[baseline_id]
    selected_id = baseline_id
    for config_id, raw_evidence in zip(challenger_ids, evidence_list):
        evidence = _mapping(raw_evidence, f"candidate evidence {config_id}")
        if set(evidence) != _SUMMARY_CANDIDATE_FIELDS or evidence["config_id"] != config_id:
            raise ValueError("Catalog benchmark summary candidate evidence has an invalid schema.")
        candidate = aggregates[config_id]
        delta = validation["deltas"][config_id]
        interval = delta["end_to_end"][selection_metric]
        expected_numbers = {
            "primary_metric": candidate["end_to_end_metrics"][selection_metric],
            "paired_delta_lower": interval["lower"],
            "selection_rate": delta["selection_rate"],
            "content_retention_delta": candidate["availability"]["content_retention_rate"]
            - baseline["availability"]["content_retention_rate"],
            "personalization_retention_delta": candidate["availability"]
            ["personalization_retention_rate"]
            - baseline["availability"]["personalization_retention_rate"],
            "p95_latency_ratio": _summary_resource_ratio(
                candidate["resources"]["warm_inference"]["p95_ms"],
                baseline["resources"]["warm_inference"]["p95_ms"],
            ),
            "serialized_model_ratio": _summary_resource_ratio(
                candidate["resources"]["model_proxy"]["serialized_model_bytes"],
                baseline["resources"]["model_proxy"]["serialized_model_bytes"],
            ),
        }
        if any(not _numbers_match(evidence[field], expected) for field, expected in expected_numbers.items()):
            raise ValueError("Catalog benchmark summary candidate evidence numbers are inconsistent.")
        passes_quality = (
            float(expected_numbers["paired_delta_lower"]) > float(parameters["min_delta"])
            and float(expected_numbers["selection_rate"])
            >= float(parameters["min_selection_rate"])
        )
        passes_reach = (
            float(expected_numbers["content_retention_delta"])
            >= float(parameters["min_content_retention_delta"])
            and float(expected_numbers["personalization_retention_delta"])
            >= float(parameters["min_personalization_retention_delta"])
        )
        passes_resources = (
            float(expected_numbers["p95_latency_ratio"])
            <= float(parameters["max_latency_ratio"])
            and float(expected_numbers["serialized_model_ratio"])
            <= float(parameters["max_serialized_model_ratio"])
        )
        expected_flags = {
            "passes_quality": passes_quality,
            "passes_reach": passes_reach,
            "passes_resources": passes_resources,
            "passes": passes_quality and passes_reach and passes_resources,
        }
        if any(evidence[field] is not expected for field, expected in expected_flags.items()):
            raise ValueError("Catalog benchmark summary candidate promotion gates are inconsistent.")
        if selected_id == baseline_id and expected_flags["passes"]:
            selected_id = config_id

    expected_decision = "promote_to_fixed_test" if selected_id != baseline_id else "retain_baseline"
    if selection["selected_config_id"] != selected_id or selection["decision"] != expected_decision:
        raise ValueError("Catalog benchmark summary selection decision is inconsistent.")
    if not isinstance(selection["decision_rule"], str) or not selection["decision_rule"].strip():
        raise ValueError("Catalog benchmark summary selection decision rule must be documented.")
    return dict(selection)


def _validate_summary_fixed_test(
    value: object,
    *,
    parameters: Mapping[str, object],
    configs_by_id: Mapping[str, Mapping[str, object]],
    weights: Mapping[str, float],
    selection: Mapping[str, object],
    selection_metric: str,
) -> dict[str, Any]:
    fixed_test = _mapping(value, "summary fixed_test")
    if set(fixed_test) != {
        "partition",
        "evaluated_after_selection",
        "evaluated_config_ids",
        "aggregate_by_config",
        "paired_delta_selected_vs_baseline",
    }:
        raise ValueError("Catalog benchmark summary fixed_test has an invalid schema.")
    baseline_id = str(parameters["baseline_config_id"])
    selected_id = str(selection["selected_config_id"])
    expected_ids = [baseline_id] + ([selected_id] if selected_id != baseline_id else [])
    if (
        fixed_test["partition"] != "test"
        or fixed_test["evaluated_after_selection"] is not True
        or fixed_test["evaluated_config_ids"] != expected_ids
    ):
        raise ValueError("Catalog benchmark summary fixed test violates selection isolation.")
    aggregates = _validate_summary_aggregates(
        fixed_test["aggregate_by_config"],
        name="fixed_test.aggregate_by_config",
        expected_ids=set(expected_ids),
        configs_by_id=configs_by_id,
        weights=weights,
        expected_repetitions=1,
        expected_sampled=int(parameters["test_users"]),
        k=int(parameters["k"]),
    )
    raw_delta = fixed_test["paired_delta_selected_vs_baseline"]
    if selected_id == baseline_id:
        if raw_delta is not None:
            raise ValueError("Catalog benchmark summary fixed test exposes an unselected challenger.")
        delta = None
    else:
        delta = _validate_summary_delta(
            raw_delta,
            expected_config_id=selected_id,
            baseline_aggregate=aggregates[baseline_id],
            candidate_aggregate=aggregates[selected_id],
            expected_sample_size=int(parameters["test_users"]),
            selection_metric=selection_metric,
            k=int(parameters["k"]),
            confidence_level=float(parameters["confidence_level"]),
            bootstrap_resamples=int(parameters["bootstrap_resamples"]),
        )
    return {"aggregate_by_id": aggregates, "delta": delta}


def _validate_summary_deployment(
    value: object,
    *,
    parameters: Mapping[str, object],
    configs_by_id: Mapping[str, Mapping[str, object]],
    selection: Mapping[str, object],
    fixed_test: Mapping[str, object],
    selection_metric: str,
) -> None:
    deployment = _mapping(value, "summary deployment")
    baseline_id = str(parameters["baseline_config_id"])
    selected_id = str(selection["selected_config_id"])
    if deployment.get("partition") != "fixed_test":
        raise ValueError("Catalog benchmark summary deployment must use fixed test evidence.")
    if selected_id == baseline_id:
        if set(deployment) != {"partition", "deployment_config_id", "decision", "reason"}:
            raise ValueError("Catalog benchmark summary baseline deployment has an invalid schema.")
        if (
            deployment["deployment_config_id"] != baseline_id
            or deployment["decision"] != "retain_baseline"
        ):
            raise ValueError("Catalog benchmark summary baseline deployment is inconsistent.")
    else:
        expected_fields = {
            "partition",
            "validation_selected_config_id",
            "deployment_config_id",
            "decision",
            "metric",
            "paired_delta_mean",
            "paired_delta_lower",
            "min_delta",
            "passes",
            "reason",
        }
        if set(deployment) != expected_fields:
            raise ValueError("Catalog benchmark summary challenger deployment has an invalid schema.")
        interval = fixed_test["delta"]["end_to_end"][selection_metric]
        passes = float(interval["lower"]) > float(parameters["min_delta"])
        expected_deployment_id = selected_id if passes else baseline_id
        expected_decision = "promote" if passes else "retain_baseline_after_test"
        if (
            deployment["validation_selected_config_id"] != selected_id
            or deployment["deployment_config_id"] != expected_deployment_id
            or deployment["decision"] != expected_decision
            or deployment["metric"] != selection_metric
            or not _numbers_match(deployment["paired_delta_mean"], interval["mean"])
            or not _numbers_match(deployment["paired_delta_lower"], interval["lower"])
            or not _numbers_match(deployment["min_delta"], parameters["min_delta"])
            or deployment["passes"] is not passes
        ):
            raise ValueError("Catalog benchmark summary challenger deployment is inconsistent.")
    if deployment.get("deployment_config_id") not in configs_by_id:
        raise ValueError("Catalog benchmark summary deployment references an invalid configuration.")
    if not isinstance(deployment.get("reason"), str) or not deployment["reason"].strip():
        raise ValueError("Catalog benchmark summary deployment reason must be documented.")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Catalog benchmark {name} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"Catalog benchmark {name} contains a non-string key.")
    return value


def _validate_finite_json(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{name}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} contains a non-string key.")
            _validate_finite_json(item, f"{name}.{key}")
        return
    raise ValueError(f"{name} contains a non-JSON value.")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite JSON constant: {value}")


def _validate_created_at(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Catalog benchmark summary created_at must be an ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "Catalog benchmark summary created_at must be an ISO-8601 timestamp."
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError("Catalog benchmark summary created_at must include a timezone.")


def _validate_summary_algorithms(value: object) -> dict[str, object]:
    algorithms = _mapping(value, "summary algorithms")
    if set(algorithms) != _ALGORITHM_FIELDS:
        raise ValueError("Catalog benchmark summary algorithms have an invalid schema.")
    for field in (
        "hybrid_algorithm_id",
        "collaborative_algorithm_id",
        "content_algorithm_id",
    ):
        if not isinstance(algorithms[field], str) or not algorithms[field].strip():
            raise ValueError(f"Catalog benchmark summary algorithms.{field} must be non-empty.")
    for field in (
        "hybrid_model_format_version",
        "collaborative_model_format_version",
        "content_model_format_version",
    ):
        _positive_integer(algorithms[field], f"algorithms.{field}")
    return dict(algorithms)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number.")
    return parsed


def _fraction(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return parsed


def _numbers_match(value: object, expected: object) -> bool:
    try:
        return math.isclose(
            _finite_number(value, "number"),
            _finite_number(expected, "expected number"),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except ValueError:
        return False


def _summary_resource_ratio(candidate: object, baseline: object) -> float:
    candidate_value = _finite_non_negative_number(candidate, "candidate resource")
    baseline_value = _finite_non_negative_number(baseline, "baseline resource")
    if baseline_value <= 0.0:
        return 1.0 if candidate_value <= 0.0 else 1e308
    return candidate_value / baseline_value


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
