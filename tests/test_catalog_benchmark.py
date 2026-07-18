from __future__ import annotations

from copy import deepcopy

import pytest

from book_recommender.catalog_benchmark import (
    CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION,
    CATALOG_BENCHMARK_ARTIFACT_TYPE,
    CATALOG_BENCHMARK_PROTOCOL,
    benchmark_catalog_configs,
    catalog_config_id,
    catalog_config_registry,
    normalize_hybrid_weights,
    parse_catalog_grid,
    validate_resource_metrics,
)


def valid_resources() -> dict:
    return {
        "preparation": {
            "trial_count": 2,
            "seconds": [0.1, 0.2],
            "median_seconds": 0.15,
            "p95_seconds": 0.2,
        },
        "fit": {
            "trial_count": 2,
            "seconds": [0.3, 0.5],
            "median_seconds": 0.4,
            "p95_seconds": 0.5,
        },
        "warm_inference": {
            "warmup_calls": 2,
            "measured_calls": 4,
            "cohort_hash": "a" * 64,
            "mean_ms": 1.4,
            "p50_ms": 1.2,
            "p95_ms": 1.8,
            "max_ms": 2.0,
        },
        "model_proxy": {
            "serialized_model_bytes": 100,
            "matrix_payload_bytes": 80,
            "content_matrix_bytes": 50,
            "collaborative_matrix_bytes": 30,
            "dense_array_bytes": 10,
        },
    }


def test_catalog_benchmark_contract_constants_are_explicit() -> None:
    assert CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION == 1
    assert CATALOG_BENCHMARK_ARTIFACT_TYPE == "catalog_size_benchmark"
    assert CATALOG_BENCHMARK_PROTOCOL == (
        "canonical_work_fixed_test_repeated_validation_paired_catalog_v1"
    )


def test_catalog_configs_are_order_independent_deduplicated_and_include_baseline() -> None:
    forward, forward_baseline = benchmark_catalog_configs(
        catalog_grid="3000,10000;1000,3000;3000,10000",
        baseline_catalog="5000,20000",
    )
    reverse, reverse_baseline = benchmark_catalog_configs(
        catalog_grid="1000,3000;3000,10000",
        baseline_catalog="5000,20000",
    )

    assert forward == reverse
    assert forward_baseline == reverse_baseline == "ib5000_cb20000"
    assert [config["config_id"] for config in forward] == [
        "ib1000_cb3000",
        "ib3000_cb10000",
        "ib5000_cb20000",
    ]


def test_catalog_id_and_registry_reject_forged_identity() -> None:
    config = {"max_interaction_books": 1000, "max_content_books": 3000}
    assert catalog_config_id(config) == "ib1000_cb3000"
    assert catalog_config_id(1000, 3000) == "ib1000_cb3000"

    registry = catalog_config_registry([{"config_id": "ib1000_cb3000", **config}])
    assert list(registry) == ["ib1000_cb3000"]

    with pytest.raises(ValueError, match="inconsistent config_id"):
        catalog_config_registry([{"config_id": "ib999_cb3000", **config}])
    with pytest.raises(ValueError, match="duplicate"):
        catalog_config_registry(
            [
                {"config_id": "ib1000_cb3000", **config},
                {"config_id": "ib1000_cb3000", **config},
            ]
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("1000,3000;", "empty configuration"),
        ("1000", "exactly 2"),
        ("1000,3000,5000", "exactly 2"),
        ("1000,", "exactly 2"),
        ("one,3000", "only integers"),
        ("0,3000", "positive integer"),
        ("3000,1000", "greater than or equal"),
    ],
)
def test_catalog_grid_rejects_invalid_limits(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_catalog_grid(value)


def test_baseline_must_be_one_catalog_pair() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        benchmark_catalog_configs(
            catalog_grid="1000,3000",
            baseline_catalog="1000,3000;3000,10000",
        )


def test_fixed_hybrid_weights_are_normalized() -> None:
    assert normalize_hybrid_weights(4, 4.5, 1.5) == {
        "collaborative_weight": 0.4,
        "content_weight": 0.45,
        "popularity_weight": 0.15,
    }

    with pytest.raises(ValueError, match="positive"):
        normalize_hybrid_weights(0, 0, 0)
    with pytest.raises(ValueError, match="finite"):
        normalize_hybrid_weights(float("nan"), 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        normalize_hybrid_weights(-1, 1, 1)


def test_resource_metrics_schema_accepts_consistent_portable_measurements() -> None:
    validated = validate_resource_metrics(valid_resources())

    assert validated["fit"]["trial_count"] == 2
    assert validated["warm_inference"]["measured_calls"] == 4
    assert validated["model_proxy"]["serialized_model_bytes"] == 100


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("fit"), "invalid schema"),
        (lambda value: value["fit"].update(trial_count=3), "trial_count samples"),
        (lambda value: value["fit"]["seconds"].__setitem__(0, float("inf")), "finite"),
        (lambda value: value["fit"].update(median_seconds=0.6), "out of order"),
        (lambda value: value["warm_inference"].update(measured_calls=0), "positive integer"),
        (lambda value: value["warm_inference"].update(cohort_hash="invalid"), "SHA-256"),
        (lambda value: value["warm_inference"].update(p50_ms=1.9), "out of order"),
        (
            lambda value: value["model_proxy"].update(serialized_model_bytes=-1),
            "non-negative integer",
        ),
        (
            lambda value: value["model_proxy"].update(matrix_payload_bytes=True),
            "non-negative integer",
        ),
    ],
)
def test_resource_metrics_schema_rejects_invalid_measurements(mutation, message: str) -> None:
    resources = deepcopy(valid_resources())
    mutation(resources)

    with pytest.raises(ValueError, match=message):
        validate_resource_metrics(resources)
