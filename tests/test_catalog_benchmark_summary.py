from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from book_recommender.catalog_benchmark import (
    CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION,
    CATALOG_BENCHMARK_ARTIFACT_TYPE,
    CATALOG_BENCHMARK_PROTOCOL,
    CATALOG_BENCHMARK_SUMMARY_SCHEMA_VERSION,
    CATALOG_BENCHMARK_SUMMARY_TYPE,
    build_catalog_benchmark_summary,
    read_catalog_benchmark_summary,
    validate_catalog_benchmark_summary,
)
from scripts import catalog_benchmark_summary as summary_script


METRICS = ("precision", "recall", "hit_rate", "map", "mrr", "ndcg")
BASELINE = "ib2_cb3"
CHALLENGER = "ib4_cb6"
SECOND_CHALLENGER = "ib5_cb7"


def metric_mapping(value: float) -> dict[str, float]:
    return {f"{metric}_at_10": value for metric in METRICS}


def diagnostics(*, common: bool) -> dict[str, float]:
    result = {
        "catalog_coverage_at_10": 0.5,
        "intra_list_diversity_at_10": 0.7,
        "novelty_at_10": 8.0,
        "top_10pct_popularity_share_at_10": 0.5,
        "top_1pct_popularity_share_at_10": 0.1,
        "top_5pct_popularity_share_at_10": 0.3,
    }
    if common:
        result.update(
            coverage_vs_common_reference_at_k=0.4,
            unique_recommended_works_at_k=5.0,
        )
    return result


def availability(*, sampled: int, interaction: int, content: int, personal: int) -> dict:
    return {
        "sampled_holdouts": sampled,
        "target_in_interaction_catalog": interaction,
        "interaction_retention_rate": interaction / sampled,
        "target_in_content_catalog": content,
        "content_retention_rate": content / sampled,
        "content_only_targets": content - interaction,
        "content_only_retention_rate": (content - interaction) / sampled,
        "history_available_holdouts": sampled,
        "history_retention_rate": 1.0,
        "positive_seed_available_holdouts": sampled,
        "positive_seed_retention_rate": 1.0,
        "personalization_evaluable_holdouts": personal,
        "personalization_retention_rate": personal / sampled,
        "full_recommendation_lists": sampled,
        "full_list_rate": 1.0,
        "empty_recommendation_lists": 0,
        "empty_list_rate": 0.0,
        "mean_list_fill_rate": 1.0,
        "status_counts": {"personalized_rankable": personal, "fallback": sampled - personal},
    }


def resources(*, trials: int, sampled: int, latency: float, size: int) -> dict:
    return {
        "preparation": {"trial_count": trials, "median_seconds": 1.0, "p95_seconds": 1.2},
        "fit": {"trial_count": trials, "median_seconds": 0.5, "p95_seconds": 0.7},
        "warm_inference": {
            "warmup_calls": trials,
            "measured_calls": sampled,
            "cohort_hash": "c" * 64,
            "mean_ms": latency * 0.6,
            "p50_ms": latency * 0.5,
            "p95_ms": latency,
            "max_ms": latency * 1.2,
        },
        "model_proxy": {
            "serialized_model_bytes": size,
            "matrix_payload_bytes": size - 10,
            "content_matrix_bytes": size - 30,
            "collaborative_matrix_bytes": 20,
            "dense_array_bytes": 10,
        },
    }


def aggregate(
    config_id: str,
    interaction_books: int,
    content_books: int,
    *,
    repetitions: int,
    sampled: int,
    metric: float,
    target_interaction: int,
    target_content: int,
    personal: int,
    latency: float,
    size: int,
) -> dict:
    return {
        "config_id": config_id,
        "collaborative_weight": 0.4,
        "content_weight": 0.45,
        "popularity_weight": 0.15,
        "repetitions": repetitions,
        "sampled_holdouts": sampled,
        "conditional_holdouts": personal,
        "end_to_end_metrics": metric_mapping(metric),
        "conditional_metrics": metric_mapping(metric),
        "max_interaction_books": interaction_books,
        "max_content_books": content_books,
        "availability": availability(
            sampled=sampled,
            interaction=target_interaction,
            content=target_content,
            personal=personal,
        ),
        "resources": resources(
            trials=repetitions,
            sampled=sampled,
            latency=latency,
            size=size,
        ),
        "config_relative_diagnostics": diagnostics(common=False),
        "common_reference_diagnostics": diagnostics(common=True),
    }


def paired_delta(
    config_id: str,
    *,
    mean: float,
    lower: float,
    sample_size: int,
    conditional_sample_size: int,
) -> dict:
    interval = {
        "mean": mean,
        "lower": lower,
        "upper": mean + 0.05,
        "confidence_level": 0.95,
        "sample_size": sample_size,
        "n_resamples": 10,
    }
    return {
        "config_id": config_id,
        "end_to_end": {f"{metric}_at_10": deepcopy(interval) for metric in METRICS},
        "conditional": {
            f"{metric}_at_10": deepcopy(interval) | {"sample_size": conditional_sample_size}
            for metric in METRICS
        },
        "primary_metric": "ndcg_at_10",
        "selection_rate": 1.0,
    }


def valid_summary() -> dict:
    validation_aggregates = [
        aggregate(
            BASELINE,
            2,
            3,
            repetitions=2,
            sampled=4,
            metric=0.1,
            target_interaction=2,
            target_content=2,
            personal=2,
            latency=2.0,
            size=100,
        ),
        aggregate(
            CHALLENGER,
            4,
            6,
            repetitions=2,
            sampled=4,
            metric=0.2,
            target_interaction=3,
            target_content=4,
            personal=4,
            latency=3.0,
            size=200,
        ),
    ]
    fixed_aggregates = [
        aggregate(
            BASELINE,
            2,
            3,
            repetitions=1,
            sampled=2,
            metric=0.1,
            target_interaction=1,
            target_content=1,
            personal=1,
            latency=2.0,
            size=100,
        ),
        aggregate(
            CHALLENGER,
            4,
            6,
            repetitions=1,
            sampled=2,
            metric=0.09,
            target_interaction=2,
            target_content=2,
            personal=2,
            latency=3.0,
            size=200,
        ),
    ]
    validation_delta = paired_delta(
        CHALLENGER,
        mean=0.1,
        lower=0.05,
        sample_size=4,
        conditional_sample_size=2,
    )
    fixed_delta = paired_delta(
        CHALLENGER,
        mean=-0.01,
        lower=-0.03,
        sample_size=2,
        conditional_sample_size=1,
    )
    return {
        "schema_version": CATALOG_BENCHMARK_SUMMARY_SCHEMA_VERSION,
        "artifact_type": CATALOG_BENCHMARK_SUMMARY_TYPE,
        "created_at": "2026-07-17T00:00:00+00:00",
        "protocol": CATALOG_BENCHMARK_PROTOCOL,
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
        "algorithms": {
            "hybrid_algorithm_id": "hybrid-v1",
            "hybrid_model_format_version": 1,
            "collaborative_algorithm_id": "collaborative-v1",
            "collaborative_model_format_version": 1,
            "content_algorithm_id": "content-v1",
            "content_model_format_version": 1,
        },
        "source": {
            "raw_artifact_sha256": "a" * 64,
            "source_dataset_fingerprint": "b" * 64,
        },
        "parameters": {
            "validation_users": 2,
            "test_users": 2,
            "repetitions": 2,
            "seed_stride": 17,
            "random_state": 42,
            "test_random_state": 42,
            "min_interactions": 2,
            "min_relevant_rating": 1.0,
            "min_user_ratings": 2,
            "min_book_ratings": 2,
            "max_ratings": None,
            "k": 10,
            "neighbors": 3,
            "hybrid_candidate_pool_size": 20,
            "hybrid_dynamic_weights": True,
            "positive_rating_threshold": 7.0,
            "hybrid_weights": {
                "collaborative_weight": 0.4,
                "content_weight": 0.45,
                "popularity_weight": 0.15,
            },
            "catalog_configs": [
                {"config_id": BASELINE, "max_interaction_books": 2, "max_content_books": 3},
                {
                    "config_id": CHALLENGER,
                    "max_interaction_books": 4,
                    "max_content_books": 6,
                },
            ],
            "baseline_config_id": BASELINE,
            "selection_metric": "ndcg",
            "min_delta": 0.0,
            "min_selection_rate": 0.6,
            "min_content_retention_delta": 0.0,
            "min_personalization_retention_delta": 0.0,
            "max_latency_ratio": 3.0,
            "max_serialized_model_ratio": 4.0,
            "confidence_level": 0.95,
            "selection_confidence_level": 0.95,
            "selection_multiplicity_adjustment": "none",
            "bootstrap_resamples": 10,
        },
        "validation": {
            "partition": "validation",
            "repetitions": 2,
            "aggregate_by_config": validation_aggregates,
            "paired_deltas_vs_baseline": {CHALLENGER: validation_delta},
        },
        "selection": {
            "partition": "validation",
            "scope": "end_to_end",
            "metric": "ndcg_at_10",
            "baseline_config_id": BASELINE,
            "selected_config_id": CHALLENGER,
            "decision": "promote_to_fixed_test",
            "min_delta": 0.0,
            "min_selection_rate": 0.6,
            "min_content_retention_delta": 0.0,
            "min_personalization_retention_delta": 0.0,
            "max_latency_ratio": 3.0,
            "max_serialized_model_ratio": 4.0,
            "confidence_level": 0.95,
            "multiplicity_adjustment": "none",
            "ranked_validation_config_ids": [CHALLENGER, BASELINE],
            "candidate_evidence": [
                {
                    "config_id": CHALLENGER,
                    "primary_metric": 0.2,
                    "paired_delta_lower": 0.05,
                    "selection_rate": 1.0,
                    "content_retention_delta": 0.5,
                    "personalization_retention_delta": 0.5,
                    "p95_latency_ratio": 1.5,
                    "serialized_model_ratio": 2.0,
                    "passes_quality": True,
                    "passes_reach": True,
                    "passes_resources": True,
                    "passes": True,
                }
            ],
            "decision_rule": "All validation promotion gates must pass.",
        },
        "fixed_test": {
            "partition": "test",
            "evaluated_after_selection": True,
            "evaluated_config_ids": [BASELINE, CHALLENGER],
            "aggregate_by_config": fixed_aggregates,
            "paired_delta_selected_vs_baseline": fixed_delta,
        },
        "deployment": {
            "partition": "fixed_test",
            "validation_selected_config_id": CHALLENGER,
            "deployment_config_id": BASELINE,
            "decision": "retain_baseline_after_test",
            "metric": "ndcg_at_10",
            "paired_delta_mean": -0.01,
            "paired_delta_lower": -0.03,
            "min_delta": 0.0,
            "passes": False,
            "reason": "Fixed-test lower bound did not pass.",
        },
    }


def raw_from_summary(summary: dict, raw_sha256: str) -> dict:
    raw = {
        "schema_version": CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": CATALOG_BENCHMARK_ARTIFACT_TYPE,
        "created_at": summary["created_at"],
        "protocol": summary["protocol"],
        "relevance_unit": summary["relevance_unit"],
        "split_unit": summary["split_unit"],
        "algorithms": deepcopy(summary["algorithms"]),
        "parameters": deepcopy(summary["parameters"]),
        "dataset": {"source_fingerprint": summary["source"]["source_dataset_fingerprint"]},
        "selection": deepcopy(summary["selection"]),
        "deployment": deepcopy(summary["deployment"]),
        "validation": deepcopy(summary["validation"]),
        "fixed_test": deepcopy(summary["fixed_test"]),
    }
    raw["validation"].update(repeats=[{"large": "payload"}], per_user_rows=[{"user": 1}])
    raw["fixed_test"].update(configurations=[{"large": "payload"}], per_user_rows=[{"user": 1}])
    for partition in (raw["validation"], raw["fixed_test"]):
        for record in partition["aggregate_by_config"]:
            trials = record["resources"]["preparation"]["trial_count"]
            record["resources"]["preparation"]["seconds"] = [1.0] * trials
            record["resources"]["fit"]["seconds"] = [0.5] * trials
            record.update(
                per_split=[{"large": "payload"}],
                actual_catalog_trials=[{"large": "payload"}],
                model_structure_trials=[{"large": "payload"}],
            )
    summary["source"]["raw_artifact_sha256"] = raw_sha256
    return raw


def with_second_challenger(payload: dict) -> dict:
    result = deepcopy(payload)
    result["parameters"]["catalog_configs"].append(
        {
            "config_id": SECOND_CHALLENGER,
            "max_interaction_books": 5,
            "max_content_books": 7,
        }
    )
    result["parameters"]["selection_multiplicity_adjustment"] = "bonferroni_familywise"
    result["parameters"]["selection_confidence_level"] = 0.975
    result["selection"]["confidence_level"] = 0.975
    result["selection"]["multiplicity_adjustment"] = "bonferroni_familywise"
    result["selection"]["ranked_validation_config_ids"].append(SECOND_CHALLENGER)
    result["validation"]["aggregate_by_config"].append(
        aggregate(
            SECOND_CHALLENGER,
            5,
            7,
            repetitions=2,
            sampled=4,
            metric=0.05,
            target_interaction=2,
            target_content=2,
            personal=2,
            latency=2.0,
            size=100,
        )
    )
    second_delta = paired_delta(
        SECOND_CHALLENGER,
        mean=-0.05,
        lower=-0.1,
        sample_size=4,
        conditional_sample_size=2,
    )
    result["validation"]["paired_deltas_vs_baseline"][SECOND_CHALLENGER] = second_delta
    result["selection"]["candidate_evidence"].append(
        {
            "config_id": SECOND_CHALLENGER,
            "primary_metric": 0.05,
            "paired_delta_lower": -0.1,
            "selection_rate": 1.0,
            "content_retention_delta": 0.0,
            "personalization_retention_delta": 0.0,
            "p95_latency_ratio": 1.0,
            "serialized_model_ratio": 1.0,
            "passes_quality": False,
            "passes_reach": True,
            "passes_resources": True,
            "passes": False,
        }
    )
    for delta in result["validation"]["paired_deltas_vs_baseline"].values():
        for scope in ("end_to_end", "conditional"):
            for interval in delta[scope].values():
                interval["confidence_level"] = 0.975
    return result


def test_build_summary_strips_repeat_and_per_user_payloads() -> None:
    expected = valid_summary()
    raw = raw_from_summary(expected, "d" * 64)

    summary = build_catalog_benchmark_summary(raw, raw_sha256="d" * 64)

    assert summary == expected
    assert "repeats" not in summary["validation"]
    assert "per_user_rows" not in summary["validation"]
    assert "per_split" not in summary["validation"]["aggregate_by_config"][0]
    assert "seconds" not in summary["validation"]["aggregate_by_config"][0]["resources"]["fit"]
    assert "per_repeat_primary_delta" not in summary["validation"]["paired_deltas_vs_baseline"][
        CHALLENGER
    ]


def test_summary_validator_rejects_nonfinite_identity_and_gate_tampering() -> None:
    payload = valid_summary()
    payload["validation"]["aggregate_by_config"][0]["end_to_end_metrics"]["ndcg_at_10"] = float(
        "nan"
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_catalog_benchmark_summary(payload)

    payload = valid_summary()
    payload["validation"]["aggregate_by_config"][0]["config_id"] = "forged"
    with pytest.raises(ValueError, match="configuration IDs"):
        validate_catalog_benchmark_summary(payload)

    payload = valid_summary()
    payload["selection"]["candidate_evidence"][0]["passes"] = False
    with pytest.raises(ValueError, match="promotion gates"):
        validate_catalog_benchmark_summary(payload)

    payload = valid_summary()
    payload["deployment"]["deployment_config_id"] = CHALLENGER
    with pytest.raises(ValueError, match="deployment is inconsistent"):
        validate_catalog_benchmark_summary(payload)


def test_summary_validator_binds_paired_intervals_to_aggregates_and_cohorts() -> None:
    payload = valid_summary()
    payload["validation"]["paired_deltas_vs_baseline"][CHALLENGER]["end_to_end"][
        "ndcg_at_10"
    ]["mean"] = 0.11
    with pytest.raises(ValueError, match="paired mean is inconsistent"):
        validate_catalog_benchmark_summary(payload)

    payload = valid_summary()
    payload["validation"]["paired_deltas_vs_baseline"][CHALLENGER]["end_to_end"][
        "ndcg_at_10"
    ]["upper"] = 1.1
    with pytest.raises(ValueError, match="interval is out of order"):
        validate_catalog_benchmark_summary(payload)

    payload = valid_summary()
    payload["validation"]["paired_deltas_vs_baseline"][CHALLENGER]["end_to_end"][
        "ndcg_at_10"
    ]["sample_size"] = 3
    with pytest.raises(ValueError, match="end-to-end paired sample size"):
        validate_catalog_benchmark_summary(payload)

    payload = valid_summary()
    payload["validation"]["paired_deltas_vs_baseline"][CHALLENGER]["conditional"][
        "ndcg_at_10"
    ]["sample_size"] = 1
    with pytest.raises(ValueError, match="conditional paired sample sizes"):
        validate_catalog_benchmark_summary(payload)

    payload = valid_summary()
    for interval in payload["validation"]["paired_deltas_vs_baseline"][CHALLENGER][
        "conditional"
    ].values():
        interval["sample_size"] = 3
    with pytest.raises(ValueError, match="conditional paired sample size is invalid"):
        validate_catalog_benchmark_summary(payload)


def test_summary_validator_enforces_multiplicity_confidence_formula() -> None:
    payload = valid_summary()
    payload["parameters"]["selection_confidence_level"] = 0.96
    with pytest.raises(ValueError, match="selection_confidence_level"):
        validate_catalog_benchmark_summary(payload)

    payload = with_second_challenger(valid_summary())
    validate_catalog_benchmark_summary(payload)
    payload["parameters"]["selection_confidence_level"] = 0.97
    with pytest.raises(ValueError, match="selection_confidence_level"):
        validate_catalog_benchmark_summary(payload)


def test_read_summary_checks_source_and_expected_algorithms(tmp_path) -> None:
    path = tmp_path / "summary.json"
    payload = valid_summary()
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_catalog_benchmark_summary(path)["selection"]["selected_config_id"] == CHALLENGER
    with pytest.raises(ValueError, match="algorithm contract"):
        read_catalog_benchmark_summary(path, expected_algorithms={"different": "contract"})

    payload["source"]["raw_artifact_sha256"] = "not-a-digest"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        read_catalog_benchmark_summary(path)


def test_generator_validates_raw_before_atomic_summary_write(monkeypatch, tmp_path) -> None:
    output = tmp_path / "nested" / "summary.json"
    raw_path = tmp_path / "raw.json"
    expected = valid_summary()
    raw = raw_from_summary(expected, "0" * 64)
    raw_bytes = json.dumps(raw, allow_nan=False).encode("utf-8")
    raw_path.write_bytes(raw_bytes)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()

    calls = []

    def validated_reader(path):
        calls.append(path)
        return {"artifact_sha256": raw_sha}

    monkeypatch.setattr(summary_script, "read_catalog_benchmark_metrics", validated_reader)
    generated = summary_script.generate_summary(raw_path, output)

    assert calls == [raw_path]
    assert generated["source"]["raw_artifact_sha256"] == raw_sha
    assert output.exists()
    assert not list(output.parent.glob("*.tmp"))

    output.unlink()
    monkeypatch.setattr(
        summary_script,
        "read_catalog_benchmark_metrics",
        lambda _: (_ for _ in ()).throw(ValueError("invalid raw artifact")),
    )
    with pytest.raises(ValueError, match="invalid raw artifact"):
        summary_script.generate_summary(raw_path, output)
    assert not output.exists()


def test_generator_rejects_same_raw_and_output_path_without_touching_raw(monkeypatch, tmp_path) -> None:
    raw_path = tmp_path / "raw.json"
    original = b'{"raw": "must remain intact"}'
    raw_path.write_bytes(original)
    monkeypatch.setattr(
        summary_script,
        "read_catalog_benchmark_metrics",
        lambda _: pytest.fail("same-path guard must run before raw validation"),
    )

    with pytest.raises(ValueError, match="paths must be different"):
        summary_script.generate_summary(raw_path, raw_path)

    assert raw_path.read_bytes() == original
