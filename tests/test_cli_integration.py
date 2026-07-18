from argparse import Namespace
import json

import pandas as pd
import pytest

from book_recommender import cli
from book_recommender.artifacts import load_model_bundle, manifest_path
from book_recommender.content_based import book_work_key
from book_recommender.data import BookDataset, clean_books, clean_ratings, clean_users
from book_recommender.evaluation import ValidationTestEvaluationDataset
from book_recommender.hybrid import HybridRecommender


def evaluation_dataset(user_count: int = 12) -> BookDataset:
    books = clean_books(
        pd.DataFrame(
            [
                {
                    "ISBN": f"B{index}",
                    "Book-Title": f"Book {index}",
                    "Book-Author": f"Author {index % 3}",
                    "Publisher": f"Publisher {index % 2}",
                }
                for index in range(8)
            ]
        )
    )
    rating_rows = []
    for user_id in range(1, user_count + 1):
        item_indices = [(user_id + offset) % 8 for offset in range(5)]
        scores = [10, 9, 8, 6, 4]
        rating_rows.extend(
            {
                "User-ID": user_id,
                "ISBN": f"B{item_index}",
                "Book-Rating": score,
            }
            for item_index, score in zip(item_indices, scores)
        )
    ratings = clean_ratings(pd.DataFrame(rating_rows))
    users = clean_users(
        pd.DataFrame(
            {
                "User-ID": range(1, user_count + 1),
                "Location": "x",
                "Age": 30,
            }
        )
    )
    return BookDataset(books=books, ratings=ratings, users=users)


def evaluation_args(tmp_path, output_name: str) -> Namespace:
    return Namespace(
        data_dir=tmp_path,
        sample_users=8,
        validation_fraction=0.5,
        min_interactions=3,
        min_relevant_rating=8.0,
        min_user_ratings=1,
        min_book_ratings=1,
        max_books=None,
        max_ratings=None,
        max_content_books=None,
        k=3,
        neighbors=3,
        hybrid_candidate_pool_size=8,
        hybrid_collaborative_weight=0.4,
        hybrid_content_weight=0.45,
        hybrid_popularity_weight=0.15,
        hybrid_min_content_rating=8.0,
        positive_rating_threshold=8.0,
        disable_hybrid_dynamic_weights=False,
        random_state=7,
        bootstrap_resamples=100,
        confidence_level=0.95,
        output=tmp_path / output_name,
    )


def benchmark_args(tmp_path, output_name: str) -> Namespace:
    return Namespace(
        data_dir=tmp_path,
        validation_users=4,
        test_users=4,
        repetitions=2,
        seed_stride=7,
        min_interactions=3,
        min_relevant_rating=8.0,
        min_user_ratings=1,
        min_book_ratings=1,
        max_books=None,
        max_ratings=None,
        max_content_books=None,
        k=3,
        neighbors=3,
        hybrid_candidate_pool_size=8,
        positive_rating_threshold=8.0,
        disable_hybrid_dynamic_weights=False,
        weight_grid="0.4,0.45,0.15;0.3,0.6,0.1",
        baseline_weights="0.4,0.45,0.15",
        selection_metric="ndcg",
        min_delta=0.0,
        min_selection_rate=0.5,
        random_state=7,
        test_random_state=19,
        bootstrap_resamples=20,
        confidence_level=0.95,
        output=tmp_path / output_name,
    )


def catalog_benchmark_args(tmp_path, output_name: str) -> Namespace:
    return Namespace(
        data_dir=tmp_path,
        catalog_grid="4,6;6,8",
        baseline_catalog="4,6",
        hybrid_weights="0.4,0.45,0.15",
        validation_users=4,
        test_users=4,
        repetitions=2,
        seed_stride=7,
        min_interactions=3,
        min_relevant_rating=8.0,
        min_user_ratings=1,
        min_book_ratings=1,
        max_ratings=None,
        k=3,
        neighbors=3,
        hybrid_candidate_pool_size=cli.DEFAULT_HYBRID_CANDIDATE_POOL_SIZE,
        positive_rating_threshold=8.0,
        disable_hybrid_dynamic_weights=False,
        selection_metric="ndcg",
        min_delta=0.0,
        min_selection_rate=0.5,
        min_content_retention_delta=0.0,
        min_personalization_retention_delta=0.0,
        max_latency_ratio=100.0,
        max_serialized_model_ratio=100.0,
        random_state=7,
        test_random_state=19,
        bootstrap_resamples=20,
        confidence_level=0.95,
        output=tmp_path / output_name,
    )


def valid_evaluation_payload() -> dict:
    metrics = {
        "precision_at_3": 0.1,
        "recall_at_3": 0.3,
        "hit_rate_at_3": 0.3,
        "map_at_3": 0.2,
        "mrr_at_3": 0.2,
        "ndcg_at_3": 0.25,
    }
    return {
        "schema_version": cli.EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "evaluation",
        "protocol": cli.EVALUATION_PROTOCOL,
        "evaluation_partition": "test",
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
        "algorithms": cli.evaluation_algorithm_contract(),
        "parameters": {"neighbors": 3},
        "dataset": {
            "source_fingerprint": "a" * 64,
            "training_fingerprint": "b" * 64,
        },
        "availability": {},
        "summary": {
            "conditional": [],
            "end_to_end": [],
            "by_model": {
                "hybrid": {
                    "conditional_metrics": metrics,
                    "end_to_end_metrics": metrics,
                }
            },
        },
        "confidence_intervals": {},
        "per_user_rows": [],
    }


def valid_benchmark_payload() -> dict:
    metrics = {
        "precision_at_3": 0.1,
        "recall_at_3": 0.3,
        "hit_rate_at_3": 0.3,
        "map_at_3": 0.2,
        "mrr_at_3": 0.2,
        "ndcg_at_3": 0.25,
    }
    weights = {
        "collaborative_weight": 0.4,
        "content_weight": 0.45,
        "popularity_weight": 0.15,
    }
    config_id = cli.weight_config_id(weights)
    configuration = {
        "config_id": config_id,
        **weights,
        "end_to_end_metrics": metrics,
        "conditional_metrics": metrics,
    }
    validation_aggregate = {
        **configuration,
        "repetitions": 2,
    }
    fixed_test_aggregate = {
        **configuration,
        "repetitions": 1,
    }
    return {
        "schema_version": cli.BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "repeated_split_benchmark",
        "protocol": cli.BENCHMARK_PROTOCOL,
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
        "algorithms": cli.evaluation_algorithm_contract(),
        "parameters": {
            "neighbors": 3,
            "k": 3,
            "repetitions": 2,
            "selection_metric": "ndcg",
            "min_delta": 0.0,
            "min_selection_rate": 0.6,
            "confidence_level": 0.95,
            "selection_confidence_level": 0.95,
            "selection_multiplicity_adjustment": "none",
            "baseline_config_id": config_id,
            "weight_configs": [{"config_id": config_id, **weights}],
        },
        "dataset": {"source_fingerprint": "a" * 64},
        "selection": {
            "partition": "validation",
            "scope": "end_to_end",
            "metric": "ndcg_at_3",
            "baseline_config_id": config_id,
            "selected_config_id": config_id,
            "decision": "retain_baseline",
            "min_delta": 0.0,
            "min_selection_rate": 0.6,
            "confidence_level": 0.95,
            "multiplicity_adjustment": "none",
        },
        "validation": {
            "partition": "validation",
            "repetitions": 2,
            "repeats": [
                {"split_id": "validation-01", "configurations": [configuration]},
                {"split_id": "validation-02", "configurations": [configuration]},
            ],
            "aggregate_by_config": [validation_aggregate],
            "paired_deltas_vs_baseline": {},
            "per_user_rows": [
                {
                    "partition": "validation",
                    "config_id": config_id,
                    "split_id": "validation-01",
                    "user_id": 1,
                    "holdout_work_key": "validation-work-1",
                },
                {
                    "partition": "validation",
                    "config_id": config_id,
                    "split_id": "validation-02",
                    "user_id": 2,
                    "holdout_work_key": "validation-work-2",
                },
            ],
        },
        "fixed_test": {
            "partition": "test",
            "evaluated_after_selection": True,
            "evaluated_config_ids": [config_id],
            "configurations": [configuration],
            "aggregate_by_config": [fixed_test_aggregate],
            "paired_delta_selected_vs_baseline": None,
            "per_user_rows": [
                {
                    "partition": "test",
                    "config_id": config_id,
                    "split_id": "fixed-test",
                    "user_id": 3,
                    "holdout_work_key": "test-work",
                }
            ],
        },
    }


def fixed_evaluation_data(
    dataset: BookDataset,
    test_holdouts: pd.DataFrame,
) -> ValidationTestEvaluationDataset:
    validation = test_holdouts.head(1).copy()
    empty = test_holdouts.iloc[0:0].copy()
    return ValidationTestEvaluationDataset(
        train_dataset=dataset,
        train_ratings_before_filter=dataset.ratings,
        validation_holdout_ratings=validation,
        validation_evaluable_holdout_ratings=empty,
        test_holdout_ratings=test_holdouts,
        test_evaluable_holdout_ratings=empty,
    )


def test_parser_rejects_negative_recommendation_sizes():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["recommend-collab", "--user-id", "1", "--top-n", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate-collab", "--k", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate-collab", "--random-state", str(2**32)])


def test_evaluation_metric_reader_accepts_only_compatible_contract(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(valid_evaluation_payload()), encoding="utf-8")

    metrics = cli.read_evaluation_metrics(
        path,
        expected_source_dataset_fingerprint="a" * 64,
        expected_parameters={"neighbors": 3},
    )

    assert metrics["protocol"] == cli.EVALUATION_PROTOCOL
    assert metrics["summary"]["by_model"]["hybrid"]["end_to_end_metrics"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("evaluation_partition", "validation"),
        ("relevance_unit", "isbn"),
        ("algorithms", {}),
    ],
)
def test_evaluation_metric_reader_rejects_stale_contract(tmp_path, field, value):
    payload = valid_evaluation_payload()
    payload[field] = value
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        cli.read_evaluation_metrics(path)


def test_evaluation_metric_reader_rejects_dataset_config_and_nonfinite_values(tmp_path):
    payload = valid_evaluation_payload()
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different source dataset"):
        cli.read_evaluation_metrics(
            path,
            expected_source_dataset_fingerprint="c" * 64,
        )
    with pytest.raises(ValueError, match="parameter"):
        cli.read_evaluation_metrics(path, expected_parameters={"neighbors": 99})

    payload["summary"]["by_model"]["hybrid"]["end_to_end_metrics"][
        "hit_rate_at_3"
    ] = float("inf")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        cli.read_evaluation_metrics(path)


def test_benchmark_metric_reader_requires_compatible_selection_and_fixed_test(tmp_path):
    payload = valid_benchmark_payload()
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    expected_weights = {
        "collaborative_weight": 0.4,
        "content_weight": 0.45,
        "popularity_weight": 0.15,
    }

    metrics = cli.read_benchmark_metrics(
        path,
        expected_source_dataset_fingerprint="a" * 64,
        expected_parameters={"neighbors": 3},
        expected_weights=expected_weights,
    )

    assert metrics["protocol"] == cli.BENCHMARK_PROTOCOL
    assert len(metrics["artifact_sha256"]) == 64
    payload = valid_benchmark_payload()
    payload.pop("validation")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        cli.read_benchmark_metrics(path)

    payload = valid_benchmark_payload()
    payload["fixed_test"]["evaluated_config_ids"].append("unselected")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="restricted"):
        cli.read_benchmark_metrics(path)


def test_evaluate_cli_writes_independent_test_metrics(monkeypatch, tmp_path):
    dataset = evaluation_dataset()
    monkeypatch.setattr(cli, "load_clean_dataset", lambda _: dataset)
    args = evaluation_args(tmp_path, "evaluation.json")

    cli.evaluate_collab_model(args)

    payload = json.loads(args.output.read_text(encoding="utf-8"))
    assert payload["evaluation_partition"] == "test"
    assert payload["availability"]["sampled_holdouts"] == 4
    assert payload["summary"]["conditional"]
    assert payload["schema_version"] == 2
    assert payload["relevance_unit"] == "canonical_work"
    assert payload["split_unit"] == "user_work"
    assert payload["summary"]["end_to_end"]
    assert "hybrid" in payload["confidence_intervals"]
    assert "content_profile_strict" in payload["summary"]["by_model"]
    assert len(payload["per_user_rows"]) == 5 * 4
    assert payload["dataset"]["candidate_works"] > 0


def test_evaluate_cli_records_all_zero_when_no_target_is_in_catalog(monkeypatch, tmp_path):
    dataset = evaluation_dataset()
    holdouts = pd.DataFrame(
        [
            {
                "user_id": 1,
                "isbn": "MISSING",
                "book_rating": 10.0,
                "work_key": "missing canonical work",
            }
        ]
    )
    evaluation_data = fixed_evaluation_data(dataset, holdouts)
    monkeypatch.setattr(cli, "load_clean_dataset", lambda _: dataset)
    monkeypatch.setattr(
        cli,
        "prepare_validation_test_evaluation",
        lambda *_, **__: evaluation_data,
    )
    args = evaluation_args(tmp_path, "zero-candidate.json")

    cli.evaluate_collab_model(args)

    payload = json.loads(args.output.read_text(encoding="utf-8"))
    assert payload["availability"]["candidate_available_holdouts"] == 0
    assert len(payload["per_user_rows"]) == 5
    assert all(row["hit_rate_at_3"] == 0.0 for row in payload["per_user_rows"])


def test_evaluate_cli_runs_cold_user_fallback_in_end_to_end_scope(monkeypatch, tmp_path):
    dataset = evaluation_dataset()
    target = dataset.books.iloc[0]
    work_key = book_work_key(target.book_title, target.book_author, target.isbn)
    holdouts = pd.DataFrame(
        [
            {
                "user_id": 999,
                "isbn": str(target.isbn),
                "book_rating": 10.0,
                "work_key": work_key,
            }
        ]
    )
    evaluation_data = fixed_evaluation_data(dataset, holdouts)
    monkeypatch.setattr(cli, "load_clean_dataset", lambda _: dataset)
    monkeypatch.setattr(
        cli,
        "prepare_validation_test_evaluation",
        lambda *_, **__: evaluation_data,
    )
    args = evaluation_args(tmp_path, "cold-user.json")

    cli.evaluate_collab_model(args)

    payload = json.loads(args.output.read_text(encoding="utf-8"))
    rows = payload["per_user_rows"]
    assert len(rows) == 5
    assert all(row["evaluation_status"] == "cold_user_fallback" for row in rows)
    assert all(row["positive_seed_count"] == 0 for row in rows)
    strict_rows = [row for row in rows if row["model"].endswith("_strict")]
    assert all(row["hit_rate_at_3"] == 0.0 for row in strict_rows)


def test_tuning_cli_uses_validation_partition(monkeypatch, tmp_path):
    dataset = evaluation_dataset()
    monkeypatch.setattr(cli, "load_clean_dataset", lambda _: dataset)
    args = evaluation_args(tmp_path, "tuning.json")
    args.weight_grid = "0.4,0.5,0.1;0.5,0.4,0.1"

    cli.tune_hybrid_weights(args)

    payload = json.loads(args.output.read_text(encoding="utf-8"))
    assert payload["evaluation_partition"] == "validation"
    assert payload["dataset"]["test_sampled_holdouts_reserved"] == 4
    assert len(payload["results"]) == 2
    assert payload["best_config"]["end_to_end_metrics"]


def test_repeated_benchmark_keeps_fixed_test_out_of_validation_selection(monkeypatch, tmp_path):
    dataset = evaluation_dataset()
    monkeypatch.setattr(cli, "load_clean_dataset", lambda _: dataset)
    args = benchmark_args(tmp_path, "benchmark.json")
    args.hybrid_candidate_pool_size = cli.DEFAULT_HYBRID_CANDIDATE_POOL_SIZE

    cli.benchmark_hybrid_weights(args)

    payload = json.loads(args.output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "repeated_split_benchmark"
    assert payload["protocol"] == cli.BENCHMARK_PROTOCOL
    assert payload["selection"]["partition"] == "validation"
    assert payload["fixed_test"]["evaluated_after_selection"] is True
    assert len(payload["validation"]["repeats"]) == 2
    assert payload["dataset"]["fixed_test_users"] == 4

    test_users = {row["user_id"] for row in payload["fixed_test"]["per_user_rows"]}
    validation_users = {row["user_id"] for row in payload["validation"]["per_user_rows"]}
    assert test_users.isdisjoint(validation_users)

    baseline = payload["selection"]["baseline_config_id"]
    selected = payload["selection"]["selected_config_id"]
    assert set(payload["fixed_test"]["evaluated_config_ids"]) == {baseline, selected}
    assert set(row["config_id"] for row in payload["fixed_test"]["per_user_rows"]) == {
        baseline,
        selected,
    }
    assert len(payload["validation"]["aggregate_by_config"]) == 2
    verified = cli.read_benchmark_metrics(args.output)
    assert verified["selection"]["selected_config_id"] == selected

    selected_weights = next(
        config for config in payload["parameters"]["weight_configs"] if config["config_id"] == selected
    )
    monkeypatch.setattr(cli, "load_hybrid_prepared_dataset", lambda _: dataset)
    model_output = tmp_path / "benchmark-lineage.joblib"
    cli.train_hybrid_model(
        Namespace(
            data_dir=tmp_path,
            min_user_ratings=1,
            min_book_ratings=1,
            max_books=None,
            max_ratings=None,
            max_content_books=None,
            neighbors=3,
            collaborative_weight=selected_weights["collaborative_weight"],
            content_weight=selected_weights["content_weight"],
            popularity_weight=selected_weights["popularity_weight"],
            positive_rating_threshold=8.0,
            disable_dynamic_weights=False,
            random_state=42,
            evaluation_artifact=None,
            benchmark_artifact=args.output,
            output=model_output,
        )
    )
    _, manifest = load_model_bundle(model_output)
    assert manifest.metrics["protocol"] == cli.BENCHMARK_PROTOCOL
    assert manifest.metrics["artifact_sha256"] == verified["artifact_sha256"]


def test_benchmark_weight_configs_are_order_independent() -> None:
    forward, forward_baseline = cli.benchmark_weight_configs(
        weight_grid="0.4,0.45,0.15;0.3,0.6,0.1",
        baseline_weights="0.4,0.45,0.15",
    )
    reverse, reverse_baseline = cli.benchmark_weight_configs(
        weight_grid="0.3,0.6,0.1;0.4,0.45,0.15",
        baseline_weights="0.4,0.45,0.15",
    )

    assert forward == reverse
    assert forward_baseline == reverse_baseline


def test_repeated_benchmark_rejects_underpowered_requested_cohorts(monkeypatch, tmp_path):
    dataset = evaluation_dataset()
    monkeypatch.setattr(cli, "load_clean_dataset", lambda _: dataset)
    args = benchmark_args(tmp_path, "underpowered.json")
    args.test_users = 50

    with pytest.raises(ValueError, match="fixed test cohort"):
        cli.benchmark_hybrid_weights(args)


def test_catalog_benchmark_reuses_cohorts_and_opens_test_after_selection(
    monkeypatch,
    tmp_path,
):
    dataset = evaluation_dataset()
    monkeypatch.setattr(cli, "load_clean_dataset", lambda _: dataset)
    args = catalog_benchmark_args(tmp_path, "catalog-benchmark.json")
    events = []
    original_evaluator = cli.evaluate_hybrid_weight_configurations

    def recording_evaluator(**kwargs):
        events.append((kwargs["partition"], str(kwargs["configs"][0]["config_id"])))
        return original_evaluator(**kwargs)

    monkeypatch.setattr(cli, "evaluate_hybrid_weight_configurations", recording_evaluator)
    cli.benchmark_catalog_sizes(args)

    payload = json.loads(args.output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == cli.CATALOG_BENCHMARK_ARTIFACT_TYPE
    assert payload["selection"]["partition"] == "validation"
    assert payload["fixed_test"]["evaluated_after_selection"] is True
    assert [partition for partition, _ in events[:4]] == ["validation"] * 4
    assert all(partition == "test" for partition, _ in events[4:])

    validation = pd.DataFrame(payload["validation"]["per_user_rows"])
    fixed_test = pd.DataFrame(payload["fixed_test"]["per_user_rows"])
    assert set(validation["user_id"]).isdisjoint(set(fixed_test["user_id"]))
    for _, split_rows in validation.groupby("split_id"):
        cohorts = [
            set(zip(rows["user_id"], rows["holdout_work_key"]))
            for _, rows in split_rows.groupby("config_id")
        ]
        assert all(cohort == cohorts[0] for cohort in cohorts[1:])

    baseline = payload["selection"]["baseline_config_id"]
    selected = payload["selection"]["selected_config_id"]
    assert payload["fixed_test"]["evaluated_config_ids"] == list(
        dict.fromkeys([baseline, selected])
    )
    assert all(
        record["resources"]["warm_inference"]["measured_calls"] > 0
        for record in payload["validation"]["aggregate_by_config"]
    )
    verified = cli.read_catalog_benchmark_metrics(args.output)
    assert verified["deployment"] == payload["deployment"]
    deployment_id = payload["deployment"]["deployment_config_id"]
    deployment_catalog = next(
        config
        for config in payload["parameters"]["catalog_configs"]
        if config["config_id"] == deployment_id
    )
    cli.read_catalog_benchmark_metrics(
        args.output,
        expected_catalog={
            "max_interaction_books": deployment_catalog["max_interaction_books"],
            "max_content_books": deployment_catalog["max_content_books"],
        },
        expected_weights=payload["parameters"]["hybrid_weights"],
    )
    with pytest.raises(ValueError, match="serving catalog"):
        cli.read_catalog_benchmark_metrics(
            args.output,
            expected_catalog={
                "max_interaction_books": 999,
                "max_content_books": 999,
            },
        )

    monkeypatch.setattr(cli, "load_hybrid_prepared_dataset", lambda _: dataset)
    model_output = tmp_path / "catalog-lineage.joblib"
    cli.train_hybrid_model(
        Namespace(
            data_dir=tmp_path,
            min_user_ratings=1,
            min_book_ratings=1,
            max_books=deployment_catalog["max_interaction_books"],
            max_ratings=None,
            max_content_books=deployment_catalog["max_content_books"],
            neighbors=3,
            collaborative_weight=0.4,
            content_weight=0.45,
            popularity_weight=0.15,
            positive_rating_threshold=8.0,
            disable_dynamic_weights=False,
            random_state=19,
            evaluation_artifact=None,
            benchmark_artifact=None,
            catalog_benchmark_artifact=args.output,
            output=model_output,
        )
    )
    _, manifest = load_model_bundle(model_output)
    assert manifest.metrics["protocol"] == cli.CATALOG_BENCHMARK_PROTOCOL
    assert manifest.metrics["artifact_sha256"] == verified["artifact_sha256"]


def test_catalog_benchmark_reader_rejects_tampered_cohort_and_resources(
    monkeypatch,
    tmp_path,
):
    dataset = evaluation_dataset()
    monkeypatch.setattr(cli, "load_clean_dataset", lambda _: dataset)
    args = catalog_benchmark_args(tmp_path, "catalog-benchmark.json")
    cli.benchmark_catalog_sizes(args)
    valid_payload = json.loads(args.output.read_text(encoding="utf-8"))

    payload = json.loads(json.dumps(valid_payload))
    payload["selection"]["partition"] = "test"
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation-only"):
        cli.read_catalog_benchmark_metrics(args.output)

    payload = json.loads(json.dumps(valid_payload))
    target_config = payload["parameters"]["catalog_configs"][1]["config_id"]
    target_row = next(
        row
        for row in payload["validation"]["per_user_rows"]
        if row["config_id"] == target_config and row["split_id"] == "validation-01"
    )
    target_row["holdout_work_key"] = "tampered-work"
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="do not share cohort"):
        cli.read_catalog_benchmark_metrics(args.output)

    payload = json.loads(json.dumps(valid_payload))
    payload["validation"]["aggregate_by_config"][0]["resources"]["fit"][
        "median_seconds"
    ] = -1.0
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="finite non-negative"):
        cli.read_catalog_benchmark_metrics(args.output)


def test_train_hybrid_writes_verified_composite_bundle(monkeypatch, tmp_path):
    dataset = evaluation_dataset()
    monkeypatch.setattr(cli, "load_hybrid_prepared_dataset", lambda _: dataset)
    output = tmp_path / "hybrid.joblib"
    args = Namespace(
        neighbors=3,
        collaborative_weight=0.4,
        content_weight=0.45,
        popularity_weight=0.15,
        positive_rating_threshold=8.0,
        disable_dynamic_weights=False,
        evaluation_artifact=None,
        output=output,
        min_user_ratings=1,
        min_book_ratings=1,
        max_books=None,
        max_ratings=None,
    )

    cli.train_hybrid_model(args)
    model, manifest = load_model_bundle(
        output,
        expected_type=HybridRecommender,
        expected_model_type="hybrid",
    )

    assert output.is_file()
    assert manifest_path(output).is_file()
    assert manifest.training_config["neighbors"] == 3
    assert model.content_model is not None
    assert model.collaborative_model is not None
