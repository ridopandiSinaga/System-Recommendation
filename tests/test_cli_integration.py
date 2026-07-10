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
