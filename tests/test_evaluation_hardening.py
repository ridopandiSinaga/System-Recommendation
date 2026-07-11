from __future__ import annotations

import math

import pandas as pd
import pytest

from book_recommender.data import BookDataset
from book_recommender.evaluation import (
    bootstrap_mean_confidence_interval,
    bootstrap_paired_ranking_deltas_at_k,
    bootstrap_ranking_metrics_at_k,
    build_fixed_test_work_level_split,
    build_work_level_holdouts,
    build_validation_test_work_level_split,
    build_validation_test_leave_one_out_split,
    catalog_coverage_at_k,
    popularity_share_at_k,
    prepare_validation_test_evaluation,
    ranking_metrics_at_k,
    remove_work_level_holdouts,
    summarize_holdout_availability,
    summarize_ranking_metrics_at_k,
)


def _ratings_for_users(user_count: int = 8) -> pd.DataFrame:
    rows = []
    for user_id in range(1, user_count + 1):
        rows.extend(
            [
                {"user_id": user_id, "isbn": f"history-a-{user_id}", "book_rating": 5},
                {"user_id": user_id, "isbn": f"history-b-{user_id}", "book_rating": 6},
                {"user_id": user_id, "isbn": f"relevant-{user_id}", "book_rating": 10},
            ]
        )
    return pd.DataFrame(rows)


def test_prepare_validation_test_evaluation_uses_shared_train_catalog() -> None:
    ratings = _ratings_for_users(8)
    anchor_rows = [
        {"user_id": 99, "isbn": f"relevant-{user_id}", "book_rating": 5}
        for user_id in range(1, 9)
    ]
    ratings = pd.concat([ratings, pd.DataFrame(anchor_rows)], ignore_index=True)
    isbns = sorted(ratings["isbn"].unique())
    books = pd.DataFrame(
        {
            "isbn": isbns,
            "book_title": [f"Title {isbn}" for isbn in isbns],
            "book_author": "Author",
            "publisher": "Publisher",
        }
    )
    users = pd.DataFrame(
        {
            "user_id": list(range(1, 9)) + [99],
            "location": "x",
            "age": 30,
        }
    )

    prepared = prepare_validation_test_evaluation(
        BookDataset(books=books, ratings=ratings, users=users),
        sample_users=8,
        validation_fraction=0.5,
        min_interactions=3,
        min_relevant_rating=8,
        random_state=7,
    )

    validation_users = set(prepared.validation_holdout_ratings["user_id"])
    test_users = set(prepared.test_holdout_ratings["user_id"])
    assert validation_users.isdisjoint(test_users)
    assert len(prepared.validation_holdout_ratings) == 4
    assert len(prepared.test_holdout_ratings) == 4
    assert prepared.validation_availability.sampled_holdouts == 4
    assert prepared.test_availability.sampled_holdouts == 4


def test_work_level_split_removes_every_edition_of_target_work() -> None:
    books = pd.DataFrame(
        [
            {"isbn": "A-1", "book_title": "Alpha", "book_author": "Author"},
            {"isbn": "A-2", "book_title": "Alpha", "book_author": "Author"},
            {"isbn": "B", "book_title": "Beta", "book_author": "Author"},
            {"isbn": "C", "book_title": "Gamma", "book_author": "Author"},
        ]
    )
    ratings = pd.DataFrame(
        [
            {"user_id": user_id, "isbn": "A-1", "book_rating": 10}
            for user_id in (1, 2)
        ]
        + [
            {"user_id": user_id, "isbn": "A-2", "book_rating": 9}
            for user_id in (1, 2)
        ]
        + [
            {"user_id": user_id, "isbn": "B", "book_rating": 6}
            for user_id in (1, 2)
        ]
        + [
            {"user_id": user_id, "isbn": "C", "book_rating": 5}
            for user_id in (1, 2)
        ]
    )

    split = build_validation_test_work_level_split(
        ratings,
        books,
        sample_users=2,
        validation_fraction=0.5,
        min_interactions=3,
        min_relevant_rating=8,
        random_state=4,
    )

    holdouts = pd.concat(
        [split.validation_holdout_ratings, split.test_holdout_ratings],
        ignore_index=True,
    )
    assert holdouts["work_key"].nunique() == 1
    for user_id in (1, 2):
        user_train = split.train_ratings[split.train_ratings["user_id"] == user_id]
        assert not set(user_train["isbn"]).intersection({"A-1", "A-2"})


def test_fixed_test_cohort_is_excluded_from_repeated_validation_and_training_targets() -> None:
    ratings = _ratings_for_users(10)
    isbns = sorted(ratings["isbn"].unique())
    books = pd.DataFrame(
        {
            "isbn": isbns,
            "book_title": [f"Title {isbn}" for isbn in isbns],
            "book_author": "Author",
            "publisher": "Publisher",
        }
    )

    outer = build_fixed_test_work_level_split(
        ratings,
        books,
        test_users=3,
        min_interactions=3,
        min_relevant_rating=8,
        random_state=11,
    )
    validation = build_work_level_holdouts(
        outer.development_ratings,
        books,
        sample_users=4,
        min_interactions=3,
        min_relevant_rating=8,
        random_state=12,
        excluded_user_ids=outer.test_users,
    )
    assert set(outer.test_holdout_ratings["user_id"]).isdisjoint(validation["user_id"])

    for holdout in outer.test_holdout_ratings.itertuples(index=False):
        remaining = outer.development_ratings.loc[
            outer.development_ratings["user_id"] == holdout.user_id,
            "isbn",
        ]
        assert f"relevant-{holdout.user_id}" not in set(remaining)

    validation_train = remove_work_level_holdouts(
        outer.development_ratings,
        books,
        validation,
    )
    for holdout in validation.itertuples(index=False):
        assert f"relevant-{holdout.user_id}" not in set(
            validation_train.loc[
                validation_train["user_id"] == holdout.user_id,
                "isbn",
            ]
        )


def test_holdout_availability_reports_candidate_and_history_retention() -> None:
    holdouts = pd.DataFrame(
        [
            {"user_id": 1, "isbn": "A"},
            {"user_id": 2, "isbn": "B"},
            {"user_id": 3, "isbn": "C"},
            {"user_id": 4, "isbn": "D"},
        ]
    )

    availability = summarize_holdout_availability(
        holdouts,
        candidate_items=["A", "B"],
        history_users=[1, 3],
    )

    assert availability.sampled_holdouts == 4
    assert availability.candidate_available_holdouts == 2
    assert availability.user_history_available_holdouts == 2
    assert availability.evaluable_holdouts == 1
    assert availability.dropped_holdouts == 3
    assert availability.candidate_retention_rate == 0.5
    assert availability.user_history_retention_rate == 0.5
    assert availability.evaluation_retention_rate == 0.25
    assert availability.to_dict()["dropped_holdouts"] == 3


def test_ranking_summary_counts_dropped_holdouts_as_zero_metric_rows() -> None:
    rows = [
        ranking_metrics_at_k(["A", "X"], ["A"], k=2),
        ranking_metrics_at_k(["X", "Y"], ["B"], k=2),
    ]

    summary = summarize_ranking_metrics_at_k(rows, k=2, sampled_holdouts=4)

    assert summary.sampled_holdouts == 4
    assert summary.evaluable_holdouts == 2
    assert summary.dropped_holdouts == 2
    assert summary.evaluation_retention_rate == 0.5
    assert summary.conditional_metrics["hit_rate_at_2"] == 0.5
    assert summary.unconditional_metrics["hit_rate_at_2"] == 0.25
    assert summary.conditional_metrics["precision_at_2"] == 0.25
    assert summary.unconditional_metrics["precision_at_2"] == 0.125
    assert summary.to_dict()["unconditional_metrics"]["recall_at_2"] == 0.25


def test_ranking_summary_supports_a_sample_with_no_evaluable_holdouts() -> None:
    summary = summarize_ranking_metrics_at_k([], k=10, sampled_holdouts=3)

    assert summary.evaluable_holdouts == 0
    assert summary.dropped_holdouts == 3
    assert summary.evaluation_retention_rate == 0.0
    assert set(summary.conditional_metrics.values()) == {0.0}
    assert set(summary.unconditional_metrics.values()) == {0.0}


def test_validation_test_holdouts_are_deterministic_and_user_disjoint() -> None:
    ratings = _ratings_for_users()

    split = build_validation_test_leave_one_out_split(
        ratings,
        sample_users=6,
        validation_fraction=0.5,
        min_interactions=3,
        min_relevant_rating=8,
        random_state=17,
    )
    repeated = build_validation_test_leave_one_out_split(
        ratings,
        sample_users=6,
        validation_fraction=0.5,
        min_interactions=3,
        min_relevant_rating=8,
        random_state=17,
    )

    pd.testing.assert_frame_equal(split.train_ratings, repeated.train_ratings)
    pd.testing.assert_frame_equal(
        split.validation_holdout_ratings, repeated.validation_holdout_ratings
    )
    pd.testing.assert_frame_equal(split.test_holdout_ratings, repeated.test_holdout_ratings)
    validation_users = set(split.validation_holdout_ratings["user_id"])
    test_users = set(split.test_holdout_ratings["user_id"])
    assert validation_users.isdisjoint(test_users)
    assert split.validation_users == 3
    assert split.test_users == 3
    assert split.sampled_users == 6
    assert len(split.train_ratings) == len(ratings) - split.sampled_users
    assert set(split.validation_holdout_ratings["book_rating"]) == {10}
    assert set(split.test_holdout_ratings["book_rating"]) == {10}


def test_bootstrap_mean_interval_is_deterministic_and_reports_sample_size() -> None:
    first = bootstrap_mean_confidence_interval(
        [0.0, 1.0, 1.0, 0.0],
        confidence_level=0.90,
        n_resamples=500,
        random_state=9,
    )
    second = bootstrap_mean_confidence_interval(
        [0.0, 1.0, 1.0, 0.0],
        confidence_level=0.90,
        n_resamples=500,
        random_state=9,
    )

    assert first == second
    assert first.mean == 0.5
    assert first.lower <= first.mean <= first.upper
    assert first.confidence_level == 0.90
    assert first.sample_size == 4
    assert first.n_resamples == 500
    assert first.to_dict()["sample_size"] == 4


def test_bootstrap_ranking_metrics_can_include_unavailable_holdouts() -> None:
    hit = ranking_metrics_at_k(["A"], ["A"], k=2)

    intervals = bootstrap_ranking_metrics_at_k(
        [hit],
        k=2,
        sampled_holdouts=4,
        n_resamples=200,
        random_state=11,
    )

    assert intervals["hit_rate_at_2"].mean == 0.25
    assert intervals["precision_at_2"].mean == 0.125
    assert intervals["hit_rate_at_2"].sample_size == 4


def test_paired_bootstrap_uses_macro_mean_over_repeated_splits() -> None:
    metric_names = ranking_metrics_at_k([], [], k=3).keys()
    baseline_rows = []
    challenger_rows = []
    for split_id, count, challenger_value in (("split-a", 100, 1.0), ("split-b", 1, 0.0)):
        for user_id in range(count):
            shared = {
                "split_id": split_id,
                "user_id": user_id,
                "holdout_work_key": f"{split_id}-{user_id}",
                "candidate_available": True,
                "history_available": True,
                "positive_seed_available": True,
                "evaluation_status": "personalized",
            }
            baseline_rows.append({**shared, **{name: 0.0 for name in metric_names}})
            challenger_rows.append(
                {**shared, **{name: challenger_value for name in metric_names}}
            )

    first = bootstrap_paired_ranking_deltas_at_k(
        baseline_rows,
        challenger_rows,
        k=3,
        n_resamples=200,
        random_state=13,
    )
    second = bootstrap_paired_ranking_deltas_at_k(
        list(reversed(baseline_rows)),
        list(reversed(challenger_rows)),
        k=3,
        n_resamples=200,
        random_state=13,
    )

    assert first == second
    assert first["ndcg_at_3"].mean == pytest.approx(0.5)
    assert first["ndcg_at_3"].sample_size == 101

    custom_pair_columns = bootstrap_paired_ranking_deltas_at_k(
        baseline_rows,
        challenger_rows,
        k=3,
        pair_columns=("user_id", "holdout_work_key"),
        cluster_column="split_id",
        n_resamples=50,
        random_state=13,
    )
    assert custom_pair_columns["ndcg_at_3"].mean == pytest.approx(0.5)


def test_paired_bootstrap_rejects_mismatched_pair_keys_or_statuses() -> None:
    metrics = ranking_metrics_at_k([], [], k=2)
    baseline = {
        "split_id": "repeat-1",
        "user_id": 1,
        "holdout_work_key": "target",
        "candidate_available": True,
        "history_available": True,
        "positive_seed_available": True,
        "evaluation_status": "personalized",
        **metrics,
    }
    challenger = {**baseline, "holdout_work_key": "different"}
    with pytest.raises(ValueError, match="identical pair keys"):
        bootstrap_paired_ranking_deltas_at_k([baseline], [challenger], k=2)

    challenger = {**baseline, "candidate_available": False}
    with pytest.raises(ValueError, match="disagree"):
        bootstrap_paired_ranking_deltas_at_k([baseline], [challenger], k=2)


@pytest.mark.parametrize("invalid_k", [0, -1, 1.5, True])
def test_k_is_validated_even_for_empty_inputs(invalid_k: object) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        catalog_coverage_at_k([], [], k=invalid_k)  # type: ignore[arg-type]


def test_other_evaluation_options_are_validated() -> None:
    ratings = _ratings_for_users()

    with pytest.raises(ValueError, match="validation_fraction"):
        build_validation_test_leave_one_out_split(ratings, validation_fraction=1.0)
    with pytest.raises(ValueError, match="sample_users must be at least 2"):
        build_validation_test_leave_one_out_split(ratings, sample_users=1)
    with pytest.raises(ValueError, match="top_percent"):
        popularity_share_at_k([], pd.Series(dtype=float), top_percent=0, k=5)
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_mean_confidence_interval([])
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_confidence_interval([1.0, math.nan])
    with pytest.raises(ValueError, match="cannot exceed"):
        metric_row = ranking_metrics_at_k(["A"], ["A"], k=1)
        summarize_ranking_metrics_at_k(
            [metric_row, metric_row],
            k=1,
            sampled_holdouts=1,
        )
