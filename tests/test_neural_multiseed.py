"""Split and uncertainty invariants without importing TensorFlow."""
import numpy as np
import pandas as pd
import pytest

from scripts.colab_neural_multiseed import (
    CANDIDATES, FROZEN_INPUT_SHA256, build_development_split, evaluate_with_fallback,
    fit_statistics, user_cluster_bootstrap,
)


def fixture_frames():
    books = pd.DataFrame([
        {"isbn": isbn, "book_title": title, "book_author": "Author"}
        for isbn, title in (("A1", "Work A"), ("A2", "Work A"),
                            ("B", "Work B"), ("C", "Work C"))
    ])
    ratings = pd.DataFrame([
        {"user_id": user, "isbn": isbn, "book_rating": (user + offset) % 10 + 1}
        for user in range(1, 9) for offset, isbn in enumerate(("A1", "A2", "B", "C"))
    ] + [{"user_id": 9, "isbn": isbn, "book_rating": 8} for isbn in ("A1", "A2")])
    return books, ratings


def test_two_training_inputs_only_and_twelve_paired_candidates():
    assert set(FROZEN_INPUT_SHA256) == {"train_books.csv", "train_ratings.csv"}
    assert len(CANDIDATES) == len({row["name"] for row in CANDIDATES}) == 12
    for split in (17, 29):
        for seed in (42, 2026, 7):
            pair = [row for row in CANDIDATES if row["split_seed"] == split
                    and row["initialization_seed"] == seed]
            assert {row["architecture"]: row["l2"] for row in pair} == {"mlp": 1e-5, "residual": 0}


def test_split_preserves_all_edition_rows_and_fit_history():
    books, ratings = fixture_frames()
    partitions, membership, summary = build_development_split(books, ratings, 17, np, pd,
                                                              users_per_partition=2)
    assert summary["selection_users"] == summary["assessment_users"] == 2
    assert sum(map(len, partitions.values())) == len(ratings)
    assert set(partitions["fit"].user_id) == set(ratings.user_id)
    assert set(partitions["selection"].user_id).isdisjoint(partitions["assessment"].user_id)
    assert not any(row["user_id"] == 9 for row in membership)
    combined = pd.concat([frame.assign(partition=name) for name, frame in partitions.items()])
    assert combined.groupby(["user_id", "work_key"]).partition.nunique().max() == 1
    assert set(combined.source_row) == set(range(len(ratings)))


def test_split_is_reproducible_and_does_not_use_rating_labels():
    books, ratings = fixture_frames()
    _, membership, summary = build_development_split(books, ratings, 17, np, pd, users_per_partition=2)
    changed = ratings.copy()
    changed["book_rating"] = 11 - changed.book_rating
    _, altered_membership, altered_summary = build_development_split(books, changed, 17, np, pd,
                                                                    users_per_partition=2)
    assert membership == altered_membership
    assert summary["membership_sha256"] == altered_summary["membership_sha256"]
    assert summary["partition_row_sha256"] == altered_summary["partition_row_sha256"]
    _, other_membership, _ = build_development_split(books, ratings, 29, np, pd,
                                                    users_per_partition=2)
    assert other_membership != membership


def test_split_refuses_insufficient_eligible_users():
    books, ratings = fixture_frames()
    with pytest.raises(ValueError, match="Insufficient"):
        build_development_split(books, ratings, 17, np, pd, users_per_partition=5)


def test_cluster_bootstrap_keeps_user_multiplicity_and_is_reproducible():
    users = [1, 1, 2, 3]
    targets = [0, 0, 0, 0]
    candidate = [2, 2, 0, 1]
    reference = [0, 0, 0, 0]
    result = user_cluster_bootstrap(users, targets, candidate, reference, np, draws=113, seed=19)
    assert result["clusters"] == 3
    assert result["pairs"] == 4
    assert result["point_estimate"] == pytest.approx(1.5)
    # Independent explicit cluster-resampling oracle; repeated sampled users must
    # contribute their complete rows repeatedly, rather than via a set/isin mask.
    indices_by_user = {1: [0, 1], 2: [2], 3: [3]}
    chosen = np.random.default_rng(19).integers(3, size=(113, 3)) + 1
    differences = []
    for sample in chosen:
        row_indices = [index for user in sample for index in indices_by_user[user]]
        differences.append(np.sqrt(np.mean(np.asarray(candidate)[row_indices] ** 2)))
    np.testing.assert_allclose(result["interval_95_percent"], np.quantile(differences, [0.025, 0.975]))
    assert result == user_cluster_bootstrap(users, targets, candidate, reference, np, draws=113, seed=19)


@pytest.mark.parametrize("users,targets,prediction,reference", [
    ([], [], [], []), ([1], [1], [1, 2], [1]), ([1], [1], [np.nan], [1]),
])
def test_bootstrap_rejects_invalid_inputs(users, targets, prediction, reference):
    with pytest.raises(ValueError):
        user_cluster_bootstrap(users, targets, prediction, reference, np)


def test_cold_predictions_use_identical_train_only_fallback_for_every_model():
    fit = pd.DataFrame({"user_id": [1, 1, 2], "isbn": ["a", "b", "a"], "book_rating": [8, 6, 4]})
    frame = pd.DataFrame({"user_id": [1, 1, 3, 3], "isbn": ["a", "z", "a", "z"],
                          "book_rating": [7, 7, 6, 6]})

    class Predictor:
        user_to_index = {1: 0, 2: 1}
        isbn_to_index = {"a": 0, "b": 1}

        def _predict_encoded(self, pairs):
            return np.full((len(pairs), 1), 0.5)

    statistics = fit_statistics(fit)
    predictions = []
    for kind in ("neural", "global_mean", "user_mean"):
        metrics, values, known = evaluate_with_fallback(Predictor(), frame, statistics, np, predictor=kind)
        predictions.append(values)
        assert metrics["known_pairs"]["pairs"] == 1
        assert metrics["all_pairs"]["pairs"] == 4
        assert known.tolist() == [True, False, False, False]
    for values in predictions:
        np.testing.assert_allclose(values[1:], [7, 6, 6])
    assert len({values[0] for values in predictions}) == 3
