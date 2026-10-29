"""Fit-only supervision and sampling locally; neural mechanics on the GPU host."""
import os

import numpy as np
import pandas as pd
import pytest

from book_recommender.ranking_neural import prepare_work_data, sample_negative_works


def frames():
    books = pd.DataFrame([
        ("A1", "River", "Ada", "Firstpress"),
        ("A2", "River (Paperback edition)", "Ada", "Secondpress"),
        ("B", "Mountain", "Bob", "Highpress"),
        ("C", "Sea", "Cara", "Oceanpress"),
        ("D", "Leakageword", "Unobservedauthor", "Unobservedpublisher"),
    ], columns=["isbn", "book_title", "book_author", "publisher"])
    fit = pd.DataFrame([
        (20, "A1", 3), (20, "A2", 9), (20, "B", 4),
        (10, "A1", 8), (10, "C", 10), (30, "B", 5),
    ], columns=["user_id", "isbn", "book_rating"])
    return books, fit


def test_work_supervision_uses_mean_for_mse_and_any_positive_edition_for_bpr():
    data = prepare_work_data(*frames(), max_features=64)
    assert data.users == [10, 20, 30]
    river = data.work_to_index[data.isbn_to_work["A1"]]
    assert data.isbn_to_work["A1"] == data.isbn_to_work["A2"]
    pair = np.flatnonzero((data.pairs == [data.user_to_index[20], river]).all(axis=1))
    assert len(pair) == 1
    assert data.targets[pair[0]] == pytest.approx((6 - 1) / 9)
    assert [data.user_to_index[20], river] in data.positive_pairs.tolist()
    assert data.popularity[river] == 2  # Two distinct positive users, not three editions.
    assert data.positive_user_mask.tolist() == [True, True, False]
    assert len(data.pairs) == 5
    assert data.fit_mean_rating == pytest.approx((8 + 10 + 6 + 4 + 5) / 5)


def test_metadata_vocabulary_and_work_catalog_are_fitted_only_on_fit_editions():
    books, fit = frames()
    data = prepare_work_data(books, fit, max_features=64)
    assert "D" not in data.isbn_to_work
    assert "leakageword" not in data.metadata_vocabulary
    assert "unobservedauthor" not in data.metadata_vocabulary
    assert "unobservedpublisher" not in data.metadata_vocabulary
    assert data.metadata_features.shape == (3, 64)
    assert data.metadata_features.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(data.metadata_features, axis=1), 1, atol=1e-6)
    assert "firstpress" in data.metadata_vocabulary
    assert "secondpress" in data.metadata_vocabulary
    # Introducing more excluded catalog metadata must not alter any features.
    books.loc[books.isbn == "D", "book_title"] = "New heldout-only terms"
    other = prepare_work_data(books.sample(frac=1, random_state=1),
                              fit.sample(frac=1, random_state=2), max_features=64)
    assert other.metadata_vocabulary == data.metadata_vocabulary
    assert other.metadata_documents == data.metadata_documents
    np.testing.assert_array_equal(other.metadata_features, data.metadata_features)
    np.testing.assert_array_equal(other.pairs, data.pairs)
    np.testing.assert_array_equal(other.targets, data.targets)


def test_sampler_excludes_low_ratings_and_every_observed_edition():
    data = prepare_work_data(*frames(), max_features=16)
    user = data.user_to_index[20]
    samples = sample_negative_works(np.repeat(user, 100), data.seen_work_indices,
                                    len(data.work_keys), np.random.default_rng(42))
    expected = data.work_to_index[data.isbn_to_work["C"]]
    assert set(samples) == {expected}
    assert data.work_to_index[data.isbn_to_work["B"]] in data.seen_work_indices[user]


def test_sampler_is_seeded_uniform_over_available_works_and_handles_empty_batch():
    seen = (frozenset({0, 4}), frozenset({1, 2, 3, 4}))
    users = np.zeros(6000, dtype=np.int32)
    first = sample_negative_works(users, seen, 5, np.random.default_rng(9))
    second = sample_negative_works(users, seen, 5, np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert set(first) == {1, 2, 3}
    counts = np.bincount(first, minlength=5)[[1, 2, 3]]
    assert np.all(np.abs(counts - 2000) < 140)
    np.testing.assert_array_equal(
        sample_negative_works([1, 1], seen, 5, np.random.default_rng(9)), [0, 0],
    )
    assert sample_negative_works([], seen, 5, np.random.default_rng(9)).shape == (0,)


def test_no_fit_positive_is_explicit_and_all_seen_user_cannot_be_sampled():
    books, fit = frames()
    fit["book_rating"] = 3
    data = prepare_work_data(books, fit, max_features=16)
    assert data.positive_pairs.shape == (0, 2)
    assert not data.positive_user_mask.any()
    assert not data.popularity.any()
    with pytest.raises(ValueError, match="full catalog"):
        sample_negative_works([0], (frozenset({0, 1}),), 2, np.random.default_rng(0))


@pytest.mark.parametrize("problem", ["missing_metadata", "invalid_rating", "duplicate_isbn"])
def test_fit_data_validation(problem):
    books, fit = frames()
    if problem == "missing_metadata":
        fit.loc[0, "isbn"] = "absent"
    elif problem == "invalid_rating":
        fit.loc[0, "book_rating"] = 0
    else:
        books = pd.concat([books, books.iloc[:1]])
    with pytest.raises(ValueError):
        prepare_work_data(books, fit)


requires_tensorflow = pytest.mark.skipif(
    os.environ.get("BOOK_REC_RUN_TF_TESTS") != "1",
    reason="Set BOOK_REC_RUN_TF_TESTS=1 in the GPU test environment",
)


@requires_tensorflow
def test_tensorflow_matched_initialization_catalog_scores_and_metadata_effect():
    from book_recommender.ranking_neural import create_ranking_model

    features = np.eye(3, 4, dtype=np.float32)
    identity = create_ranking_model(2, 3, features * 0, random_state=42, fit_mean_rating=7)
    metadata = create_ranking_model(2, 3, features, random_state=42, fit_mean_rating=7)
    for before, after in zip(identity.trainable_variables, metadata.trainable_variables):
        np.testing.assert_array_equal(before.numpy(), after.numpy())
    pairs = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=np.int32)
    np.testing.assert_allclose(metadata(pairs).numpy().reshape(2, 3),
                               metadata.score_catalog(np.array([0, 1], dtype=np.int32)), atol=1e-7)
    assert np.any(identity(pairs).numpy() != metadata(pairs).numpy())
    assert not metadata.metadata_features.trainable
    assert not metadata.global_logit.trainable


@requires_tensorflow
def test_tensorflow_regularization_matches_declared_formula():
    from book_recommender.ranking_neural import create_ranking_model

    model = create_ranking_model(2, 3, np.eye(3, 4, dtype=np.float32))
    model.work_bias.assign([.1, .2, -.3])
    pairs = np.array([[0, 0], [0, 2], [1, 1]], dtype=np.int32)
    user_weights = model.user_embedding.embeddings.numpy()
    work_weights = model.work_embedding.embeddings.numpy()
    expected_observed = np.mean(np.sum(user_weights[pairs[:, 0]] ** 2, axis=1)
                                + np.sum(work_weights[pairs[:, 1]] ** 2, axis=1)
                                + model.work_bias.numpy()[pairs[:, 1]] ** 2)
    kernels = [layer.kernel.numpy() for layer in [model.metadata_projection, model.user_hidden,
               model.user_output, model.work_hidden, model.work_output]]
    expected = 1e-5 * (expected_observed + sum(np.sum(k ** 2) for k in kernels)
                      / sum(k.size for k in kernels))
    assert float(model.regularization_loss(pairs)) == pytest.approx(expected, rel=1e-5)


@requires_tensorflow
@pytest.mark.parametrize("objective", ["mse", "bpr"])
def test_tensorflow_loss_gradients_update_and_save_load_roundtrip(tmp_path, objective):
    import tensorflow as tf
    from book_recommender.ranking_neural import (
        create_ranking_model, load_ranking_model, make_train_step,
    )

    model = create_ranking_model(2, 3, np.eye(3, 4, dtype=np.float32))
    users = np.array([0, 1], dtype=np.int32)
    items = np.array([0, 1], dtype=np.int32)
    negatives = np.array([2, 2], dtype=np.int32)
    targets = np.array([.9, .8], dtype=np.float32)
    positive_scores = model(np.column_stack([users, items])).numpy().ravel()
    if objective == "mse":
        expected_loss = np.mean((1 / (1 + np.exp(-positive_scores)) - targets) ** 2)
    else:
        negative_scores = model(np.column_stack([users, negatives])).numpy().ravel()
        expected_loss = np.mean(np.logaddexp(0, negative_scores - positive_scores))
    before = [variable.numpy().copy() for variable in model.trainable_variables]
    fixed_metadata = model.metadata_features.numpy().copy()
    fixed_global = model.global_logit.numpy().copy()
    step = make_train_step(model, tf.keras.optimizers.Adam(.001), objective=objective)
    trace = step(users, items, targets, negatives)
    assert float(trace["data_loss"]) == pytest.approx(expected_loss, rel=1e-5)
    assert float(trace["loss"]) == pytest.approx(
        float(trace["data_loss"] + trace["regularization_loss"]), rel=1e-6,
    )
    assert np.isfinite(float(trace["gradient_norm"])) and float(trace["gradient_norm"]) > 0
    assert any(np.any(old != new.numpy()) for old, new in zip(before, model.trainable_variables))
    np.testing.assert_array_equal(model.metadata_features.numpy(), fixed_metadata)
    np.testing.assert_array_equal(model.global_logit.numpy(), fixed_global)
    path = tmp_path / f"{objective}.keras"
    model.save(path)
    restored = load_ranking_model(path)
    for original, loaded in zip(model.weights, restored.weights):
        np.testing.assert_array_equal(original.numpy(), loaded.numpy())
    np.testing.assert_array_equal(model.score_catalog(users).numpy(),
                                  restored.score_catalog(users).numpy())
