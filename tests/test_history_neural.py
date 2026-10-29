"""Fit-history isolation locally and optional deterministic TensorFlow mechanics."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from book_recommender.history_neural import (
    ItemCosineBaseline, build_history_data, build_item_cosine,
)
from book_recommender.ranking_neural import prepare_work_data


def work_data():
    books = pd.DataFrame([
        ("A1", "Work A", "Author"), ("A2", "Work A (Paperback edition)", "Author"),
        *[(letter, f"Work {letter}", "Author") for letter in "BCDEF"],
    ], columns=["isbn", "book_title", "book_author"])
    fit = pd.DataFrame([
        (10, "A1", 8), (10, "A2", 9), (10, "B", 9), (10, "C", 3),
        (20, "A1", 8), (20, "C", 9), (30, "B", 10), (30, "D", 4),
        (40, "D", 5), (40, "E", 6), (40, "F", 7), (50, "E", 8), (50, "F", 9),
    ], columns=["user_id", "isbn", "book_rating"])
    return prepare_work_data(books, fit, max_features=16)


def test_binary_work_histories_and_training_eligibility_exclude_singletons():
    data = build_history_data(work_data())
    assert sparse.isspmatrix_csr(data.positive_csr)
    assert data.positive_csr.shape == (5, 6)
    assert data.positive_counts.tolist() == [2, 2, 1, 0, 2]
    assert data.positive_csr.nnz == 7
    assert data.positive_values.tolist() == [0, 1, 0, 2, 1, 4, 5]
    assert data.positive_row_splits.tolist() == [0, 2, 4, 5, 5, 7]
    assert data.eligible_pairs.tolist() == [[0, 0], [0, 1], [1, 0], [1, 2], [4, 4], [4, 5]]
    assert data.popularity.tolist() == [2, 2, 1, 0, 1, 1]


def test_softmax_mask_excludes_all_observations_except_positive_training_target():
    data = build_history_data(work_data())
    mask = data.softmax_candidate_mask(np.array([0, 1]), np.array([0, 2]))
    assert mask.dtype == bool
    # User0: mask other positive B and explicitly low-rated C; target A restored.
    assert mask[0].tolist() == [True, False, False, True, True, True]
    assert mask[1].tolist() == [False, True, True, True, True, True]
    with pytest.raises(ValueError, match="positive fit work"):
        data.softmax_candidate_mask(np.array([0]), np.array([2]))


def test_cosine_significance_formula_and_source_row_pruning(tmp_path):
    data = build_history_data(work_data())
    full = build_item_cosine(data, k=50, shrink=10)
    matrix = full.similarity.toarray()
    assert np.count_nonzero(np.diag(matrix)) == 0
    assert matrix[0, 1] == pytest.approx(.5/11)
    assert matrix[0, 2] == pytest.approx(1/np.sqrt(2)/11)
    assert matrix[4, 5] == pytest.approx(1/11)
    assert not matrix[3].any()  # A work with no positive users has no similarity.
    pruned = build_item_cosine(data, k=1, shrink=0)
    assert np.diff(pruned.similarity.indptr).max() == 1
    # Retaining neighbours per SOURCE row keeps B->A, even though A's strongest
    # incoming edge is C->A. Column pruning would incorrectly remove B->A.
    assert pruned.score_catalog(np.array([2]))[0, 0] == .5
    assert pruned.score_catalog(np.array([0]))[0, 2] == pytest.approx(1/np.sqrt(2))
    assert not pruned.score_catalog(np.array([3])).any()
    sparse.save_npz(tmp_path / "similarity.npz", pruned.similarity)
    restored = ItemCosineBaseline(data.positive_csr, sparse.load_npz(tmp_path / "similarity.npz"))
    np.testing.assert_array_equal(restored.score_catalog(np.arange(5)),
                                  pruned.score_catalog(np.arange(5)))


def test_cosine_pruning_breaks_exact_similarity_ties_by_work_index():
    history = sparse.csr_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]], dtype=float))
    model = build_item_cosine(SimpleNamespace(positive_csr=history), k=1, shrink=0)
    assert model.similarity.indices[model.similarity.indptr[0]:model.similarity.indptr[1]].tolist() == [1]


requires_tensorflow = pytest.mark.skipif(
    os.environ.get("BOOK_REC_RUN_TF_TESTS") != "1",
    reason="Set BOOK_REC_RUN_TF_TESTS=1 on the remote GPU host",
)


@requires_tensorflow
def test_tensorflow_target_exclusion_empty_history_and_persisted_csr(tmp_path):
    import tensorflow as tf
    from book_recommender.history_neural import create_history_model, load_history_model

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    data = build_history_data(work_data())
    model = create_history_model(data, random_state=42)
    embeddings = np.zeros((6, 32), dtype=np.float32)
    embeddings[:, 0] = [1, 2, 3, 4, 5, 6]
    model.context_embedding.embeddings.assign(embeddings)
    users, targets = np.array([0, 1, 2, 3], np.int32), np.array([0, 2, 1, 0], np.int32)
    pooled, counts = model.context_summary(users, exclude_items=targets)
    np.testing.assert_array_equal(counts.numpy(), [1, 1, 0, 0])
    np.testing.assert_array_equal(pooled.numpy()[:, 0], [2, 1, 0, 0])
    whole, whole_counts = model.context_summary(users)
    np.testing.assert_array_equal(whole_counts, [2, 2, 1, 0])
    np.testing.assert_allclose(whole.numpy()[:, 0], [3/np.sqrt(2), 4/np.sqrt(2), 2, 0])
    before = model.score_targets(np.array([0]), np.array([0]), exclude_items=np.array([0])).numpy()
    embeddings[0] = 1e3
    model.context_embedding.embeddings.assign(embeddings)
    # Editing only the excluded target's CONTEXT embedding cannot change score.
    np.testing.assert_array_equal(before, model.score_targets(
        np.array([0]), np.array([0]), exclude_items=np.array([0])).numpy())
    assert model.context_embedding.embeddings is not model.output_embedding.embeddings
    assert np.isfinite(model.score_catalog(users).numpy()).all()
    pairs = np.column_stack([users, targets])
    np.testing.assert_allclose(model(pairs).numpy().ravel(),
                               model.score_catalog(users).numpy()[np.arange(4), targets], atol=1e-6)
    path = tmp_path / "history.keras"
    model.save(path)
    restored = load_history_model(path)
    for original, loaded in zip(model.weights, restored.weights):
        np.testing.assert_array_equal(original.numpy(), loaded.numpy())
    np.testing.assert_array_equal(restored.score_catalog(users).numpy(), model.score_catalog(users).numpy())


@requires_tensorflow
@pytest.mark.parametrize("objective", ["bpr", "softmax"])
def test_tensorflow_deterministic_training_matches_loss_and_updates_context(objective):
    import tensorflow as tf
    from book_recommender.history_neural import create_history_model, make_history_train_step

    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    data = build_history_data(work_data())
    model = create_history_model(data, random_state=42)
    matched = create_history_model(data, random_state=42)
    for original, other in zip(model.trainable_variables, matched.trainable_variables):
        np.testing.assert_array_equal(original.numpy(), other.numpy())
    users, targets, negatives = np.array([0, 1], np.int32), np.array([0, 2], np.int32), np.array([5, 4], np.int32)
    mask = data.softmax_candidate_mask(users, targets)
    if objective == "bpr":
        pos = model.score_targets(users, targets, exclude_items=targets).numpy()
        neg = model.score_targets(users, negatives, exclude_items=targets).numpy()
        expected_loss = np.logaddexp(0, neg-pos).mean()
    else:
        scores = model.score_catalog(users, exclude_items=targets).numpy()
        allowed = np.where(mask, scores, -np.inf)
        maximum = allowed.max(axis=1)
        expected_loss = (maximum+np.log(np.exp(allowed-maximum[:, None]).sum(axis=1))
                         - scores[np.arange(2), targets]).mean()
    expected_reg = 1e-5*sum(np.sum(variable.numpy()**2) for variable in model.trainable_variables)
    expected_reg /= sum(np.prod(variable.shape) for variable in model.trainable_variables)
    before = model.context_embedding.embeddings.numpy().copy()
    step = make_history_train_step(model, tf.keras.optimizers.Adam(.001), objective=objective)
    trace = step(users, targets, negatives, mask)
    assert float(trace["data_loss"]) == pytest.approx(expected_loss, rel=2e-6)
    assert float(trace["regularization_loss"]) == pytest.approx(expected_reg, rel=2e-6)
    assert float(trace["gradient_norm"]) > 0
    assert np.any(before != model.context_embedding.embeddings.numpy())
    for value in trace.values():
        assert np.isfinite(float(value))
    # Deterministic mode covers ragged gather, segment reduction, and optimizer.
    repeat = make_history_train_step(matched, tf.keras.optimizers.Adam(.001), objective=objective)
    repeat(users, targets, negatives, mask)
    for original, other in zip(model.trainable_variables, matched.trainable_variables):
        np.testing.assert_array_equal(original.numpy(), other.numpy())
