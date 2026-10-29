"""Joint ridge math locally; optional TensorFlow factorization checks on Colab."""
import os

import numpy as np
import pytest

from book_recommender.factorization import fit_joint_bias


def test_joint_bias_matches_independent_ridge_normal_equations():
    pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 2]])
    targets = np.array([0.2, 0.8, 0.1, 0.6, 0.7, 0.9])
    design = np.zeros((len(pairs), 6))
    design[np.arange(len(pairs)), pairs[:, 0]] = 1
    design[np.arange(len(pairs)), 3 + pairs[:, 1]] = 1
    expected = np.linalg.solve(design.T @ design + np.diag([5.] * 3 + [20.] * 3),
                               design.T @ (targets - targets.mean()))
    result = fit_joint_bias(pairs, targets, 3, 3, user_shrinkage=5, item_shrinkage=20)
    np.testing.assert_allclose(np.r_[result["user_bias"], result["book_bias"]], expected, atol=1e-9)
    assert result["converged"]
    assert result["iterations"] < 1000
    assert result["convergence_history"][-1]["sum_squared_error_plus_ridge"] <= result["convergence_history"][0]["sum_squared_error_plus_ridge"]


def test_zero_ridge_predictions_match_least_squares_despite_nonunique_biases():
    pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0.2, 0.9, 0.1, 0.5])
    design = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]])
    expected = targets.mean() + design @ np.linalg.lstsq(design, targets - targets.mean(), rcond=None)[0]
    result = fit_joint_bias(pairs, targets, 2, 2)
    actual = result["global_mean"] + result["user_bias"][pairs[:, 0]] + result["book_bias"][pairs[:, 1]]
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_iteration_cap_is_reported_without_claiming_convergence():
    result = fit_joint_bias([[0, 0], [0, 1], [1, 0]], [0.1, 0.8, 0.4], 2, 2, max_iterations=1)
    assert result["iterations"] == 1
    assert not result["converged"]
    assert result["final_max_parameter_delta"] > 1e-10


@pytest.mark.parametrize("pairs,targets,users,books,kwargs", [
    ([], [], 1, 1, {}), ([[0, 0]], [np.nan], 1, 1, {}),
    ([[0, 0]], [0.5], 2, 1, {}), ([[1, 0]], [0.5], 1, 1, {}),
    ([[0, 0]], [0.5], 1, 1, {"user_shrinkage": -1}),
])
def test_joint_bias_rejects_invalid_fit_inputs(pairs, targets, users, books, kwargs):
    with pytest.raises(ValueError):
        fit_joint_bias(pairs, targets, users, books, **kwargs)


requires_tensorflow = pytest.mark.skipif(os.environ.get("BOOK_REC_RUN_TF_TESTS") != "1",
                                       reason="Set BOOK_REC_RUN_TF_TESTS=1 in the GPU test environment")


@requires_tensorflow
def test_tensorflow_dot_product_and_observed_pair_penalty():
    from book_recommender.factorization import create_factorization_model

    model = create_factorization_model(2, 2, embedding_size=2, global_mean=0.5,
                                        user_bias=[0.1, 0.2], book_bias=[-0.1, 0.3])
    users = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    books = np.array([[0.5, 1.], [2., 0.]], dtype=np.float32)
    model.user_embedding.embeddings.assign(users)
    model.book_embedding.embeddings.assign(books)
    pairs = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.int32)
    expected = [3.0, 3.0, 7.0]
    np.testing.assert_allclose(model(pairs).numpy().ravel(), expected, atol=1e-6)
    penalty = np.mean([5 + 1.25 + .01 + .01, 5 + 1.25 + .01 + .01,
                       25 + 4 + .04 + .09])
    assert float(model.observed_pair_penalty(pairs).numpy()) == pytest.approx(penalty)
    assert model.losses == []  # No additional Keras-wide L2 penalty.
    assert not model.global_mean.trainable
    assert model.user_bias.trainable and model.book_bias.trainable


@requires_tensorflow
def test_tensorflow_factorization_roundtrip_restores_all_weights(tmp_path):
    from book_recommender.factorization import create_factorization_model, load_factorization_model

    model = create_factorization_model(3, 2, global_mean=.7, user_bias=[.1, .2, -.1], book_bias=[0, .05])
    model.user_embedding.embeddings.assign_add(np.ones((3, 16), dtype=np.float32) * .2)
    path = tmp_path / "mf.keras"
    model.save(path)
    restored = load_factorization_model(path)
    assert len(restored.weights) == len(model.weights)
    for before, after in zip(model.weights, restored.weights):
        np.testing.assert_array_equal(before.numpy(), after.numpy())
    pairs = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int32)
    np.testing.assert_array_equal(model(pairs).numpy(), restored(pairs).numpy())


@requires_tensorflow
def test_tensorflow_gradient_updates_factors_and_biases_without_moving_global_mean():
    import tensorflow as tf
    from book_recommender.factorization import create_factorization_model

    model = create_factorization_model(2, 2, global_mean=.5)
    pairs = np.array([[0, 0], [1, 1]], dtype=np.int32)
    targets = tf.constant([[.1], [.9]])
    before = {variable.path: variable.numpy().copy() for variable in model.trainable_variables}
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((model(pairs) - targets) ** 2) + .001 * model.observed_pair_penalty(pairs)
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(gradient is not None for gradient in gradients)
    tf.keras.optimizers.Adam(.005).apply_gradients(zip(gradients, model.trainable_variables))
    assert all(np.any(variable.numpy() != before[variable.path]) for variable in model.trainable_variables)
    assert float(model.global_mean.numpy()) == .5
