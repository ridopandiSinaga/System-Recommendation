"""Behavioral checks for the separately trained additive neural residual."""
import json

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from book_recommender.residual_neural import (  # noqa: E402
    ResidualNeuralRecommender,
    ResidualRatingModel,
)


def books():
    return pd.DataFrame(
        [("A", "Alpha", "Author A", "Press"), ("A2", "Alpha", "Author A", "Press"),
         ("B", "Beta", "Author B", "Press"), ("C", "Gamma", "Author C", "Press"),
         ("UNSEEN", "Metadata only", "Author D", "Press")],
        columns=["isbn", "book_title", "book_author", "publisher"],
    )


def ratings():
    return pd.DataFrame(
        [(1, "A", 9), (1, "B", 3), (2, "A", 7), (2, "B", 5), (2, "C", 9)],
        columns=["user_id", "isbn", "book_rating"],
    )


@pytest.fixture(autouse=True)
def clear_tensorflow_state():
    tf.keras.backend.clear_session()
    yield
    tf.keras.backend.clear_session()


def model(**kwargs):
    return ResidualNeuralRecommender(embedding_size=4, hidden_units=(8, 4), **kwargs)


def test_initial_predictions_exactly_match_independent_train_bias_formula():
    recommender = model(user_shrinkage=2, item_shrinkage=3).initialize(books(), ratings())
    pairs, _, _ = recommender.encode_ratings(ratings())
    # Independently derive the baseline in rating units, not using model weights.
    global_mean = ratings().book_rating.mean()
    user_bias = {1: (12 - 2 * global_mean) / 4, 2: (21 - 3 * global_mean) / 5}
    residuals = ratings().assign(
        residual=lambda rows: rows.book_rating - global_mean - rows.user_id.map(user_bias),
    )
    item_bias = residuals.groupby("isbn").residual.agg(["sum", "count"])
    item_bias = (item_bias["sum"] / (item_bias["count"] + 3)).to_dict()
    expected = np.array([
        (global_mean + user_bias[row.user_id] + item_bias[row.isbn] - 1) / 9
        for row in ratings().itertuples(index=False)
    ])[:, None]

    assert isinstance(recommender.model, ResidualRatingModel)
    np.testing.assert_allclose(recommender.model(pairs).numpy(), expected, atol=1e-7)
    np.testing.assert_array_equal(
        recommender.model(pairs).numpy(), recommender.baseline_predictions(pairs, clip=False),
    )
    assert not recommender.model.global_mean.trainable
    assert not recommender.model.user_bias.trainable
    assert not recommender.model.book_bias.trainable


def test_baseline_uses_training_only_and_evaluation_cannot_add_vocabulary():
    train = pd.concat([
        ratings(), pd.DataFrame([(777, "UNSEEN", 0)], columns=ratings().columns),
    ], ignore_index=True)
    recommender = model().initialize(books(), train)
    frozen = [value.numpy().copy() for value in recommender.model.non_trainable_variables]
    held_out = pd.DataFrame(
        [(1, "C", 1), (1, "UNSEEN", 10), (999, "A", 10), (999, "UNSEEN", 10)],
        columns=ratings().columns,
    )

    report = recommender.evaluate_ratings(held_out)

    assert set(recommender.user_to_index) == {1, 2}
    assert set(recommender.isbn_to_index) == {"A", "B", "C"}
    assert recommender.baseline_summary["train_pairs"] == 5
    assert report["known_pairs"] == 1
    assert report["unknown_pairs"] == 3
    for actual, expected in zip(recommender.model.non_trainable_variables, frozen):
        np.testing.assert_array_equal(actual.numpy(), expected)
    cold = recommender.recommend_for_user(999)
    assert cold.source.eq("popularity").all()
    assert "UNSEEN" not in set(cold.isbn)


def test_zero_scale_remains_bias_only_through_real_adam_updates():
    recommender = model(residual_scale=0, l2=0).initialize(books(), ratings())
    pairs, _, _ = recommender.encode_ratings(ratings())
    before = recommender.baseline_predictions(pairs)

    recommender.fit(books(), ratings(), epochs=2, batch_size=5)

    np.testing.assert_array_equal(recommender._predict_encoded(pairs), before)
    assert recommender.first_step_trace["embedding_data_gradient_norm_before"] == 0
    assert recommender.first_step_trace["embedding_delta_norm"] == 0
    assert int(recommender.optimizer.iterations.numpy()) == 2
    assert np.isfinite(list(recommender.first_step_trace["gradient_norms"].values())).all()
    assert recommender.recommend_for_user(1).source.eq("bias_only").all()
    assert recommender.recommend_for_user(999).source.eq("popularity").all()


def test_zero_head_defers_embedding_data_gradient_then_updates_embeddings():
    recommender = model(l2=0, learning_rate=0.02).initialize(books(), ratings())
    pairs, targets, _ = recommender.encode_ratings(ratings())
    before = recommender.model.user_embedding.embeddings.numpy().copy()

    trace = recommender.trace_training_step(pairs, targets)
    _, _, _, _, gradients = recommender.loss_and_gradients(pairs, targets)

    assert trace["embedding_data_gradient_norm_before"] == 0
    assert trace["embedding_delta_norm"] == 0
    np.testing.assert_array_equal(recommender.model.user_embedding.embeddings.numpy(), before)
    assert float(tf.linalg.global_norm([gradients[0]]).numpy()) > 0
    recommender._compiled_step(pairs, targets)
    assert np.linalg.norm(recommender.model.user_embedding.embeddings.numpy() - before) > 0


def test_real_residual_learns_interaction_that_user_and_item_means_cannot_explain():
    interactions = pd.DataFrame(
        [(1, "A", 10), (1, "B", 1), (2, "A", 1), (2, "B", 10)],
        columns=ratings().columns,
    )
    recommender = model(l2=0, learning_rate=0.03).initialize(books(), interactions)
    initial_mse = recommender.evaluate_ratings(interactions)["mse_normalized"]

    recommender.fit(books(), interactions, epochs=80, batch_size=4)

    assert initial_mse == pytest.approx(0.25)
    assert recommender.evaluate_ratings(interactions)["mse_normalized"] < initial_mse * 0.1
    pairs, _, _ = recommender.encode_ratings(interactions)
    np.testing.assert_allclose(recommender.baseline_predictions(pairs), 0.5)


def test_reports_clip_linear_predictions_but_training_loss_stays_raw():
    recommender = model(l2=0).initialize(books(), ratings())
    pairs, targets, _ = recommender.encode_ratings(ratings())
    recommender.model.output_layer.bias.assign([2.0])
    raw, mse, _, _, _ = recommender.loss_and_gradients(pairs, targets)

    assert (raw.numpy() > 1).all()
    assert float(mse.numpy()) == pytest.approx(np.mean((raw.numpy() - targets) ** 2))
    np.testing.assert_array_equal(recommender._predict_encoded(pairs), np.ones((5, 1)))
    report = recommender.evaluate_ratings(ratings())
    assert report["mse_normalized"] == pytest.approx(np.mean((1 - targets) ** 2))
    recommendations = recommender.recommend_for_user(1)
    assert recommendations.score.eq(10).all()


def test_canonical_validation_overlap_is_rejected_before_updates():
    validation = pd.DataFrame([(1, "A2", 8)], columns=ratings().columns)
    recommender = model()

    with pytest.raises(ValueError, match="same user/canonical work"):
        recommender.fit(books(), ratings(), validation_ratings=validation, epochs=1)

    assert int(recommender.optimizer.iterations.numpy()) == 0


def test_roundtrip_preserves_biases_nondefault_config_predictions_and_history(tmp_path):
    recommender = model(
        user_shrinkage=3, item_shrinkage=7, residual_scale=0.25, learning_rate=0.02,
    ).fit(books(), ratings(), epochs=2, batch_size=5)
    pairs, _, _ = recommender.encode_ratings(ratings())
    expected = recommender._predict_encoded(pairs)
    artifact = tmp_path / "residual"

    recommender.save(artifact)
    loaded = ResidualNeuralRecommender.load(artifact)

    metadata = json.loads((artifact / "metadata.json").read_text())
    assert metadata["architecture"] == "residual"
    assert metadata["model_kind"] == "bias_residual_neural"
    assert metadata["predictor_kind"] == "residual_neural"
    assert loaded.user_shrinkage == 3
    assert loaded.item_shrinkage == 7
    assert loaded.residual_scale == 0.25
    assert loaded.history == recommender.history
    assert loaded.first_step_trace == recommender.first_step_trace
    assert loaded.recommend_for_user(1).source.eq("residual_neural").all()
    assert len(loaded.model.weights) == len(recommender.model.weights)
    for restored_weight, trained_weight in zip(loaded.model.weights, recommender.model.weights):
        np.testing.assert_array_equal(restored_weight.numpy(), trained_weight.numpy())
    np.testing.assert_allclose(loaded._predict_encoded(pairs), expected)
    np.testing.assert_array_equal(loaded.user_bias, recommender.user_bias)
    np.testing.assert_array_equal(loaded.book_bias, recommender.book_bias)
    np.testing.assert_array_equal(
        loaded.baseline_predictions(pairs), recommender.baseline_predictions(pairs),
    )
    pd.testing.assert_frame_equal(
        loaded.recommend_for_user(1), recommender.recommend_for_user(1),
    )


def test_artifact_type_marker_prevents_loading_wrong_architecture(tmp_path):
    artifact = tmp_path / "wrong"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(json.dumps({
        "format_version": 1, "architecture": "mlp", "model_kind": "bias_residual_neural",
    }))

    with pytest.raises(ValueError, match="Unsupported residual"):
        ResidualNeuralRecommender.load(artifact)


@pytest.mark.parametrize("parameter", ["user_shrinkage", "item_shrinkage", "residual_scale"])
def test_negative_regularization_or_scale_is_rejected(parameter):
    with pytest.raises(ValueError, match=parameter):
        model(**{parameter: -1})
