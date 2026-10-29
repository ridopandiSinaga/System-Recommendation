"""Exercise the real optional TensorFlow model on small CPU datasets."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from book_recommender.neural import NeuralRatingModel, NeuralRecommender  # noqa: E402


def sample_books():
    return pd.DataFrame(
        [
            ("A", "Alpha", "Author A", "First Press"),
            ("A2", "Alpha", "Author A", "Second Press"),
            ("B", "Beta", "Author B", "First Press"),
            ("C", "Gamma", "Author C", "First Press"),
            ("C2", "Gamma", "Author C", "Second Press"),
            ("D", "Delta", "Author D", "First Press"),
            ("CONTENT", "Metadata only", "Author E", "First Press"),
        ],
        columns=["isbn", "book_title", "book_author", "publisher"],
    )


def sample_ratings():
    return pd.DataFrame(
        [
            (1, "A", 9), (1, "B", 2),
            (2, "A2", 8), (2, "C", 8), (2, "C2", 7), (2, "D", 3),
            (3, "B", 7), (3, "C", 3), (3, "D", 6),
        ],
        columns=["user_id", "isbn", "book_rating"],
    )


@pytest.fixture(autouse=True)
def clear_tensorflow_state():
    tf.keras.backend.clear_session()
    yield
    tf.keras.backend.clear_session()


@pytest.fixture
def initialized():
    return NeuralRecommender(
        embedding_size=4, hidden_units=(8, 4), learning_rate=0.02, random_state=42,
    ).initialize(sample_books(), sample_ratings())


def test_forward_prediction_does_not_depend_on_other_pairs_in_batch(initialized):
    model = initialized.model
    pair = np.array([[0, 0]], dtype=np.int32)
    batch = np.array([[1, 2], [0, 0], [2, 1], [0, 0]], dtype=np.int32)

    single_prediction = model(pair, training=False).numpy()
    batch_predictions = model(batch, training=False).numpy()

    assert isinstance(model, NeuralRatingModel)
    assert batch_predictions.shape == (4, 1)
    assert np.isfinite(batch_predictions).all()
    assert ((batch_predictions > 0) & (batch_predictions < 1)).all()
    np.testing.assert_allclose(batch_predictions[[1, 3]], single_prediction.repeat(2, axis=0))


def test_backpropagation_produces_finite_gradients_and_adam_changes_embedding(initialized):
    pairs, targets, _ = initialized.encode_ratings(initialized.ratings)
    predictions, mse, regularization, loss, gradients = initialized.loss_and_gradients(
        pairs, targets,
    )

    assert predictions.shape == targets.shape
    assert float(mse.numpy()) == pytest.approx(
        np.mean(np.square(targets - predictions.numpy())),
    )
    assert float(loss.numpy()) == pytest.approx(float((mse + regularization).numpy()))
    assert len(gradients) == len(initialized.model.trainable_variables)
    for gradient in gradients:
        assert gradient is not None
        assert np.isfinite(tf.convert_to_tensor(gradient).numpy()).all()
    assert float(tf.linalg.global_norm(gradients).numpy()) > 0

    trace = initialized.trace_training_step(pairs, targets)

    assert trace["optimizer"] == "Adam"
    assert trace["optimizer_iterations"] == 1
    assert trace["embedding_delta_norm"] > 0
    assert np.isfinite(list(trace["gradient_norms"].values())).all()
    assert np.linalg.norm(trace["user_embedding_gradient"]) > 0
    assert not np.array_equal(trace["user_embedding_before"], trace["user_embedding_after"])
    np.testing.assert_allclose(
        initialized.model.user_embedding.embeddings[pairs[0, 0]].numpy(),
        trace["user_embedding_after"],
    )


def test_real_training_learns_a_simple_rating_pattern():
    ratings = pd.DataFrame(
        [(1, "A", 10), (1, "B", 10), (2, "A", 1), (2, "B", 1)],
        columns=["user_id", "isbn", "book_rating"],
    )
    recommender = NeuralRecommender(
        embedding_size=4, hidden_units=(8, 4), learning_rate=0.03, l2=0,
    ).initialize(sample_books(), ratings)
    initial_mse = recommender.evaluate_ratings(ratings)["mse_normalized"]

    recommender.fit(sample_books(), ratings, epochs=35, batch_size=4)

    final_mse = recommender.evaluate_ratings(ratings)["mse_normalized"]
    assert final_mse < initial_mse * 0.25
    assert len(recommender.history) == 35
    assert recommender.history[-1]["train_mse"] == pytest.approx(final_mse)
    assert recommender.first_step_trace["embedding_delta_norm"] > 0
    assert int(recommender.optimizer.iterations.numpy()) == 35


def test_zero_ratings_and_metadata_only_books_do_not_create_embeddings():
    ratings = pd.concat([
        sample_ratings(),
        pd.DataFrame([(777, "CONTENT", 0)], columns=sample_ratings().columns),
    ], ignore_index=True)
    recommender = NeuralRecommender(embedding_size=4, hidden_units=(8,)).initialize(
        sample_books(), ratings,
    )

    assert set(recommender.user_to_index) == {1, 2, 3}
    assert set(recommender.isbn_to_index) == {"A", "A2", "B", "C", "C2", "D"}
    assert "CONTENT" in set(recommender.books.isbn)
    assert len(recommender.ratings) == len(sample_ratings())
    assert recommender.model.user_embedding.embeddings.shape == (3, 4)
    assert recommender.model.book_embedding.embeddings.shape == (6, 4)


def test_unknown_evaluation_pairs_are_counted_and_implicit_rows_excluded(initialized):
    evaluation = pd.DataFrame(
        [(1, "A", 9), (999, "A", 8), (1, "CONTENT", 7), (1, "ABSENT", 6),
         (777, "CONTENT", 0)],
        columns=["user_id", "isbn", "book_rating"],
    )

    pairs, targets, unknown = initialized.encode_ratings(evaluation)
    report = initialized.evaluate_ratings(evaluation)

    assert pairs.shape == (1, 2)
    np.testing.assert_allclose(targets, [[8 / 9]])
    assert unknown == 3
    assert report["explicit_pairs"] == 4
    assert report["known_pairs"] == 1
    assert report["unknown_pairs"] == 3
    assert report["scope"] == "known_user_and_book_pairs_only"
    expected = np.mean(np.square(targets - initialized.model(pairs).numpy()))
    assert report["mse_normalized"] == pytest.approx(expected)
    assert report["rmse_rating_scale"] == pytest.approx(9 * np.sqrt(expected))


def test_all_unknown_evaluation_returns_no_rating_metric(initialized):
    evaluation = pd.DataFrame(
        [(999, "CONTENT", 8)], columns=["user_id", "isbn", "book_rating"],
    )

    report = initialized.evaluate_ratings(evaluation)

    assert report["known_pairs"] == 0
    assert report["unknown_pairs"] == 1
    assert report["mse_normalized"] is None
    assert report["rmse_rating_scale"] is None


@pytest.mark.parametrize("validation_isbn", ["A", "A2"])
def test_validation_rejects_same_user_work_even_for_other_editions(validation_isbn):
    validation = pd.DataFrame(
        [(1, validation_isbn, 5)], columns=["user_id", "isbn", "book_rating"],
    )

    with pytest.raises(ValueError, match="same user/canonical work"):
        NeuralRecommender(embedding_size=4, hidden_units=(8,)).fit(
            sample_books(), sample_ratings(), validation_ratings=validation, epochs=1,
        )


def test_validation_with_no_known_pairs_is_rejected():
    validation = pd.DataFrame(
        [(999, "CONTENT", 8)], columns=["user_id", "isbn", "book_rating"],
    )

    with pytest.raises(ValueError, match="no known user-book pairs"):
        NeuralRecommender(embedding_size=4, hidden_units=(8,)).fit(
            sample_books(), sample_ratings(), validation_ratings=validation, epochs=1,
        )


def test_best_validation_weights_are_restored_after_real_updates(monkeypatch):
    recommender = NeuralRecommender(
        embedding_size=4, hidden_units=(8, 4), learning_rate=0.02,
    )
    validation = pd.DataFrame(
        [(3, "A", 8)], columns=["user_id", "isbn", "book_rating"],
    )
    original_mse = recommender._mse
    validation_weights = []
    validation_losses = iter([0.04, 0.09, 0.1])

    def controlled_validation_mse(pairs, targets):
        if len(pairs) == 1:
            validation_weights.append(recommender.model.get_weights())
            return next(validation_losses)
        return original_mse(pairs, targets)

    monkeypatch.setattr(recommender, "_mse", controlled_validation_mse)
    recommender.fit(
        sample_books(), sample_ratings(), validation_ratings=validation,
        epochs=6, batch_size=9, patience=2,
    )

    assert len(recommender.history) == 3
    assert recommender.best_epoch == 1
    for restored, best in zip(recommender.model.get_weights(), validation_weights[0]):
        np.testing.assert_array_equal(restored, best)
    assert any(
        not np.array_equal(first, last)
        for first, last in zip(validation_weights[0], validation_weights[-1])
    )


def test_neural_recommendations_exclude_read_works_and_deduplicate_editions(
    initialized, monkeypatch,
):
    # Deliberately rank a later ISBN edition highest to catch reranking before sorting.
    scores = {
        initialized.isbn_to_index[isbn]: score
        for isbn, score in {"C": 0.2, "C2": 0.9, "D": 0.8}.items()
    }
    monkeypatch.setattr(
        initialized, "_predict_encoded",
        lambda pairs: np.array([[scores[index]] for index in pairs[:, 1]], dtype=np.float32),
    )
    recommendations = initialized.recommend_for_user(1, top_n=10)

    assert len(recommendations) == 2
    assert recommendations.isbn.tolist() == ["C2", "D"]
    assert set(recommendations.book_title) == {"Gamma", "Delta"}
    assert recommendations.source.eq("neural").all()
    assert recommendations.score.between(1, 10).all()
    assert recommendations.score.is_monotonic_decreasing


def test_cold_user_fallback_is_identified_and_deduplicated(initialized):
    recommendations = initialized.recommend_for_user(999, top_n=10)

    assert len(recommendations) == 4
    assert recommendations.source.eq("popularity").all()
    assert recommendations.book_title.is_unique
    assert set(recommendations.book_title) == {"Alpha", "Beta", "Gamma", "Delta"}
    assert "CONTENT" not in set(recommendations.isbn)
    assert recommendations.isbn.tolist() == ["A", "C", "B", "D"]
    assert recommendations.score.is_monotonic_decreasing


def test_keras_artifact_roundtrip_preserves_predictions_and_recommendations(tmp_path):
    recommender = NeuralRecommender(
        embedding_size=4, hidden_units=(8, 4), learning_rate=0.02,
    ).fit(sample_books(), sample_ratings(), epochs=2, batch_size=9)
    pairs, _, _ = recommender.encode_ratings(sample_ratings())
    predictions = recommender.model(pairs, training=False).numpy()
    recommendations = recommender.recommend_for_user(1, top_n=10)
    artifact = tmp_path / "neural"

    recommender.save(artifact)
    loaded = NeuralRecommender.load(artifact)

    assert (artifact / "network.keras").is_file()
    assert (artifact / "metadata.json").is_file()
    assert loaded.user_to_index == recommender.user_to_index
    assert loaded.isbn_to_index == recommender.isbn_to_index
    assert loaded.history == recommender.history
    assert loaded.first_step_trace == recommender.first_step_trace
    np.testing.assert_allclose(loaded.model(pairs, training=False).numpy(), predictions)
    pd.testing.assert_frame_equal(loaded.recommend_for_user(1, top_n=10), recommendations)
    assert loaded.recommend_for_user(999, top_n=1).source.eq("popularity").all()


def test_epoch_callback_checkpoint_restores_adam_and_replays_next_epoch(tmp_path):
    config = {"embedding_size": 4, "hidden_units": (8, 4), "learning_rate": 0.02}
    recommender = NeuralRecommender(**config)
    validation = pd.DataFrame(
        [(3, "A", 8)], columns=["user_id", "isbn", "book_rating"],
    )
    snapshots = []
    checkpoint_paths = []

    def checkpoint_epoch(current, record):
        assert current is recommender
        assert current.history[-1] == record
        assert len(current.history) == record["epoch"]
        assert int(current.optimizer.iterations.numpy()) == record["epoch"]
        assert current.best_epoch == min(
            current.history, key=lambda epoch: epoch["val_mse"],
        )["epoch"]
        assert current.first_step_trace["embedding_delta_norm"] > 0
        pairs, _, _ = current.encode_ratings(sample_ratings())
        snapshots.append({
            "predictions": current.model(pairs, training=False).numpy(),
            "optimizer": [variable.numpy().copy() for variable in current.optimizer.variables],
            "rng_state": deepcopy(current.training_rng_state),
        })
        if record["epoch"] == 1:
            checkpoint_paths.append(
                tf.train.Checkpoint(model=current.model, optimizer=current.optimizer).save(
                    str(tmp_path / "training"),
                ),
            )
        # The callback receives a copy; annotations must not alter training history.
        record["callback_only"] = True

    recommender.fit(
        sample_books(), sample_ratings(), validation_ratings=validation,
        epochs=2, batch_size=9, epoch_callback=checkpoint_epoch,
    )

    assert len(snapshots) == 2
    assert all("callback_only" not in record for record in recommender.history)
    restored = NeuralRecommender(**config).initialize(sample_books(), sample_ratings())
    restored.optimizer.build(restored.model.trainable_variables)
    checkpoint = tf.train.Checkpoint(model=restored.model, optimizer=restored.optimizer)
    checkpoint.restore(checkpoint_paths[0]).assert_consumed()
    pairs, targets, _ = restored.encode_ratings(sample_ratings())
    np.testing.assert_allclose(
        restored.model(pairs, training=False).numpy(), snapshots[0]["predictions"],
    )
    assert int(restored.optimizer.iterations.numpy()) == 1
    for variable, saved in zip(restored.optimizer.variables, snapshots[0]["optimizer"]):
        np.testing.assert_array_equal(variable.numpy(), saved)

    # Replay the next epoch with the saved RNG state and restored Adam moments.
    rng = np.random.default_rng()
    rng.bit_generator.state = snapshots[0]["rng_state"]
    order = rng.permutation(len(pairs))
    restored._compiled_step(pairs[order], targets[order])
    assert int(restored.optimizer.iterations.numpy()) == 2
    np.testing.assert_allclose(
        restored.model(pairs, training=False).numpy(), snapshots[1]["predictions"],
        rtol=1e-6, atol=1e-7,
    )
    for variable, saved in zip(restored.optimizer.variables, snapshots[1]["optimizer"]):
        np.testing.assert_allclose(variable.numpy(), saved, rtol=1e-6, atol=1e-7)
