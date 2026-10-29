"""Neural rating prediction with an inspectable TensorFlow training step.

TensorFlow is an optional dependency: baseline serving never imports this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from book_recommender.collaborative import _validate_non_negative_float, _validate_user_id
from book_recommender.content_based import (
    _validate_positive_int,
    book_work_key,
    rerank_unique_works,
)
from book_recommender.data import clean_books, clean_ratings


@tf.keras.utils.register_keras_serializable(package="book_recommender")
class NeuralRatingModel(tf.keras.Model):
    """Learn user/book embeddings and nonlinear interactions between them."""

    def __init__(
        self, num_users: int, num_books: int, embedding_size: int = 16,
        hidden_units: tuple[int, ...] = (32, 16), l2: float = 1e-5, **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.hidden_units = tuple(hidden_units)
        self.l2 = l2
        regularizer = tf.keras.regularizers.L2(l2)
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_size, embeddings_regularizer=regularizer,
            name="user_embedding",
        )
        self.book_embedding = tf.keras.layers.Embedding(
            num_books, embedding_size, embeddings_regularizer=regularizer,
            name="book_embedding",
        )
        self.hidden = [
            tf.keras.layers.Dense(
                units, activation="relu", kernel_regularizer=regularizer,
                name=f"hidden_{index + 1}",
            )
            for index, units in enumerate(hidden_units)
        ]
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="rating")

    def call(self, inputs, training=False):
        user_vector = self.user_embedding(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        # Every row is one user-book pair; never contract across the batch axis.
        x = tf.concat([user_vector, book_vector, user_vector * book_vector], axis=1)
        for layer in self.hidden:
            x = layer(x)
        return self.output_layer(x)

    def get_config(self):
        return {
            **super().get_config(), "num_users": self.num_users, "num_books": self.num_books,
            "embedding_size": self.embedding_size, "hidden_units": self.hidden_units,
            "l2": self.l2,
        }


class NeuralRecommender:
    """Explicit-rating MLP with train-only vocabularies and popularity fallback.

    The fixed target scale is (rating - 1) / 9. Unknown IDs are excluded from
    rating metrics and counted explicitly; they never index random embeddings.
    """

    FORMAT_VERSION = 1

    def __init__(
        self, embedding_size: int = 16, hidden_units: tuple[int, ...] = (32, 16),
        learning_rate: float = 0.001, l2: float = 1e-5, random_state: int = 42,
    ):
        self.embedding_size = _validate_positive_int(embedding_size, "embedding_size")
        self.hidden_units = tuple(_validate_positive_int(n, "hidden_units") for n in hidden_units)
        if not self.hidden_units:
            raise ValueError("At least one hidden layer is required for the neural model.")
        self.learning_rate = _validate_non_negative_float(learning_rate, "learning_rate")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive.")
        self.l2 = _validate_non_negative_float(l2, "l2")
        if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
            raise ValueError("random_state must be an integer from 0 through 2**32 - 1.")
        if not 0 <= random_state < 2**32:
            raise ValueError("random_state must be an integer from 0 through 2**32 - 1.")
        self.random_state = int(random_state)
        self.model = None
        self.history: list[dict] = []
        self.first_step_trace: dict = {}

    def initialize(self, books: pd.DataFrame, ratings: pd.DataFrame) -> "NeuralRecommender":
        """Build untrained embeddings from explicit training interactions only."""
        self.books = clean_books(books)
        self.ratings = clean_ratings(ratings)
        self.ratings = self.ratings[self.ratings.isbn.isin(self.books.isbn)].reset_index(drop=True)
        if self.ratings.empty or self.ratings.book_rating.min() < 1:
            raise ValueError("Neural training requires explicit ratings on the 1-10 scale.")
        self.user_to_index = {
            int(user): index for index, user in enumerate(sorted(self.ratings.user_id.unique()))
        }
        self.isbn_to_index = {
            str(isbn): index for index, isbn in enumerate(sorted(self.ratings.isbn.unique()))
        }
        self.work_keys = {
            str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
            for row in self.books.itertuples(index=False)
        }
        tf.keras.utils.set_random_seed(self.random_state)
        self.model = NeuralRatingModel(
            len(self.user_to_index), len(self.isbn_to_index), self.embedding_size,
            self.hidden_units, self.l2,
        )
        self.model(tf.zeros((1, 2), dtype=tf.int32))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Bind a fresh graph to each freshly initialized network/optimizer.
        self._compiled_step = tf.function(self._train_batch, reduce_retracing=True)
        self.history = []
        self.first_step_trace = {}
        return self

    def encode_ratings(self, ratings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
        self._check_initialized()
        explicit = clean_ratings(ratings)
        if not explicit.empty and explicit.book_rating.min() < 1:
            raise ValueError("Neural targets must be explicit ratings on the 1-10 scale.")
        known = explicit[
            explicit.user_id.isin(self.user_to_index) & explicit.isbn.isin(self.isbn_to_index)
        ]
        pairs = np.column_stack([
            known.user_id.map(self.user_to_index).to_numpy(dtype=np.int32),
            known.isbn.map(self.isbn_to_index).to_numpy(dtype=np.int32),
        ])
        targets = ((known.book_rating.to_numpy(dtype=np.float32) - 1.0) / 9.0).reshape(-1, 1)
        return pairs, targets, len(explicit) - len(known)

    def loss_and_gradients(self, pairs, targets):
        """Forward propagation, regression loss, and backpropagation."""
        with tf.GradientTape() as tape:
            predictions = self.model(pairs, training=True)
            mse = tf.reduce_mean(tf.square(tf.cast(targets, tf.float32) - predictions))
            regularization = tf.add_n(self.model.losses)
            loss = mse + regularization
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return predictions, mse, regularization, loss, gradients

    def _train_batch(self, pairs, targets):
        _, _, _, loss, gradients = self.loss_and_gradients(pairs, targets)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def trace_training_step(self, pairs, targets) -> dict:
        """Perform one real Adam update and report its numerical evidence."""
        self._check_initialized()
        pairs = np.asarray(pairs, dtype=np.int32)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        if not len(pairs) or pairs.shape != (len(targets), 2):
            raise ValueError("A trace requires a nonempty batch of encoded user-book pairs.")
        user_index = int(pairs[0, 0])
        before = self.model.user_embedding.embeddings[user_index].numpy().copy()
        predictions, mse, regularization, loss, gradients = self.loss_and_gradients(pairs, targets)
        # Only the demonstration densifies this small gradient for one displayed row.
        embedding_gradient = tf.convert_to_tensor(gradients[0])[user_index].numpy()
        gradient_norms = {
            variable.path: float(tf.linalg.global_norm([gradient]).numpy())
            for variable, gradient in zip(self.model.trainable_variables, gradients)
        }
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        after = self.model.user_embedding.embeddings[user_index].numpy().copy()
        predictions_after = self.model(pairs, training=False).numpy()
        return {
            "encoded_pairs": pairs[:8].tolist(),
            "targets_normalized": targets[:8, 0].tolist(),
            "predictions_before": predictions.numpy()[:8, 0].tolist(),
            "predictions_after": predictions_after[:8, 0].tolist(),
            "mse_before": float(mse.numpy()),
            "regularization_loss": float(regularization.numpy()),
            "total_loss_before": float(loss.numpy()),
            "mse_after": float(np.mean(np.square(targets - predictions_after))),
            "gradient_norms": gradient_norms,
            "user_embedding_before": before.tolist(),
            "user_embedding_gradient": embedding_gradient.tolist(),
            "user_embedding_after": after.tolist(),
            "embedding_delta_norm": float(np.linalg.norm(after - before)),
            "optimizer": "Adam", "optimizer_iterations": int(self.optimizer.iterations.numpy()),
        }

    def fit(
        self, books: pd.DataFrame, ratings: pd.DataFrame, *,
        validation_ratings: pd.DataFrame | None = None, epochs: int = 10,
        batch_size: int = 256, patience: int = 3,
        epoch_callback: Callable[["NeuralRecommender", dict], None] | None = None,
    ) -> "NeuralRecommender":
        epochs = _validate_positive_int(epochs, "epochs")
        batch_size = _validate_positive_int(batch_size, "batch_size")
        patience = _validate_positive_int(patience, "patience")
        self.initialize(books, ratings)
        pairs, targets, _ = self.encode_ratings(self.ratings)
        if validation_ratings is not None:
            self._check_validation_overlap(validation_ratings)
            val_pairs, val_targets, _ = self.encode_ratings(validation_ratings)
            if not len(val_pairs):
                raise ValueError("Validation has no known user-book pairs after train-only filtering.")
        rng = np.random.default_rng(self.random_state)
        best_mse, best_weights, wait = float("inf"), None, 0
        self.best_epoch = None
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(pairs))
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                if not self.first_step_trace:
                    self.first_step_trace = self.trace_training_step(pairs[indices], targets[indices])
                else:
                    self._compiled_step(pairs[indices], targets[indices])
            train_mse = self._mse(pairs, targets)
            record = {"epoch": epoch, "train_mse": train_mse, "train_rmse": 9 * np.sqrt(train_mse)}
            if validation_ratings is not None:
                val_mse = self._mse(val_pairs, val_targets)
                record.update(val_mse=val_mse, val_rmse=9 * np.sqrt(val_mse))
                if val_mse < best_mse:
                    best_mse, best_weights, wait = val_mse, self.model.get_weights(), 0
                    self.best_epoch = epoch
                else:
                    wait += 1
            self.history.append(record)
            self.training_rng_state = rng.bit_generator.state
            if epoch_callback is not None:
                epoch_callback(self, record.copy())
            if validation_ratings is not None and wait >= patience:
                break
        if best_weights is not None:
            self.model.set_weights(best_weights)
        return self

    def _check_validation_overlap(self, validation_ratings):
        # Reject leakage across editions, not just duplicate user/ISBN pairs.
        training_pairs = {
            (int(row.user_id), self.work_keys[str(row.isbn)])
            for row in self.ratings.itertuples(index=False)
        }
        for row in clean_ratings(validation_ratings).itertuples(index=False):
            key = self.work_keys.get(str(row.isbn))
            if key is not None and (int(row.user_id), key) in training_pairs:
                raise ValueError("Training and validation contain the same user/canonical work.")

    def _predict_encoded(self, pairs, batch_size=1024):
        if not len(pairs):
            return np.empty((0, 1), dtype=np.float32)
        return np.concatenate([
            self.model(pairs[start:start + batch_size], training=False).numpy()
            for start in range(0, len(pairs), batch_size)
        ])

    def _mse(self, pairs, targets):
        return float(np.mean(np.square(targets - self._predict_encoded(pairs))))

    def evaluate_ratings(self, ratings: pd.DataFrame) -> dict:
        pairs, targets, unknown = self.encode_ratings(ratings)
        mse = self._mse(pairs, targets) if len(pairs) else None
        return {
            "explicit_pairs": len(pairs) + unknown, "known_pairs": len(pairs),
            "unknown_pairs": unknown, "mse_normalized": mse,
            "rmse_rating_scale": None if mse is None else 9 * np.sqrt(mse),
            "scope": "known_user_and_book_pairs_only",
        }

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        self._check_initialized()
        user_id = _validate_user_id(user_id)
        top_n = _validate_positive_int(top_n, "top_n")
        read = self.ratings[self.ratings.user_id == user_id]
        excluded = {self.work_keys[str(isbn)] for isbn in read.isbn}
        if user_id in self.user_to_index:
            candidates = [
                isbn for isbn in self.isbn_to_index if self.work_keys[isbn] not in excluded
            ]
            pairs = np.array([
                [self.user_to_index[user_id], self.isbn_to_index[isbn]] for isbn in candidates
            ], dtype=np.int32).reshape(-1, 2)
            scored = pd.DataFrame({
                "isbn": candidates, "score": 1 + 9 * self._predict_encoded(pairs).ravel(),
                "source": "neural",
            })
        else:
            scored = self.ratings.groupby("isbn").agg(
                mean_rating=("book_rating", "mean"), rating_count=("book_rating", "size"),
            ).reset_index()
            scored["score"] = scored.mean_rating * np.log1p(scored.rating_count)
            scored["source"] = "popularity"
        result = scored.merge(self.books, on="isbn", how="left").sort_values(
            ["score", "isbn"], ascending=[False, True], kind="stable",
        )
        return rerank_unique_works(result, top_n=top_n, excluded_work_keys=excluded)

    def save(self, path: str | Path) -> None:
        self._check_initialized()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "network.keras")
        payload = {
            "format_version": self.FORMAT_VERSION,
            "config": {
                "embedding_size": self.embedding_size, "hidden_units": self.hidden_units,
                "learning_rate": self.learning_rate, "l2": self.l2,
                "random_state": self.random_state,
            },
            "books": json.loads(self.books.to_json(orient="records")),
            "ratings": json.loads(self.ratings.to_json(orient="records")),
            "history": self.history, "first_step_trace": self.first_step_trace,
            "best_epoch": getattr(self, "best_epoch", None),
        }
        (path / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NeuralRecommender":
        path = Path(path)
        payload = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported neural artifact version.")
        recommender = cls(**payload["config"]).initialize(
            pd.DataFrame(payload["books"]), pd.DataFrame(payload["ratings"]),
        )
        model = tf.keras.models.load_model(path / "network.keras", compile=False, safe_mode=True)
        if not isinstance(model, NeuralRatingModel):
            raise ValueError("Artifact is not a neural rating model.")
        if (model.num_users, model.num_books) != (
            len(recommender.user_to_index), len(recommender.isbn_to_index),
        ):
            raise ValueError("Neural model dimensions do not match the stored vocabulary.")
        recommender.model = model
        recommender._compiled_step = tf.function(recommender._train_batch, reduce_retracing=True)
        recommender.history = payload["history"]
        recommender.first_step_trace = payload["first_step_trace"]
        recommender.best_epoch = payload["best_epoch"]
        return recommender

    def _check_initialized(self):
        if self.model is None:
            raise RuntimeError("Initialize or fit the neural recommender first.")
