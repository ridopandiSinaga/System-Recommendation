"""A train-derived rating baseline plus a small nonlinear learned residual.

The model predicts on the same fixed ``(rating - 1) / 9`` scale as the original
neural example. Its linear output trains against raw targets. Rating reports,
early stopping, and recommendation scores clip predictions to the valid scale.
This module is separate so the original educational experiment stays reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from book_recommender.collaborative import _validate_non_negative_float
from book_recommender.neural import NeuralRecommender


@tf.keras.utils.register_keras_serializable(package="book_recommender")
class ResidualRatingModel(tf.keras.Model):
    """Frozen additive biases and a zero-initialized linear residual head.

    Zero initialization makes the initial prediction exactly the bias baseline.
    The head learns on the first update; its nonzero weights then pass the data
    gradient into the embeddings. L2 can update embeddings on the first step.
    Baseline weights are nontrainable and are included in Keras checkpoints.
    """

    def __init__(
        self, num_users: int, num_books: int, embedding_size: int = 16,
        hidden_units: tuple[int, ...] = (32, 16), l2: float = 1e-5,
        residual_scale: float = 1.0, **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.hidden_units = tuple(hidden_units)
        self.l2 = l2
        self.residual_scale = residual_scale
        regularizer = tf.keras.regularizers.L2(l2)
        # Keep user embeddings first among trainable variables for the inherited
        # inspectable training step. The bias weights below are nontrainable.
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
                units, activation="leaky_relu", kernel_regularizer=regularizer,
                name=f"hidden_{index + 1}",
            ) for index, units in enumerate(hidden_units)
        ]
        self.output_layer = tf.keras.layers.Dense(
            1, activation=None, kernel_initializer="zeros", bias_initializer="zeros",
            name="rating_residual",
        )
        self.global_mean = self.add_weight(
            name="global_mean", shape=(), initializer="zeros", trainable=False,
        )
        self.user_bias = self.add_weight(
            name="user_bias", shape=(num_users,), initializer="zeros", trainable=False,
        )
        self.book_bias = self.add_weight(
            name="book_bias", shape=(num_books,), initializer="zeros", trainable=False,
        )

    def build(self, input_shape):
        # Keras must create every child variable before deserializing its weights.
        # Automatic tracing otherwise invents float inputs for the integer gather
        # in baseline(), leaving child layers unbuilt when restoring an artifact.
        self.user_embedding.build((None,))
        self.book_embedding.build((None,))
        width = 3 * self.embedding_size
        for layer in self.hidden:
            layer.build((None, width))
            width = layer.units
        self.output_layer.build((None, width))
        super().build(input_shape)

    def baseline(self, inputs):
        values = (
            self.global_mean + tf.gather(self.user_bias, inputs[:, 0])
            + tf.gather(self.book_bias, inputs[:, 1])
        )
        return values[:, None]

    def call(self, inputs, training=False):
        user = self.user_embedding(inputs[:, 0])
        book = self.book_embedding(inputs[:, 1])
        features = tf.concat([user, book, user * book], axis=1)
        for layer in self.hidden:
            features = layer(features)
        # Multiplication, even for scale=0, keeps every gradient path connected.
        return self.baseline(inputs) + self.residual_scale * self.output_layer(features)

    def get_config(self):
        return {
            **super().get_config(), "num_users": self.num_users, "num_books": self.num_books,
            "embedding_size": self.embedding_size, "hidden_units": self.hidden_units,
            "l2": self.l2, "residual_scale": self.residual_scale,
        }


class ResidualNeuralRecommender(NeuralRecommender):
    """Fit all biases from explicit training ratings, then learn their residual.

    User bias is ``sum(y - global) / (count + user_shrinkage)``. Book bias is
    ``sum(y - global - user_bias) / (count + item_shrinkage)``. Validation and
    test ratings never contribute to these statistics or the vocabularies.

    Unknown IDs retain the original recommender's explicit metric exclusion and
    popularity fallback. The public bias arrays also support separately labelled
    fallback evaluation, with zero bias for an unknown user or unknown book.
    """

    MODEL_KIND = "bias_residual_neural"

    def __init__(
        self, embedding_size: int = 16, hidden_units: tuple[int, ...] = (32, 16),
        learning_rate: float = 0.001, l2: float = 1e-5, random_state: int = 42,
        user_shrinkage: float = 0.0, item_shrinkage: float = 20.0,
        residual_scale: float = 1.0,
    ):
        super().__init__(embedding_size, hidden_units, learning_rate, l2, random_state)
        self.user_shrinkage = _validate_non_negative_float(user_shrinkage, "user_shrinkage")
        self.item_shrinkage = _validate_non_negative_float(item_shrinkage, "item_shrinkage")
        self.residual_scale = _validate_non_negative_float(residual_scale, "residual_scale")

    def initialize(self, books: pd.DataFrame, ratings: pd.DataFrame):
        # Reuse the original cleaning, train-only vocabulary, and work identity.
        super().initialize(books, ratings)
        pairs, targets, _ = self.encode_ratings(self.ratings)
        targets = targets[:, 0].astype(np.float64)
        self.global_mean_normalized = float(targets.mean())
        user_count = np.bincount(pairs[:, 0], minlength=len(self.user_to_index))
        user_sum = np.bincount(
            pairs[:, 0], weights=targets - self.global_mean_normalized,
            minlength=len(self.user_to_index),
        )
        self.user_bias = (user_sum / (user_count + self.user_shrinkage)).astype(np.float32)
        book_count = np.bincount(pairs[:, 1], minlength=len(self.isbn_to_index))
        book_sum = np.bincount(
            pairs[:, 1],
            weights=targets - self.global_mean_normalized - self.user_bias[pairs[:, 0]],
            minlength=len(self.isbn_to_index),
        )
        self.book_bias = (book_sum / (book_count + self.item_shrinkage)).astype(np.float32)
        tf.keras.utils.set_random_seed(self.random_state)
        self.model = ResidualRatingModel(
            len(self.user_to_index), len(self.isbn_to_index), self.embedding_size,
            self.hidden_units, self.l2, self.residual_scale,
        )
        self.model(tf.zeros((1, 2), dtype=tf.int32))
        self.model.global_mean.assign(self.global_mean_normalized)
        self.model.user_bias.assign(self.user_bias)
        self.model.book_bias.assign(self.book_bias)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self._compiled_step = tf.function(self._train_batch, reduce_retracing=True)
        self.baseline_summary = {
            "train_pairs": len(self.ratings), "users": len(self.user_to_index),
            "books": len(self.isbn_to_index),
            "global_mean_rating": 1 + 9 * self.global_mean_normalized,
            "user_shrinkage": self.user_shrinkage, "item_shrinkage": self.item_shrinkage,
            "biases_trainable": False,
            "normalization": "(rating - 1) / 9",
            "prediction_reporting": "clip normalized prediction to [0, 1]",
            "training_objective": "raw linear normalized MSE plus L2",
            "hidden_activation": "leaky_relu (negative_slope=0.2)",
        }
        return self

    def baseline_predictions(self, pairs, *, clip=True):
        """Return train-only additive baseline predictions on the normalized scale."""
        self._check_initialized()
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        predictions = self.model.baseline(pairs).numpy()
        return np.clip(predictions, 0.0, 1.0) if clip else predictions

    def _predict_encoded(self, pairs, batch_size=1024):
        # This is shared by evaluation, early stopping, and recommendations.
        return np.clip(super()._predict_encoded(pairs, batch_size=batch_size), 0.0, 1.0)

    @property
    def predictor_kind(self):
        return "bias_only" if self.residual_scale == 0 else "residual_neural"

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        recommendations = super().recommend_for_user(user_id, top_n=top_n)
        recommendations.loc[recommendations.source == "neural", "source"] = self.predictor_kind
        return recommendations

    def trace_training_step(self, pairs, targets):
        pairs = np.asarray(pairs, dtype=np.int32)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        baseline = self.baseline_predictions(pairs, clip=False)
        with tf.GradientTape() as tape:
            raw = self.model(pairs, training=True)
            data_mse = tf.reduce_mean(tf.square(targets - raw))
        embedding_data_gradient = tape.gradient(data_mse, self.model.user_embedding.embeddings)
        trace = super().trace_training_step(pairs, targets)
        trace.update({
            "model_kind": self.MODEL_KIND,
            "baseline_predictions_normalized": baseline[:8, 0].tolist(),
            "residual_scale": self.residual_scale,
            "embedding_data_gradient_norm_before": float(
                tf.linalg.global_norm([embedding_data_gradient]).numpy()
            ),
            "head_initialization": "zeros; first data update learns the residual head",
            "embedding_first_step_note": (
                "At initialization the zero head blocks the data gradient to embeddings; "
                "L2 may update them. Data gradients propagate after the head learns."
            ),
            "training_objective": "raw linear normalized MSE plus L2",
            "prediction_reporting": "clip normalized prediction to [0, 1]",
        })
        return trace

    def save(self, path: str | Path) -> None:
        super().save(path)
        metadata_path = Path(path) / "metadata.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["architecture"] = "residual"
        payload["model_kind"] = self.MODEL_KIND
        payload["predictor_kind"] = self.predictor_kind
        payload["config"].update({
            "user_shrinkage": self.user_shrinkage, "item_shrinkage": self.item_shrinkage,
            "residual_scale": self.residual_scale,
        })
        payload["baseline_summary"] = self.baseline_summary
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        payload = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if (
            payload.get("format_version") != cls.FORMAT_VERSION
            or payload.get("architecture") != "residual"
            or payload.get("model_kind") != cls.MODEL_KIND
        ):
            raise ValueError("Unsupported residual neural artifact.")
        recommender = cls(**payload["config"]).initialize(
            pd.DataFrame(payload["books"]), pd.DataFrame(payload["ratings"]),
        )
        model = tf.keras.models.load_model(path / "network.keras", compile=False, safe_mode=True)
        if not isinstance(model, ResidualRatingModel):
            raise ValueError("Artifact is not a residual neural rating model.")
        if (model.num_users, model.num_books) != (
            len(recommender.user_to_index), len(recommender.isbn_to_index),
        ):
            raise ValueError("Residual model dimensions do not match the stored vocabulary.")
        recommender.model = model
        recommender.global_mean_normalized = float(model.global_mean.numpy())
        recommender.user_bias = model.user_bias.numpy().copy()
        recommender.book_bias = model.book_bias.numpy().copy()
        recommender._compiled_step = tf.function(recommender._train_batch, reduce_retracing=True)
        recommender.history = payload["history"]
        recommender.first_step_trace = payload["first_step_trace"]
        recommender.best_epoch = payload["best_epoch"]
        return recommender
