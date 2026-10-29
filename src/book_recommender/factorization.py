"""Joint additive biases and an optional TensorFlow biased matrix factorization.

NumPy bias fitting can be imported without loading TensorFlow. The Keras model
class is registered lazily by ``create_factorization_model`` or its loader.
"""
from __future__ import annotations

import numpy as np

_MODEL_CLASS = None


def fit_joint_bias(pairs, targets, num_users, num_books, *, user_shrinkage=0.0,
                   item_shrinkage=0.0, max_iterations=1000, tolerance=1e-10):
    """Minimize sum squared residuals plus additive bias ridge penalties.

    The global mean is fixed at the mean of fit targets. Alternating exact
    coordinate updates see fit observations only. Targets use the normalized
    rating scale; hyperparameters penalize the summed objective, not its mean.
    """
    pairs = np.asarray(pairs, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if pairs.shape != (len(targets), 2) or not len(targets):
        raise ValueError("Bias fitting requires nonempty aligned user/book pairs and targets.")
    if not np.isfinite(targets).all() or not np.isfinite([user_shrinkage, item_shrinkage]).all():
        raise ValueError("Bias targets and penalties must be finite.")
    if (min(user_shrinkage, item_shrinkage) < 0 or max_iterations < 1 or tolerance <= 0
            or min(num_users, num_books) < 1):
        raise ValueError("Invalid bias dimensions, penalties, iteration limit, or tolerance.")
    if ((pairs < 0).any() or (pairs[:, 0] >= num_users).any()
            or (pairs[:, 1] >= num_books).any()):
        raise ValueError("Encoded bias pair lies outside its vocabulary.")
    user_count = np.bincount(pairs[:, 0], minlength=num_users)
    book_count = np.bincount(pairs[:, 1], minlength=num_books)
    if (user_count == 0).any() or (book_count == 0).any():
        raise ValueError("Bias vocabulary must contain only IDs observed in fit data.")
    global_mean = float(targets.mean())
    user_bias, book_bias = np.zeros(num_users), np.zeros(num_books)
    history = []
    converged = False
    for iteration in range(1, max_iterations + 1):
        next_user = np.bincount(pairs[:, 0], weights=targets - global_mean - book_bias[pairs[:, 1]],
                                minlength=num_users) / (user_count + user_shrinkage)
        next_book = np.bincount(pairs[:, 1], weights=targets - global_mean - next_user[pairs[:, 0]],
                                minlength=num_books) / (book_count + item_shrinkage)
        delta = float(max(np.max(np.abs(next_user - user_bias)), np.max(np.abs(next_book - book_bias))))
        user_bias, book_bias = next_user, next_book
        if iteration == 1 or iteration % 50 == 0 or delta <= tolerance or iteration == max_iterations:
            error = targets - global_mean - user_bias[pairs[:, 0]] - book_bias[pairs[:, 1]]
            objective = float(np.sum(error ** 2) + user_shrinkage * np.sum(user_bias ** 2)
                              + item_shrinkage * np.sum(book_bias ** 2))
            history.append({"iteration": iteration, "max_parameter_delta": delta,
                            "sum_squared_error_plus_ridge": objective})
        if delta <= tolerance:
            converged = True
            break
    return {
        "global_mean": global_mean, "user_bias": user_bias, "book_bias": book_bias,
        "converged": converged, "iterations": iteration, "final_max_parameter_delta": delta,
        "fit_objective": history[-1]["sum_squared_error_plus_ridge"], "convergence_history": history,
        "user_shrinkage": float(user_shrinkage), "item_shrinkage": float(item_shrinkage),
    }


def _factorization_model_class():
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="book_recommender")
    class BiasedMatrixFactorization(tf.keras.Model):
        """Fixed fit mean, trainable user/item biases, and a 16D dot product."""

        def __init__(self, num_users, num_books, embedding_size=16, random_state=42, **kwargs):
            super().__init__(**kwargs)
            self.num_users = int(num_users)
            self.num_books = int(num_books)
            self.embedding_size = int(embedding_size)
            self.random_state = int(random_state)
            self.user_embedding = tf.keras.layers.Embedding(
                self.num_users, self.embedding_size,
                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=self.random_state),
                name="user_embedding",
            )
            self.book_embedding = tf.keras.layers.Embedding(
                self.num_books, self.embedding_size,
                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=self.random_state + 1),
                name="book_embedding",
            )
            self.global_mean = self.add_weight(name="global_mean", shape=(), initializer="zeros", trainable=False)
            self.user_bias = self.add_weight(name="user_bias", shape=(self.num_users,), initializer="zeros", trainable=True)
            self.book_bias = self.add_weight(name="book_bias", shape=(self.num_books,), initializer="zeros", trainable=True)

        def build(self, input_shape):
            self.user_embedding.build((None,))
            self.book_embedding.build((None,))
            super().build(input_shape)

        def call(self, inputs, training=False):
            user = self.user_embedding(inputs[:, 0])
            book = self.book_embedding(inputs[:, 1])
            prediction = (self.global_mean + tf.gather(self.user_bias, inputs[:, 0])
                          + tf.gather(self.book_bias, inputs[:, 1]) + tf.reduce_sum(user * book, axis=1))
            return prediction[:, None]

        def observed_pair_penalty(self, inputs):
            user = self.user_embedding(inputs[:, 0])
            book = self.book_embedding(inputs[:, 1])
            return tf.reduce_mean(
                tf.reduce_sum(tf.square(user), axis=1) + tf.reduce_sum(tf.square(book), axis=1)
                + tf.square(tf.gather(self.user_bias, inputs[:, 0]))
                + tf.square(tf.gather(self.book_bias, inputs[:, 1]))
            )

        def get_config(self):
            return {**super().get_config(), "num_users": self.num_users, "num_books": self.num_books,
                    "embedding_size": self.embedding_size, "random_state": self.random_state}

    _MODEL_CLASS = BiasedMatrixFactorization
    return _MODEL_CLASS


def create_factorization_model(num_users, num_books, *, embedding_size=16, random_state=42,
                               global_mean=0.0, user_bias=None, book_bias=None):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(random_state)
    model = _factorization_model_class()(num_users, num_books, embedding_size, random_state)
    model(tf.zeros((1, 2), dtype=tf.int32))
    model.global_mean.assign(global_mean)
    if user_bias is not None:
        model.user_bias.assign(user_bias)
    if book_bias is not None:
        model.book_bias.assign(book_bias)
    return model


def load_factorization_model(path):
    import tensorflow as tf

    model_class = _factorization_model_class()
    model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    if not isinstance(model, model_class):
        raise ValueError("Artifact is not a biased matrix factorization model.")
    return model
