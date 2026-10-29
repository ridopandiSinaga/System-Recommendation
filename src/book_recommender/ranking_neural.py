"""Fit-only work data and optional TensorFlow towers for ranking experiments.

TensorFlow is imported lazily. A sampled unobserved work is a contrastive
candidate, not a claim that its user dislikes it. MSE and BPR use different
supervision policies: a work's mean explicit rating versus any edition rated 8+.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from book_recommender.content_based import book_work_key

_MODEL_CLASS = None


@dataclass
class WorkTrainingData:
    users: list[int]
    work_keys: list[str]
    user_to_index: dict[int, int]
    isbn_to_work: dict[str, str]
    work_to_index: dict[str, int]
    pairs: np.ndarray
    targets: np.ndarray
    positive_pairs: np.ndarray
    seen_work_indices: tuple[frozenset[int], ...]
    positive_user_mask: np.ndarray
    popularity: np.ndarray
    metadata_features: np.ndarray
    metadata_vocabulary: dict[str, int]
    metadata_documents: list[str]
    fit_mean_rating: float


def prepare_work_data(books, fit, *, max_features=1024):
    """Aggregate fit observations and fit TF-IDF on fit-observed editions only.

    Canonical work IDs and text features never inspect selection or assessment
    observations. Duplicate editions contribute one user/work target (their
    mean) and at most one positive label (their maximum is at least eight).
    Padding makes the metadata branch's parameter shapes identical in all arms.
    """
    if isinstance(max_features, bool) or not isinstance(max_features, (int, np.integer)):
        raise ValueError("max_features must be a positive integer.")
    if max_features < 1:
        raise ValueError("max_features must be a positive integer.")
    if not {"isbn", "book_title", "book_author"}.issubset(books.columns):
        raise ValueError("Book catalog requires ISBN, title, and author columns.")
    if not {"user_id", "isbn", "book_rating"}.issubset(fit.columns) or fit.empty:
        raise ValueError("Fit observations require nonempty user, ISBN, and rating columns.")
    catalog = books.copy()
    ratings = fit[["user_id", "isbn", "book_rating"]].copy()
    if catalog["isbn"].isna().any() or ratings.isna().any().any():
        raise ValueError("Fit IDs and ratings cannot be missing.")
    catalog["isbn"] = catalog["isbn"].astype(str)
    ratings["isbn"] = ratings["isbn"].astype(str)
    if catalog["isbn"].duplicated().any():
        raise ValueError("Book catalog must have unique ISBNs.")
    numeric_users = pd.to_numeric(ratings["user_id"], errors="raise")
    numeric_ratings = pd.to_numeric(ratings["book_rating"], errors="raise")
    if (not np.isfinite(numeric_users).all() or (numeric_users % 1 != 0).any()
            or not np.isfinite(numeric_ratings).all()
            or not numeric_ratings.between(1, 10).all()):
        raise ValueError("User IDs must be integers and explicit ratings must be finite in [1, 10].")
    ratings["user_id"] = numeric_users.astype(np.int64)
    ratings["book_rating"] = numeric_ratings.astype(float)
    observed_isbns = set(ratings["isbn"])
    if not observed_isbns.issubset(set(catalog["isbn"])):
        raise ValueError("Every fit ISBN must have catalog metadata.")
    catalog = catalog[catalog["isbn"].isin(observed_isbns)].sort_values("isbn")
    if "publisher" not in catalog:
        catalog["publisher"] = ""
    catalog["work_key"] = [
        book_work_key(row.book_title, row.book_author, row.isbn)
        for row in catalog.itertuples(index=False)
    ]
    isbn_to_work = dict(zip(catalog["isbn"], catalog["work_key"]))
    ratings["work_key"] = ratings["isbn"].map(isbn_to_work)
    grouped = ratings.groupby(["user_id", "work_key"], sort=True)["book_rating"].agg(
        mean_rating="mean", max_rating="max",
    ).reset_index()
    users = sorted(int(user) for user in ratings["user_id"].unique())
    works = sorted(catalog["work_key"].unique().tolist())
    user_to_index = {user: index for index, user in enumerate(users)}
    work_to_index = {work: index for index, work in enumerate(works)}
    pairs = np.column_stack([
        grouped["user_id"].map(user_to_index), grouped["work_key"].map(work_to_index),
    ]).astype(np.int32)
    positive_pairs = pairs[grouped["max_rating"].to_numpy() >= 8].copy()
    seen = [set() for _ in users]
    for user_index, work_index in pairs:
        seen[user_index].add(int(work_index))
    positive_user_mask = np.zeros(len(users), dtype=bool)
    positive_user_mask[positive_pairs[:, 0]] = True
    popularity = np.bincount(positive_pairs[:, 1], minlength=len(works))
    documents_by_work = {}
    for work, rows in catalog.groupby("work_key", sort=True):
        # Repeated editions do not repeat the same field's text in a document.
        fragments = []
        for field in ["book_title", "book_author", "publisher"]:
            fragments.extend(sorted(set(rows[field].fillna("").astype(str)) - {""}))
        document = " ".join(fragments).strip()
        documents_by_work[work] = document or "missingmetadata"
    documents = [documents_by_work[work] for work in works]
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), max_features=int(max_features),
        token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
    )
    transformed = vectorizer.fit_transform(documents).toarray()
    features = np.zeros((len(works), int(max_features)), dtype=np.float32)
    features[:, :transformed.shape[1]] = transformed
    mean_ratings = grouped["mean_rating"].to_numpy(dtype=np.float64)
    return WorkTrainingData(
        users=users, work_keys=works, user_to_index=user_to_index,
        isbn_to_work=isbn_to_work, work_to_index=work_to_index,
        pairs=pairs, targets=((mean_ratings - 1) / 9).astype(np.float32),
        positive_pairs=positive_pairs,
        seen_work_indices=tuple(frozenset(values) for values in seen),
        positive_user_mask=positive_user_mask, popularity=popularity,
        metadata_features=features,
        metadata_vocabulary={str(word): int(index)
                             for word, index in vectorizer.vocabulary_.items()},
        metadata_documents=documents, fit_mean_rating=float(mean_ratings.mean()),
    )


def sample_negative_works(user_indices, seen_work_indices, num_works, rng):
    """Sample uniformly from each user's unobserved fit-catalog works.

    All observed works are excluded, including low-rated works and alternate
    editions. Rejection sampling switches to an exact complement for dense
    histories, so nearly complete users cannot cause an unbounded loop.
    """
    users = np.asarray(user_indices, dtype=np.int64)
    if users.ndim != 1 or num_works < 1:
        raise ValueError("Negative sampling requires one-dimensional users and a positive catalog size.")
    if (users < 0).any() or (users >= len(seen_work_indices)).any():
        raise ValueError("Encoded user lies outside the fit vocabulary.")
    result = np.empty(len(users), dtype=np.int32)
    for user in np.unique(users):
        seen = seen_work_indices[int(user)]
        if any(work < 0 or work >= num_works for work in seen):
            raise ValueError("Observed work lies outside the fit vocabulary.")
        if len(seen) >= num_works:
            raise ValueError("Cannot sample an unobserved work for a user who observed the full catalog.")
        positions = np.flatnonzero(users == user)
        if len(seen) > num_works // 2:
            available = np.array([work for work in range(num_works) if work not in seen])
            result[positions] = rng.choice(available, size=len(positions))
            continue
        remaining = positions
        # At most half the vocabulary is excluded. The exact-complement tail
        # preserves uniformity even in the exceptionally unlikely long tail.
        for _ in range(32):
            draws = rng.integers(0, num_works, size=len(remaining), dtype=np.int32)
            valid = np.fromiter((int(draw) not in seen for draw in draws), dtype=bool)
            result[remaining[valid]] = draws[valid]
            remaining = remaining[~valid]
            if not len(remaining):
                break
        if len(remaining):
            available = np.array([work for work in range(num_works) if work not in seen])
            result[remaining] = rng.choice(available, size=len(remaining))
    return result


def _ranking_model_class():
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="book_recommender")
    class WorkRankingTowers(tf.keras.Model):
        """Matched 32D neural towers with an optional metadata input branch."""

        def __init__(self, num_users, num_works, metadata_size=1024, random_state=42, **kwargs):
            super().__init__(**kwargs)
            self.num_users = int(num_users)
            self.num_works = int(num_works)
            self.metadata_size = int(metadata_size)
            self.random_state = int(random_state)
            self.user_embedding = tf.keras.layers.Embedding(
                self.num_users, 32, name="user_embedding",
                embeddings_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.05, seed=self.random_state),
            )
            self.work_embedding = tf.keras.layers.Embedding(
                self.num_works, 32, name="work_embedding",
                embeddings_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.05, seed=self.random_state + 1),
            )
            def dense(units, name, seed_offset, **options):
                return tf.keras.layers.Dense(
                    units, name=name,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(
                        seed=self.random_state + seed_offset), **options,
                )
            self.metadata_projection = dense(32, "metadata_projection", 2, use_bias=False)
            self.user_hidden = dense(64, "user_hidden", 3, activation="relu")
            self.user_output = dense(32, "user_output", 4)
            self.work_hidden = dense(64, "work_hidden", 5, activation="relu")
            self.work_output = dense(32, "work_output", 6)
            self.metadata_features = self.add_weight(
                name="metadata_features", shape=(self.num_works, self.metadata_size),
                initializer="zeros", trainable=False,
            )
            self.global_logit = self.add_weight(
                name="global_logit", shape=(), initializer="zeros", trainable=False,
            )
            self.work_bias = self.add_weight(
                name="work_bias", shape=(self.num_works,), initializer="zeros", trainable=True,
            )

        def build(self, input_shape):
            self.user_embedding.build((None,))
            self.work_embedding.build((None,))
            self.metadata_projection.build((None, self.metadata_size))
            self.user_hidden.build((None, 32))
            self.user_output.build((None, 64))
            self.work_hidden.build((None, 64))
            self.work_output.build((None, 64))
            super().build(input_shape)

        def latent_vectors(self, user_indices, work_indices):
            user_vectors = self.user_output(self.user_hidden(self.user_embedding(user_indices)))
            work_input = tf.concat([
                self.work_embedding(work_indices),
                self.metadata_projection(tf.gather(self.metadata_features, work_indices)),
            ], axis=1)
            work_vectors = self.work_output(self.work_hidden(work_input))
            return user_vectors, work_vectors

        def call(self, inputs, training=False):
            users, works = inputs[:, 0], inputs[:, 1]
            user_vectors, work_vectors = self.latent_vectors(users, works)
            scores = (tf.reduce_sum(user_vectors * work_vectors, axis=1) / tf.sqrt(32.0)
                      + tf.gather(self.work_bias, works) + self.global_logit)
            return scores[:, None]

        def score_catalog(self, user_indices):
            user_vectors, work_vectors = self.latent_vectors(
                user_indices, tf.range(self.num_works, dtype=tf.int32),
            )
            return (tf.matmul(user_vectors, work_vectors, transpose_b=True) / tf.sqrt(32.0)
                    + self.work_bias[None, :] + self.global_logit)

        def regularization_loss(self, pairs, regularization=1e-5):
            users, works = pairs[:, 0], pairs[:, 1]
            observed_penalty = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.user_embedding(users)), axis=1)
                + tf.reduce_sum(tf.square(self.work_embedding(works)), axis=1)
                + tf.square(tf.gather(self.work_bias, works)),
            )
            kernels = [layer.kernel for layer in [
                self.metadata_projection, self.user_hidden, self.user_output,
                self.work_hidden, self.work_output,
            ]]
            kernel_parameters = sum(int(np.prod(kernel.shape)) for kernel in kernels)
            kernel_penalty = tf.add_n([tf.reduce_sum(tf.square(kernel)) for kernel in kernels])
            return tf.cast(regularization, tf.float32) * (
                observed_penalty + kernel_penalty / kernel_parameters
            )

        def get_config(self):
            return {**super().get_config(), "num_users": self.num_users,
                    "num_works": self.num_works, "metadata_size": self.metadata_size,
                    "random_state": self.random_state}

    _MODEL_CLASS = WorkRankingTowers
    return _MODEL_CLASS


def create_ranking_model(num_users, num_works, metadata_features, *, random_state=42,
                         fit_mean_rating=5.5):
    """Create identical trainable initial weights for matched seed/shape arms."""
    import tensorflow as tf

    features = np.asarray(metadata_features, dtype=np.float32)
    if (min(num_users, num_works) < 1 or features.ndim != 2
            or features.shape[0] != num_works or features.shape[1] < 1
            or not np.isfinite(features).all()
            or not np.isfinite(fit_mean_rating) or not 1 <= fit_mean_rating <= 10):
        raise ValueError("Invalid model dimensions, metadata matrix, or fit mean rating.")
    tf.keras.utils.set_random_seed(random_state)
    model = _ranking_model_class()(
        num_users, num_works, metadata_size=features.shape[1], random_state=random_state,
    )
    model(tf.zeros((1, 2), dtype=tf.int32))
    model.metadata_features.assign(features)
    normalized_mean = np.clip((fit_mean_rating - 1) / 9, 1e-6, 1 - 1e-6)
    model.global_logit.assign(float(np.log(normalized_mean / (1 - normalized_mean))))
    return model


def load_ranking_model(path):
    import tensorflow as tf

    model_class = _ranking_model_class()
    model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    if not isinstance(model, model_class):
        raise ValueError("Artifact is not a work-ranking tower model.")
    return model


def make_train_step(model, optimizer, *, objective, regularization=1e-5):
    """Build one compiled MSE or BPR update with finite-gradient assertions.

    BPR regularization observes both positive and negative pairs. The same
    regularization coefficient does not imply equal influence across losses.
    """
    import tensorflow as tf

    objective = objective.lower()
    if objective not in {"mse", "bpr", "mse_id", "bpr_id", "bpr_metadata"}:
        raise ValueError("Objective must be mse or bpr.")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Regularization must be finite and nonnegative.")
    is_mse = objective.startswith("mse")

    @tf.function(reduce_retracing=True)
    def step(users, items, targets, negative_items):
        users = tf.cast(users, tf.int32)
        items = tf.cast(items, tf.int32)
        pairs = tf.stack([users, items], axis=1)
        with tf.GradientTape() as tape:
            positive_scores = tf.reshape(model(pairs, training=True), [-1])
            if is_mse:
                data_loss = tf.reduce_mean(tf.square(
                    tf.sigmoid(positive_scores) - tf.cast(tf.reshape(targets, [-1]), tf.float32),
                ))
                penalty_pairs = pairs
            else:
                negative_pairs = tf.stack([users, tf.cast(negative_items, tf.int32)], axis=1)
                negative_scores = tf.reshape(model(negative_pairs, training=True), [-1])
                data_loss = tf.reduce_mean(tf.nn.softplus(negative_scores - positive_scores))
                penalty_pairs = tf.concat([pairs, negative_pairs], axis=0)
            penalty = model.regularization_loss(penalty_pairs, regularization=regularization)
            loss = data_loss + penalty
            tf.debugging.assert_all_finite(loss, "Nonfinite ranking loss")
        gradients = tape.gradient(loss, model.trainable_variables)
        for gradient in gradients:
            if gradient is None:
                raise ValueError("A trainable ranking parameter has no gradient.")
            values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
            tf.debugging.assert_all_finite(values, "Nonfinite ranking gradient")
        gradient_norm = tf.linalg.global_norm(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return {"loss": loss, "data_loss": data_loss, "regularization_loss": penalty,
                "gradient_norm": gradient_norm}

    return step
