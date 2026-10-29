"""Fit-only positive histories, item cosine, and optional neural history encoders.

Training removes its target work from the user's context before either loss.
The context/output item tables are independent; no user-ID parameter is learned.
TensorFlow is imported lazily and sparse histories are never globally densified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

_MODEL_CLASS = None


@dataclass
class HistoryTrainingData:
    positive_csr: sparse.csr_matrix
    positive_values: np.ndarray
    positive_row_splits: np.ndarray
    eligible_pairs: np.ndarray
    positive_counts: np.ndarray
    seen_work_indices: tuple[frozenset[int], ...]
    popularity: np.ndarray

    @property
    def num_users(self):
        return self.positive_csr.shape[0]

    @property
    def num_works(self):
        return self.positive_csr.shape[1]

    def softmax_candidate_mask(self, users, targets):
        """Keep the training target and all unobserved works, masking other fit works."""
        users, targets = _validate_pairs(users, targets, self.num_users, self.num_works)
        if not np.asarray(self.positive_csr[users, targets]).reshape(-1).all():
            raise ValueError("Every training target must be a positive fit work.")
        mask = np.ones((len(users), self.num_works), dtype=bool)
        for row, user in enumerate(users):
            mask[row, list(self.seen_work_indices[int(user)])] = False
        mask[np.arange(len(users)), targets] = True
        return mask


def _validate_pairs(users, items, num_users, num_works):
    users, items = np.asarray(users), np.asarray(items)
    if users.ndim != 1 or items.ndim != 1 or len(users) != len(items):
        raise ValueError("Encoded users and works must be aligned one-dimensional arrays.")
    if (not np.issubdtype(users.dtype, np.integer)
            or not np.issubdtype(items.dtype, np.integer)):
        raise ValueError("Encoded users and works must be integers.")
    if (users < 0).any() or (users >= num_users).any() or (items < 0).any() or (items >= num_works).any():
        raise ValueError("Encoded user or work lies outside the fit vocabulary.")
    return users.astype(np.int32), items.astype(np.int32)


def build_history_data(work_data):
    """Create binary canonical-work histories using positive fit observations only."""
    positive_pairs = np.asarray(work_data.positive_pairs, dtype=np.int32)
    if positive_pairs.ndim != 2 or positive_pairs.shape[1] != 2:
        raise ValueError("Positive fit pairs must have shape (n, 2).")
    num_users, num_works = len(work_data.users), len(work_data.work_keys)
    _validate_pairs(positive_pairs[:, 0], positive_pairs[:, 1], num_users, num_works)
    history = sparse.csr_matrix(
        (np.ones(len(positive_pairs), dtype=np.float32),
         (positive_pairs[:, 0], positive_pairs[:, 1])), shape=(num_users, num_works),
    )
    history.sum_duplicates()
    history.data[:] = 1
    history.sort_indices()
    counts = np.diff(history.indptr).astype(np.int32)
    canonical_pairs = np.column_stack([
        np.repeat(np.arange(num_users, dtype=np.int32), counts), history.indices,
    ]).astype(np.int32)
    eligible = canonical_pairs[counts[canonical_pairs[:, 0]] >= 2].copy()
    popularity = np.asarray(history.sum(axis=0)).reshape(-1).astype(np.int64)
    if not np.array_equal(popularity, work_data.popularity):
        raise ValueError("Positive history and fit popularity disagree.")
    for user, item in canonical_pairs:
        if int(item) not in work_data.seen_work_indices[int(user)]:
            raise ValueError("Positive work is absent from the full fit history.")
    return HistoryTrainingData(
        positive_csr=history,
        positive_values=history.indices.astype(np.int32, copy=True),
        positive_row_splits=history.indptr.astype(np.int64, copy=True),
        eligible_pairs=eligible, positive_counts=counts,
        seen_work_indices=work_data.seen_work_indices, popularity=popularity,
    )


class ItemCosineBaseline:
    """Positive-history rows multiplied by source-work-to-target-work similarity."""

    def __init__(self, positive_csr, similarity):
        self.positive_csr = sparse.csr_matrix(positive_csr, dtype=np.float32)
        self.similarity = sparse.csr_matrix(similarity, dtype=np.float64)
        if self.similarity.shape != (self.positive_csr.shape[1],) * 2:
            raise ValueError("Similarity dimensions must match the history work vocabulary.")

    def score_catalog(self, users):
        users = np.asarray(users)
        if (users.ndim != 1 or not np.issubdtype(users.dtype, np.integer)
                or (users < 0).any() or (users >= self.positive_csr.shape[0]).any()):
            raise ValueError("Encoded users must lie inside the fit vocabulary.")
        return (self.positive_csr[users] @ self.similarity).toarray()


def build_item_cosine(history_data, *, k=50, shrink=0.0):
    """Fit significance-weighted cosine, retaining k targets per source row.

    S_ij = C_ij / sqrt(c_i*c_j) * C_ij / (C_ij + shrink), diagonal zero.
    C counts distinct positive users shared by the two works. Equal similarities
    retain the lower work index; work indices are lexicographic canonical keys.
    The evaluator separately resolves recommendation score ties by popularity.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    if not np.isfinite(shrink) or shrink < 0:
        raise ValueError("shrink must be finite and nonnegative.")
    history = history_data.positive_csr.astype(np.float64)
    cooccurrence = (history.T @ history).tocsr()
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    popularity = np.asarray(history.sum(axis=0)).reshape(-1)
    rows, columns, values = [], [], []
    for source in range(cooccurrence.shape[0]):
        start, stop = cooccurrence.indptr[source:source+2]
        targets = cooccurrence.indices[start:stop]
        shared = cooccurrence.data[start:stop]
        if not len(targets):
            continue
        scores = shared / np.sqrt(popularity[source]*popularity[targets])
        scores *= shared / (shared+shrink)
        order = np.lexsort((targets, -scores))[:k]
        rows.extend([source] * len(order))
        columns.extend(targets[order].tolist())
        values.extend(scores[order].tolist())
    similarity = sparse.csr_matrix((values, (rows, columns)), shape=cooccurrence.shape)
    similarity.sort_indices()
    return ItemCosineBaseline(history_data.positive_csr, similarity)


def _history_model_class():
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="book_recommender")
    class HistoryRankingEncoder(tf.keras.Model):
        """Untied context/output embeddings and a shared nonlinear history encoder."""

        def __init__(self, num_users, num_works, num_positive_entries, random_state=42, **kwargs):
            super().__init__(**kwargs)
            self.num_users = int(num_users)
            self.num_works = int(num_works)
            self.num_positive_entries = int(num_positive_entries)
            self.random_state = int(random_state)
            self.context_embedding = tf.keras.layers.Embedding(
                self.num_works, 32, name="context_embedding",
                embeddings_initializer=tf.keras.initializers.RandomNormal(
                    stddev=.05, seed=self.random_state),
            )
            self.output_embedding = tf.keras.layers.Embedding(
                self.num_works, 32, name="output_embedding",
                embeddings_initializer=tf.keras.initializers.RandomNormal(
                    stddev=.05, seed=self.random_state+1),
            )
            self.history_hidden = tf.keras.layers.Dense(
                64, activation="tanh", name="history_hidden",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state+2),
            )
            self.history_output = tf.keras.layers.Dense(
                32, name="history_output",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.random_state+3),
            )
            self.work_bias = self.add_weight(
                name="work_bias", shape=(self.num_works,), initializer="zeros", trainable=True,
            )
            self.positive_values = self.add_weight(
                name="positive_values", shape=(self.num_positive_entries,),
                dtype="int32", initializer="zeros", trainable=False,
            )
            self.positive_row_splits = self.add_weight(
                name="positive_row_splits", shape=(self.num_users+1,),
                dtype="int64", initializer="zeros", trainable=False,
            )

        def build(self, input_shape):
            self.context_embedding.build((None,))
            self.output_embedding.build((None,))
            self.history_hidden.build((None, 32))
            self.history_output.build((None, 64))
            super().build(input_shape)

        def context_summary(self, users, exclude_items=None):
            """Return sum/sqrt(count) embeddings and remaining context counts."""
            users = tf.cast(users, tf.int32)
            histories = tf.RaggedTensor.from_row_splits(
                tf.convert_to_tensor(self.positive_values),
                tf.convert_to_tensor(self.positive_row_splits), validate=False,
            )
            batch = tf.gather(histories, users)
            work_indices, row_ids = batch.flat_values, batch.value_rowids()
            keep = tf.ones_like(work_indices, dtype=tf.float32)
            if exclude_items is not None:
                target_for_entry = tf.gather(tf.cast(exclude_items, tf.int32), row_ids)
                keep = tf.cast(tf.not_equal(work_indices, target_for_entry), tf.float32)
            batch_size = tf.shape(users)[0]
            embedded = self.context_embedding(work_indices)
            summed = tf.math.unsorted_segment_sum(embedded*keep[:, None], row_ids, batch_size)
            counts = tf.math.unsorted_segment_sum(keep, row_ids, batch_size)
            pooled = summed/tf.sqrt(tf.maximum(counts, 1.0))[:, None]
            return pooled, counts

        def user_vectors(self, users, exclude_items=None):
            pooled, _ = self.context_summary(users, exclude_items)
            return self.history_output(self.history_hidden(pooled))

        def score_from_user_vectors(self, vectors, items=None):
            if items is None:
                return (tf.matmul(vectors, self.output_embedding.embeddings, transpose_b=True)
                        / tf.sqrt(32.0) + self.work_bias[None, :])
            items = tf.cast(items, tf.int32)
            return (tf.reduce_sum(vectors*self.output_embedding(items), axis=1)/tf.sqrt(32.0)
                    + tf.gather(self.work_bias, items))

        def score_targets(self, users, items, exclude_items=None):
            return self.score_from_user_vectors(self.user_vectors(users, exclude_items), items)

        def score_catalog(self, users, exclude_items=None):
            return self.score_from_user_vectors(self.user_vectors(users, exclude_items))

        def call(self, inputs, training=False):
            return self.score_targets(inputs[:, 0], inputs[:, 1])[:, None]

        def regularization_loss(self, regularization=1e-5):
            parameters = sum(int(np.prod(variable.shape)) for variable in self.trainable_variables)
            squared = tf.add_n([tf.reduce_sum(tf.square(variable)) for variable in self.trainable_variables])
            return tf.cast(regularization, tf.float32)*squared/parameters

        def get_config(self):
            return {**super().get_config(), "num_users": self.num_users,
                    "num_works": self.num_works, "num_positive_entries": self.num_positive_entries,
                    "random_state": self.random_state}

    _MODEL_CLASS = HistoryRankingEncoder
    return _MODEL_CLASS


def create_history_model(history_data, *, random_state=42):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(random_state)
    model = _history_model_class()(
        history_data.num_users, history_data.num_works,
        num_positive_entries=len(history_data.positive_values), random_state=random_state,
    )
    model.build((None, 2))
    model.positive_values.assign(history_data.positive_values)
    model.positive_row_splits.assign(history_data.positive_row_splits)
    model(tf.zeros((1, 2), dtype=tf.int32))
    return model


def load_history_model(path):
    import tensorflow as tf

    model_class = _history_model_class()
    model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    if not isinstance(model, model_class):
        raise ValueError("Artifact is not a positive-history neural encoder.")
    return model


def make_history_train_step(model, optimizer, *, objective, regularization=1e-5):
    """Compile one target-excluded BPR or masked full-catalogue cross-entropy step."""
    import tensorflow as tf

    objective = objective.removeprefix("history_")
    if objective not in {"bpr", "softmax"}:
        raise ValueError("History objective must be bpr or softmax.")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Regularization must be finite and nonnegative.")

    @tf.function(reduce_retracing=True)
    def step(users, targets, negatives, candidate_mask):
        users, targets = tf.cast(users, tf.int32), tf.cast(targets, tf.int32)
        with tf.GradientTape() as tape:
            # Both BPR scores share exactly this target-excluded history vector.
            vectors = model.user_vectors(users, exclude_items=targets)
            if objective == "bpr":
                positive_scores = model.score_from_user_vectors(vectors, targets)
                negative_scores = model.score_from_user_vectors(vectors, tf.cast(negatives, tf.int32))
                data_loss = tf.reduce_mean(tf.nn.softplus(negative_scores-positive_scores))
            else:
                logits = model.score_from_user_vectors(vectors)
                mask = tf.cast(candidate_mask, tf.bool)
                tf.debugging.assert_equal(tf.shape(mask), tf.shape(logits))
                target_mask = tf.gather_nd(mask, tf.stack([tf.range(tf.shape(users)[0]), targets], axis=1))
                tf.debugging.assert_equal(tf.reduce_all(target_mask), True,
                                          message="The training target must remain in the candidate set")
                masked_logits = tf.where(mask, logits, tf.constant(-1e9, dtype=tf.float32))
                data_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets, logits=masked_logits,
                ))
            penalty = model.regularization_loss(regularization)
            loss = data_loss+penalty
            tf.debugging.assert_all_finite(loss, "Nonfinite history training loss")
        gradients = tape.gradient(loss, model.trainable_variables)
        for gradient in gradients:
            if gradient is None:
                raise ValueError("A history encoder parameter has no gradient.")
            values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
            tf.debugging.assert_all_finite(values, "Nonfinite history gradient")
        gradient_norm = tf.linalg.global_norm(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return {"loss": loss, "data_loss": data_loss, "regularization_loss": penalty,
                "gradient_norm": gradient_norm}

    return step
