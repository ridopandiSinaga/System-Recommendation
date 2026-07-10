from __future__ import annotations

from collections import defaultdict
import math
import numbers
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from book_recommender.content_based import (
    _validate_positive_int,
    book_work_key,
    rerank_unique_works,
)
from book_recommender.data import clean_books, clean_ratings


class ItemBasedCollaborativeRecommender:
    """Item-item adjusted-cosine filtering over a sparse explicit-rating matrix."""

    MODEL_FORMAT_VERSION = 3
    ALGORITHM_ID = "adjusted_cosine_canonical_work_positive_seed_v3"

    def __init__(
        self,
        n_neighbors: int = 50,
        positive_rating_threshold: float = 8.0,
        bias_regularization: float = 5.0,
    ):
        self.model_format_version = self.MODEL_FORMAT_VERSION
        self.algorithm_id = self.ALGORITHM_ID
        self.n_neighbors = _validate_positive_int(n_neighbors, "n_neighbors")
        self.positive_rating_threshold = _validate_rating_threshold(
            positive_rating_threshold,
            "positive_rating_threshold",
        )
        self.bias_regularization = _validate_non_negative_float(
            bias_regularization,
            "bias_regularization",
        )
        self.neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
        self.books: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.item_user_matrix: csr_matrix | None = None
        self.isbn_to_index: dict[str, int] = {}
        self.index_to_isbn: dict[int, str] = {}
        self.isbn_to_work_key: dict[str, str] = {}
        self.user_to_index: dict[int, int] = {}
        self.popularity: pd.DataFrame | None = None

    def fit(self, books: pd.DataFrame, ratings: pd.DataFrame) -> "ItemBasedCollaborativeRecommender":
        books = clean_books(books)
        ratings = clean_ratings(ratings)
        if ratings.empty:
            raise ValueError("Ratings data is empty after preprocessing.")

        self.books = books.drop_duplicates("isbn").reset_index(drop=True).copy()
        valid_isbns = set(self.books["isbn"])
        self.ratings = ratings[ratings["isbn"].isin(valid_isbns)].reset_index(drop=True).copy()
        if self.ratings.empty:
            raise ValueError("Ratings data has no rows matching the validated book catalog.")

        users = sorted(self.ratings["user_id"].unique())
        items = self.books["isbn"].tolist()
        self.user_to_index = {int(user_id): index for index, user_id in enumerate(users)}
        self.isbn_to_index = {str(isbn): index for index, isbn in enumerate(items)}
        self.index_to_isbn = {index: isbn for isbn, index in self.isbn_to_index.items()}
        self.isbn_to_work_key = {
            str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
            for row in self.books.itertuples(index=False)
        }

        row_indices = self.ratings["isbn"].map(self.isbn_to_index).to_numpy()
        col_indices = self.ratings["user_id"].map(self.user_to_index).to_numpy()
        values = self._bias_adjusted_ratings()
        self.item_user_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(items), len(users)),
        )
        self.item_user_matrix.eliminate_zeros()
        self.neighbors.fit(self.item_user_matrix)
        self.popularity = self._build_popularity()
        return self

    def score_for_user(
        self,
        user_id: int,
        seed_items: int = 5,
        candidate_pool_size: int | None = None,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        user_id = _validate_user_id(user_id)
        seed_items = _validate_positive_int(seed_items, "seed_items")
        if candidate_pool_size is not None:
            candidate_pool_size = _validate_positive_int(
                candidate_pool_size,
                "candidate_pool_size",
            )
        user_history = self.ratings[self.ratings["user_id"] == int(user_id)].copy()
        if user_history.empty:
            return pd.DataFrame(columns=["isbn", "score"])

        read_isbns = set(user_history["isbn"].astype(str))
        threshold = self._positive_rating_threshold()
        seed_history = user_history[user_history["book_rating"] >= threshold].copy()
        seed_history["_work_key"] = seed_history["isbn"].astype(str).map(
            self.isbn_to_work_key
        )
        seed_history = (
            seed_history.sort_values(
                ["book_rating", "isbn"],
                ascending=[False, True],
                kind="stable",
            )
            .drop_duplicates("_work_key")
            .head(seed_items)
        )
        if seed_history.empty:
            return pd.DataFrame(columns=["isbn", "score"])

        scores: dict[str, float] = defaultdict(float)

        n_neighbors = min(self.item_user_matrix.shape[0], self.n_neighbors + 1)
        for row in seed_history.itertuples(index=False):
            isbn = str(row.isbn)
            if isbn not in self.isbn_to_index:
                continue
            item_index = self.isbn_to_index[isbn]
            distances, indices = self.neighbors.kneighbors(
                self.item_user_matrix[item_index],
                n_neighbors=n_neighbors,
            )
            # Threshold-relative strength prevents a merely non-zero rating from being treated
            # as approval. At the default threshold, ratings 8/9/10 receive 1/3, 2/3, and 1.
            rating_weight = (float(row.book_rating) - threshold + 1.0) / (11.0 - threshold)
            for distance, neighbor_index in zip(distances[0], indices[0]):
                candidate_isbn = self.index_to_isbn[int(neighbor_index)]
                if candidate_isbn == isbn or candidate_isbn in read_isbns:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                if similarity <= 0.0:
                    continue
                scores[candidate_isbn] += similarity * rating_weight

        if not scores:
            return pd.DataFrame(columns=["isbn", "score"])

        scored = pd.DataFrame(
            [{"isbn": isbn, "score": score} for isbn, score in scores.items()]
        ).sort_values(["score", "isbn"], ascending=[False, True], kind="stable")
        if candidate_pool_size is not None:
            scored = scored.head(candidate_pool_size)
        return scored.reset_index(drop=True)

    def recommend_for_user(self, user_id: int, top_n: int = 10, seed_items: int = 5) -> pd.DataFrame:
        self._check_is_fitted()
        user_id = _validate_user_id(user_id)
        top_n = _validate_positive_int(top_n, "top_n")
        seed_items = _validate_positive_int(seed_items, "seed_items")
        user_history = self.ratings[self.ratings["user_id"] == int(user_id)].copy()
        if user_history.empty:
            return self.recommend_popular(top_n=top_n)

        read_isbns = set(user_history["isbn"].astype(str))
        read_work_keys = self._work_keys_for_isbns(read_isbns)
        scored = self.score_for_user(
            user_id=user_id,
            seed_items=seed_items,
            candidate_pool_size=None,
        )
        if scored.empty:
            return self.recommend_popular(
                top_n=top_n,
                exclude_isbns=read_isbns,
                exclude_work_keys=read_work_keys,
            )

        result = scored.merge(self.books, on="isbn", how="left")
        result = rerank_unique_works(
            result,
            top_n=top_n,
            excluded_work_keys=read_work_keys,
        )
        if len(result) >= top_n:
            return result.head(top_n).reset_index(drop=True)

        selected_isbns = set(result["isbn"].astype(str)) if not result.empty else set()
        selected_work_keys = self._work_keys_for_isbns(selected_isbns)
        filler = self.recommend_popular(
            top_n=top_n - len(result),
            exclude_isbns=read_isbns.union(selected_isbns),
            exclude_work_keys=read_work_keys.union(selected_work_keys),
        )
        return pd.concat([result, filler], ignore_index=True, sort=False).head(top_n)

    def recommend_popular(
        self,
        top_n: int = 10,
        exclude_isbns: set[str] | None = None,
        exclude_work_keys: set[str] | None = None,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        top_n = _validate_positive_int(top_n, "top_n")
        exclude_isbns = {str(isbn) for isbn in (exclude_isbns or set())}
        excluded_works = set(exclude_work_keys or set())
        excluded_works.update(self._work_keys_for_isbns(exclude_isbns))
        popular = self.popularity[~self.popularity["isbn"].astype(str).isin(exclude_isbns)]
        popular = popular.merge(self.books, on="isbn", how="left")
        return rerank_unique_works(
            popular,
            top_n=top_n,
            excluded_work_keys=excluded_works,
        )

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ItemBasedCollaborativeRecommender":
        model = joblib.load(path)
        if not isinstance(model, ItemBasedCollaborativeRecommender):
            raise TypeError(f"Artifact is not a {ItemBasedCollaborativeRecommender.__name__}")
        if (
            getattr(model, "model_format_version", None)
            != ItemBasedCollaborativeRecommender.MODEL_FORMAT_VERSION
            or getattr(model, "algorithm_id", None)
            != ItemBasedCollaborativeRecommender.ALGORITHM_ID
        ):
            raise ValueError("Legacy collaborative artifact is incompatible; retrain the model.")
        return model

    def _build_popularity(self) -> pd.DataFrame:
        popularity = (
            self.ratings.groupby("isbn")
            .agg(mean_rating=("book_rating", "mean"), rating_count=("book_rating", "size"))
            .reset_index()
        )
        popularity["score"] = popularity["mean_rating"] * np.log1p(popularity["rating_count"])
        return popularity.sort_values(
            ["score", "rating_count", "isbn"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)

    def _bias_adjusted_ratings(self) -> np.ndarray:
        """Center ratings by a regularized user mean for adjusted-cosine similarity."""

        ratings = self.ratings["book_rating"].astype("float64")
        global_mean = float(ratings.mean())
        regularization = self._bias_regularization()
        user_stats = (
            self.ratings.assign(_rating=ratings)
            .groupby("user_id")["_rating"]
            .agg(["sum", "count"])
        )
        user_bias = (
            user_stats["sum"] + regularization * global_mean
        ) / (user_stats["count"] + regularization)
        centered = ratings - self.ratings["user_id"].map(user_bias).astype("float64")
        return centered.to_numpy(dtype="float32")

    def _positive_rating_threshold(self) -> float:
        # ``getattr`` keeps artifacts produced before this field was introduced usable.
        return float(getattr(self, "positive_rating_threshold", 8.0))

    def _bias_regularization(self) -> float:
        return float(getattr(self, "bias_regularization", 5.0))

    def _work_keys_for_isbns(self, isbns: set[str]) -> set[str]:
        if not isbns:
            return set()
        rows = self.books[self.books["isbn"].astype(str).isin(isbns)]
        return {
            book_work_key(row.book_title, row.book_author, row.isbn)
            for row in rows.itertuples(index=False)
        }

    def _check_is_fitted(self) -> None:
        if (
            self.books is None
            or self.ratings is None
            or self.item_user_matrix is None
            or self.popularity is None
        ):
            raise RuntimeError("Recommender has not been fitted.")


def _validate_user_id(user_id: object) -> int:
    if (
        isinstance(user_id, bool)
        or not isinstance(user_id, numbers.Integral)
        or int(user_id) <= 0
    ):
        raise ValueError("user_id must be a positive integer.")
    return int(user_id)


def _validate_rating_threshold(value: object, name: str) -> float:
    parsed = _validate_finite_float(value, name)
    if not 1.0 <= parsed <= 10.0:
        raise ValueError(f"{name} must be between 1 and 10.")
    return parsed


def _validate_non_negative_float(value: object, name: str) -> float:
    parsed = _validate_finite_float(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _validate_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number.")
    return parsed
