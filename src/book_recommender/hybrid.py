from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.collaborative import (
    _validate_finite_float,
    _validate_rating_threshold,
    _validate_user_id,
)
from book_recommender.content_based import (
    ContentBasedRecommender,
    _validate_positive_int,
    book_work_key,
)
from book_recommender.data import clean_books, clean_ratings


class HybridRecommender:
    """Hybrid recommender combining item-CF, metadata profile, and popularity."""

    MODEL_FORMAT_VERSION = 3
    ALGORITHM_ID = "hybrid_canonical_work_seed_dedup_v3"

    def __init__(
        self,
        n_neighbors: int = 50,
        collaborative_weight: float = 0.40,
        content_weight: float = 0.45,
        popularity_weight: float = 0.15,
        dynamic_weights: bool = True,
        min_content_rating: float = 8.0,
        positive_rating_threshold: float | None = None,
    ):
        self.model_format_version = self.MODEL_FORMAT_VERSION
        self.algorithm_id = self.ALGORITHM_ID
        self.n_neighbors = _validate_positive_int(n_neighbors, "n_neighbors")
        self.collaborative_weight = _validate_non_negative_weight(
            collaborative_weight,
            "collaborative_weight",
        )
        self.content_weight = _validate_non_negative_weight(content_weight, "content_weight")
        self.popularity_weight = _validate_non_negative_weight(
            popularity_weight,
            "popularity_weight",
        )
        if self.collaborative_weight + self.content_weight + self.popularity_weight <= 0.0:
            raise ValueError("At least one hybrid weight must be greater than zero.")
        if not isinstance(dynamic_weights, (bool, np.bool_)):
            raise ValueError("dynamic_weights must be a boolean.")
        self.dynamic_weights = bool(dynamic_weights)

        legacy_threshold = _validate_rating_threshold(min_content_rating, "min_content_rating")
        effective_threshold = (
            legacy_threshold
            if positive_rating_threshold is None
            else _validate_rating_threshold(
                positive_rating_threshold,
                "positive_rating_threshold",
            )
        )
        # Keep the legacy field synchronized so existing CLI configuration and artifacts retain
        # their meaning while exposing the clearer positive-rating terminology.
        self.min_content_rating = effective_threshold
        self.positive_rating_threshold = effective_threshold

        self.books: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.content_model: ContentBasedRecommender | None = None
        self.collaborative_model: ItemBasedCollaborativeRecommender | None = None
        self.isbn_to_index: dict[str, int] = {}
        self.popularity_scores: np.ndarray | None = None

    def fit(self, books: pd.DataFrame, ratings: pd.DataFrame) -> "HybridRecommender":
        self.books = clean_books(books)
        ratings = clean_ratings(ratings)
        valid_isbns = set(self.books["isbn"].astype(str))
        self.ratings = ratings[ratings["isbn"].astype(str).isin(valid_isbns)].reset_index(drop=True).copy()
        if self.ratings.empty:
            raise ValueError("Ratings data has no rows matching the validated book catalog.")

        self.content_model = ContentBasedRecommender().fit(self.books)
        self.collaborative_model = ItemBasedCollaborativeRecommender(
            n_neighbors=self.n_neighbors,
            positive_rating_threshold=self._positive_rating_threshold(),
        ).fit(self.books, self.ratings)
        self.isbn_to_index = {
            str(isbn): index
            for index, isbn in enumerate(self.books["isbn"].astype(str).tolist())
        }
        self.popularity_scores = self._build_popularity_scores()
        return self

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        seed_items: int = 5,
        candidate_pool_size: int = 500,
        allow_popularity_fallback: bool = True,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        user_id = _validate_user_id(user_id)
        top_n = _validate_positive_int(top_n, "top_n")
        seed_items = _validate_positive_int(seed_items, "seed_items")
        candidate_pool_size = _validate_positive_int(
            candidate_pool_size,
            "candidate_pool_size",
        )
        if not isinstance(allow_popularity_fallback, bool):
            raise ValueError("allow_popularity_fallback must be a boolean.")
        user_history = self.ratings[self.ratings["user_id"] == int(user_id)].copy()
        if user_history.empty:
            if allow_popularity_fallback:
                return self._popular_with_hybrid_columns(top_n=top_n)
            return self._empty_recommendations()

        read_isbns = {str(isbn) for isbn in user_history["isbn"]}
        read_work_keys = self._work_keys_for_isbns(read_isbns)
        collaborative_scores = self._collaborative_scores(
            user_id=user_id,
            seed_items=seed_items,
            candidate_pool_size=candidate_pool_size,
        )
        content_scores = self._content_profile_scores(user_history, seed_items=seed_items)
        popularity_scores = normalize_scores(self.popularity_scores)

        positive_isbns = set(
            user_history.loc[
                user_history["book_rating"] >= self._positive_rating_threshold(),
                "isbn",
            ].astype(str)
        )
        positive_history_count = len(self._work_keys_for_isbns(positive_isbns))
        weights = self._weights_for_history(
            positive_history_count,
            content_scores,
            collaborative_scores,
            allow_popularity_fallback=allow_popularity_fallback,
        )
        final_scores = (
            weights["collaborative"] * collaborative_scores
            + weights["content"] * content_scores
            + weights["popularity"] * popularity_scores
        )

        for isbn in read_isbns:
            index = self.isbn_to_index.get(isbn)
            if index is not None:
                final_scores[index] = -np.inf

        score_columns = [
            "isbn",
            "score",
            "collaborative_score",
            "content_score",
            "popularity_score",
            "source",
            "reason",
        ]
        rows = []
        selected_work_keys: set[str] = set()
        for index in np.argsort(-final_scores, kind="stable"):
            score = float(final_scores[index])
            if not np.isfinite(score) or score <= 0.0:
                continue
            isbn = str(self.books.iloc[index]["isbn"])
            work_key = self._work_key_at(int(index))
            if work_key in read_work_keys or work_key in selected_work_keys:
                continue
            source = strongest_source(
                collaborative=weights["collaborative"] * collaborative_scores[index],
                content=weights["content"] * content_scores[index],
                popularity=weights["popularity"] * popularity_scores[index],
            )
            rows.append(
                {
                    "isbn": isbn,
                    "score": score,
                    "collaborative_score": float(collaborative_scores[index]),
                    "content_score": float(content_scores[index]),
                    "popularity_score": float(popularity_scores[index]),
                    "source": source,
                    "reason": recommendation_reason(
                        source,
                        positive_rating_threshold=self._positive_rating_threshold(),
                        is_fallback=weights["popularity"] >= 1.0 - 1e-12,
                    ),
                }
            )
            selected_work_keys.add(work_key)
            if len(rows) >= top_n:
                break

        result = pd.DataFrame(rows, columns=score_columns)
        if len(result) < top_n and allow_popularity_fallback:
            result = self._fill_with_popular(result, read_isbns, read_work_keys, top_n)
        return result.merge(self.books, on="isbn", how="left").head(top_n).reset_index(drop=True)

    @staticmethod
    def _empty_recommendations() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "isbn",
                "score",
                "collaborative_score",
                "content_score",
                "popularity_score",
                "source",
                "reason",
            ]
        )

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "HybridRecommender":
        model = joblib.load(path)
        if not isinstance(model, HybridRecommender):
            raise TypeError(f"Artifact is not a {HybridRecommender.__name__}")
        if (
            getattr(model, "model_format_version", None) != HybridRecommender.MODEL_FORMAT_VERSION
            or getattr(model, "algorithm_id", None) != HybridRecommender.ALGORITHM_ID
        ):
            raise ValueError("Legacy hybrid artifact is incompatible; retrain the model.")
        return model

    def _collaborative_scores(
        self,
        user_id: int,
        seed_items: int,
        candidate_pool_size: int,
    ) -> np.ndarray:
        scored = self.collaborative_model.score_for_user(
            user_id=user_id,
            seed_items=seed_items,
            candidate_pool_size=candidate_pool_size,
        )
        scores = np.zeros(len(self.books), dtype="float64")
        if scored.empty:
            return scores

        normalized = normalize_scores(scored["score"].to_numpy(dtype="float64"))
        for isbn, score in zip(scored["isbn"].astype(str), normalized):
            index = self.isbn_to_index.get(str(isbn))
            if index is not None:
                scores[index] = float(score)
        return scores

    def _content_profile_scores(self, user_history: pd.DataFrame, seed_items: int) -> np.ndarray:
        threshold = self._positive_rating_threshold()
        profile_history = user_history[user_history["book_rating"] >= threshold].copy()
        profile_history["_work_key"] = profile_history["isbn"].astype(str).map(
            lambda isbn: self.content_model.work_keys[self.isbn_to_index[isbn]]
        )
        profile_history = (
            profile_history.sort_values(
                ["book_rating", "isbn"],
                ascending=[False, True],
                kind="stable",
            )
            .drop_duplicates("_work_key")
            .head(seed_items)
        )
        if profile_history.empty:
            return np.zeros(len(self.books), dtype="float64")

        indexed_history = [
            (self.isbn_to_index[str(row.isbn)], float(row.book_rating))
            for row in profile_history.itertuples(index=False)
            if str(row.isbn) in self.isbn_to_index
        ]
        if not indexed_history:
            return np.zeros(len(self.books), dtype="float64")

        indices = [index for index, _ in indexed_history]
        ratings = np.array([rating for _, rating in indexed_history], dtype="float64")
        weights = np.clip((ratings - threshold + 1.0) / (11.0 - threshold), 0.0, 1.0)
        profile = self.content_model.matrix[indices].multiply(weights[:, None]).sum(axis=0)
        similarities = cosine_similarity(csr_matrix(profile), self.content_model.matrix).ravel()
        return normalize_scores(np.maximum(similarities, 0.0))

    def _build_popularity_scores(self) -> np.ndarray:
        score_by_isbn = (
            self.collaborative_model.popularity.set_index("isbn")["score"].astype("float64").to_dict()
        )
        return np.array(
            [float(score_by_isbn.get(str(isbn), 0.0)) for isbn in self.books["isbn"].astype(str)],
            dtype="float64",
        )

    def _weights_for_history(
        self,
        positive_history_count: int,
        content_scores: np.ndarray,
        collaborative_scores: np.ndarray,
        allow_popularity_fallback: bool = True,
    ) -> dict[str, float]:
        if positive_history_count <= 0:
            return (
                {"collaborative": 0.0, "content": 0.0, "popularity": 1.0}
                if allow_popularity_fallback
                else {"collaborative": 0.0, "content": 0.0, "popularity": 0.0}
            )

        weights = {
            "collaborative": self.collaborative_weight,
            "content": self.content_weight,
            "popularity": self.popularity_weight,
        }
        if self.dynamic_weights:
            if positive_history_count <= 2:
                # Relative multipliers keep every configured/tuned base weight active while
                # shifting sparse-history users toward metadata.
                weights["collaborative"] *= 0.50
                weights["content"] *= 1.40
            elif positive_history_count <= 5:
                weights["collaborative"] *= 0.80
                weights["content"] *= 1.20

        if not np.any(content_scores > 0.0):
            if weights["popularity"] > 0.0:
                weights["popularity"] += weights["content"]
            weights["content"] = 0.0
        if not np.any(collaborative_scores > 0.0):
            if weights["popularity"] > 0.0:
                weights["popularity"] += weights["collaborative"]
            weights["collaborative"] = 0.0
        return normalize_weights(
            weights,
            fallback_to_popularity=allow_popularity_fallback,
        )

    def _popular_with_hybrid_columns(self, top_n: int) -> pd.DataFrame:
        popular = self.collaborative_model.recommend_popular(top_n=top_n).copy()
        popular["collaborative_score"] = 0.0
        popular["content_score"] = 0.0
        popular["popularity_score"] = normalize_scores(popular["score"].to_numpy(dtype="float64"))
        popular["score"] = popular["popularity_score"]
        popular["source"] = "popularity"
        popular["reason"] = recommendation_reason("popularity", is_fallback=True)
        return popular.reset_index(drop=True)

    def _fill_with_popular(
        self,
        result: pd.DataFrame,
        read_isbns: set[str],
        read_work_keys: set[str],
        top_n: int,
    ) -> pd.DataFrame:
        existing = set(result["isbn"].astype(str)) if not result.empty else set()
        exclude = read_isbns.union(existing)
        existing_work_keys = self._work_keys_for_isbns(existing)
        popular = self.collaborative_model.recommend_popular(
            top_n=top_n,
            exclude_isbns=exclude,
            exclude_work_keys=read_work_keys.union(existing_work_keys),
        )
        if popular.empty:
            return result

        filler = popular[["isbn", "score"]].copy()
        filler["collaborative_score"] = 0.0
        filler["content_score"] = 0.0
        filler["popularity_score"] = normalize_scores(filler["score"].to_numpy(dtype="float64"))
        filler["score"] = filler["popularity_score"]
        filler["source"] = "popularity"
        filler["reason"] = recommendation_reason("popularity", is_fallback=True)
        combined = pd.concat([result, filler[result.columns]], ignore_index=True)
        return combined.head(top_n)

    def _positive_rating_threshold(self) -> float:
        return float(
            getattr(
                self,
                "positive_rating_threshold",
                getattr(self, "min_content_rating", 8.0),
            )
        )

    def _work_key_at(self, index: int) -> str:
        row = self.books.iloc[index]
        return book_work_key(row["book_title"], row["book_author"], row["isbn"])

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
            or self.content_model is None
            or self.collaborative_model is None
            or self.popularity_scores is None
        ):
            raise RuntimeError("Recommender has not been fitted.")


def normalize_scores(values) -> np.ndarray:
    scores = np.asarray(values, dtype="float64")
    if scores.size == 0:
        return scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    max_score = float(np.max(scores))
    if max_score <= 0.0:
        return np.zeros_like(scores, dtype="float64")
    return scores / max_score


def normalize_weights(
    weights: dict[str, float],
    *,
    fallback_to_popularity: bool = True,
) -> dict[str, float]:
    clipped = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return {
            "collaborative": 0.0,
            "content": 0.0,
            "popularity": 1.0 if fallback_to_popularity else 0.0,
        }
    return {key: value / total for key, value in clipped.items()}


def strongest_source(collaborative: float, content: float, popularity: float) -> str:
    sources = {
        "collaborative": float(collaborative),
        "content": float(content),
        "popularity": float(popularity),
    }
    return max(sources, key=sources.get)


def recommendation_reason(
    source: str,
    positive_rating_threshold: float = 8.0,
    is_fallback: bool = False,
) -> str:
    threshold = f"{positive_rating_threshold:g}/10"
    reasons = {
        "collaborative": (
            f"Item-rating patterns are similar to books the user rated {threshold} or higher."
        ),
        "content": f"Metadata is similar to books the user rated {threshold} or higher.",
        "popularity": (
            "Popular fallback from books with strong rating history."
            if is_fallback
            else "Popularity is the strongest signal in this blended recommendation."
        ),
    }
    return reasons.get(source, "Combined hybrid signal from user history and book metadata.")


def _validate_non_negative_weight(value: object, name: str) -> float:
    parsed = _validate_finite_float(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed
