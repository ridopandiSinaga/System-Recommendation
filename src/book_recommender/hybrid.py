from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender


class HybridRecommender:
    """Hybrid recommender combining item-CF, metadata profile, and popularity."""

    def __init__(
        self,
        n_neighbors: int = 50,
        collaborative_weight: float = 0.45,
        content_weight: float = 0.45,
        popularity_weight: float = 0.10,
        dynamic_weights: bool = True,
        min_content_rating: float = 8.0,
    ):
        self.n_neighbors = n_neighbors
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        self.dynamic_weights = dynamic_weights
        self.min_content_rating = min_content_rating

        self.books: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.content_model: ContentBasedRecommender | None = None
        self.collaborative_model: ItemBasedCollaborativeRecommender | None = None
        self.isbn_to_index: dict[str, int] = {}
        self.popularity_scores: np.ndarray | None = None

    def fit(self, books: pd.DataFrame, ratings: pd.DataFrame) -> "HybridRecommender":
        self.books = books.drop_duplicates("isbn").reset_index(drop=True).copy()
        valid_isbns = set(self.books["isbn"].astype(str))
        self.ratings = ratings[ratings["isbn"].astype(str).isin(valid_isbns)].reset_index(drop=True).copy()
        if self.ratings.empty:
            raise ValueError("Ratings data is empty after preprocessing.")

        self.content_model = ContentBasedRecommender().fit(self.books)
        self.collaborative_model = ItemBasedCollaborativeRecommender(
            n_neighbors=self.n_neighbors
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
    ) -> pd.DataFrame:
        self._check_is_fitted()
        user_history = self.ratings[self.ratings["user_id"] == int(user_id)].copy()
        if user_history.empty:
            return self._popular_with_hybrid_columns(top_n=top_n)

        read_isbns = {str(isbn) for isbn in user_history["isbn"]}
        collaborative_scores = self._collaborative_scores(
            user_id=user_id,
            seed_items=seed_items,
            candidate_pool_size=candidate_pool_size,
        )
        content_scores = self._content_profile_scores(user_history, seed_items=seed_items)
        popularity_scores = normalize_scores(self.popularity_scores)

        weights = self._weights_for_history(len(user_history), content_scores, collaborative_scores)
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
        ]
        rows = []
        for index in np.argsort(-final_scores):
            score = float(final_scores[index])
            if not np.isfinite(score) or score <= 0.0:
                continue
            isbn = str(self.books.iloc[index]["isbn"])
            rows.append(
                {
                    "isbn": isbn,
                    "score": score,
                    "collaborative_score": float(collaborative_scores[index]),
                    "content_score": float(content_scores[index]),
                    "popularity_score": float(popularity_scores[index]),
                    "source": strongest_source(
                        collaborative=weights["collaborative"] * collaborative_scores[index],
                        content=weights["content"] * content_scores[index],
                        popularity=weights["popularity"] * popularity_scores[index],
                    ),
                }
            )
            if len(rows) >= top_n:
                break

        result = pd.DataFrame(rows, columns=score_columns)
        if len(result) < top_n:
            result = self._fill_with_popular(result, read_isbns, top_n)
        return result.merge(self.books, on="isbn", how="left").head(top_n).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "HybridRecommender":
        model = joblib.load(path)
        if not isinstance(model, HybridRecommender):
            raise TypeError(f"Artifact is not a {HybridRecommender.__name__}")
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
        profile_history = user_history[user_history["book_rating"] >= self.min_content_rating]
        if profile_history.empty:
            profile_history = user_history.sort_values("book_rating", ascending=False).head(seed_items)

        indexed_history = [
            (self.isbn_to_index[str(row.isbn)], float(row.book_rating))
            for row in profile_history.itertuples(index=False)
            if str(row.isbn) in self.isbn_to_index
        ]
        if not indexed_history:
            return np.zeros(len(self.books), dtype="float64")

        indices = [index for index, _ in indexed_history]
        ratings = np.array([rating for _, rating in indexed_history], dtype="float64")
        weights = np.clip(ratings / 10.0, 0.1, 1.0)
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
        history_count: int,
        content_scores: np.ndarray,
        collaborative_scores: np.ndarray,
    ) -> dict[str, float]:
        if history_count <= 0:
            return {"collaborative": 0.0, "content": 0.0, "popularity": 1.0}

        if self.dynamic_weights:
            if history_count <= 2:
                weights = {"collaborative": 0.20, "content": 0.65, "popularity": 0.15}
            elif history_count <= 5:
                weights = {"collaborative": 0.35, "content": 0.50, "popularity": 0.15}
            else:
                weights = {
                    "collaborative": self.collaborative_weight,
                    "content": self.content_weight,
                    "popularity": self.popularity_weight,
                }
        else:
            weights = {
                "collaborative": self.collaborative_weight,
                "content": self.content_weight,
                "popularity": self.popularity_weight,
            }

        if not np.any(content_scores > 0.0):
            weights["popularity"] += weights["content"]
            weights["content"] = 0.0
        if not np.any(collaborative_scores > 0.0):
            weights["popularity"] += weights["collaborative"]
            weights["collaborative"] = 0.0
        return normalize_weights(weights)

    def _popular_with_hybrid_columns(self, top_n: int) -> pd.DataFrame:
        popular = self.collaborative_model.recommend_popular(top_n=top_n).copy()
        popular["collaborative_score"] = 0.0
        popular["content_score"] = 0.0
        popular["popularity_score"] = normalize_scores(popular["score"].to_numpy(dtype="float64"))
        popular["score"] = popular["popularity_score"]
        popular["source"] = "popularity"
        return popular.reset_index(drop=True)

    def _fill_with_popular(
        self,
        result: pd.DataFrame,
        read_isbns: set[str],
        top_n: int,
    ) -> pd.DataFrame:
        existing = set(result["isbn"].astype(str)) if not result.empty else set()
        exclude = read_isbns.union(existing)
        popular = self.collaborative_model.recommend_popular(top_n=top_n + len(exclude), exclude_isbns=exclude)
        if popular.empty:
            return result

        filler = popular[["isbn", "score"]].copy()
        filler["collaborative_score"] = 0.0
        filler["content_score"] = 0.0
        filler["popularity_score"] = normalize_scores(filler["score"].to_numpy(dtype="float64"))
        filler["score"] = filler["popularity_score"]
        filler["source"] = "popularity"
        combined = pd.concat([result, filler[result.columns]], ignore_index=True)
        return combined.head(top_n)

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


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return {"collaborative": 0.0, "content": 0.0, "popularity": 1.0}
    return {key: value / total for key, value in clipped.items()}


def strongest_source(collaborative: float, content: float, popularity: float) -> str:
    sources = {
        "collaborative": float(collaborative),
        "content": float(content),
        "popularity": float(popularity),
    }
    return max(sources, key=sources.get)
