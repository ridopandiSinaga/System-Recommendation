from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class ItemBasedCollaborativeRecommender:
    """Lightweight item-item collaborative filtering over a sparse rating matrix."""

    def __init__(self, n_neighbors: int = 50):
        self.n_neighbors = n_neighbors
        self.neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
        self.books: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.item_user_matrix: csr_matrix | None = None
        self.isbn_to_index: dict[str, int] = {}
        self.index_to_isbn: dict[int, str] = {}
        self.user_to_index: dict[int, int] = {}
        self.popularity: pd.DataFrame | None = None

    def fit(self, books: pd.DataFrame, ratings: pd.DataFrame) -> "ItemBasedCollaborativeRecommender":
        required_books = {"isbn", "book_title", "book_author", "publisher"}
        required_ratings = {"user_id", "isbn", "book_rating"}
        missing_books = required_books.difference(books.columns)
        missing_ratings = required_ratings.difference(ratings.columns)
        if missing_books:
            raise ValueError(f"Books data is missing columns: {', '.join(sorted(missing_books))}")
        if missing_ratings:
            raise ValueError(f"Ratings data is missing columns: {', '.join(sorted(missing_ratings))}")
        if ratings.empty:
            raise ValueError("Ratings data is empty after preprocessing.")

        self.books = books.drop_duplicates("isbn").reset_index(drop=True).copy()
        valid_isbns = set(self.books["isbn"])
        self.ratings = ratings[ratings["isbn"].isin(valid_isbns)].reset_index(drop=True).copy()

        users = sorted(self.ratings["user_id"].unique())
        items = self.books["isbn"].tolist()
        self.user_to_index = {int(user_id): index for index, user_id in enumerate(users)}
        self.isbn_to_index = {str(isbn): index for index, isbn in enumerate(items)}
        self.index_to_isbn = {index: isbn for isbn, index in self.isbn_to_index.items()}

        row_indices = self.ratings["isbn"].map(self.isbn_to_index).to_numpy()
        col_indices = self.ratings["user_id"].map(self.user_to_index).to_numpy()
        values = self.ratings["book_rating"].astype("float32").to_numpy()
        self.item_user_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(items), len(users)),
        )
        self.neighbors.fit(self.item_user_matrix)
        self.popularity = self._build_popularity()
        return self

    def recommend_for_user(self, user_id: int, top_n: int = 10, seed_items: int = 5) -> pd.DataFrame:
        self._check_is_fitted()
        user_history = self.ratings[self.ratings["user_id"] == int(user_id)].copy()
        if user_history.empty:
            return self.recommend_popular(top_n=top_n)

        read_isbns = set(user_history["isbn"])
        seed_history = user_history.sort_values("book_rating", ascending=False).head(seed_items)
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
            rating_weight = float(row.book_rating) / 10.0
            for distance, neighbor_index in zip(distances[0], indices[0]):
                candidate_isbn = self.index_to_isbn[int(neighbor_index)]
                if candidate_isbn == isbn or candidate_isbn in read_isbns:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                if similarity <= 0.0:
                    continue
                scores[candidate_isbn] += similarity * rating_weight

        if not scores:
            return self.recommend_popular(top_n=top_n, exclude_isbns=read_isbns)

        scored = pd.DataFrame(
            [{"isbn": isbn, "score": score} for isbn, score in scores.items()]
        ).sort_values("score", ascending=False)
        result = scored.merge(self.books, on="isbn", how="left")
        return result.head(top_n).reset_index(drop=True)

    def recommend_popular(
        self,
        top_n: int = 10,
        exclude_isbns: set[str] | None = None,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        exclude_isbns = exclude_isbns or set()
        popular = self.popularity[~self.popularity["isbn"].isin(exclude_isbns)].head(top_n)
        return popular.merge(self.books, on="isbn", how="left").reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ItemBasedCollaborativeRecommender":
        model = joblib.load(path)
        if not isinstance(model, ItemBasedCollaborativeRecommender):
            raise TypeError(f"Artifact is not a {ItemBasedCollaborativeRecommender.__name__}")
        return model

    def _build_popularity(self) -> pd.DataFrame:
        popularity = (
            self.ratings.groupby("isbn")
            .agg(mean_rating=("book_rating", "mean"), rating_count=("book_rating", "size"))
            .reset_index()
        )
        popularity["score"] = popularity["mean_rating"] * np.log1p(popularity["rating_count"])
        return popularity.sort_values(["score", "rating_count"], ascending=False).reset_index(drop=True)

    def _check_is_fitted(self) -> None:
        if (
            self.books is None
            or self.ratings is None
            or self.item_user_matrix is None
            or self.popularity is None
        ):
            raise RuntimeError("Recommender has not been fitted.")
