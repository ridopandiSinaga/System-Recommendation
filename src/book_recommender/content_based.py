from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class ContentBasedRecommender:
    def __init__(self, max_features: int = 50_000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=max_features,
        )
        self.neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
        self.books: pd.DataFrame | None = None
        self.matrix = None

    def fit(self, books: pd.DataFrame) -> "ContentBasedRecommender":
        required = {"isbn", "book_title", "book_author", "publisher"}
        missing = required.difference(books.columns)
        if missing:
            raise ValueError(f"Books data is missing columns: {', '.join(sorted(missing))}")

        self.books = books.reset_index(drop=True).copy()
        text = self._metadata_text(self.books)
        self.matrix = self.vectorizer.fit_transform(text)
        self.neighbors.fit(self.matrix)
        return self

    def recommend_similar(
        self,
        title: str | None = None,
        isbn: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        query_index = self._resolve_query_index(title=title, isbn=isbn)
        n_neighbors = min(len(self.books), top_n + 1)
        distances, indices = self.neighbors.kneighbors(self.matrix[query_index], n_neighbors=n_neighbors)

        rows = []
        query_isbn = self.books.iloc[query_index]["isbn"]
        for distance, index in zip(distances[0], indices[0]):
            row = self.books.iloc[index].copy()
            if row["isbn"] == query_isbn:
                continue
            row["score"] = 1.0 - float(distance)
            rows.append(row)
            if len(rows) >= top_n:
                break

        return pd.DataFrame(rows).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ContentBasedRecommender":
        model = joblib.load(path)
        if not isinstance(model, ContentBasedRecommender):
            raise TypeError(f"Artifact is not a {ContentBasedRecommender.__name__}")
        return model

    @staticmethod
    def _metadata_text(books: pd.DataFrame) -> pd.Series:
        return (
            books["book_title"].fillna("")
            + " "
            + books["book_author"].fillna("")
            + " "
            + books["publisher"].fillna("")
        )

    def _resolve_query_index(self, title: str | None, isbn: str | None) -> int:
        if self.books is None:
            raise RuntimeError("Recommender has not been fitted.")
        if isbn:
            matches = self.books.index[self.books["isbn"].astype(str).str.casefold() == str(isbn).casefold()]
            if len(matches):
                return int(matches[0])
            raise ValueError(f"ISBN not found: {isbn}")

        if not title:
            raise ValueError("Provide either title or isbn.")

        normalized = title.casefold()
        exact = self.books.index[self.books["book_title"].astype(str).str.casefold() == normalized]
        if len(exact):
            return int(exact[0])

        contains = self.books.index[
            self.books["book_title"].astype(str).str.casefold().str.contains(normalized, regex=False)
        ]
        if len(contains):
            return int(contains[0])

        raise ValueError(f"Book title not found: {title}")

    def _check_is_fitted(self) -> None:
        if self.books is None or self.matrix is None:
            raise RuntimeError("Recommender has not been fitted.")
