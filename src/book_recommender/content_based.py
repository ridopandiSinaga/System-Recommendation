from __future__ import annotations

import numbers
from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from book_recommender.data import clean_books


class BookNotFoundError(ValueError):
    """Raised when a requested book is not present in the fitted catalog."""


class ContentBasedRecommender:
    MODEL_FORMAT_VERSION = 2
    ALGORITHM_ID = "tfidf_cosine_work_dedup_v2"

    def __init__(self, max_features: int = 50_000):
        self.model_format_version = self.MODEL_FORMAT_VERSION
        self.algorithm_id = self.ALGORITHM_ID
        self.max_features = _validate_positive_int(max_features, "max_features")
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=self.max_features,
        )
        self.neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
        self.books: pd.DataFrame | None = None
        self.matrix = None
        self.work_keys: list[str] | None = None

    def fit(self, books: pd.DataFrame) -> "ContentBasedRecommender":
        self.books = clean_books(books)
        if self.books.empty:
            raise ValueError("Books data is empty after validation.")
        text = self._metadata_text(self.books)
        self.matrix = self.vectorizer.fit_transform(text)
        self.neighbors.fit(self.matrix)
        self.work_keys = [
            book_work_key(row.book_title, row.book_author, row.isbn)
            for row in self.books.itertuples(index=False)
        ]
        return self

    def recommend_similar(
        self,
        title: str | None = None,
        isbn: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        top_n = _validate_positive_int(top_n, "top_n")
        query_index = self._resolve_query_index(title=title, isbn=isbn)
        query_row = self.books.iloc[query_index]
        query_isbn = str(query_row["isbn"])
        work_keys = self._work_keys()
        query_work_key = work_keys[query_index]
        catalog_size = len(self.books)
        pool_size = min(catalog_size, max(32, top_n * 4 + 1))

        while True:
            distances, indices = self.neighbors.kneighbors(
                self.matrix[query_index],
                n_neighbors=pool_size,
            )
            candidate_indices = indices[0]
            candidates = self.books.iloc[candidate_indices].copy()
            candidates["score"] = 1.0 - distances[0].astype(float)
            candidates["_work_key"] = [work_keys[int(index)] for index in candidate_indices]
            candidates = candidates[
                (candidates["isbn"].astype(str) != query_isbn)
                & (candidates["_work_key"] != query_work_key)
            ].sort_values(
                ["score", "isbn"],
                ascending=[False, True],
                kind="stable",
            )
            selected = candidates.drop_duplicates("_work_key", keep="first").head(top_n)
            if len(selected) >= top_n:
                cutoff = float(selected.iloc[-1]["score"])
                boundary = 1.0 - float(distances[0][-1])
                if pool_size == catalog_size or boundary < cutoff - np.finfo(float).eps:
                    return selected.drop(columns="_work_key").reset_index(drop=True)
            if pool_size == catalog_size:
                return selected.drop(columns="_work_key").reset_index(drop=True)
            pool_size = min(catalog_size, pool_size * 2)

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ContentBasedRecommender":
        model = joblib.load(path)
        if not isinstance(model, ContentBasedRecommender):
            raise TypeError(f"Artifact is not a {ContentBasedRecommender.__name__}")
        if (
            getattr(model, "model_format_version", None)
            != ContentBasedRecommender.MODEL_FORMAT_VERSION
            or getattr(model, "algorithm_id", None) != ContentBasedRecommender.ALGORITHM_ID
        ):
            raise ValueError("Legacy content artifact is incompatible; retrain the model.")
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
            raise BookNotFoundError(f"ISBN not found: {isbn}")

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

        raise BookNotFoundError(f"Book title not found: {title}")

    def _check_is_fitted(self) -> None:
        if self.books is None or self.matrix is None:
            raise RuntimeError("Recommender has not been fitted.")

    def _work_keys(self) -> list[str]:
        if self.books is None:
            raise RuntimeError("Recommender has not been fitted.")
        work_keys = getattr(self, "work_keys", None)
        if work_keys is None or len(work_keys) != len(self.books):
            work_keys = [
                book_work_key(row.book_title, row.book_author, row.isbn)
                for row in self.books.itertuples(index=False)
            ]
            self.work_keys = work_keys
        return work_keys


_EDITION_QUALIFIER = re.compile(
    r"[\(\[][^\)\]]*\b(?:edition|paperback|hardcover|mass\s+market|reprint|large\s+print)\b"
    r"[^\)\]]*[\)\]]",
    flags=re.IGNORECASE,
)


def book_work_key(title: object, author: object, isbn: object | None = None) -> str:
    """Return a deterministic identity for a work across ISBN editions.

    ISBN identifies a physical edition, while a normalized title/author pair identifies the
    work well enough for output-level diversity. ISBN is only used as a fallback for malformed
    rows without a title, avoiding accidental collapse of unrelated missing-title records.
    """

    normalized_title = _normalize_work_component(title, strip_edition_qualifiers=True)
    normalized_author = _normalize_work_component(author)
    if normalized_title:
        return f"title:{normalized_title}|author:{normalized_author}"
    return f"isbn:{_normalize_work_component(isbn)}"


def rerank_unique_works(
    candidates: pd.DataFrame,
    top_n: int,
    excluded_work_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Preserve candidate order while allowing at most one edition of each work."""

    seen = set(excluded_work_keys or set())
    selected_indices: list[object] = []
    for index, row in candidates.iterrows():
        key = book_work_key(row.get("book_title"), row.get("book_author"), row.get("isbn"))
        if key in seen:
            continue
        seen.add(key)
        selected_indices.append(index)
        if len(selected_indices) >= top_n:
            break

    if not selected_indices:
        return candidates.iloc[0:0].copy().reset_index(drop=True)
    return candidates.loc[selected_indices].copy().reset_index(drop=True)


def _normalize_work_component(value: object, strip_edition_qualifiers: bool = False) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    if strip_edition_qualifiers:
        text = _EDITION_QUALIFIER.sub(" ", text)
    text = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(text.split())


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)
