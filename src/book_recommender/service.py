from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from book_recommender.artifacts import ArtifactManifest, load_model_bundle
from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender, book_work_key
from book_recommender.hybrid import HybridRecommender


@dataclass(frozen=True)
class RecommenderService:
    """Load-once inference boundary shared by UI and API adapters."""

    model: HybridRecommender
    manifest: ArtifactManifest

    @classmethod
    def load(cls, path: str | Path) -> "RecommenderService":
        model, manifest = load_model_bundle(
            path,
            expected_type=HybridRecommender,
            expected_model_type="hybrid",
        )
        _validate_hybrid_model(model, manifest)
        return cls(model=model, manifest=manifest)

    @property
    def books(self) -> pd.DataFrame:
        return self.model.books

    @property
    def ratings(self) -> pd.DataFrame:
        return self.model.ratings

    @property
    def content_model(self):
        return self.model.content_model

    @property
    def collaborative_model(self):
        return self.model.collaborative_model

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        return self.model.recommend_for_user(user_id=user_id, top_n=top_n)

    def recommend_similar(
        self,
        *,
        title: str | None = None,
        isbn: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        return self.model.content_model.recommend_similar(title=title, isbn=isbn, top_n=top_n)

    def model_info(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "artifact_id": self.manifest.artifact_id,
            "model_type": self.manifest.model_type,
            "created_at": self.manifest.created_at,
            "code_sha": self.manifest.code_sha,
            "dataset_fingerprint": self.manifest.dataset.get("fingerprint"),
            "books": int(len(self.books)),
            "ratings": int(len(self.ratings)),
        }


def _validate_hybrid_model(model: HybridRecommender, manifest: ArtifactManifest) -> None:
    if (
        getattr(model, "model_format_version", None) != HybridRecommender.MODEL_FORMAT_VERSION
        or getattr(model, "algorithm_id", None) != HybridRecommender.ALGORITHM_ID
    ):
        raise ValueError("Hybrid artifact algorithm/version is incompatible; retrain it.")
    model._check_is_fitted()
    if not isinstance(model.content_model, ContentBasedRecommender):
        raise ValueError("Hybrid artifact has an invalid content model.")
    if not isinstance(model.collaborative_model, ItemBasedCollaborativeRecommender):
        raise ValueError("Hybrid artifact has an invalid collaborative model.")
    if (
        getattr(model.content_model, "model_format_version", None)
        != ContentBasedRecommender.MODEL_FORMAT_VERSION
        or getattr(model.content_model, "algorithm_id", None)
        != ContentBasedRecommender.ALGORITHM_ID
    ):
        raise ValueError("Hybrid artifact content algorithm/version is incompatible.")
    if (
        getattr(model.collaborative_model, "model_format_version", None)
        != ItemBasedCollaborativeRecommender.MODEL_FORMAT_VERSION
        or getattr(model.collaborative_model, "algorithm_id", None)
        != ItemBasedCollaborativeRecommender.ALGORITHM_ID
    ):
        raise ValueError("Hybrid artifact collaborative algorithm/version is incompatible.")
    model.content_model._check_is_fitted()
    model.collaborative_model._check_is_fitted()

    required_books = {"isbn", "book_title", "book_author", "publisher"}
    required_ratings = {"user_id", "isbn", "book_rating"}
    if not required_books.issubset(model.books.columns):
        raise ValueError("Hybrid artifact has an invalid books schema.")
    if not required_ratings.issubset(model.ratings.columns):
        raise ValueError("Hybrid artifact has an invalid ratings schema.")
    if len(model.content_model.books) != len(model.books):
        raise ValueError("Hybrid artifact content catalog length is inconsistent.")
    if model.content_model.matrix.shape[0] != len(model.books):
        raise ValueError("Hybrid artifact content matrix shape is inconsistent.")
    if model.collaborative_model.item_user_matrix.shape[0] != len(model.books):
        raise ValueError("Hybrid artifact collaborative matrix shape is inconsistent.")
    if len(model.isbn_to_index) != len(model.books):
        raise ValueError("Hybrid artifact ISBN index is inconsistent.")

    root_isbns = model.books["isbn"].astype(str).tolist()
    if len(root_isbns) != len(set(root_isbns)):
        raise ValueError("Hybrid artifact contains duplicate ISBNs.")
    if model.content_model.books["isbn"].astype(str).tolist() != root_isbns:
        raise ValueError("Hybrid artifact content catalog order is inconsistent.")
    if model.collaborative_model.books["isbn"].astype(str).tolist() != root_isbns:
        raise ValueError("Hybrid artifact collaborative catalog order is inconsistent.")
    expected_isbn_to_index = {isbn: index for index, isbn in enumerate(root_isbns)}
    expected_index_to_isbn = {index: isbn for isbn, index in expected_isbn_to_index.items()}
    if model.isbn_to_index != expected_isbn_to_index:
        raise ValueError("Hybrid artifact ISBN index values are inconsistent.")
    if model.collaborative_model.isbn_to_index != expected_isbn_to_index:
        raise ValueError("Hybrid artifact collaborative ISBN index is inconsistent.")
    if model.collaborative_model.index_to_isbn != expected_index_to_isbn:
        raise ValueError("Hybrid artifact collaborative reverse ISBN index is inconsistent.")

    matrix = model.collaborative_model.item_user_matrix
    if matrix.shape != (len(root_isbns), len(model.collaborative_model.user_to_index)):
        raise ValueError("Hybrid artifact collaborative matrix dimensions are inconsistent.")
    if sorted(model.collaborative_model.user_to_index.values()) != list(
        range(len(model.collaborative_model.user_to_index))
    ):
        raise ValueError("Hybrid artifact user index is inconsistent.")
    rating_users = set(model.ratings["user_id"].astype(int))
    if set(model.collaborative_model.user_to_index) != rating_users:
        raise ValueError("Hybrid artifact user index does not match ratings.")
    if not np.isfinite(matrix.data).all() or not np.isfinite(model.content_model.matrix.data).all():
        raise ValueError("Hybrid artifact matrices contain non-finite values.")

    expected_work_keys = [
        book_work_key(row.book_title, row.book_author, row.isbn)
        for row in model.books.itertuples(index=False)
    ]
    if model.content_model.work_keys != expected_work_keys:
        raise ValueError("Hybrid artifact canonical-work index is inconsistent.")
    expected_isbn_to_work = dict(zip(root_isbns, expected_work_keys))
    if model.collaborative_model.isbn_to_work_key != expected_isbn_to_work:
        raise ValueError("Hybrid artifact collaborative work index is inconsistent.")
    popularity_scores = np.asarray(model.popularity_scores)
    if popularity_scores.shape != (len(root_isbns),) or not np.isfinite(
        popularity_scores
    ).all():
        raise ValueError("Hybrid artifact popularity scores are inconsistent.")
    popularity = model.collaborative_model.popularity
    required_popularity = {"isbn", "mean_rating", "rating_count", "score"}
    if not required_popularity.issubset(popularity.columns):
        raise ValueError("Hybrid artifact popularity table has an invalid schema.")
    if not set(popularity["isbn"].astype(str)).issubset(set(root_isbns)):
        raise ValueError("Hybrid artifact popularity table contains unknown ISBNs.")
    numeric_popularity = popularity[["mean_rating", "rating_count", "score"]].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(numeric_popularity).all():
        raise ValueError("Hybrid artifact popularity table contains non-finite values.")

    if not model.collaborative_model.ratings.equals(model.ratings):
        raise ValueError("Hybrid artifact nested ratings are inconsistent.")
    expected_books_schema = {str(column): str(dtype) for column, dtype in model.books.dtypes.items()}
    expected_ratings_schema = {
        str(column): str(dtype) for column, dtype in model.ratings.dtypes.items()
    }
    if manifest.dataset.get("books_rows") != len(model.books):
        raise ValueError("Artifact manifest books row count does not match the model.")
    if manifest.dataset.get("ratings_rows") != len(model.ratings):
        raise ValueError("Artifact manifest ratings row count does not match the model.")
    if manifest.dataset.get("books_schema") != expected_books_schema:
        raise ValueError("Artifact manifest books schema does not match the model.")
    if manifest.dataset.get("ratings_schema") != expected_ratings_schema:
        raise ValueError("Artifact manifest ratings schema does not match the model.")
    if getattr(model, "artifact_dataset_fingerprint", None) != manifest.dataset.get(
        "fingerprint"
    ):
        raise ValueError("Artifact manifest dataset fingerprint does not match the model.")
