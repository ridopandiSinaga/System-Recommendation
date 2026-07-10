import pandas as pd
import pytest

from book_recommender.artifacts import fingerprint_dataset, save_model_bundle
from book_recommender.data import clean_books, clean_ratings
from book_recommender.hybrid import HybridRecommender
from book_recommender.service import RecommenderService


def test_service_loads_verified_composite_bundle(tmp_path):
    books = clean_books(
        pd.DataFrame(
            [
                {"ISBN": "A", "Book-Title": "Alpha", "Book-Author": "One", "Publisher": "P"},
                {"ISBN": "B", "Book-Title": "Beta", "Book-Author": "One", "Publisher": "P"},
                {"ISBN": "C", "Book-Title": "Gamma", "Book-Author": "Two", "Publisher": "Q"},
            ]
        )
    )
    ratings = clean_ratings(
        pd.DataFrame(
            [
                {"User-ID": 1, "ISBN": "A", "Book-Rating": 10},
                {"User-ID": 1, "ISBN": "B", "Book-Rating": 9},
                {"User-ID": 2, "ISBN": "A", "Book-Rating": 9},
                {"User-ID": 2, "ISBN": "C", "Book-Rating": 8},
            ]
        )
    )
    model = HybridRecommender(n_neighbors=2).fit(books, ratings)
    artifact = tmp_path / "hybrid.joblib"
    save_model_bundle(
        model,
        artifact,
        model_type="hybrid",
        dataset_fingerprint=fingerprint_dataset(books, ratings),
        training_config={"n_neighbors": 2},
    )

    service = RecommenderService.load(artifact)

    assert service.model_info()["status"] == "ready"
    assert service.model_info()["books"] == 3
    assert not service.recommend_for_user(1, top_n=1).empty
    assert service.recommend_similar(isbn="A", top_n=1).iloc[0]["isbn"] == "B"


def test_service_rejects_nested_catalog_order_mismatch(tmp_path):
    books = clean_books(
        pd.DataFrame(
            [
                {"ISBN": "A", "Book-Title": "Alpha", "Book-Author": "One", "Publisher": "P"},
                {"ISBN": "B", "Book-Title": "Beta", "Book-Author": "Two", "Publisher": "Q"},
            ]
        )
    )
    ratings = clean_ratings(
        pd.DataFrame(
            [
                {"User-ID": 1, "ISBN": "A", "Book-Rating": 10},
                {"User-ID": 1, "ISBN": "B", "Book-Rating": 9},
            ]
        )
    )
    model = HybridRecommender(n_neighbors=2).fit(books, ratings)
    model.content_model.books = model.content_model.books.iloc[::-1].reset_index(drop=True)
    artifact = tmp_path / "corrupt-hybrid.joblib"
    save_model_bundle(
        model,
        artifact,
        model_type="hybrid",
        dataset_fingerprint=fingerprint_dataset(books, ratings),
        training_config={},
    )

    with pytest.raises(ValueError, match="content catalog order"):
        RecommenderService.load(artifact)
