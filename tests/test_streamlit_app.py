from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.streamlit_app import normalize_cover_url
from book_recommender.artifacts import fingerprint_dataset, save_model_bundle
from book_recommender.data import clean_books, clean_ratings
from book_recommender.hybrid import HybridRecommender


def _serving_artifact(tmp_path: Path) -> Path:
    books = clean_books(
        pd.DataFrame(
            [
                {"ISBN": "A", "Book-Title": "Alpha", "Book-Author": "One", "Publisher": "P"},
                {"ISBN": "B", "Book-Title": "Beta", "Book-Author": "One", "Publisher": "P"},
                {"ISBN": "C", "Book-Title": "Gamma", "Book-Author": "Two", "Publisher": "Q"},
                {"ISBN": "D", "Book-Title": "Delta", "Book-Author": "Three", "Publisher": "R"},
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
                {"User-ID": 3, "ISBN": "B", "Book-Rating": 9},
                {"User-ID": 3, "ISBN": "D", "Book-Rating": 8},
            ]
        )
    )
    model = HybridRecommender(n_neighbors=3).fit(books, ratings)
    artifact = tmp_path / "hybrid.joblib"
    save_model_bundle(
        model,
        artifact,
        model_type="hybrid",
        dataset_fingerprint=fingerprint_dataset(books, ratings),
        training_config={"n_neighbors": 3},
    )
    return artifact


def test_streamlit_app_loads_one_verified_bundle(monkeypatch, tmp_path):
    artifact = _serving_artifact(tmp_path)
    monkeypatch.setenv("BOOK_RECOMMENDER_ARTIFACT", str(artifact))
    monkeypatch.delenv("BOOK_RECOMMENDER_ALLOW_RUNTIME_TRAINING", raising=False)
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "Book Recommender"
    assert any("Artifact:" in success.value for success in app.sidebar.success)
    assert len(app.tabs) == 3


def test_streamlit_app_hides_artifact_failure_details(monkeypatch, tmp_path):
    missing = tmp_path / "secret-model-path" / "hybrid.joblib"
    monkeypatch.setenv("BOOK_RECOMMENDER_ARTIFACT", str(missing))
    monkeypatch.delenv("BOOK_RECOMMENDER_ALLOW_RUNTIME_TRAINING", raising=False)
    st.cache_resource.clear()
    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert app.error
    assert "Model service is unavailable" in app.error[0].value
    assert "secret-model-path" not in app.error[0].value


def test_cover_urls_are_restricted_to_known_dataset_hosts():
    assert (
        normalize_cover_url("http://images.amazon.com/images/P/example.jpg")
        == "https://images.amazon.com/images/P/example.jpg"
    )
    assert normalize_cover_url("https://tracker.example/cover.jpg") is None
    assert normalize_cover_url("javascript:alert(1)") is None
