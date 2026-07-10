import numpy as np
import pandas as pd
import pytest
import joblib

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender, book_work_key
from book_recommender.hybrid import HybridRecommender


def hardening_books() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "isbn": "A-1",
                "book_title": "Shared Work",
                "book_author": "Ana Author",
                "publisher": "First Press",
            },
            {
                "isbn": "A-2",
                "book_title": "Shared Work (Paperback Edition)",
                "book_author": "Ana Author",
                "publisher": "Second Press",
            },
            {
                "isbn": "B",
                "book_title": "Shared Patterns",
                "book_author": "Ana Author",
                "publisher": "First Press",
            },
            {
                "isbn": "C",
                "book_title": "Garden Guide",
                "book_author": "Green Writer",
                "publisher": "Home Press",
            },
            {
                "isbn": "D",
                "book_title": "Ocean Notes",
                "book_author": "Blue Writer",
                "publisher": "Nature Press",
            },
        ]
    )


def hardening_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": 1, "isbn": "A-1", "book_rating": 10.0},
            {"user_id": 1, "isbn": "B", "book_rating": 9.0},
            {"user_id": 1, "isbn": "C", "book_rating": 2.0},
            {"user_id": 2, "isbn": "A-2", "book_rating": 10.0},
            {"user_id": 2, "isbn": "B", "book_rating": 9.0},
            {"user_id": 2, "isbn": "D", "book_rating": 3.0},
            {"user_id": 3, "isbn": "A-1", "book_rating": 9.0},
            {"user_id": 3, "isbn": "A-2", "book_rating": 8.0},
            {"user_id": 3, "isbn": "C", "book_rating": 7.0},
            {"user_id": 99, "isbn": "D", "book_rating": 2.0},
        ]
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ContentBasedRecommender(max_features=0),
        lambda: ContentBasedRecommender(max_features=1.5),
        lambda: ItemBasedCollaborativeRecommender(n_neighbors=0),
        lambda: ItemBasedCollaborativeRecommender(positive_rating_threshold=0),
        lambda: ItemBasedCollaborativeRecommender(positive_rating_threshold=np.inf),
        lambda: ItemBasedCollaborativeRecommender(bias_regularization=-1),
        lambda: HybridRecommender(n_neighbors=0),
        lambda: HybridRecommender(collaborative_weight=np.nan),
        lambda: HybridRecommender(
            collaborative_weight=0,
            content_weight=0,
            popularity_weight=0,
        ),
        lambda: HybridRecommender(dynamic_weights=1),
        lambda: HybridRecommender(positive_rating_threshold=11),
    ],
)
def test_constructor_numeric_arguments_are_validated(factory):
    with pytest.raises(ValueError):
        factory()


def test_recommendation_numeric_arguments_are_validated():
    books = hardening_books()
    ratings = hardening_ratings()
    content = ContentBasedRecommender().fit(books)
    collaborative = ItemBasedCollaborativeRecommender(n_neighbors=3).fit(books, ratings)
    hybrid = HybridRecommender(n_neighbors=3).fit(books, ratings)

    with pytest.raises(ValueError, match="top_n"):
        content.recommend_similar(isbn="A-1", top_n=0)
    with pytest.raises(ValueError, match="top_n"):
        collaborative.recommend_for_user(1, top_n=-1)
    with pytest.raises(ValueError, match="seed_items"):
        collaborative.score_for_user(1, seed_items=0)
    with pytest.raises(ValueError, match="candidate_pool_size"):
        collaborative.score_for_user(1, candidate_pool_size=0)
    with pytest.raises(ValueError, match="top_n"):
        collaborative.recommend_popular(top_n=0)
    with pytest.raises(ValueError, match="user_id"):
        hybrid.recommend_for_user(1.5)
    with pytest.raises(ValueError, match="user_id"):
        collaborative.recommend_for_user(0)
    with pytest.raises(ValueError, match="user_id"):
        hybrid.recommend_for_user(-1)
    with pytest.raises(ValueError, match="top_n"):
        hybrid.recommend_for_user(1, top_n=0)
    with pytest.raises(ValueError, match="seed_items"):
        hybrid.recommend_for_user(1, seed_items=0)
    with pytest.raises(ValueError, match="candidate_pool_size"):
        hybrid.recommend_for_user(1, candidate_pool_size=-1)
    with pytest.raises(ValueError, match="allow_popularity_fallback"):
        hybrid.recommend_for_user(1, allow_popularity_fallback=1)


def test_adjusted_cosine_matrix_centers_each_users_explicit_ratings():
    model = ItemBasedCollaborativeRecommender(
        n_neighbors=3,
        bias_regularization=0,
    ).fit(hardening_books(), hardening_ratings())

    column_sums = np.asarray(model.item_user_matrix.sum(axis=0)).ravel()
    assert np.allclose(column_sums, 0.0, atol=1e-6)
    assert np.any(model.item_user_matrix.data < 0.0)
    assert np.any(model.item_user_matrix.data > 0.0)


def test_model_fit_normalizes_identifiers_and_fails_on_empty_catalog_join():
    books = hardening_books().copy()
    numeric_books = books.iloc[:2].copy()
    numeric_books["isbn"] = [1, 2]
    numeric_ratings = pd.DataFrame(
        [
            {"user_id": 1, "isbn": "1", "book_rating": 9},
            {"user_id": 2, "isbn": "2", "book_rating": 8},
        ]
    )

    model = ItemBasedCollaborativeRecommender(n_neighbors=1).fit(
        numeric_books,
        numeric_ratings,
    )
    assert set(model.books["isbn"]) == {"1", "2"}

    mismatched = pd.DataFrame(
        [{"user_id": 1, "isbn": "missing", "book_rating": 9}]
    )
    with pytest.raises(ValueError, match="no rows matching"):
        ItemBasedCollaborativeRecommender().fit(numeric_books, mismatched)
    with pytest.raises(ValueError, match="no rows matching"):
        HybridRecommender().fit(numeric_books, mismatched)


def test_legacy_raw_collaborative_artifact_fails_closed(tmp_path):
    model = ItemBasedCollaborativeRecommender(n_neighbors=3).fit(
        hardening_books(),
        hardening_ratings(),
    )
    del model.model_format_version
    del model.algorithm_id
    artifact = tmp_path / "legacy.joblib"
    joblib.dump(model, artifact)

    with pytest.raises(ValueError, match="Legacy collaborative artifact"):
        ItemBasedCollaborativeRecommender.load(artifact)


def test_low_explicit_ratings_are_not_used_as_preference_seeds():
    books = hardening_books()
    ratings = hardening_ratings()
    collaborative = ItemBasedCollaborativeRecommender(n_neighbors=3).fit(books, ratings)
    hybrid = HybridRecommender(n_neighbors=3).fit(books, ratings)

    assert collaborative.score_for_user(99).empty

    recommendations = hybrid.recommend_for_user(99, top_n=2)

    assert not recommendations.empty
    assert set(recommendations["source"]) == {"popularity"}
    assert np.allclose(recommendations["collaborative_score"], 0.0)
    assert np.allclose(recommendations["content_score"], 0.0)
    assert all("fallback" in reason.casefold() for reason in recommendations["reason"])


def test_strict_ablation_does_not_leak_popularity_when_preference_signal_is_absent():
    model = HybridRecommender(
        n_neighbors=3,
        collaborative_weight=0.0,
        content_weight=1.0,
        popularity_weight=0.0,
        dynamic_weights=False,
    ).fit(hardening_books(), hardening_ratings())

    recommendations = model.recommend_for_user(
        99,
        top_n=3,
        allow_popularity_fallback=False,
    )

    assert recommendations.empty


def test_dynamic_weight_confidence_counts_only_positive_preferences(monkeypatch):
    model = HybridRecommender(n_neighbors=3).fit(hardening_books(), hardening_ratings())
    observed_counts = []
    original = model._weights_for_history

    def capture_count(positive_history_count, *args, **kwargs):
        observed_counts.append(positive_history_count)
        return original(positive_history_count, *args, **kwargs)

    monkeypatch.setattr(model, "_weights_for_history", capture_count)
    model.recommend_for_user(1, top_n=1)

    assert observed_counts == [2]
    assert len(model.ratings[model.ratings["user_id"] == 1]) == 3


def test_dynamic_weight_confidence_counts_unique_positive_works(monkeypatch):
    model = HybridRecommender(n_neighbors=3).fit(hardening_books(), hardening_ratings())
    observed_counts = []
    original = model._weights_for_history

    def capture_count(positive_history_count, *args, **kwargs):
        observed_counts.append(positive_history_count)
        return original(positive_history_count, *args, **kwargs)

    monkeypatch.setattr(model, "_weights_for_history", capture_count)
    model.recommend_for_user(3, top_n=1)

    assert observed_counts == [1]


def test_content_output_excludes_other_editions_of_query_work():
    model = ContentBasedRecommender().fit(hardening_books())

    recommendations = model.recommend_similar(isbn="A-1", top_n=4)

    assert "A-2" not in set(recommendations["isbn"])
    assert len(recommendations) == 3


def test_content_query_does_not_materialize_the_full_catalog(monkeypatch):
    rows = [
        {
            "isbn": "ISBN-0",
            "book_title": "Quantum Python Guide",
            "book_author": "Ada Expert",
            "publisher": "Technical Press",
        }
    ]
    rows.extend(
        {
            "isbn": f"ISBN-{index}",
            "book_title": f"Quantum Python Advanced {index}",
            "book_author": "Ada Expert",
            "publisher": "Technical Press",
        }
        for index in range(1, 21)
    )
    rows.extend(
        {
            "isbn": f"ISBN-{index}",
            "book_title": f"Gardening Subject {index}",
            "book_author": f"Writer {index}",
            "publisher": "Garden House",
        }
        for index in range(21, 500)
    )
    books = pd.DataFrame(rows)
    model = ContentBasedRecommender().fit(books)
    original = model.neighbors.kneighbors
    requested_sizes = []

    def observed_kneighbors(*args, **kwargs):
        requested_sizes.append(kwargs["n_neighbors"])
        return original(*args, **kwargs)

    monkeypatch.setattr(model.neighbors, "kneighbors", observed_kneighbors)
    recommendations = model.recommend_similar(isbn="ISBN-0", top_n=10)

    assert len(recommendations) == 10
    assert max(requested_sizes) < len(books)


def test_all_recommendation_outputs_contain_unique_works():
    books = hardening_books()
    ratings = hardening_ratings()
    collaborative = ItemBasedCollaborativeRecommender(n_neighbors=4).fit(books, ratings)
    hybrid = HybridRecommender(n_neighbors=4).fit(books, ratings)

    outputs = [
        collaborative.recommend_popular(top_n=4),
        collaborative.recommend_for_user(user_id=404, top_n=4),
        hybrid.recommend_for_user(user_id=404, top_n=4),
    ]
    for recommendations in outputs:
        work_keys = [
            book_work_key(row.book_title, row.book_author, row.isbn)
            for row in recommendations.itertuples(index=False)
        ]
        assert len(work_keys) == len(set(work_keys))
        assert len(recommendations) == 4


def test_personalized_reason_names_the_configured_positive_threshold():
    model = HybridRecommender(
        n_neighbors=4,
        collaborative_weight=0.0,
        content_weight=1.0,
        popularity_weight=0.0,
        dynamic_weights=False,
        positive_rating_threshold=9.0,
    ).fit(hardening_books(), hardening_ratings())

    recommendations = model.recommend_for_user(1, top_n=1)

    assert recommendations.iloc[0]["source"] == "content"
    assert "9/10 or higher" in recommendations.iloc[0]["reason"]
