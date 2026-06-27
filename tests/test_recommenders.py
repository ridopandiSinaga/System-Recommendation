import pandas as pd

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender
from book_recommender.data import clean_books, clean_ratings
from book_recommender.evaluation import (
    average_precision_at_k,
    build_leave_one_out_split,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    ranking_metrics_at_k,
    recall_at_k,
)


def sample_books():
    return clean_books(
        pd.DataFrame(
            [
                {
                    "ISBN": "A",
                    "Book-Title": "Python Basics",
                    "Book-Author": "Ada Smith",
                    "Publisher": "Tech Press",
                },
                {
                    "ISBN": "B",
                    "Book-Title": "Python Data Projects",
                    "Book-Author": "Ada Smith",
                    "Publisher": "Tech Press",
                },
                {
                    "ISBN": "C",
                    "Book-Title": "Gardening Notes",
                    "Book-Author": "Lee Green",
                    "Publisher": "Home Press",
                },
            ]
        )
    )


def sample_ratings():
    return clean_ratings(
        pd.DataFrame(
            [
                {"User-ID": 1, "ISBN": "A", "Book-Rating": 10},
                {"User-ID": 1, "ISBN": "B", "Book-Rating": 9},
                {"User-ID": 2, "ISBN": "A", "Book-Rating": 10},
                {"User-ID": 3, "ISBN": "B", "Book-Rating": 8},
                {"User-ID": 3, "ISBN": "C", "Book-Rating": 2},
            ]
        )
    )


def test_content_based_recommends_similar_metadata():
    recommender = ContentBasedRecommender().fit(sample_books())
    recommendations = recommender.recommend_similar(title="Python Basics", top_n=1)

    assert recommendations.iloc[0]["isbn"] == "B"


def test_content_based_artifact_roundtrip(tmp_path):
    artifact = tmp_path / "content.joblib"
    recommender = ContentBasedRecommender().fit(sample_books())

    recommender.save(artifact)
    loaded = ContentBasedRecommender.load(artifact)
    recommendations = loaded.recommend_similar(title="Python Basics", top_n=1)

    assert recommendations.iloc[0]["isbn"] == "B"


def test_collaborative_recommends_unread_similar_item():
    recommender = ItemBasedCollaborativeRecommender(n_neighbors=2).fit(sample_books(), sample_ratings())
    recommendations = recommender.recommend_for_user(user_id=2, top_n=1)

    assert recommendations.iloc[0]["isbn"] == "B"


def test_collaborative_artifact_roundtrip(tmp_path):
    artifact = tmp_path / "collab.joblib"
    recommender = ItemBasedCollaborativeRecommender(n_neighbors=2).fit(
        sample_books(),
        sample_ratings(),
    )

    recommender.save(artifact)
    loaded = ItemBasedCollaborativeRecommender.load(artifact)
    recommendations = loaded.recommend_for_user(user_id=2, top_n=1)

    assert recommendations.iloc[0]["isbn"] == "B"


def test_ranking_metrics():
    recommended = ["B", "C", "A", "D"]
    relevant = {"A", "B"}

    assert precision_at_k(recommended, relevant, k=2) == 0.5
    assert recall_at_k(recommended, relevant, k=2) == 0.5
    assert hit_rate_at_k(recommended, relevant, k=2) == 1.0
    assert average_precision_at_k(recommended, relevant, k=4) == 0.8333333333333333
    assert round(ndcg_at_k(recommended, relevant, k=4), 4) == 0.9197
    assert precision_at_k(["A"], relevant, k=4) == 0.25

    metrics = ranking_metrics_at_k(recommended, relevant, k=4)

    assert metrics["precision_at_4"] == 0.5
    assert metrics["recall_at_4"] == 1.0
    assert metrics["hit_rate_at_4"] == 1.0
    assert metrics["map_at_4"] == 0.8333333333333333
    assert round(metrics["ndcg_at_4"], 4) == 0.9197


def test_leave_one_out_split_removes_holdout_rating():
    train, holdout = build_leave_one_out_split(
        sample_ratings(),
        sample_users=1,
        min_interactions=2,
        min_relevant_rating=8,
        random_state=42,
    )

    assert len(holdout) == 1
    assert len(train) == len(sample_ratings()) - 1
    holdout_row = holdout.iloc[0]
    overlap = train[
        (train["user_id"] == holdout_row["user_id"])
        & (train["isbn"] == holdout_row["isbn"])
        & (train["book_rating"] == holdout_row["book_rating"])
    ]
    assert overlap.empty
