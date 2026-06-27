import pandas as pd

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender
from book_recommender.data import clean_books, clean_ratings
from book_recommender.evaluation import precision_at_k, recall_at_k


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


def test_collaborative_recommends_unread_similar_item():
    recommender = ItemBasedCollaborativeRecommender(n_neighbors=2).fit(sample_books(), sample_ratings())
    recommendations = recommender.recommend_for_user(user_id=2, top_n=1)

    assert recommendations.iloc[0]["isbn"] == "B"


def test_ranking_metrics():
    recommended = ["B", "C", "D"]
    relevant = {"A", "B"}

    assert precision_at_k(recommended, relevant, k=2) == 0.5
    assert recall_at_k(recommended, relevant, k=2) == 0.5
