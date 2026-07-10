import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.cli import parse_weight_grid, rank_tuning_results
from book_recommender.content_based import ContentBasedRecommender
from book_recommender.data import BookDataset, clean_books, clean_ratings
from book_recommender.hybrid import HybridRecommender
from book_recommender.evaluation import (
    average_precision_at_k,
    build_leave_one_out_split,
    catalog_coverage_at_k,
    hit_rate_at_k,
    intra_list_diversity_at_k,
    mean_novelty_at_k,
    ndcg_at_k,
    popularity_share_at_k,
    precision_at_k,
    prepare_leave_one_out_evaluation,
    ranking_metrics_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    recommendation_diagnostics_at_k,
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


def test_hybrid_recommends_unread_item_with_component_scores():
    recommender = HybridRecommender(n_neighbors=2).fit(sample_books(), sample_ratings())
    recommendations = recommender.recommend_for_user(user_id=2, top_n=1, candidate_pool_size=5)

    assert recommendations.iloc[0]["isbn"] == "B"
    assert recommendations.iloc[0]["score"] > 0
    assert "collaborative_score" in recommendations.columns
    assert "content_score" in recommendations.columns
    assert "popularity_score" in recommendations.columns
    assert "reason" in recommendations.columns
    assert recommendations.iloc[0]["reason"]
    assert recommendations.iloc[0]["source"] in {"collaborative", "content", "popularity"}


def test_hybrid_unknown_user_uses_popularity_fallback():
    recommender = HybridRecommender(n_neighbors=2).fit(sample_books(), sample_ratings())
    recommendations = recommender.recommend_for_user(user_id=999, top_n=1)

    assert not recommendations.empty
    assert recommendations.iloc[0]["source"] == "popularity"
    assert recommendations.iloc[0]["reason"] == "Popular fallback from books with strong rating history."


def test_hybrid_artifact_roundtrip(tmp_path):
    artifact = tmp_path / "hybrid.joblib"
    recommender = HybridRecommender(n_neighbors=2).fit(sample_books(), sample_ratings())

    recommender.save(artifact)
    loaded = HybridRecommender.load(artifact)
    recommendations = loaded.recommend_for_user(user_id=2, top_n=1, candidate_pool_size=5)

    assert recommendations.iloc[0]["isbn"] == "B"


def test_parse_weight_grid_normalizes_configs():
    configs = parse_weight_grid("2,1,1;0.5,0.4,0.1")

    assert configs[0]["collaborative_weight"] == 0.5
    assert configs[0]["content_weight"] == 0.25
    assert configs[0]["popularity_weight"] == 0.25
    assert configs[1]["collaborative_weight"] == 0.5
    assert configs[1]["content_weight"] == 0.4
    assert configs[1]["popularity_weight"] == 0.1


@pytest.mark.parametrize("value", ["nan,1,1", "inf,1,1", "-inf,1,1"])
def test_parse_weight_grid_rejects_non_finite_weights(value):
    with pytest.raises(ValueError, match="finite"):
        parse_weight_grid(value)


def test_rank_tuning_results_prefers_hit_rate_then_ndcg():
    results = [
        {
            "config_id": "low",
            "hit_rate_at_10": 0.1,
            "ndcg_at_10": 0.5,
            "catalog_coverage_at_10": 0.9,
            "top_10pct_popularity_share_at_10": 0.1,
        },
        {
            "config_id": "best",
            "hit_rate_at_10": 0.2,
            "ndcg_at_10": 0.1,
            "catalog_coverage_at_10": 0.1,
            "top_10pct_popularity_share_at_10": 0.9,
        },
        {
            "config_id": "tie_break",
            "hit_rate_at_10": 0.2,
            "ndcg_at_10": 0.2,
            "catalog_coverage_at_10": 0.1,
            "top_10pct_popularity_share_at_10": 0.9,
        },
    ]

    ranked = rank_tuning_results(results, k=10)

    assert ranked[0]["config_id"] == "tie_break"
    assert ranked[1]["config_id"] == "best"


def test_ranking_metrics():
    recommended = ["B", "C", "A", "D"]
    relevant = {"A", "B"}

    assert precision_at_k(recommended, relevant, k=2) == 0.5
    assert recall_at_k(recommended, relevant, k=2) == 0.5
    assert hit_rate_at_k(recommended, relevant, k=2) == 1.0
    assert average_precision_at_k(recommended, relevant, k=4) == 0.8333333333333333
    assert reciprocal_rank_at_k(recommended, relevant, k=4) == 1.0
    assert round(ndcg_at_k(recommended, relevant, k=4), 4) == 0.9197
    assert precision_at_k(["A"], relevant, k=4) == 0.25

    metrics = ranking_metrics_at_k(recommended, relevant, k=4)

    assert metrics["precision_at_4"] == 0.5
    assert metrics["recall_at_4"] == 1.0
    assert metrics["hit_rate_at_4"] == 1.0
    assert metrics["map_at_4"] == 0.8333333333333333
    assert metrics["mrr_at_4"] == 1.0
    assert round(metrics["ndcg_at_4"], 4) == 0.9197


def test_recommendation_diagnostics():
    recommendation_lists = [["A", "B"], ["B", "C"]]
    catalog = ["A", "B", "C", "D"]
    popularity = pd.Series({"A": 10, "B": 5, "C": 1, "D": 1})
    features = csr_matrix(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    item_to_index = {isbn: index for index, isbn in enumerate(catalog)}

    assert catalog_coverage_at_k(recommendation_lists, catalog, k=2) == 0.75
    assert popularity_share_at_k(recommendation_lists, popularity, top_percent=0.25, k=2) == 0.25
    assert mean_novelty_at_k(recommendation_lists, popularity, total_interactions=17, k=2) > 0
    assert intra_list_diversity_at_k(["A", "B"], features, item_to_index, k=2) == 1.0

    diagnostics = recommendation_diagnostics_at_k(
        recommendation_lists,
        catalog_items=catalog,
        popularity_counts=popularity,
        k=2,
        item_features=features,
        item_to_index=item_to_index,
    )

    assert diagnostics["catalog_coverage_at_2"] == 0.75
    assert diagnostics["top_1pct_popularity_share_at_2"] == 0.25
    assert diagnostics["intra_list_diversity_at_2"] == 1.0


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


def test_leave_one_out_evaluation_filters_candidates_from_train_only():
    books = clean_books(
        pd.DataFrame(
            [
                {"ISBN": "A", "Book-Title": "Alpha", "Book-Author": "Author", "Publisher": "Pub"},
                {"ISBN": "B", "Book-Title": "Beta", "Book-Author": "Author", "Publisher": "Pub"},
                {"ISBN": "D", "Book-Title": "Delta", "Book-Author": "Author", "Publisher": "Pub"},
            ]
        )
    )
    ratings = clean_ratings(
        pd.DataFrame(
            [
                {"User-ID": 1, "ISBN": "A", "Book-Rating": 7},
                {"User-ID": 1, "ISBN": "B", "Book-Rating": 6},
                {"User-ID": 1, "ISBN": "D", "Book-Rating": 10},
            ]
        )
    )
    users = pd.DataFrame({"user_id": [1], "location": ["x"], "age": [30]})

    evaluation_data = prepare_leave_one_out_evaluation(
        BookDataset(books=books, ratings=ratings, users=users),
        sample_users=1,
        min_interactions=3,
        min_relevant_rating=10,
        min_user_ratings=1,
        min_book_ratings=1,
        max_books=None,
        random_state=42,
    )

    assert evaluation_data.holdout_ratings.iloc[0]["isbn"] == "D"
    assert "D" not in set(evaluation_data.train_dataset.books["isbn"])
    assert "D" not in set(evaluation_data.train_dataset.ratings["isbn"])
    assert evaluation_data.dropped_holdouts == 1
