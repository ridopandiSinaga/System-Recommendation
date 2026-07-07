from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from book_recommender.data import BookDataset, prepare_interaction_dataset


@dataclass(frozen=True)
class LeaveOneOutEvaluationDataset:
    train_dataset: BookDataset
    train_ratings_before_filter: pd.DataFrame
    holdout_ratings: pd.DataFrame
    evaluable_holdout_ratings: pd.DataFrame
    dropped_holdouts: int


def precision_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    recommended_k = _unique_top_k(recommended, k)
    if not recommended_k:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    recommended_k = _unique_top_k(recommended, k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(relevant_set)


def hit_rate_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    recommended_k = _unique_top_k(recommended, k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return float(any(item in relevant_set for item in recommended_k))


def average_precision_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    recommended_k = _unique_top_k(recommended, k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(recommended_k, start=1):
        if item in relevant_set:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant_set), k)


def reciprocal_rank_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    recommended_k = _unique_top_k(recommended, k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    for rank, item in enumerate(recommended_k, start=1):
        if item in relevant_set:
            return 1 / rank
    return 0.0


def ndcg_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    recommended_k = _unique_top_k(recommended, k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recommended_k, start=1):
        if item in relevant_set:
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_set), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def ranking_metrics_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> dict[str, float]:
    return {
        f"precision_at_{k}": precision_at_k(recommended, relevant, k),
        f"recall_at_{k}": recall_at_k(recommended, relevant, k),
        f"hit_rate_at_{k}": hit_rate_at_k(recommended, relevant, k),
        f"map_at_{k}": average_precision_at_k(recommended, relevant, k),
        f"mrr_at_{k}": reciprocal_rank_at_k(recommended, relevant, k),
        f"ndcg_at_{k}": ndcg_at_k(recommended, relevant, k),
    }


def catalog_coverage_at_k(
    recommendation_lists: Iterable[Iterable[str]],
    catalog_items: Iterable[str],
    k: int,
) -> float:
    catalog_set = {str(item) for item in catalog_items}
    if not catalog_set:
        return 0.0

    recommended_items = set()
    for recommendations in recommendation_lists:
        recommended_items.update(str(item) for item in _unique_top_k(recommendations, k))
    return len(recommended_items.intersection(catalog_set)) / len(catalog_set)


def novelty_at_k(
    recommended: Iterable[str],
    popularity_counts: pd.Series,
    total_interactions: int,
    k: int,
) -> float:
    recommended_k = _unique_top_k(recommended, k)
    if not recommended_k or total_interactions <= 0:
        return 0.0

    item_count = max(int(len(popularity_counts)), 1)
    denominator = total_interactions + item_count
    values = []
    for item in recommended_k:
        count = int(popularity_counts.get(item, 0))
        probability = (count + 1) / denominator
        values.append(-math.log2(probability))
    return float(np.mean(values)) if values else 0.0


def mean_novelty_at_k(
    recommendation_lists: Iterable[Iterable[str]],
    popularity_counts: pd.Series,
    total_interactions: int,
    k: int,
) -> float:
    values = [
        novelty_at_k(recommendations, popularity_counts, total_interactions, k)
        for recommendations in recommendation_lists
    ]
    return float(np.mean(values)) if values else 0.0


def popularity_share_at_k(
    recommendation_lists: Iterable[Iterable[str]],
    popularity_counts: pd.Series,
    top_percent: float,
    k: int,
) -> float:
    if popularity_counts.empty:
        return 0.0
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be greater than 0 and less than or equal to 1.")

    top_n = max(1, math.ceil(len(popularity_counts) * top_percent))
    top_items = set(popularity_counts.sort_values(ascending=False).head(top_n).index)
    total = 0
    hits = 0
    for recommendations in recommendation_lists:
        for item in _unique_top_k(recommendations, k):
            total += 1
            if item in top_items:
                hits += 1
    if total == 0:
        return 0.0
    return hits / total


def intra_list_diversity_at_k(
    recommended: Iterable[str],
    item_features,
    item_to_index: dict[str, int],
    k: int,
) -> float:
    indices = [item_to_index[str(item)] for item in _unique_top_k(recommended, k) if str(item) in item_to_index]
    if len(indices) < 2:
        return 0.0

    similarities = cosine_similarity(item_features[indices])
    upper_triangle = similarities[np.triu_indices(len(indices), k=1)]
    distances = 1.0 - upper_triangle
    return float(np.mean(distances)) if len(distances) else 0.0


def mean_intra_list_diversity_at_k(
    recommendation_lists: Iterable[Iterable[str]],
    item_features,
    item_to_index: dict[str, int],
    k: int,
) -> float:
    values = [
        intra_list_diversity_at_k(recommendations, item_features, item_to_index, k)
        for recommendations in recommendation_lists
    ]
    return float(np.mean(values)) if values else 0.0


def recommendation_diagnostics_at_k(
    recommendation_lists: list[list[str]],
    catalog_items: Iterable[str],
    popularity_counts: pd.Series,
    k: int,
    item_features=None,
    item_to_index: dict[str, int] | None = None,
) -> dict[str, float]:
    total_interactions = int(popularity_counts.sum())
    diagnostics = {
        f"catalog_coverage_at_{k}": catalog_coverage_at_k(recommendation_lists, catalog_items, k),
        f"novelty_at_{k}": mean_novelty_at_k(recommendation_lists, popularity_counts, total_interactions, k),
        f"top_1pct_popularity_share_at_{k}": popularity_share_at_k(
            recommendation_lists,
            popularity_counts,
            0.01,
            k,
        ),
        f"top_5pct_popularity_share_at_{k}": popularity_share_at_k(
            recommendation_lists,
            popularity_counts,
            0.05,
            k,
        ),
        f"top_10pct_popularity_share_at_{k}": popularity_share_at_k(
            recommendation_lists,
            popularity_counts,
            0.10,
            k,
        ),
    }
    if item_features is not None and item_to_index is not None:
        diagnostics[f"intra_list_diversity_at_{k}"] = mean_intra_list_diversity_at_k(
            recommendation_lists,
            item_features,
            item_to_index,
            k,
        )
    return diagnostics


def build_leave_one_out_split(
    ratings: pd.DataFrame,
    sample_users: int = 100,
    min_interactions: int = 3,
    min_relevant_rating: float = 8.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"user_id", "isbn", "book_rating"}
    missing_columns = required_columns.difference(ratings.columns)
    if missing_columns:
        raise ValueError(f"Ratings data is missing columns: {', '.join(sorted(missing_columns))}")
    if sample_users <= 0:
        raise ValueError("sample_users must be positive.")
    if min_interactions <= 1:
        raise ValueError("min_interactions must be greater than 1.")

    indexed = ratings.reset_index().rename(columns={"index": "_row_id"})
    user_counts = indexed["user_id"].value_counts()
    eligible_users = user_counts[user_counts >= min_interactions].index
    candidates = indexed[
        indexed["user_id"].isin(eligible_users)
        & (indexed["book_rating"].astype(float) >= min_relevant_rating)
    ]
    if candidates.empty:
        raise ValueError("No eligible holdout ratings found for the requested evaluation settings.")

    users = candidates["user_id"].drop_duplicates()
    users = users.sample(n=min(sample_users, len(users)), random_state=random_state)

    holdout_rows = []
    for offset, user_id in enumerate(users):
        user_candidates = candidates[candidates["user_id"] == user_id]
        holdout_rows.append(user_candidates.sample(n=1, random_state=random_state + offset))

    holdout = pd.concat(holdout_rows, ignore_index=True)
    train = indexed[~indexed["_row_id"].isin(holdout["_row_id"])]

    return (
        train.drop(columns=["_row_id"]).reset_index(drop=True),
        holdout.drop(columns=["_row_id"]).reset_index(drop=True),
    )


def prepare_leave_one_out_evaluation(
    dataset: BookDataset,
    sample_users: int = 100,
    min_interactions: int = 3,
    min_relevant_rating: float = 8.0,
    min_user_ratings: int = 1,
    min_book_ratings: int = 1,
    max_books: int | None = None,
    max_ratings: int | None = None,
    random_state: int = 42,
) -> LeaveOneOutEvaluationDataset:
    train_ratings, holdout_ratings = build_leave_one_out_split(
        dataset.ratings,
        sample_users=sample_users,
        min_interactions=min_interactions,
        min_relevant_rating=min_relevant_rating,
        random_state=random_state,
    )
    train_dataset = prepare_interaction_dataset(
        books=dataset.books,
        ratings=train_ratings,
        users=dataset.users,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
        max_books=max_books,
        max_ratings=max_ratings,
        random_state=random_state,
    )

    candidate_isbns = set(train_dataset.books["isbn"].astype(str))
    users_with_history = set(train_dataset.ratings["user_id"].astype(int))
    evaluable_holdout_ratings = holdout_ratings[
        holdout_ratings["isbn"].astype(str).isin(candidate_isbns)
        & holdout_ratings["user_id"].astype(int).isin(users_with_history)
    ].reset_index(drop=True)

    return LeaveOneOutEvaluationDataset(
        train_dataset=train_dataset,
        train_ratings_before_filter=train_ratings.reset_index(drop=True),
        holdout_ratings=holdout_ratings.reset_index(drop=True),
        evaluable_holdout_ratings=evaluable_holdout_ratings,
        dropped_holdouts=int(len(holdout_ratings) - len(evaluable_holdout_ratings)),
    )


def _unique_top_k(items: Iterable[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive.")

    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= k:
            break
    return result
