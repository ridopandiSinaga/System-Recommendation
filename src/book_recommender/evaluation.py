from __future__ import annotations

from collections.abc import Iterable
import math

import pandas as pd


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
        f"ndcg_at_{k}": ndcg_at_k(recommended, relevant, k),
    }


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
