from __future__ import annotations

from collections.abc import Iterable


def precision_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    recommended_k = list(recommended)[:k]
    if not recommended_k:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(recommended_k)


def recall_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    recommended_k = list(recommended)[:k]
    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / len(relevant_set)
