from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from book_recommender.content_based import book_work_key
from book_recommender.data import BookDataset, prepare_interaction_dataset, select_content_catalog


@dataclass(frozen=True)
class HoldoutAvailability:
    """Availability counts for sampled holdouts after train-only filtering."""

    sampled_holdouts: int
    candidate_available_holdouts: int
    user_history_available_holdouts: int
    evaluable_holdouts: int

    def __post_init__(self) -> None:
        counts = {
            "sampled_holdouts": self.sampled_holdouts,
            "candidate_available_holdouts": self.candidate_available_holdouts,
            "user_history_available_holdouts": self.user_history_available_holdouts,
            "evaluable_holdouts": self.evaluable_holdouts,
        }
        for name, value in counts.items():
            _validate_non_negative_integer(value, name)
            if name != "sampled_holdouts" and value > self.sampled_holdouts:
                raise ValueError(f"{name} cannot exceed sampled_holdouts.")
        if self.evaluable_holdouts > self.candidate_available_holdouts:
            raise ValueError("evaluable_holdouts cannot exceed candidate_available_holdouts.")
        if self.evaluable_holdouts > self.user_history_available_holdouts:
            raise ValueError("evaluable_holdouts cannot exceed user_history_available_holdouts.")

    @property
    def dropped_holdouts(self) -> int:
        return self.sampled_holdouts - self.evaluable_holdouts

    @property
    def candidate_retention_rate(self) -> float:
        return _safe_ratio(self.candidate_available_holdouts, self.sampled_holdouts)

    @property
    def user_history_retention_rate(self) -> float:
        return _safe_ratio(self.user_history_available_holdouts, self.sampled_holdouts)

    @property
    def evaluation_retention_rate(self) -> float:
        return _safe_ratio(self.evaluable_holdouts, self.sampled_holdouts)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "sampled_holdouts": self.sampled_holdouts,
            "candidate_available_holdouts": self.candidate_available_holdouts,
            "user_history_available_holdouts": self.user_history_available_holdouts,
            "evaluable_holdouts": self.evaluable_holdouts,
            "dropped_holdouts": self.dropped_holdouts,
            "candidate_retention_rate": self.candidate_retention_rate,
            "user_history_retention_rate": self.user_history_retention_rate,
            "evaluation_retention_rate": self.evaluation_retention_rate,
        }


@dataclass(frozen=True)
class RankingMetricsSummary:
    """Conditional and all-holdout ranking metrics for one evaluation run."""

    k: int
    sampled_holdouts: int
    evaluable_holdouts: int
    conditional_metrics: dict[str, float]
    unconditional_metrics: dict[str, float]

    @property
    def dropped_holdouts(self) -> int:
        return self.sampled_holdouts - self.evaluable_holdouts

    @property
    def evaluation_retention_rate(self) -> float:
        return _safe_ratio(self.evaluable_holdouts, self.sampled_holdouts)

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "sampled_holdouts": self.sampled_holdouts,
            "evaluable_holdouts": self.evaluable_holdouts,
            "dropped_holdouts": self.dropped_holdouts,
            "evaluation_retention_rate": self.evaluation_retention_rate,
            "conditional_metrics": dict(self.conditional_metrics),
            "unconditional_metrics": dict(self.unconditional_metrics),
        }


@dataclass(frozen=True)
class LeaveOneOutValidationTestSplit:
    """Disjoint validation/test holdouts plus the histories used for model training."""

    train_ratings: pd.DataFrame
    validation_holdout_ratings: pd.DataFrame
    test_holdout_ratings: pd.DataFrame

    @property
    def validation_users(self) -> int:
        return int(self.validation_holdout_ratings["user_id"].nunique())

    @property
    def test_users(self) -> int:
        return int(self.test_holdout_ratings["user_id"].nunique())

    @property
    def sampled_users(self) -> int:
        return self.validation_users + self.test_users


@dataclass(frozen=True)
class ValidationTestEvaluationDataset:
    """Prepared train catalog plus disjoint validation and test evaluation partitions."""

    train_dataset: BookDataset
    train_ratings_before_filter: pd.DataFrame
    validation_holdout_ratings: pd.DataFrame
    validation_evaluable_holdout_ratings: pd.DataFrame
    test_holdout_ratings: pd.DataFrame
    test_evaluable_holdout_ratings: pd.DataFrame

    @property
    def validation_availability(self) -> HoldoutAvailability:
        return summarize_holdout_availability(
            self.validation_holdout_ratings,
            candidate_items=_catalog_work_keys(self.train_dataset.books),
            history_users=self.train_dataset.ratings["user_id"],
            item_column="work_key",
        )

    @property
    def test_availability(self) -> HoldoutAvailability:
        return summarize_holdout_availability(
            self.test_holdout_ratings,
            candidate_items=_catalog_work_keys(self.train_dataset.books),
            history_users=self.train_dataset.ratings["user_id"],
            item_column="work_key",
        )


@dataclass(frozen=True)
class FixedTestWorkLevelSplit:
    """A fixed external test target set and the development ratings left after hiding it.

    The test users remain in ``development_ratings`` with their held-out canonical work
    removed. This preserves their pre-holdout history for the final personalized test
    while callers can exclude those users from repeated validation sampling.
    """

    development_ratings: pd.DataFrame
    test_holdout_ratings: pd.DataFrame

    @property
    def test_users(self) -> set[object]:
        return set(self.test_holdout_ratings["user_id"].tolist())


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    """Percentile bootstrap interval for an arithmetic mean."""

    mean: float
    lower: float
    upper: float
    confidence_level: float
    sample_size: int
    n_resamples: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "sample_size": self.sample_size,
            "n_resamples": self.n_resamples,
        }


@dataclass(frozen=True)
class LeaveOneOutEvaluationDataset:
    train_dataset: BookDataset
    train_ratings_before_filter: pd.DataFrame
    holdout_ratings: pd.DataFrame
    evaluable_holdout_ratings: pd.DataFrame
    dropped_holdouts: int

    @property
    def availability(self) -> HoldoutAvailability:
        return summarize_holdout_availability(
            self.holdout_ratings,
            candidate_items=self.train_dataset.books["isbn"],
            history_users=self.train_dataset.ratings["user_id"],
        )

    @property
    def sampled_holdouts(self) -> int:
        return int(len(self.holdout_ratings))

    @property
    def evaluable_holdouts(self) -> int:
        return int(len(self.evaluable_holdout_ratings))

    @property
    def candidate_retention_rate(self) -> float:
        return self.availability.candidate_retention_rate

    @property
    def evaluation_retention_rate(self) -> float:
        return self.availability.evaluation_retention_rate


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


def mean_ranking_metrics_at_k(
    metric_rows: Iterable[Mapping[str, float]] | pd.DataFrame,
    *,
    k: int,
) -> dict[str, float]:
    """Return the arithmetic mean for rows that explicitly define the evaluation scope."""

    _validate_k(k)
    rows = _coerce_ranking_metric_rows(metric_rows, k)
    metric_names = _ranking_metric_names(k)
    if not rows:
        return {metric: 0.0 for metric in metric_names}
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in metric_names
    }


def summarize_ranking_metrics_at_k(
    metric_rows: Iterable[Mapping[str, float]] | pd.DataFrame,
    *,
    k: int,
    sampled_holdouts: int,
) -> RankingMetricsSummary:
    """Aggregate evaluable rows and count unavailable holdouts as all-zero rows.

    ``conditional_metrics`` describes only rows for which the held-out item and user
    history survived train-only filtering. ``unconditional_metrics`` uses every sampled
    holdout as the denominator, so each unavailable holdout is an explicit miss.
    """

    _validate_k(k)
    _validate_positive_integer(sampled_holdouts, "sampled_holdouts")
    rows = _coerce_ranking_metric_rows(metric_rows, k)
    if len(rows) > sampled_holdouts:
        raise ValueError("The number of metric rows cannot exceed sampled_holdouts.")

    metric_names = _ranking_metric_names(k)
    conditional_metrics = {}
    unconditional_metrics = {}
    for metric_name in metric_names:
        values = np.asarray([row[metric_name] for row in rows], dtype=float)
        conditional_metrics[metric_name] = float(values.mean()) if len(values) else 0.0
        unconditional_metrics[metric_name] = float(values.sum() / sampled_holdouts)

    return RankingMetricsSummary(
        k=int(k),
        sampled_holdouts=int(sampled_holdouts),
        evaluable_holdouts=len(rows),
        conditional_metrics=conditional_metrics,
        unconditional_metrics=unconditional_metrics,
    )


def bootstrap_mean_confidence_interval(
    values: Iterable[float],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 2_000,
    random_state: int = 42,
) -> BootstrapConfidenceInterval:
    """Return a deterministic percentile-bootstrap confidence interval for a mean."""

    if isinstance(confidence_level, bool) or not isinstance(
        confidence_level, (int, float, np.integer, np.floating)
    ):
        raise ValueError("confidence_level must be a number between 0 and 1.")
    confidence_level = float(confidence_level)
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be greater than 0 and less than 1.")
    _validate_positive_integer(n_resamples, "n_resamples")
    _validate_random_state(random_state)

    try:
        observed = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("values must contain only numeric values.") from error
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("values must be a non-empty one-dimensional collection.")
    if not np.isfinite(observed).all():
        raise ValueError("values must contain only finite values.")

    rng = np.random.default_rng(int(random_state))
    bootstrap_means = np.empty(int(n_resamples), dtype=float)
    # Keep temporary index arrays bounded for large evaluation samples.
    batch_size = max(1, min(int(n_resamples), 1_000_000 // len(observed)))
    for start in range(0, int(n_resamples), batch_size):
        stop = min(start + batch_size, int(n_resamples))
        indices = rng.integers(0, len(observed), size=(stop - start, len(observed)))
        bootstrap_means[start:stop] = observed[indices].mean(axis=1)

    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return BootstrapConfidenceInterval(
        mean=float(observed.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        sample_size=int(len(observed)),
        n_resamples=int(n_resamples),
    )


def bootstrap_ranking_metrics_at_k(
    metric_rows: Iterable[Mapping[str, float]] | pd.DataFrame,
    *,
    k: int,
    sampled_holdouts: int | None = None,
    confidence_level: float = 0.95,
    n_resamples: int = 2_000,
    random_state: int = 42,
) -> dict[str, BootstrapConfidenceInterval]:
    """Bootstrap ranking means, optionally padding unavailable holdouts with zeros.

    When ``sampled_holdouts`` is omitted, intervals are conditional on evaluability.
    Supplying it produces unconditional intervals over all sampled holdouts.
    """

    _validate_k(k)
    rows = _coerce_ranking_metric_rows(metric_rows, k)
    if sampled_holdouts is None:
        if not rows:
            raise ValueError("metric_rows cannot be empty when sampled_holdouts is omitted.")
        sample_size = len(rows)
    else:
        _validate_positive_integer(sampled_holdouts, "sampled_holdouts")
        sample_size = int(sampled_holdouts)
        if len(rows) > sample_size:
            raise ValueError("The number of metric rows cannot exceed sampled_holdouts.")

    intervals = {}
    for metric_name in _ranking_metric_names(k):
        values = [row[metric_name] for row in rows]
        values.extend([0.0] * (sample_size - len(values)))
        intervals[metric_name] = bootstrap_mean_confidence_interval(
            values,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )
    return intervals


def bootstrap_paired_ranking_deltas_at_k(
    baseline_rows: Iterable[Mapping[str, object]] | pd.DataFrame,
    challenger_rows: Iterable[Mapping[str, object]] | pd.DataFrame,
    *,
    k: int,
    scope: str = "end_to_end",
    pair_columns: tuple[str, ...] = ("split_id", "user_id", "holdout_work_key"),
    cluster_column: str | None = "split_id",
    require_matching_status: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 2_000,
    random_state: int = 42,
) -> dict[str, BootstrapConfidenceInterval]:
    """Estimate paired challenger-minus-baseline ranking deltas.

    By default, paired rows must have identical availability and evaluation statuses. This
    prevents a weight-only comparison from mistaking different holdout support for a model
    improvement. Set ``require_matching_status=False`` when availability is itself allowed
    to change, such as when comparing catalog configurations. In that mode, ``end_to_end``
    keeps every paired holdout while ``conditional`` uses only the common rankable support
    of the baseline and challenger.

    When ``cluster_column`` is supplied, bootstrap samples first draw repeated split IDs and
    then user-work pairs within every selected split. The point estimate is therefore a macro
    average over splits rather than a pooled row average.
    """

    _validate_k(k)
    _validate_fraction(confidence_level, "confidence_level")
    _validate_positive_integer(n_resamples, "n_resamples")
    _validate_random_state(random_state)
    if scope not in {"end_to_end", "conditional"}:
        raise ValueError("scope must be either 'end_to_end' or 'conditional'.")
    if not pair_columns or len(set(pair_columns)) != len(pair_columns):
        raise ValueError("pair_columns must contain one or more unique column names.")
    if cluster_column is not None and not isinstance(cluster_column, str):
        raise ValueError("cluster_column must be a string or None.")
    if not isinstance(require_matching_status, (bool, np.bool_)):
        raise ValueError("require_matching_status must be a boolean.")

    metric_names = _ranking_metric_names(k)
    status_columns = (
        "candidate_available",
        "history_available",
        "positive_seed_available",
        "evaluation_status",
    )
    baseline = _coerce_paired_ranking_rows(
        baseline_rows,
        name="baseline_rows",
        metric_names=metric_names,
        required_columns=(*pair_columns, *status_columns),
    )
    challenger = _coerce_paired_ranking_rows(
        challenger_rows,
        name="challenger_rows",
        metric_names=metric_names,
        required_columns=(*pair_columns, *status_columns),
    )
    if cluster_column is not None:
        for name, frame in (("baseline_rows", baseline), ("challenger_rows", challenger)):
            if cluster_column not in frame.columns:
                raise ValueError(f"{name} is missing cluster column {cluster_column!r}.")
            if frame[cluster_column].isna().any():
                raise ValueError(f"{name} contains a missing cluster value.")

    _validate_unique_pair_keys(baseline, pair_columns, "baseline_rows")
    _validate_unique_pair_keys(challenger, pair_columns, "challenger_rows")
    try:
        paired = baseline.merge(
            challenger,
            on=list(pair_columns),
            how="outer",
            suffixes=("_baseline", "_challenger"),
            indicator=True,
            validate="one_to_one",
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Paired ranking rows cannot be aligned by pair_columns.") from error
    if not paired["_merge"].eq("both").all():
        raise ValueError("baseline_rows and challenger_rows must contain identical pair keys.")

    paired_cluster_column = cluster_column
    if cluster_column is not None and cluster_column not in pair_columns:
        baseline_cluster = f"{cluster_column}_baseline"
        challenger_cluster = f"{cluster_column}_challenger"
        if not paired[baseline_cluster].eq(paired[challenger_cluster]).all():
            raise ValueError(f"Paired rows disagree on {cluster_column!r}.")
        paired_cluster_column = baseline_cluster

    if require_matching_status:
        for column in status_columns:
            baseline_column = f"{column}_baseline"
            challenger_column = f"{column}_challenger"
            if not paired[baseline_column].eq(paired[challenger_column]).all():
                raise ValueError(f"Paired rows disagree on {column!r}.")
    for column in status_columns[:3]:
        for side in ("baseline", "challenger"):
            if not _series_contains_only_booleans(paired[f"{column}_{side}"]):
                raise ValueError(f"Paired rows must contain boolean {column!r} values.")

    sort_columns = list(pair_columns)
    if paired_cluster_column is not None and paired_cluster_column not in sort_columns:
        sort_columns.insert(0, paired_cluster_column)
    paired = paired.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if scope == "conditional":
        conditional_mask = (
            paired["candidate_available_baseline"]
            & paired["positive_seed_available_baseline"]
        )
        if not require_matching_status:
            conditional_mask &= (
                paired["candidate_available_challenger"]
                & paired["positive_seed_available_challenger"]
            )
        paired = paired.loc[conditional_mask].reset_index(drop=True)
    if paired.empty:
        raise ValueError(f"No paired rows are available for the {scope!r} scope.")

    intervals = {}
    for metric_name in metric_names:
        deltas = (
            paired[f"{metric_name}_challenger"].to_numpy(dtype=float)
            - paired[f"{metric_name}_baseline"].to_numpy(dtype=float)
        )
        if cluster_column is None:
            intervals[metric_name] = bootstrap_mean_confidence_interval(
                deltas,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                random_state=random_state,
            )
            continue

        intervals[metric_name] = _hierarchical_bootstrap_mean_confidence_interval(
            deltas,
            paired[paired_cluster_column].tolist(),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )
    return intervals


def catalog_coverage_at_k(
    recommendation_lists: Iterable[Iterable[str]],
    catalog_items: Iterable[str],
    k: int,
) -> float:
    _validate_k(k)
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
    _validate_k(k)
    _validate_non_negative_integer(total_interactions, "total_interactions")
    recommended_k = _unique_top_k(recommended, k)
    if not recommended_k or total_interactions == 0:
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
    _validate_k(k)
    _validate_non_negative_integer(total_interactions, "total_interactions")
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
    _validate_k(k)
    if isinstance(top_percent, bool) or not isinstance(
        top_percent, (int, float, np.integer, np.floating)
    ):
        raise ValueError("top_percent must be a number greater than 0 and at most 1.")
    top_percent = float(top_percent)
    if not math.isfinite(top_percent) or not 0 < top_percent <= 1:
        raise ValueError("top_percent must be greater than 0 and less than or equal to 1.")
    if popularity_counts.empty:
        return 0.0

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
    _validate_k(k)
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
    _validate_k(k)
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
    _validate_k(k)
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


def summarize_holdout_availability(
    holdout_ratings: pd.DataFrame,
    *,
    candidate_items: Iterable[str],
    history_users: Iterable[object],
    item_column: str = "isbn",
) -> HoldoutAvailability:
    """Count why sampled holdouts can or cannot be used for ranking evaluation."""

    required_columns = {"user_id", item_column}
    missing_columns = required_columns.difference(holdout_ratings.columns)
    if missing_columns:
        raise ValueError(
            f"Holdout data is missing columns: {', '.join(sorted(missing_columns))}"
        )

    candidate_set = {str(item) for item in candidate_items if pd.notna(item)}
    history_user_set = {str(user_id) for user_id in history_users if pd.notna(user_id)}
    candidate_mask = holdout_ratings[item_column].notna() & holdout_ratings[
        item_column
    ].astype(str).isin(candidate_set)
    history_mask = holdout_ratings["user_id"].notna() & holdout_ratings[
        "user_id"
    ].astype(str).isin(history_user_set)
    evaluable_mask = candidate_mask & history_mask

    return HoldoutAvailability(
        sampled_holdouts=int(len(holdout_ratings)),
        candidate_available_holdouts=int(candidate_mask.sum()),
        user_history_available_holdouts=int(history_mask.sum()),
        evaluable_holdouts=int(evaluable_mask.sum()),
    )


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
    _validate_positive_integer(sample_users, "sample_users")
    _validate_positive_integer(min_interactions, "min_interactions")
    if min_interactions <= 1:
        raise ValueError("min_interactions must be greater than 1.")
    _validate_finite_number(min_relevant_rating, "min_relevant_rating")
    _validate_random_state(random_state)

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
    users = users.sample(n=min(int(sample_users), len(users)), random_state=int(random_state))

    holdout_rows = []
    for offset, user_id in enumerate(users):
        user_candidates = candidates[candidates["user_id"] == user_id]
        per_user_seed = (int(random_state) + offset) % (2**32)
        holdout_rows.append(user_candidates.sample(n=1, random_state=per_user_seed))

    holdout = pd.concat(holdout_rows, ignore_index=True)
    train = indexed[~indexed["_row_id"].isin(holdout["_row_id"])]

    return (
        train.drop(columns=["_row_id"]).reset_index(drop=True),
        holdout.drop(columns=["_row_id"]).reset_index(drop=True),
    )


def build_validation_test_leave_one_out_split(
    ratings: pd.DataFrame,
    sample_users: int = 100,
    validation_fraction: float = 0.5,
    min_interactions: int = 3,
    min_relevant_rating: float = 8.0,
    random_state: int = 42,
) -> LeaveOneOutValidationTestSplit:
    """Create disjoint user-level validation and untouched test holdouts.

    One relevant interaction is removed for every sampled user. Sampled users are then
    deterministically partitioned between validation and test, while all remaining user
    histories stay in ``train_ratings``. Model choices can therefore be made solely on
    validation users and reported once on a disjoint set of test users.
    """

    _validate_positive_integer(sample_users, "sample_users")
    if sample_users < 2:
        raise ValueError("sample_users must be at least 2 for validation/test partitioning.")
    _validate_fraction(validation_fraction, "validation_fraction")
    _validate_random_state(random_state)

    train_ratings, holdout_ratings = build_leave_one_out_split(
        ratings,
        sample_users=sample_users,
        min_interactions=min_interactions,
        min_relevant_rating=min_relevant_rating,
        random_state=random_state,
    )
    if len(holdout_ratings) < 2:
        raise ValueError(
            "At least two eligible users are required for validation/test partitioning."
        )

    rng = np.random.default_rng(int(random_state))
    order = rng.permutation(len(holdout_ratings))
    validation_size = int(math.floor(len(holdout_ratings) * float(validation_fraction) + 0.5))
    validation_size = max(1, min(validation_size, len(holdout_ratings) - 1))
    validation_indices = order[:validation_size]
    test_indices = order[validation_size:]

    return LeaveOneOutValidationTestSplit(
        train_ratings=train_ratings.reset_index(drop=True),
        validation_holdout_ratings=holdout_ratings.iloc[validation_indices].reset_index(drop=True),
        test_holdout_ratings=holdout_ratings.iloc[test_indices].reset_index(drop=True),
    )


def build_work_level_holdouts(
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    *,
    sample_users: int = 100,
    min_interactions: int = 3,
    min_relevant_rating: float = 8.0,
    random_state: int = 42,
    excluded_user_ids: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Sample one relevant canonical work per eligible user without changing ratings.

    ``excluded_user_ids`` makes it possible to reserve an outer test cohort while
    repeatedly drawing validation targets from the remaining development users.
    """

    _validate_positive_integer(sample_users, "sample_users")
    _validate_positive_integer(min_interactions, "min_interactions")
    if min_interactions <= 1:
        raise ValueError("min_interactions must be greater than 1.")
    _validate_finite_number(min_relevant_rating, "min_relevant_rating")
    _validate_random_state(random_state)
    _validate_work_level_split_inputs(ratings, books)

    work_map = _book_work_map(books)
    enriched = ratings.copy()
    enriched["isbn"] = enriched["isbn"].astype(str)
    enriched["work_key"] = enriched["isbn"].map(work_map)
    if enriched["work_key"].isna().any():
        raise ValueError("Every rating ISBN must have book metadata for work-level splitting.")

    excluded_users = set(excluded_user_ids) if excluded_user_ids is not None else set()
    work_counts = enriched.groupby("user_id")["work_key"].nunique()
    eligible_users = set(work_counts[work_counts >= min_interactions].index).difference(
        excluded_users
    )
    relevant = enriched[
        enriched["user_id"].isin(eligible_users)
        & (enriched["book_rating"].astype(float) >= min_relevant_rating)
    ]
    candidate_works = relevant[["user_id", "work_key"]].drop_duplicates()
    if candidate_works.empty:
        raise ValueError("No eligible work-level holdouts found for the requested settings.")

    candidate_users = candidate_works["user_id"].drop_duplicates().sort_values(kind="stable")
    users = candidate_users.sample(
        n=min(int(sample_users), len(candidate_users)),
        random_state=int(random_state),
    )
    holdout_rows = []
    for offset, user_id in enumerate(users):
        user_candidates = candidate_works[
            candidate_works["user_id"] == user_id
        ].sort_values("work_key", kind="stable")
        per_user_seed = (int(random_state) + offset) % (2**32)
        target_work = str(
            user_candidates.sample(n=1, random_state=per_user_seed).iloc[0]["work_key"]
        )
        representative = (
            relevant[
                (relevant["user_id"] == user_id) & (relevant["work_key"] == target_work)
            ]
            .sort_values(["book_rating", "isbn"], ascending=[False, True], kind="stable")
            .iloc[0]
            .copy()
        )
        holdout_rows.append(representative)
    return pd.DataFrame(holdout_rows).reset_index(drop=True)


def remove_work_level_holdouts(
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    holdout_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Remove every edition of each sampled user-work target from training ratings."""

    _validate_work_level_split_inputs(ratings, books)
    required_holdout_columns = {"user_id", "work_key"}
    missing_holdout_columns = required_holdout_columns.difference(holdout_ratings.columns)
    if missing_holdout_columns:
        raise ValueError(
            "holdout_ratings is missing columns: "
            f"{', '.join(sorted(missing_holdout_columns))}"
        )
    if holdout_ratings.empty:
        return ratings.reset_index(drop=True).copy()
    if holdout_ratings[["user_id", "work_key"]].isna().any().any():
        raise ValueError("holdout_ratings contains missing user_id or work_key values.")

    work_map = _book_work_map(books)
    enriched = ratings.copy()
    enriched["isbn"] = enriched["isbn"].astype(str)
    enriched["work_key"] = enriched["isbn"].map(work_map)
    if enriched["work_key"].isna().any():
        raise ValueError("Every rating ISBN must have book metadata for work-level splitting.")
    target_pairs = {
        (user_id, str(work_key))
        for user_id, work_key in holdout_ratings[["user_id", "work_key"]].itertuples(
            index=False,
            name=None,
        )
    }
    pair_index = pd.MultiIndex.from_arrays(
        [enriched["user_id"], enriched["work_key"]],
        names=["user_id", "work_key"],
    )
    target_index = pd.MultiIndex.from_tuples(
        sorted(target_pairs, key=lambda pair: (str(pair[0]), pair[1])),
        names=["user_id", "work_key"],
    )
    return enriched.loc[~pair_index.isin(target_index)].drop(columns="work_key").reset_index(
        drop=True
    )


def build_fixed_test_work_level_split(
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    *,
    test_users: int = 100,
    min_interactions: int = 3,
    min_relevant_rating: float = 8.0,
    random_state: int = 42,
) -> FixedTestWorkLevelSplit:
    """Create a fixed outer test cohort before any repeated validation tuning."""

    test_holdouts = build_work_level_holdouts(
        ratings,
        books,
        sample_users=test_users,
        min_interactions=min_interactions,
        min_relevant_rating=min_relevant_rating,
        random_state=random_state,
    )
    return FixedTestWorkLevelSplit(
        development_ratings=remove_work_level_holdouts(ratings, books, test_holdouts),
        test_holdout_ratings=test_holdouts,
    )


def prepare_work_level_evaluation_train_dataset(
    dataset: BookDataset,
    train_ratings: pd.DataFrame,
    *,
    min_user_ratings: int = 1,
    min_book_ratings: int = 1,
    max_books: int | None = None,
    max_ratings: int | None = None,
    max_content_books: int | None = None,
    random_state: int = 42,
) -> BookDataset:
    """Apply train-only interaction and content-catalog filtering for work holdouts."""

    interaction_dataset = prepare_interaction_dataset(
        books=dataset.books,
        ratings=train_ratings,
        users=dataset.users,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
        max_books=max_books,
        max_ratings=max_ratings,
        random_state=random_state,
    )
    return BookDataset(
        books=select_content_catalog(
            dataset.books,
            required_books=interaction_dataset.books,
            max_content_books=max_content_books,
        ),
        ratings=interaction_dataset.ratings,
        users=interaction_dataset.users,
    )


def build_validation_test_work_level_split(
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    sample_users: int = 100,
    validation_fraction: float = 0.5,
    min_interactions: int = 3,
    min_relevant_rating: float = 8.0,
    random_state: int = 42,
) -> LeaveOneOutValidationTestSplit:
    """Hold out a complete canonical work and partition users into validation/test.

    A user may have rated multiple ISBN editions of one work. All such editions are
    removed from that user's training history, and the holdout carries ``work_key`` so
    evaluation can treat any recommended edition of that work as the same relevant item.
    """

    _validate_positive_integer(sample_users, "sample_users")
    if sample_users < 2:
        raise ValueError("sample_users must be at least 2 for validation/test partitioning.")
    _validate_positive_integer(min_interactions, "min_interactions")
    if min_interactions <= 1:
        raise ValueError("min_interactions must be greater than 1.")
    _validate_fraction(validation_fraction, "validation_fraction")
    _validate_finite_number(min_relevant_rating, "min_relevant_rating")
    _validate_random_state(random_state)
    required_ratings = {"user_id", "isbn", "book_rating"}
    required_books = {"isbn", "book_title", "book_author"}
    missing_ratings = required_ratings.difference(ratings.columns)
    missing_books = required_books.difference(books.columns)
    if missing_ratings:
        raise ValueError(
            f"Ratings data is missing columns: {', '.join(sorted(missing_ratings))}"
        )
    if missing_books:
        raise ValueError(f"Books data is missing columns: {', '.join(sorted(missing_books))}")

    work_map = _book_work_map(books)
    enriched = ratings.copy()
    enriched["isbn"] = enriched["isbn"].astype(str)
    enriched["work_key"] = enriched["isbn"].map(work_map)
    if enriched["work_key"].isna().any():
        raise ValueError("Every rating ISBN must have book metadata for work-level splitting.")

    work_counts = enriched.groupby("user_id")["work_key"].nunique()
    eligible_users = work_counts[work_counts >= min_interactions].index
    relevant = enriched[
        enriched["user_id"].isin(eligible_users)
        & (enriched["book_rating"].astype(float) >= min_relevant_rating)
    ]
    candidate_works = relevant[["user_id", "work_key"]].drop_duplicates()
    if candidate_works.empty:
        raise ValueError("No eligible work-level holdouts found for the requested settings.")

    users = candidate_works["user_id"].drop_duplicates()
    users = users.sample(n=min(int(sample_users), len(users)), random_state=int(random_state))
    holdout_rows = []
    target_pairs: set[tuple[object, str]] = set()
    for offset, user_id in enumerate(users):
        user_candidates = candidate_works[candidate_works["user_id"] == user_id]
        per_user_seed = (int(random_state) + offset) % (2**32)
        target_work = str(
            user_candidates.sample(n=1, random_state=per_user_seed).iloc[0]["work_key"]
        )
        target_pairs.add((user_id, target_work))
        representative = (
            relevant[
                (relevant["user_id"] == user_id) & (relevant["work_key"] == target_work)
            ]
            .sort_values(["book_rating", "isbn"], ascending=[False, True], kind="stable")
            .iloc[0]
            .copy()
        )
        holdout_rows.append(representative)

    pair_index = pd.MultiIndex.from_arrays(
        [enriched["user_id"], enriched["work_key"]],
        names=["user_id", "work_key"],
    )
    target_index = pd.MultiIndex.from_tuples(
        sorted(target_pairs, key=lambda pair: (str(pair[0]), pair[1])),
        names=["user_id", "work_key"],
    )
    train_ratings = enriched.loc[~pair_index.isin(target_index)].drop(columns="work_key")
    holdouts = pd.DataFrame(holdout_rows).reset_index(drop=True)
    if len(holdouts) < 2:
        raise ValueError("At least two eligible users are required for validation/test splitting.")

    rng = np.random.default_rng(int(random_state))
    order = rng.permutation(len(holdouts))
    validation_size = int(math.floor(len(holdouts) * float(validation_fraction) + 0.5))
    validation_size = max(1, min(validation_size, len(holdouts) - 1))
    return LeaveOneOutValidationTestSplit(
        train_ratings=train_ratings.reset_index(drop=True),
        validation_holdout_ratings=holdouts.iloc[order[:validation_size]].reset_index(drop=True),
        test_holdout_ratings=holdouts.iloc[order[validation_size:]].reset_index(drop=True),
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
    users_with_history = set(train_dataset.ratings["user_id"].astype(str))
    evaluable_holdout_ratings = holdout_ratings[
        holdout_ratings["isbn"].astype(str).isin(candidate_isbns)
        & holdout_ratings["user_id"].astype(str).isin(users_with_history)
    ].reset_index(drop=True)

    availability = summarize_holdout_availability(
        holdout_ratings,
        candidate_items=candidate_isbns,
        history_users=users_with_history,
    )

    return LeaveOneOutEvaluationDataset(
        train_dataset=train_dataset,
        train_ratings_before_filter=train_ratings.reset_index(drop=True),
        holdout_ratings=holdout_ratings.reset_index(drop=True),
        evaluable_holdout_ratings=evaluable_holdout_ratings,
        dropped_holdouts=availability.dropped_holdouts,
    )


def prepare_validation_test_evaluation(
    dataset: BookDataset,
    sample_users: int = 200,
    validation_fraction: float = 0.5,
    min_interactions: int = 3,
    min_relevant_rating: float = 8.0,
    min_user_ratings: int = 1,
    min_book_ratings: int = 1,
    max_books: int | None = None,
    max_ratings: int | None = None,
    max_content_books: int | None = None,
    random_state: int = 42,
) -> ValidationTestEvaluationDataset:
    """Prepare one shared train catalog with disjoint validation and test targets."""

    split = build_validation_test_work_level_split(
        dataset.ratings,
        dataset.books,
        sample_users=sample_users,
        validation_fraction=validation_fraction,
        min_interactions=min_interactions,
        min_relevant_rating=min_relevant_rating,
        random_state=random_state,
    )
    interaction_dataset = prepare_interaction_dataset(
        books=dataset.books,
        ratings=split.train_ratings,
        users=dataset.users,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
        max_books=max_books,
        max_ratings=max_ratings,
        random_state=random_state,
    )
    train_dataset = BookDataset(
        books=select_content_catalog(
            dataset.books,
            required_books=interaction_dataset.books,
            max_content_books=max_content_books,
        ),
        ratings=interaction_dataset.ratings,
        users=interaction_dataset.users,
    )
    validation_evaluable = _evaluable_holdouts(
        split.validation_holdout_ratings,
        train_dataset,
    )
    test_evaluable = _evaluable_holdouts(split.test_holdout_ratings, train_dataset)

    return ValidationTestEvaluationDataset(
        train_dataset=train_dataset,
        train_ratings_before_filter=split.train_ratings.reset_index(drop=True),
        validation_holdout_ratings=split.validation_holdout_ratings.reset_index(drop=True),
        validation_evaluable_holdout_ratings=validation_evaluable,
        test_holdout_ratings=split.test_holdout_ratings.reset_index(drop=True),
        test_evaluable_holdout_ratings=test_evaluable,
    )


def _evaluable_holdouts(holdouts: pd.DataFrame, train_dataset: BookDataset) -> pd.DataFrame:
    item_column = "work_key" if "work_key" in holdouts.columns else "isbn"
    candidate_items = (
        set(_catalog_work_keys(train_dataset.books))
        if item_column == "work_key"
        else set(train_dataset.books["isbn"].astype(str))
    )
    users_with_history = set(train_dataset.ratings["user_id"].astype(str))
    return holdouts[
        holdouts[item_column].astype(str).isin(candidate_items)
        & holdouts["user_id"].astype(str).isin(users_with_history)
    ].reset_index(drop=True)


def _validate_work_level_split_inputs(ratings: pd.DataFrame, books: pd.DataFrame) -> None:
    required_ratings = {"user_id", "isbn", "book_rating"}
    required_books = {"isbn", "book_title", "book_author"}
    missing_ratings = required_ratings.difference(ratings.columns)
    missing_books = required_books.difference(books.columns)
    if missing_ratings:
        raise ValueError(
            f"Ratings data is missing columns: {', '.join(sorted(missing_ratings))}"
        )
    if missing_books:
        raise ValueError(f"Books data is missing columns: {', '.join(sorted(missing_books))}")


def _book_work_map(books: pd.DataFrame) -> dict[str, str]:
    return {
        str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
        for row in books[["isbn", "book_title", "book_author"]].itertuples(index=False)
    }


def _catalog_work_keys(books: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(_book_work_map(books).values()))


def _unique_top_k(items: Iterable[str], k: int) -> list[str]:
    _validate_k(k)

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


def _ranking_metric_names(k: int) -> tuple[str, ...]:
    return (
        f"precision_at_{k}",
        f"recall_at_{k}",
        f"hit_rate_at_{k}",
        f"map_at_{k}",
        f"mrr_at_{k}",
        f"ndcg_at_{k}",
    )


def _coerce_ranking_metric_rows(
    metric_rows: Iterable[Mapping[str, float]] | pd.DataFrame,
    k: int,
) -> list[dict[str, float]]:
    if isinstance(metric_rows, pd.DataFrame):
        raw_rows: list[object] = metric_rows.to_dict(orient="records")
    else:
        raw_rows = list(metric_rows)

    metric_names = _ranking_metric_names(k)
    rows = []
    for row_index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"metric_rows[{row_index}] must be a mapping.")
        missing_metrics = [name for name in metric_names if name not in raw_row]
        if missing_metrics:
            raise ValueError(
                f"metric_rows[{row_index}] is missing metrics: {', '.join(missing_metrics)}"
            )

        row = {}
        for metric_name in metric_names:
            raw_value = raw_row[metric_name]
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{metric_name} must be a numeric value between 0 and 1.")
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{metric_name} must be a finite value between 0 and 1.")
            row[metric_name] = value
        rows.append(row)
    return rows


def _coerce_paired_ranking_rows(
    rows: Iterable[Mapping[str, object]] | pd.DataFrame,
    *,
    name: str,
    metric_names: tuple[str, ...],
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    else:
        raw_rows = list(rows)
        if not all(isinstance(row, Mapping) for row in raw_rows):
            raise ValueError(f"{name} must contain only mappings.")
        frame = pd.DataFrame(raw_rows)
    if frame.empty:
        raise ValueError(f"{name} must be non-empty.")

    missing = [column for column in (*required_columns, *metric_names) if column not in frame]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")
    if frame[list(required_columns)].isna().any().any():
        raise ValueError(f"{name} contains missing pair or availability values.")

    for metric_name in metric_names:
        values = frame[metric_name].tolist()
        for value in values:
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{name}.{metric_name} must contain numeric values between 0 and 1.")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name}.{metric_name} must contain finite values between 0 and 1."
                )
        frame[metric_name] = frame[metric_name].astype(float)
    return frame


def _validate_unique_pair_keys(
    frame: pd.DataFrame,
    pair_columns: tuple[str, ...],
    name: str,
) -> None:
    if frame[list(pair_columns)].isna().any().any():
        raise ValueError(f"{name} contains a missing pair key.")
    if frame.duplicated(list(pair_columns)).any():
        raise ValueError(f"{name} contains duplicate pair keys.")


def _series_contains_only_booleans(values: pd.Series) -> bool:
    return bool(
        values.map(lambda value: isinstance(value, (bool, np.bool_))).all()
    )


def _hierarchical_bootstrap_mean_confidence_interval(
    values: np.ndarray,
    clusters: list[object],
    *,
    confidence_level: float,
    n_resamples: int,
    random_state: int,
) -> BootstrapConfidenceInterval:
    """Bootstrap a macro mean by resampling clusters and rows within clusters."""

    if len(values) != len(clusters) or len(values) == 0:
        raise ValueError("values and clusters must be non-empty and have matching lengths.")
    if not np.isfinite(values).all():
        raise ValueError("values must contain only finite values.")
    ordered_clusters = sorted(set(clusters), key=lambda value: (type(value).__name__, str(value)))
    by_cluster = [
        values[np.asarray([cluster == value for cluster in clusters], dtype=bool)]
        for value in ordered_clusters
    ]
    if not all(len(cluster_values) for cluster_values in by_cluster):
        raise ValueError("Every bootstrap cluster must contain at least one value.")

    point_mean = float(np.mean([cluster_values.mean() for cluster_values in by_cluster]))
    rng = np.random.default_rng(int(random_state))
    bootstrap_means = np.empty(int(n_resamples), dtype=float)
    cluster_count = len(by_cluster)
    for sample_index in range(int(n_resamples)):
        selected_clusters = rng.integers(0, cluster_count, size=cluster_count)
        sampled_means = []
        for cluster_index in selected_clusters:
            cluster_values = by_cluster[int(cluster_index)]
            row_indices = rng.integers(0, len(cluster_values), size=len(cluster_values))
            sampled_means.append(float(cluster_values[row_indices].mean()))
        bootstrap_means[sample_index] = float(np.mean(sampled_means))

    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return BootstrapConfidenceInterval(
        mean=point_mean,
        lower=float(lower),
        upper=float(upper),
        confidence_level=float(confidence_level),
        sample_size=int(len(values)),
        n_resamples=int(n_resamples),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validate_k(k: int) -> None:
    _validate_positive_integer(k, "k")


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _validate_finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number.")


def _validate_fraction(value: float, name: str) -> None:
    _validate_finite_number(value, name)
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must be greater than 0 and less than 1.")


def _validate_random_state(random_state: int) -> None:
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or not 0 <= int(random_state) < 2**32
    ):
        raise ValueError("random_state must be an integer between 0 and 2**32 - 1.")
