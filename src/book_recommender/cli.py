from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from threadpoolctl import threadpool_limits

from book_recommender.artifacts import (
    ArtifactError,
    fingerprint_dataset,
    load_model_bundle,
    save_model_bundle,
)
from book_recommender.benchmark_resources import (
    hybrid_sparse_payload,
    model_serialized_size_bytes,
    summarize_latency_ms,
    timed_call,
)
from book_recommender.catalog_benchmark import (
    CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION,
    CATALOG_BENCHMARK_ARTIFACT_TYPE,
    CATALOG_BENCHMARK_PROTOCOL,
    benchmark_catalog_configs,
    catalog_config_registry,
    normalize_hybrid_weights as normalize_catalog_hybrid_weights,
    validate_resource_metrics,
)
from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender, book_work_key
from book_recommender.data import (
    BookDataset,
    DatasetNotFoundError,
    dataset_summary,
    load_clean_dataset,
    prepare_dataset,
    prepare_hybrid_dataset,
    select_content_catalog,
    validate_data_dir,
)
from book_recommender.evaluation import (
    bootstrap_paired_ranking_deltas_at_k,
    bootstrap_ranking_metrics_at_k,
    build_fixed_test_work_level_split,
    build_work_level_holdouts,
    mean_ranking_metrics_at_k,
    prepare_work_level_evaluation_train_dataset,
    prepare_validation_test_evaluation,
    ranking_metrics_at_k,
    recommendation_diagnostics_at_k,
    remove_work_level_holdouts,
    summarize_holdout_availability,
)
from book_recommender.hybrid import HybridRecommender


DEFAULT_MAX_BOOKS = 5_000
DEFAULT_MAX_CONTENT_BOOKS = 20_000
DEFAULT_HYBRID_CANDIDATE_POOL_SIZE = 500
DEFAULT_POSITIVE_RATING_THRESHOLD = 8.0
EVALUATION_ARTIFACT_SCHEMA_VERSION = 2
EVALUATION_PROTOCOL = (
    "canonical_work_holdout_disjoint_validation_test_train_only_filtering_v2"
)
TUNING_PROTOCOL = "canonical_work_tuning_validation_partition_train_only_filtering_v2"
BENCHMARK_ARTIFACT_SCHEMA_VERSION = 2
BENCHMARK_PROTOCOL = "canonical_work_fixed_test_repeated_validation_paired_bootstrap_v1"
DEFAULT_HYBRID_WEIGHT_GRID = [
    (0.50, 0.40, 0.10),
    (0.45, 0.45, 0.10),
    (0.35, 0.55, 0.10),
    (0.55, 0.35, 0.10),
    (0.40, 0.45, 0.15),
    (0.30, 0.60, 0.10),
]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (ArtifactError, DatasetNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Book recommendation project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    from book_recommender.neural_cli import register_neural_commands

    register_neural_commands(subparsers)

    check = subparsers.add_parser("check-data", help="Validate that Kaggle CSV files exist.")
    add_data_dir_arg(check)
    check.set_defaults(func=check_data)

    summary = subparsers.add_parser("summary", help="Show cleaned dataset summary.")
    add_common_dataset_args(summary)
    summary.set_defaults(func=show_summary)

    content = subparsers.add_parser("recommend-content", help="Recommend books similar to a title or ISBN.")
    add_data_dir_arg(content)
    content.add_argument(
        "--max-content-books",
        type=positive_int,
        default=DEFAULT_MAX_CONTENT_BOOKS,
    )
    content.add_argument("--title", help="Book title to use as query.")
    content.add_argument("--isbn", help="ISBN to use as query.")
    content.add_argument("--top-n", type=positive_int, default=10)
    content.set_defaults(func=recommend_content)

    collab = subparsers.add_parser("recommend-collab", help="Recommend books for a user from rating history.")
    add_common_dataset_args(collab)
    collab.add_argument("--user-id", type=positive_int, required=True)
    collab.add_argument("--top-n", type=positive_int, default=10)
    collab.add_argument("--neighbors", type=positive_int, default=50)
    collab.add_argument(
        "--positive-rating-threshold",
        type=rating_threshold,
        default=DEFAULT_POSITIVE_RATING_THRESHOLD,
    )
    collab.set_defaults(func=recommend_collab)

    hybrid = subparsers.add_parser("recommend-hybrid", help="Recommend books using hybrid scoring.")
    add_common_dataset_args(hybrid)
    hybrid.add_argument("--user-id", type=positive_int, required=True)
    hybrid.add_argument("--top-n", type=positive_int, default=10)
    hybrid.add_argument("--neighbors", type=positive_int, default=50)
    hybrid.add_argument("--candidate-pool-size", type=positive_int, default=500)
    hybrid.add_argument(
        "--max-content-books",
        type=positive_int,
        default=DEFAULT_MAX_CONTENT_BOOKS,
    )
    hybrid.add_argument(
        "--positive-rating-threshold",
        type=rating_threshold,
        default=DEFAULT_POSITIVE_RATING_THRESHOLD,
    )
    hybrid.set_defaults(func=recommend_hybrid)

    evaluate_collab = subparsers.add_parser(
        "evaluate-collab",
        help="Evaluate collaborative, hybrid, and popularity ranking metrics.",
    )
    add_common_dataset_args(evaluate_collab)
    evaluate_collab.add_argument("--sample-users", type=at_least_two_int, default=200)
    evaluate_collab.add_argument("--validation-fraction", type=fraction, default=0.5)
    evaluate_collab.add_argument("--min-interactions", type=at_least_two_int, default=3)
    evaluate_collab.add_argument("--min-relevant-rating", type=rating_threshold, default=8.0)
    evaluate_collab.add_argument("--k", type=positive_int, default=50)
    evaluate_collab.add_argument("--neighbors", type=positive_int, default=50)
    evaluate_collab.add_argument(
        "--hybrid-candidate-pool-size",
        type=positive_int,
        default=DEFAULT_HYBRID_CANDIDATE_POOL_SIZE,
    )
    evaluate_collab.add_argument(
        "--max-content-books",
        type=positive_int,
        default=DEFAULT_MAX_CONTENT_BOOKS,
    )
    evaluate_collab.add_argument("--hybrid-collaborative-weight", type=non_negative_float, default=0.40)
    evaluate_collab.add_argument("--hybrid-content-weight", type=non_negative_float, default=0.45)
    evaluate_collab.add_argument("--hybrid-popularity-weight", type=non_negative_float, default=0.15)
    evaluate_collab.add_argument(
        "--positive-rating-threshold",
        type=rating_threshold,
        default=DEFAULT_POSITIVE_RATING_THRESHOLD,
    )
    evaluate_collab.add_argument("--disable-hybrid-dynamic-weights", action="store_true")
    evaluate_collab.add_argument("--random-state", type=random_state_int, default=42)
    evaluate_collab.add_argument("--bootstrap-resamples", type=positive_int, default=2000)
    evaluate_collab.add_argument("--confidence-level", type=fraction, default=0.95)
    evaluate_collab.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/recommender_evaluation.json"),
        help="Path to save evaluation metrics and parameters as JSON.",
    )
    evaluate_collab.set_defaults(func=evaluate_collab_model)

    tune_hybrid = subparsers.add_parser(
        "tune-hybrid",
        help="Evaluate several hybrid weight combinations on the same holdout split.",
    )
    add_common_dataset_args(tune_hybrid)
    tune_hybrid.add_argument("--sample-users", type=at_least_two_int, default=200)
    tune_hybrid.add_argument("--validation-fraction", type=fraction, default=0.5)
    tune_hybrid.add_argument("--min-interactions", type=at_least_two_int, default=3)
    tune_hybrid.add_argument("--min-relevant-rating", type=rating_threshold, default=8.0)
    tune_hybrid.add_argument("--k", type=positive_int, default=50)
    tune_hybrid.add_argument("--neighbors", type=positive_int, default=50)
    tune_hybrid.add_argument(
        "--hybrid-candidate-pool-size",
        type=positive_int,
        default=DEFAULT_HYBRID_CANDIDATE_POOL_SIZE,
    )
    tune_hybrid.add_argument(
        "--max-content-books",
        type=positive_int,
        default=DEFAULT_MAX_CONTENT_BOOKS,
    )
    tune_hybrid.add_argument(
        "--positive-rating-threshold",
        type=rating_threshold,
        default=DEFAULT_POSITIVE_RATING_THRESHOLD,
    )
    tune_hybrid.add_argument("--disable-hybrid-dynamic-weights", action="store_true")
    tune_hybrid.add_argument(
        "--weight-grid",
        default=None,
        help=(
            "Semicolon-separated collab,content,popularity weights. "
            "Example: '0.5,0.4,0.1;0.35,0.55,0.1'."
        ),
    )
    tune_hybrid.add_argument("--random-state", type=random_state_int, default=42)
    tune_hybrid.add_argument("--bootstrap-resamples", type=positive_int, default=2000)
    tune_hybrid.add_argument("--confidence-level", type=fraction, default=0.95)
    tune_hybrid.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/hybrid_weight_tuning.json"),
        help="Path to save tuning results as JSON.",
    )
    tune_hybrid.set_defaults(func=tune_hybrid_weights)

    benchmark_hybrid = subparsers.add_parser(
        "benchmark-hybrid",
        help="Select hybrid weights with repeated validation and one fixed untouched test cohort.",
    )
    add_common_dataset_args(benchmark_hybrid)
    benchmark_hybrid.add_argument(
        "--validation-users",
        type=positive_int,
        default=200,
        help="Number of user-work holdouts drawn for each validation repeat.",
    )
    benchmark_hybrid.add_argument(
        "--test-users",
        type=positive_int,
        default=200,
        help="Number of user-work holdouts reserved once for the final fixed test.",
    )
    benchmark_hybrid.add_argument("--repetitions", type=at_least_two_int, default=3)
    benchmark_hybrid.add_argument("--seed-stride", type=positive_int, default=1)
    benchmark_hybrid.add_argument("--min-interactions", type=at_least_two_int, default=3)
    benchmark_hybrid.add_argument("--min-relevant-rating", type=rating_threshold, default=8.0)
    benchmark_hybrid.add_argument("--k", type=positive_int, default=50)
    benchmark_hybrid.add_argument("--neighbors", type=positive_int, default=50)
    benchmark_hybrid.add_argument(
        "--hybrid-candidate-pool-size",
        type=positive_int,
        default=DEFAULT_HYBRID_CANDIDATE_POOL_SIZE,
    )
    benchmark_hybrid.add_argument(
        "--max-content-books",
        type=positive_int,
        default=DEFAULT_MAX_CONTENT_BOOKS,
    )
    benchmark_hybrid.add_argument(
        "--positive-rating-threshold",
        type=rating_threshold,
        default=DEFAULT_POSITIVE_RATING_THRESHOLD,
    )
    benchmark_hybrid.add_argument("--disable-hybrid-dynamic-weights", action="store_true")
    benchmark_hybrid.add_argument(
        "--weight-grid",
        default=None,
        help=(
            "Semicolon-separated collab,content,popularity weights. "
            "Example: '0.5,0.4,0.1;0.35,0.55,0.1'."
        ),
    )
    benchmark_hybrid.add_argument(
        "--baseline-weights",
        default="0.40,0.45,0.15",
        help="One collab,content,popularity baseline configuration used for paired deltas.",
    )
    benchmark_hybrid.add_argument(
        "--selection-metric",
        choices=("precision", "recall", "hit_rate", "map", "mrr", "ndcg"),
        default="ndcg",
    )
    benchmark_hybrid.add_argument(
        "--min-delta",
        type=non_negative_float,
        default=0.0,
        help="Required lower paired-CI improvement over the baseline.",
    )
    benchmark_hybrid.add_argument(
        "--min-selection-rate",
        type=closed_fraction,
        default=0.60,
        help="Minimum validation-repeat fraction whose primary-metric delta exceeds min-delta.",
    )
    benchmark_hybrid.add_argument("--random-state", type=random_state_int, default=100)
    benchmark_hybrid.add_argument("--test-random-state", type=random_state_int, default=42)
    benchmark_hybrid.add_argument("--bootstrap-resamples", type=positive_int, default=2000)
    benchmark_hybrid.add_argument("--confidence-level", type=fraction, default=0.95)
    benchmark_hybrid.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/repeated_split_hybrid_benchmark.json"),
        help="Path to save the repeated-validation decision and final fixed-test report.",
    )
    benchmark_hybrid.set_defaults(func=benchmark_hybrid_weights)

    benchmark_catalog = subparsers.add_parser(
        "benchmark-catalog",
        help=(
            "Compare interaction/content catalog sizes with repeated validation and one "
            "fixed untouched test cohort."
        ),
    )
    add_data_dir_arg(benchmark_catalog)
    benchmark_catalog.add_argument("--min-user-ratings", type=positive_int, default=2)
    benchmark_catalog.add_argument("--min-book-ratings", type=positive_int, default=2)
    benchmark_catalog.add_argument("--max-ratings", type=positive_int, default=None)
    benchmark_catalog.add_argument(
        "--catalog-grid",
        default="1000,3000;3000,10000;5000,20000",
        help="Semicolon-separated interaction,content limits.",
    )
    benchmark_catalog.add_argument(
        "--baseline-catalog",
        default="1000,3000",
        help="One interaction,content baseline that must be present in the grid.",
    )
    benchmark_catalog.add_argument(
        "--hybrid-weights",
        default="0.40,0.45,0.15",
        help="One fixed collaborative,content,popularity weight triple used by every catalog.",
    )
    benchmark_catalog.add_argument("--validation-users", type=positive_int, default=200)
    benchmark_catalog.add_argument("--test-users", type=positive_int, default=200)
    benchmark_catalog.add_argument("--repetitions", type=at_least_two_int, default=5)
    benchmark_catalog.add_argument("--seed-stride", type=positive_int, default=1)
    benchmark_catalog.add_argument("--min-interactions", type=at_least_two_int, default=3)
    benchmark_catalog.add_argument("--min-relevant-rating", type=rating_threshold, default=8.0)
    benchmark_catalog.add_argument("--k", type=positive_int, default=10)
    benchmark_catalog.add_argument("--neighbors", type=positive_int, default=50)
    benchmark_catalog.add_argument(
        "--hybrid-candidate-pool-size",
        type=positive_int,
        default=DEFAULT_HYBRID_CANDIDATE_POOL_SIZE,
    )
    benchmark_catalog.add_argument(
        "--positive-rating-threshold",
        type=rating_threshold,
        default=DEFAULT_POSITIVE_RATING_THRESHOLD,
    )
    benchmark_catalog.add_argument("--disable-hybrid-dynamic-weights", action="store_true")
    benchmark_catalog.add_argument(
        "--selection-metric",
        choices=("precision", "recall", "hit_rate", "map", "mrr", "ndcg"),
        default="ndcg",
    )
    benchmark_catalog.add_argument("--min-delta", type=non_negative_float, default=0.0)
    benchmark_catalog.add_argument(
        "--min-selection-rate",
        type=closed_fraction,
        default=0.60,
    )
    benchmark_catalog.add_argument(
        "--min-content-retention-delta",
        type=non_negative_float,
        default=0.0,
        help="Minimum validation gain in target availability within the content catalog.",
    )
    benchmark_catalog.add_argument(
        "--min-personalization-retention-delta",
        type=non_negative_float,
        default=0.0,
        help="Minimum validation gain in personalized-rankable holdout retention.",
    )
    benchmark_catalog.add_argument(
        "--max-latency-ratio",
        type=positive_float,
        default=3.0,
        help="Maximum challenger/baseline warm-inference p95 ratio allowed for promotion.",
    )
    benchmark_catalog.add_argument(
        "--max-serialized-model-ratio",
        type=positive_float,
        default=4.0,
        help="Maximum challenger/baseline serialized-model size ratio allowed for promotion.",
    )
    benchmark_catalog.add_argument("--random-state", type=random_state_int, default=100)
    benchmark_catalog.add_argument("--test-random-state", type=random_state_int, default=42)
    benchmark_catalog.add_argument("--bootstrap-resamples", type=positive_int, default=2000)
    benchmark_catalog.add_argument("--confidence-level", type=fraction, default=0.95)
    benchmark_catalog.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/catalog_size_benchmark.json"),
    )
    benchmark_catalog.set_defaults(func=benchmark_catalog_sizes)

    train_content = subparsers.add_parser("train-content", help="Build and save content model artifact.")
    add_data_dir_arg(train_content)
    train_content.add_argument(
        "--max-content-books",
        type=positive_int,
        default=DEFAULT_MAX_CONTENT_BOOKS,
    )
    train_content.add_argument("--output", type=Path, default=Path("models/content.joblib"))
    train_content.set_defaults(func=train_content_model)

    train_collab = subparsers.add_parser("train-collab", help="Build and save collaborative model artifact.")
    add_common_dataset_args(train_collab)
    train_collab.add_argument("--neighbors", type=positive_int, default=50)
    train_collab.add_argument(
        "--positive-rating-threshold",
        type=rating_threshold,
        default=DEFAULT_POSITIVE_RATING_THRESHOLD,
    )
    train_collab.add_argument("--output", type=Path, default=Path("models/collab.joblib"))
    train_collab.set_defaults(func=train_collab_model)

    train_hybrid = subparsers.add_parser(
        "train-hybrid",
        help="Build a verified composite hybrid artifact for serving.",
    )
    add_common_dataset_args(train_hybrid)
    train_hybrid.add_argument("--neighbors", type=positive_int, default=50)
    train_hybrid.add_argument(
        "--max-content-books",
        type=positive_int,
        default=DEFAULT_MAX_CONTENT_BOOKS,
    )
    train_hybrid.add_argument("--collaborative-weight", type=non_negative_float, default=0.40)
    train_hybrid.add_argument("--content-weight", type=non_negative_float, default=0.45)
    train_hybrid.add_argument("--popularity-weight", type=non_negative_float, default=0.15)
    train_hybrid.add_argument(
        "--positive-rating-threshold",
        type=rating_threshold,
        default=DEFAULT_POSITIVE_RATING_THRESHOLD,
    )
    train_hybrid.add_argument("--disable-dynamic-weights", action="store_true")
    train_hybrid.add_argument("--random-state", type=random_state_int, default=42)
    train_hybrid.add_argument(
        "--evaluation-artifact",
        type=Path,
        default=None,
        help="Optional evaluation JSON whose summary is embedded in the model manifest.",
    )
    train_hybrid.add_argument(
        "--benchmark-artifact",
        type=Path,
        default=None,
        help=(
            "Optional repeated-split benchmark JSON; it must select these normalized "
            "weights and is embedded in the model manifest."
        ),
    )
    train_hybrid.add_argument(
        "--catalog-benchmark-artifact",
        type=Path,
        default=None,
        help=(
            "Optional catalog-size benchmark JSON; its deployment configuration and fixed "
            "weights must match this serving bundle."
        ),
    )
    train_hybrid.add_argument("--output", type=Path, default=Path("models/hybrid.joblib"))
    train_hybrid.set_defaults(func=train_hybrid_model)

    artifact_info = subparsers.add_parser(
        "artifact-info",
        help="Verify a model bundle and print its manifest.",
    )
    artifact_info.add_argument("path", type=Path)
    artifact_info.set_defaults(func=show_artifact_info)

    return parser


def add_data_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data/raw", type=Path, help="Folder containing Kaggle CSV files.")


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    add_data_dir_arg(parser)
    parser.add_argument("--min-user-ratings", type=positive_int, default=2)
    parser.add_argument("--min-book-ratings", type=positive_int, default=2)
    parser.add_argument("--max-books", type=positive_int, default=DEFAULT_MAX_BOOKS)
    parser.add_argument("--max-ratings", type=positive_int, default=None)


def check_data(args: argparse.Namespace) -> None:
    data_dir = validate_data_dir(args.data_dir)
    print(f"Dataset files found in {data_dir.resolve()}")


def show_summary(args: argparse.Namespace) -> None:
    dataset = load_prepared_dataset(args)
    for key, value in dataset_summary(dataset).items():
        print(f"{key}: {value:,}")


def recommend_content(args: argparse.Namespace) -> None:
    if not args.title and not args.isbn:
        raise SystemExit("Provide --title or --isbn.")
    dataset = load_content_catalog(args)
    recommender = ContentBasedRecommender().fit(dataset.books)
    recommendations = recommender.recommend_similar(title=args.title, isbn=args.isbn, top_n=args.top_n)
    print_recommendations(recommendations)


def recommend_collab(args: argparse.Namespace) -> None:
    dataset = load_prepared_dataset(args)
    recommender = ItemBasedCollaborativeRecommender(
        n_neighbors=args.neighbors,
        positive_rating_threshold=args.positive_rating_threshold,
    ).fit(
        dataset.books,
        dataset.ratings,
    )
    recommendations = recommender.recommend_for_user(user_id=args.user_id, top_n=args.top_n)
    print_recommendations(recommendations)


def recommend_hybrid(args: argparse.Namespace) -> None:
    dataset = load_hybrid_prepared_dataset(args)
    recommender = HybridRecommender(
        n_neighbors=args.neighbors,
        positive_rating_threshold=args.positive_rating_threshold,
    ).fit(
        dataset.books,
        dataset.ratings,
    )
    recommendations = recommender.recommend_for_user(
        user_id=args.user_id,
        top_n=args.top_n,
        candidate_pool_size=args.candidate_pool_size,
    )
    print_recommendations(recommendations)


def train_content_model(args: argparse.Namespace) -> None:
    dataset = load_content_catalog(args)
    recommender = ContentBasedRecommender().fit(dataset.books)
    manifest = save_training_bundle(
        recommender,
        args.output,
        model_type="content",
        dataset=dataset,
        args=args,
    )
    print(f"Saved content model {manifest.artifact_id} to {args.output.resolve()}")
    print_training_summary(dataset)


def train_collab_model(args: argparse.Namespace) -> None:
    dataset = load_prepared_dataset(args)
    recommender = ItemBasedCollaborativeRecommender(
        n_neighbors=args.neighbors,
        positive_rating_threshold=args.positive_rating_threshold,
    ).fit(
        dataset.books,
        dataset.ratings,
    )
    manifest = save_training_bundle(
        recommender,
        args.output,
        model_type="collaborative",
        dataset=dataset,
        args=args,
    )
    print(f"Saved collaborative model {manifest.artifact_id} to {args.output.resolve()}")
    print_training_summary(dataset)


def train_hybrid_model(args: argparse.Namespace) -> None:
    metrics = {}
    evaluation_artifact = getattr(args, "evaluation_artifact", None)
    benchmark_artifact = getattr(args, "benchmark_artifact", None)
    catalog_benchmark_artifact = getattr(args, "catalog_benchmark_artifact", None)
    lineage_artifacts = [
        artifact
        for artifact in (
            evaluation_artifact,
            benchmark_artifact,
            catalog_benchmark_artifact,
        )
        if artifact is not None
    ]
    if len(lineage_artifacts) > 1:
        raise ValueError(
            "Use only one of --evaluation-artifact, --benchmark-artifact, or "
            "--catalog-benchmark-artifact."
        )
    if lineage_artifacts:
        clean_dataset = load_clean_dataset(args.data_dir)
        source_fingerprint = fingerprint_dataset(
            clean_dataset.books,
            clean_dataset.ratings,
            clean_dataset.users,
        ).fingerprint
        if evaluation_artifact is not None:
            metrics = read_evaluation_metrics(
                evaluation_artifact,
                expected_source_dataset_fingerprint=source_fingerprint,
                expected_parameters=evaluation_parameters_for_training(args),
            )
        elif benchmark_artifact is not None:
            metrics = read_benchmark_metrics(
                benchmark_artifact,
                expected_source_dataset_fingerprint=source_fingerprint,
                expected_parameters=benchmark_parameters_for_training(args),
                expected_weights=normalized_hybrid_weights(
                    args.collaborative_weight,
                    args.content_weight,
                    args.popularity_weight,
                ),
            )
        else:
            metrics = read_catalog_benchmark_metrics(
                catalog_benchmark_artifact,
                expected_source_dataset_fingerprint=source_fingerprint,
                expected_parameters=catalog_benchmark_parameters_for_training(args),
                expected_catalog={
                    "max_interaction_books": args.max_books,
                    "max_content_books": args.max_content_books,
                },
                expected_weights=normalized_hybrid_weights(
                    args.collaborative_weight,
                    args.content_weight,
                    args.popularity_weight,
                ),
            )
    dataset = load_hybrid_prepared_dataset(args)
    recommender = HybridRecommender(
        n_neighbors=args.neighbors,
        collaborative_weight=args.collaborative_weight,
        content_weight=args.content_weight,
        popularity_weight=args.popularity_weight,
        dynamic_weights=not args.disable_dynamic_weights,
        positive_rating_threshold=args.positive_rating_threshold,
    ).fit(dataset.books, dataset.ratings)
    manifest = save_training_bundle(
        recommender,
        args.output,
        model_type="hybrid",
        dataset=dataset,
        args=args,
        metrics=metrics,
    )
    print(f"Saved hybrid model {manifest.artifact_id} to {args.output.resolve()}")
    print_training_summary(dataset)


def show_artifact_info(args: argparse.Namespace) -> None:
    _, manifest = load_model_bundle(args.path)
    print(json.dumps(manifest.__dict__, indent=2, sort_keys=True))


def evaluate_collab_model(args: argparse.Namespace) -> None:
    clean_dataset = load_clean_dataset(args.data_dir)
    evaluation_data = prepare_validation_test_evaluation(
        clean_dataset,
        sample_users=args.sample_users,
        validation_fraction=args.validation_fraction,
        min_interactions=args.min_interactions,
        min_relevant_rating=args.min_relevant_rating,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
        max_books=args.max_books,
        max_ratings=args.max_ratings,
        max_content_books=args.max_content_books,
        random_state=args.random_state,
    )
    holdouts = evaluation_data.test_holdout_ratings
    availability = evaluation_data.test_availability
    train_dataset = evaluation_data.train_dataset
    hybrid_recommender = HybridRecommender(
        n_neighbors=args.neighbors,
        collaborative_weight=args.hybrid_collaborative_weight,
        content_weight=args.hybrid_content_weight,
        popularity_weight=args.hybrid_popularity_weight,
        dynamic_weights=not args.disable_hybrid_dynamic_weights,
        positive_rating_threshold=args.positive_rating_threshold,
    ).fit(
        train_dataset.books,
        train_dataset.ratings,
    )
    recommender = hybrid_recommender.collaborative_model
    content_model = hybrid_recommender.content_model
    work_context = build_work_evaluation_context(train_dataset, content_model)

    rows = []
    model_names = (
        "hybrid",
        "hybrid_no_popularity_strict",
        "content_profile_strict",
        "collaborative_with_popularity_fallback",
        "popularity",
    )
    recommendation_lists = {model: [] for model in model_names}
    train_user_counts = train_dataset.ratings["user_id"].value_counts()
    positive_user_counts = positive_work_counts(
        train_dataset.ratings,
        work_context["isbn_to_work"],
        threshold=args.positive_rating_threshold,
    )
    catalog_work_set = set(work_context["catalog_items"])
    history_user_set = set(train_dataset.ratings["user_id"].astype(int))
    positive_user_set = set(positive_user_counts.index.astype(int))
    for holdout in holdouts.itertuples(index=False):
        user_id = int(holdout.user_id)
        relevant = [str(holdout.work_key)]
        candidate_available = str(holdout.work_key) in catalog_work_set
        history_available = user_id in history_user_set
        positive_seed_count = int(positive_user_counts.get(user_id, 0))
        positive_seed_available = user_id in positive_user_set
        status = holdout_status(
            candidate_available,
            history_available,
            positive_seed_available,
        )
        recommendations_by_model = {
            "hybrid": recommendation_work_keys(
                hybrid_recommender.recommend_for_user(
                    user_id,
                    top_n=args.k,
                    candidate_pool_size=args.hybrid_candidate_pool_size,
                ),
                work_context["isbn_to_work"],
            ),
            "hybrid_no_popularity_strict": recommendation_work_keys(
                recommend_hybrid_with_weights(
                    hybrid_recommender,
                    user_id=user_id,
                    top_n=args.k,
                    candidate_pool_size=args.hybrid_candidate_pool_size,
                    collaborative_weight=args.hybrid_collaborative_weight,
                    content_weight=args.hybrid_content_weight,
                    popularity_weight=0.0,
                    dynamic_weights=False,
                    allow_popularity_fallback=False,
                ),
                work_context["isbn_to_work"],
            ),
            "content_profile_strict": recommendation_work_keys(
                recommend_hybrid_with_weights(
                    hybrid_recommender,
                    user_id=user_id,
                    top_n=args.k,
                    candidate_pool_size=args.hybrid_candidate_pool_size,
                    collaborative_weight=0.0,
                    content_weight=1.0,
                    popularity_weight=0.0,
                    dynamic_weights=False,
                    allow_popularity_fallback=False,
                ),
                work_context["isbn_to_work"],
            ),
            "collaborative_with_popularity_fallback": recommendation_work_keys(
                recommender.recommend_for_user(user_id, top_n=args.k),
                work_context["isbn_to_work"],
            ),
        }
        train_history = train_dataset.ratings[train_dataset.ratings["user_id"] == user_id]
        read_isbns = set(train_history["isbn"])
        recommendations_by_model["popularity"] = recommendation_work_keys(
            recommender.recommend_popular(
                top_n=args.k,
                exclude_isbns=read_isbns,
            ),
            work_context["isbn_to_work"],
        )

        train_interactions = int(train_user_counts.get(user_id, 0))
        segment = interaction_segment(positive_seed_count)
        for model, recommended in recommendations_by_model.items():
            rows.append(
                {
                    "model": model,
                    "user_id": user_id,
                    "user_segment": segment,
                    "train_interactions": train_interactions,
                    "positive_seed_count": positive_seed_count,
                    "holdout_isbn": str(holdout.isbn),
                    "holdout_work_key": str(holdout.work_key),
                    "candidate_available": candidate_available,
                    "history_available": history_available,
                    "positive_seed_available": positive_seed_available,
                    "evaluation_status": status,
                    **ranking_metrics_at_k(recommended, relevant, args.k),
                }
            )
            recommendation_lists[model].append(recommended)

    diagnostics = {
        model: recommendation_diagnostics_at_k(
            recommendations,
            catalog_items=work_context["catalog_items"],
            popularity_counts=work_context["popularity_counts"],
            k=args.k,
            item_features=work_context["item_features"],
            item_to_index=work_context["item_to_index"],
        )
        for model, recommendations in recommendation_lists.items()
    }

    conditional_rows = [
        row
        for row in rows
        if row["candidate_available"] and row["positive_seed_available"]
    ]
    summary = summarize_evaluation_rows(conditional_rows, ["model"])
    segment_summary = summarize_evaluation_rows(rows, ["model", "user_segment"])
    conditional_segment_summary = summarize_evaluation_rows(
        conditional_rows,
        ["model", "user_segment"],
    )
    metric_summaries = {}
    confidence_intervals = {}
    for model in model_names:
        model_rows = [row for row in rows if row["model"] == model]
        model_conditional_rows = [
            row
            for row in model_rows
            if row["candidate_available"] and row["positive_seed_available"]
        ]
        end_to_end_metrics = mean_ranking_metrics_at_k(model_rows, k=args.k)
        conditional_metrics = mean_ranking_metrics_at_k(model_conditional_rows, k=args.k)
        metric_summaries[model] = {
            "k": args.k,
            "sampled_holdouts": len(holdouts),
            "conditional_holdouts": len(model_conditional_rows),
            "conditional_metrics": conditional_metrics,
            "end_to_end_metrics": end_to_end_metrics,
        }
        confidence_intervals[model] = {
            "conditional": (
                {
                    metric: interval.to_dict()
                    for metric, interval in bootstrap_ranking_metrics_at_k(
                        model_conditional_rows,
                        k=args.k,
                        confidence_level=args.confidence_level,
                        n_resamples=args.bootstrap_resamples,
                        random_state=args.random_state,
                    ).items()
                }
                if model_conditional_rows
                else {}
            ),
            "end_to_end": {
                metric: interval.to_dict()
                for metric, interval in bootstrap_ranking_metrics_at_k(
                model_rows,
                k=args.k,
                sampled_holdouts=len(holdouts),
                confidence_level=args.confidence_level,
                n_resamples=args.bootstrap_resamples,
                random_state=args.random_state,
                ).items()
            },
        }
    end_to_end_summary = evaluation_summary_dataframe(metric_summaries, "end_to_end_metrics")

    print("protocol: user-work holdout with disjoint validation/test and train-only filtering")
    print("relevance_unit: canonical_work")
    print("evaluation_partition: test")
    print(f"sampled_holdouts: {availability.sampled_holdouts:,}")
    print(f"candidate_available_holdouts: {availability.candidate_available_holdouts:,}")
    print(f"user_history_available_holdouts: {availability.user_history_available_holdouts:,}")
    availability_payload = availability_with_personalization(
        availability,
        holdouts,
        catalog_work_set=catalog_work_set,
        positive_user_set=positive_user_set,
    )
    print(f"history_evaluable_holdouts: {availability.evaluable_holdouts:,}")
    print(
        "personalization_evaluable_holdouts: "
        f"{availability_payload['personalization_evaluable_holdouts']:,}"
    )
    print(f"history_dropped_holdouts: {availability.dropped_holdouts:,}")
    print(f"history_evaluation_retention_rate: {availability.evaluation_retention_rate:.4f}")
    print(
        "personalization_retention_rate: "
        f"{availability_payload['personalization_retention_rate']:.4f}"
    )
    print(f"k: {args.k}")
    print(f"train_ratings_before_filter: {len(evaluation_data.train_ratings_before_filter):,}")
    print(f"train_ratings: {len(train_dataset.ratings):,}")
    print(f"candidate_books: {len(train_dataset.books):,}")
    print(f"candidate_works: {len(work_context['catalog_items']):,}")
    print()
    print("Conditional ranking metrics (evaluable targets only):")
    print_evaluation_summary(summary)
    print()
    print("End-to-end ranking metrics (all sampled targets; unavailable target = miss):")
    print_evaluation_summary(end_to_end_summary)
    print()
    print("End-to-end segment metrics:")
    print_evaluation_summary(segment_summary)
    print()
    print("Conditional segment metrics:")
    print_evaluation_summary(conditional_segment_summary)
    print()
    print("Recommendation diagnostics:")
    print_diagnostics(diagnostics)

    if args.output:
        save_evaluation_artifact(
            args.output,
            clean_dataset=clean_dataset,
            evaluation_data=evaluation_data,
            rows=rows,
            summary=summary,
            end_to_end_summary=end_to_end_summary,
            segment_summary=segment_summary,
            conditional_segment_summary=conditional_segment_summary,
            diagnostics=diagnostics,
            metric_summaries=metric_summaries,
            confidence_intervals=confidence_intervals,
            availability=availability_payload,
            args=args,
        )
        print()
        print(f"saved_evaluation: {args.output.resolve()}")


def tune_hybrid_weights(args: argparse.Namespace) -> None:
    clean_dataset = load_clean_dataset(args.data_dir)
    evaluation_data = prepare_validation_test_evaluation(
        clean_dataset,
        sample_users=args.sample_users,
        validation_fraction=args.validation_fraction,
        min_interactions=args.min_interactions,
        min_relevant_rating=args.min_relevant_rating,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
        max_books=args.max_books,
        max_ratings=args.max_ratings,
        max_content_books=args.max_content_books,
        random_state=args.random_state,
    )
    holdouts = evaluation_data.validation_holdout_ratings
    availability = evaluation_data.validation_availability
    if availability.candidate_available_holdouts == 0:
        raise SystemExit(
            "No validation target works remain in the content candidate catalog. "
            "Try increasing --max-content-books."
        )

    train_dataset = evaluation_data.train_dataset
    hybrid_recommender = HybridRecommender(
        n_neighbors=args.neighbors,
        dynamic_weights=not args.disable_hybrid_dynamic_weights,
        positive_rating_threshold=args.positive_rating_threshold,
    ).fit(train_dataset.books, train_dataset.ratings)

    work_context = build_work_evaluation_context(
        train_dataset,
        hybrid_recommender.content_model,
    )

    weight_grid = parse_weight_grid(args.weight_grid)
    catalog_work_set = set(work_context["catalog_items"])
    history_user_set = set(train_dataset.ratings["user_id"].astype(int))
    positive_user_counts = positive_work_counts(
        train_dataset.ratings,
        work_context["isbn_to_work"],
        threshold=args.positive_rating_threshold,
    )
    positive_user_set = set(positive_user_counts.index.astype(int))
    results = []
    for config_index, weights in enumerate(weight_grid, start=1):
        hybrid_recommender.collaborative_weight = weights["collaborative_weight"]
        hybrid_recommender.content_weight = weights["content_weight"]
        hybrid_recommender.popularity_weight = weights["popularity_weight"]

        metric_rows = []
        recommendation_lists = []
        train_user_counts = train_dataset.ratings["user_id"].value_counts()
        for holdout in holdouts.itertuples(index=False):
            user_id = int(holdout.user_id)
            relevant = [str(holdout.work_key)]
            candidate_available = str(holdout.work_key) in catalog_work_set
            history_available = user_id in history_user_set
            positive_seed_count = int(positive_user_counts.get(user_id, 0))
            positive_seed_available = user_id in positive_user_set
            recommended = recommendation_work_keys(
                hybrid_recommender.recommend_for_user(
                    user_id,
                    top_n=args.k,
                    candidate_pool_size=args.hybrid_candidate_pool_size,
                ),
                work_context["isbn_to_work"],
            )
            train_interactions = int(train_user_counts.get(user_id, 0))
            metric_rows.append(
                {
                    "model": "hybrid",
                    "user_id": user_id,
                    "user_segment": interaction_segment(positive_seed_count),
                    "train_interactions": train_interactions,
                    "positive_seed_count": positive_seed_count,
                    "holdout_isbn": str(holdout.isbn),
                    "holdout_work_key": str(holdout.work_key),
                    "candidate_available": candidate_available,
                    "history_available": history_available,
                    "positive_seed_available": positive_seed_available,
                    "evaluation_status": holdout_status(
                        candidate_available,
                        history_available,
                        positive_seed_available,
                    ),
                    **ranking_metrics_at_k(recommended, relevant, args.k),
                }
            )
            recommendation_lists.append(recommended)

        end_to_end_metrics = mean_ranking_metrics_at_k(metric_rows, k=args.k)
        conditional_rows = [
            row
            for row in metric_rows
            if row["candidate_available"] and row["positive_seed_available"]
        ]
        conditional_metrics = mean_ranking_metrics_at_k(conditional_rows, k=args.k)
        confidence_intervals = {
            "end_to_end": {
                metric: interval.to_dict()
                for metric, interval in bootstrap_ranking_metrics_at_k(
                metric_rows,
                k=args.k,
                sampled_holdouts=len(holdouts),
                confidence_level=args.confidence_level,
                n_resamples=args.bootstrap_resamples,
                random_state=args.random_state,
                ).items()
            },
            "conditional": (
                {
                    metric: interval.to_dict()
                    for metric, interval in bootstrap_ranking_metrics_at_k(
                        conditional_rows,
                        k=args.k,
                        confidence_level=args.confidence_level,
                        n_resamples=args.bootstrap_resamples,
                        random_state=args.random_state,
                    ).items()
                }
                if conditional_rows
                else {}
            ),
        }
        diagnostics = recommendation_diagnostics_at_k(
            recommendation_lists,
            catalog_items=work_context["catalog_items"],
            popularity_counts=work_context["popularity_counts"],
            k=args.k,
            item_features=work_context["item_features"],
            item_to_index=work_context["item_to_index"],
        )
        result = {
            "config_id": f"w{config_index}",
            **weights,
            "dynamic_weights": not args.disable_hybrid_dynamic_weights,
            "sampled_holdouts": len(holdouts),
            "conditional_holdouts": len(conditional_rows),
            "candidate_retention_rate": availability.candidate_retention_rate,
            **end_to_end_metrics,
            "conditional_metrics": conditional_metrics,
            "end_to_end_metrics": end_to_end_metrics,
            "confidence_intervals": confidence_intervals,
            "per_user_rows": metric_rows,
            **diagnostics,
        }
        results.append(result)

    ranked_results = rank_tuning_results(results, args.k)

    print("protocol: tune canonical-work ranking on validation users; test users untouched")
    print("relevance_unit: canonical_work")
    print("evaluation_partition: validation")
    print(f"sampled_holdouts: {availability.sampled_holdouts:,}")
    print(f"candidate_available_holdouts: {availability.candidate_available_holdouts:,}")
    print(f"user_history_available_holdouts: {availability.user_history_available_holdouts:,}")
    availability_payload = availability_with_personalization(
        availability,
        holdouts,
        catalog_work_set=catalog_work_set,
        positive_user_set=positive_user_set,
    )
    print(f"history_evaluable_holdouts: {availability.evaluable_holdouts:,}")
    print(
        "personalization_evaluable_holdouts: "
        f"{availability_payload['personalization_evaluable_holdouts']:,}"
    )
    print(f"history_dropped_holdouts: {availability.dropped_holdouts:,}")
    print(f"history_evaluation_retention_rate: {availability.evaluation_retention_rate:.4f}")
    print(
        "personalization_retention_rate: "
        f"{availability_payload['personalization_retention_rate']:.4f}"
    )
    print(f"k: {args.k}")
    print(f"weight_configs: {len(ranked_results):,}")
    print(f"dynamic_weights: {not args.disable_hybrid_dynamic_weights}")
    print(f"candidate_books: {len(train_dataset.books):,}")
    print(f"candidate_works: {len(work_context['catalog_items']):,}")
    print()
    print("Hybrid weight tuning:")
    print_tuning_summary(ranked_results, args.k)

    if args.output:
        save_tuning_artifact(
            args.output,
            clean_dataset=clean_dataset,
            evaluation_data=evaluation_data,
            results=ranked_results,
            availability=availability_payload,
            args=args,
        )
        print()
        print(f"saved_tuning: {args.output.resolve()}")


def benchmark_hybrid_weights(args: argparse.Namespace) -> None:
    """Tune hybrid weights on repeated development splits, then test once externally."""

    _validate_benchmark_seed_range(args)
    clean_dataset = load_clean_dataset(args.data_dir)
    configs, baseline_config_id = benchmark_weight_configs(
        weight_grid=args.weight_grid,
        baseline_weights=args.baseline_weights,
    )
    fixed_test = build_fixed_test_work_level_split(
        clean_dataset.ratings,
        clean_dataset.books,
        test_users=args.test_users,
        min_interactions=args.min_interactions,
        min_relevant_rating=args.min_relevant_rating,
        random_state=args.test_random_state,
    )
    if len(fixed_test.test_holdout_ratings) != args.test_users:
        raise ValueError(
            "The requested fixed test cohort is unavailable after work-level eligibility "
            "filtering; reduce --test-users."
        )
    test_user_ids = fixed_test.test_users
    validation_rows: list[dict[str, object]] = []
    validation_repeats = []
    for repeat_index in range(args.repetitions):
        split_seed = args.random_state + repeat_index * args.seed_stride
        split_id = f"validation-{repeat_index + 1:02d}"
        holdouts = build_work_level_holdouts(
            fixed_test.development_ratings,
            clean_dataset.books,
            sample_users=args.validation_users,
            min_interactions=args.min_interactions,
            min_relevant_rating=args.min_relevant_rating,
            random_state=split_seed,
            excluded_user_ids=test_user_ids,
        )
        if len(holdouts) != args.validation_users:
            raise ValueError(
                "The requested validation cohort is unavailable after reserving the fixed "
                "test cohort; reduce --validation-users or --test-users."
            )
        train_ratings = remove_work_level_holdouts(
            fixed_test.development_ratings,
            clean_dataset.books,
            holdouts,
        )
        train_dataset = prepare_work_level_evaluation_train_dataset(
            clean_dataset,
            train_ratings,
            min_user_ratings=args.min_user_ratings,
            min_book_ratings=args.min_book_ratings,
            max_books=args.max_books,
            max_ratings=args.max_ratings,
            max_content_books=args.max_content_books,
            random_state=split_seed,
        )
        evaluated = evaluate_hybrid_weight_configurations(
            train_dataset=train_dataset,
            holdouts=holdouts,
            configs=configs,
            args=args,
            split_id=split_id,
            split_seed=split_seed,
            partition="validation",
        )
        validation_rows.extend(evaluated["rows"])
        validation_repeats.append(
            {
                "split_id": split_id,
                "split_seed": split_seed,
                "sampled_holdouts": int(len(holdouts)),
                "train_ratings_before_filter": int(len(train_ratings)),
                "train_ratings": int(len(train_dataset.ratings)),
                "candidate_books": int(len(train_dataset.books)),
                "candidate_works": candidate_work_count(train_dataset.books),
                "availability": evaluated["availability"],
                "configurations": evaluated["configurations"],
            }
        )

    validation_aggregate = aggregate_benchmark_configurations(
        validation_rows,
        configs=configs,
        k=args.k,
    )
    selection_metric = f"{args.selection_metric}_at_{args.k}"
    challenger_count = len(configs) - 1
    selection_confidence_level = familywise_confidence_level(
        args.confidence_level,
        challenger_count,
    )
    selection_multiplicity_adjustment = (
        "bonferroni_familywise" if challenger_count > 1 else "none"
    )
    validation_deltas = paired_benchmark_deltas(
        validation_rows,
        configs=configs,
        baseline_config_id=baseline_config_id,
        k=args.k,
        selection_metric=selection_metric,
        min_delta=args.min_delta,
        confidence_level=selection_confidence_level,
        n_resamples=args.bootstrap_resamples,
        random_state=args.random_state,
        cluster_column="split_id",
    )
    selection = select_benchmark_configuration(
        validation_aggregate,
        validation_deltas=validation_deltas,
        baseline_config_id=baseline_config_id,
        selection_metric=selection_metric,
        min_delta=args.min_delta,
        min_selection_rate=args.min_selection_rate,
        confidence_level=selection_confidence_level,
        multiplicity_adjustment=selection_multiplicity_adjustment,
    )
    selected_config_id = selection["selected_config_id"]

    final_train_dataset = prepare_work_level_evaluation_train_dataset(
        clean_dataset,
        fixed_test.development_ratings,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
        max_books=args.max_books,
        max_ratings=args.max_ratings,
        max_content_books=args.max_content_books,
        random_state=args.test_random_state,
    )
    config_by_id = {str(config["config_id"]): config for config in configs}
    final_config_ids = [baseline_config_id]
    if selected_config_id != baseline_config_id:
        final_config_ids.append(selected_config_id)
    final_configs = [config_by_id[config_id] for config_id in final_config_ids]
    final_evaluated = evaluate_hybrid_weight_configurations(
        train_dataset=final_train_dataset,
        holdouts=fixed_test.test_holdout_ratings,
        configs=final_configs,
        args=args,
        split_id="fixed-test",
        split_seed=args.test_random_state,
        partition="test",
    )
    final_rows = final_evaluated["rows"]
    final_aggregate = aggregate_benchmark_configurations(
        final_rows,
        configs=final_configs,
        k=args.k,
    )
    final_delta = None
    if selected_config_id != baseline_config_id:
        final_delta = paired_benchmark_deltas(
            final_rows,
            configs=final_configs,
            baseline_config_id=baseline_config_id,
            k=args.k,
            selection_metric=selection_metric,
            min_delta=args.min_delta,
            confidence_level=args.confidence_level,
            n_resamples=args.bootstrap_resamples,
            random_state=_stable_seed(args.test_random_state, selected_config_id),
            cluster_column=None,
        )[selected_config_id]

    save_benchmark_artifact(
        args.output,
        clean_dataset=clean_dataset,
        fixed_test_holdouts=fixed_test.test_holdout_ratings,
        final_train_dataset=final_train_dataset,
        configs=configs,
        baseline_config_id=baseline_config_id,
        validation_repeats=validation_repeats,
        validation_rows=validation_rows,
        validation_aggregate=validation_aggregate,
        validation_deltas=validation_deltas,
        selection=selection,
        final_evaluated=final_evaluated,
        final_rows=final_rows,
        final_aggregate=final_aggregate,
        final_delta=final_delta,
        args=args,
    )

    selected = next(config for config in configs if config["config_id"] == selected_config_id)
    print("protocol: fixed outer test plus repeated development-only validation")
    print("relevance_unit: canonical_work")
    print("selection_partition: validation")
    print(f"validation_repetitions: {args.repetitions}")
    print(f"fixed_test_holdouts: {len(fixed_test.test_holdout_ratings):,}")
    print(f"baseline_config: {baseline_config_id}")
    print(f"selected_config: {selected_config_id}")
    print(f"decision: {selection['decision']}")
    print(
        f"selected_weights: collaborative={selected['collaborative_weight']:.4f}, "
        f"content={selected['content_weight']:.4f}, "
        f"popularity={selected['popularity_weight']:.4f}"
    )
    print(f"selection_metric: {selection_metric}")
    print(f"selection_confidence_level: {selection_confidence_level:.4f}")
    print(f"saved_benchmark: {args.output.resolve()}")


def benchmark_catalog_sizes(args: argparse.Namespace) -> None:
    """Compare catalog reach/cost on repeated validation, then confirm once on test."""

    _validate_benchmark_seed_range(args)
    clean_dataset = load_clean_dataset(args.data_dir)
    catalog_configs, baseline_config_id = benchmark_catalog_configs(
        catalog_grid=args.catalog_grid,
        baseline_catalog=args.baseline_catalog,
    )
    parsed_weights = parse_weight_grid(args.hybrid_weights)
    if len(parsed_weights) != 1:
        raise ValueError("--hybrid-weights must contain exactly one weight configuration.")
    weights = normalize_catalog_hybrid_weights(
        parsed_weights[0]["collaborative_weight"],
        parsed_weights[0]["content_weight"],
        parsed_weights[0]["popularity_weight"],
    )
    configs = [{**config, **weights} for config in catalog_configs]

    fixed_test = build_fixed_test_work_level_split(
        clean_dataset.ratings,
        clean_dataset.books,
        test_users=args.test_users,
        min_interactions=args.min_interactions,
        min_relevant_rating=args.min_relevant_rating,
        random_state=args.test_random_state,
    )
    if len(fixed_test.test_holdout_ratings) != args.test_users:
        raise ValueError(
            "The requested fixed test cohort is unavailable after work-level eligibility "
            "filtering; reduce --test-users."
        )

    validation_rows: list[dict[str, object]] = []
    validation_repeats: list[dict[str, object]] = []
    validation_configurations: list[dict[str, object]] = []
    validation_latencies = {str(config["config_id"]): [] for config in configs}
    test_user_ids = fixed_test.test_users
    for repeat_index in range(args.repetitions):
        split_seed = args.random_state + repeat_index * args.seed_stride
        split_id = f"validation-{repeat_index + 1:02d}"
        holdouts = build_work_level_holdouts(
            fixed_test.development_ratings,
            clean_dataset.books,
            sample_users=args.validation_users,
            min_interactions=args.min_interactions,
            min_relevant_rating=args.min_relevant_rating,
            random_state=split_seed,
            excluded_user_ids=test_user_ids,
        )
        if len(holdouts) != args.validation_users:
            raise ValueError(
                "The requested validation cohort is unavailable after reserving the fixed "
                "test cohort; reduce --validation-users or --test-users."
            )
        train_ratings = remove_work_level_holdouts(
            fixed_test.development_ratings,
            clean_dataset.books,
            holdouts,
        )
        evaluated = evaluate_catalog_configurations(
            clean_dataset=clean_dataset,
            train_ratings=train_ratings,
            holdouts=holdouts,
            configs=configs,
            args=args,
            split_id=split_id,
            split_seed=split_seed,
            partition="validation",
            execution_offset=repeat_index,
        )
        validation_rows.extend(evaluated["rows"])
        validation_configurations.extend(evaluated["configurations"])
        for config_id, samples in evaluated["latency_samples_by_config"].items():
            validation_latencies[config_id].extend(samples)
        validation_repeats.append(
            {
                "split_id": split_id,
                "split_seed": int(split_seed),
                "sampled_holdouts": int(len(holdouts)),
                "train_ratings_before_catalog_filter": int(len(train_ratings)),
                "configurations": evaluated["configurations"],
            }
        )

    validation_aggregate = aggregate_catalog_configurations(
        validation_rows,
        configs=configs,
        configuration_records=validation_configurations,
        latency_samples_by_config=validation_latencies,
        k=args.k,
    )
    selection_metric = f"{args.selection_metric}_at_{args.k}"
    challenger_count = len(configs) - 1
    selection_confidence_level = familywise_confidence_level(
        args.confidence_level,
        challenger_count,
    )
    multiplicity_adjustment = "bonferroni_familywise" if challenger_count > 1 else "none"
    validation_deltas = paired_benchmark_deltas(
        validation_rows,
        configs=configs,
        baseline_config_id=baseline_config_id,
        k=args.k,
        selection_metric=selection_metric,
        min_delta=args.min_delta,
        confidence_level=selection_confidence_level,
        n_resamples=args.bootstrap_resamples,
        random_state=args.random_state,
        cluster_column="split_id",
        require_matching_status=False,
    )
    selection = select_catalog_configuration(
        validation_aggregate,
        validation_deltas=validation_deltas,
        baseline_config_id=baseline_config_id,
        selection_metric=selection_metric,
        args=args,
        confidence_level=selection_confidence_level,
        multiplicity_adjustment=multiplicity_adjustment,
    )
    selected_config_id = str(selection["selected_config_id"])

    config_by_id = {str(config["config_id"]): config for config in configs}
    final_config_ids = [baseline_config_id]
    if selected_config_id != baseline_config_id:
        final_config_ids.append(selected_config_id)
    final_configs = [config_by_id[config_id] for config_id in final_config_ids]
    final_evaluated = evaluate_catalog_configurations(
        clean_dataset=clean_dataset,
        train_ratings=fixed_test.development_ratings,
        holdouts=fixed_test.test_holdout_ratings,
        configs=final_configs,
        args=args,
        split_id="fixed-test",
        split_seed=args.test_random_state,
        partition="test",
        execution_offset=0,
    )
    final_rows = final_evaluated["rows"]
    final_aggregate = aggregate_catalog_configurations(
        final_rows,
        configs=final_configs,
        configuration_records=final_evaluated["configurations"],
        latency_samples_by_config=final_evaluated["latency_samples_by_config"],
        k=args.k,
    )
    final_delta = None
    if selected_config_id != baseline_config_id:
        final_delta = paired_benchmark_deltas(
            final_rows,
            configs=final_configs,
            baseline_config_id=baseline_config_id,
            k=args.k,
            selection_metric=selection_metric,
            min_delta=args.min_delta,
            confidence_level=args.confidence_level,
            n_resamples=args.bootstrap_resamples,
            random_state=_stable_seed(args.test_random_state, selected_config_id),
            cluster_column=None,
            require_matching_status=False,
        )[selected_config_id]
    deployment = catalog_deployment_decision(
        selection=selection,
        final_delta=final_delta,
        baseline_config_id=baseline_config_id,
        selection_metric=selection_metric,
        min_delta=args.min_delta,
    )

    save_catalog_benchmark_artifact(
        args.output,
        clean_dataset=clean_dataset,
        fixed_test_holdouts=fixed_test.test_holdout_ratings,
        configs=configs,
        baseline_config_id=baseline_config_id,
        weights=weights,
        validation_repeats=validation_repeats,
        validation_rows=validation_rows,
        validation_aggregate=validation_aggregate,
        validation_deltas=validation_deltas,
        selection=selection,
        final_evaluated=final_evaluated,
        final_rows=final_rows,
        final_aggregate=final_aggregate,
        final_delta=final_delta,
        deployment=deployment,
        args=args,
    )

    selected = config_by_id[selected_config_id]
    print("protocol: fixed outer test plus repeated catalog validation")
    print("relevance_unit: canonical_work")
    print("selection_partition: validation")
    print(f"validation_repetitions: {args.repetitions}")
    print(f"fixed_test_holdouts: {len(fixed_test.test_holdout_ratings):,}")
    print(f"baseline_config: {baseline_config_id}")
    print(f"selected_config: {selected_config_id}")
    print(f"selection_decision: {selection['decision']}")
    print(
        "selected_catalog: interaction="
        f"{selected['max_interaction_books']:,}, content={selected['max_content_books']:,}"
    )
    print(f"deployment_config: {deployment['deployment_config_id']}")
    print(f"deployment_decision: {deployment['decision']}")
    print(f"selection_metric: {selection_metric}")
    print(f"selection_confidence_level: {selection_confidence_level:.4f}")
    print(f"saved_benchmark: {args.output.resolve()}")


def evaluate_catalog_configurations(
    *,
    clean_dataset: BookDataset,
    train_ratings: pd.DataFrame,
    holdouts: pd.DataFrame,
    configs: list[dict[str, object]],
    args: argparse.Namespace,
    split_id: str,
    split_seed: int,
    partition: str,
    execution_offset: int,
) -> dict[str, object]:
    """Fit every catalog on identical holdouts and attach fair common-reference diagnostics."""

    ordered = list(configs)
    offset = int(execution_offset) % len(ordered)
    execution_order = [*ordered[offset:], *ordered[:offset]]
    evaluated_by_id: dict[str, dict[str, object]] = {}
    datasets_by_id: dict[str, BookDataset] = {}
    rows: list[dict[str, object]] = []
    latency_samples_by_config: dict[str, list[float]] = {}
    for config in execution_order:
        config_id = str(config["config_id"])

        def prepare_catalog() -> BookDataset:
            return prepare_work_level_evaluation_train_dataset(
                clean_dataset,
                train_ratings,
                min_user_ratings=args.min_user_ratings,
                min_book_ratings=args.min_book_ratings,
                max_books=int(config["max_interaction_books"]),
                max_ratings=args.max_ratings,
                max_content_books=int(config["max_content_books"]),
                random_state=split_seed,
            )

        with threadpool_limits(limits=1):
            train_dataset, preparation_ms = timed_call(prepare_catalog)
        evaluated = evaluate_hybrid_weight_configurations(
            train_dataset=train_dataset,
            holdouts=holdouts,
            configs=[config],
            args=args,
            split_id=split_id,
            split_seed=split_seed,
            partition=partition,
            collect_resources=True,
            preparation_seconds=preparation_ms / 1_000.0,
        )
        enrich_catalog_evaluation(
            evaluated,
            train_dataset=train_dataset,
            config=config,
            split_id=split_id,
            k=args.k,
        )
        evaluated_by_id[config_id] = evaluated
        datasets_by_id[config_id] = train_dataset
        rows.extend(evaluated["rows"])
        latency_samples_by_config[config_id] = evaluated["latency_samples_by_config"][
            config_id
        ]

    reference_context = build_common_catalog_reference_context(
        clean_dataset=clean_dataset,
        train_ratings=train_ratings,
        datasets=list(datasets_by_id.values()),
    )
    configurations: list[dict[str, object]] = []
    for config in configs:
        config_id = str(config["config_id"])
        evaluated = evaluated_by_id[config_id]
        recommendation_lists = evaluated["recommendation_lists_by_config"][config_id]
        configuration = evaluated["configurations"][0]
        diagnostics = recommendation_diagnostics_at_k(
            recommendation_lists,
            catalog_items=reference_context["catalog_items"],
            popularity_counts=reference_context["popularity_counts"],
            k=args.k,
            item_features=reference_context["item_features"],
            item_to_index=reference_context["item_to_index"],
        )
        diagnostics["unique_recommended_works_at_k"] = int(
            len({work for recommendations in recommendation_lists for work in recommendations[: args.k]})
        )
        diagnostics["coverage_vs_common_reference_at_k"] = float(
            diagnostics[f"catalog_coverage_at_{args.k}"]
        )
        configuration["common_reference_diagnostics"] = diagnostics
        configurations.append(configuration)

    return {
        "rows": rows,
        "configurations": configurations,
        "latency_samples_by_config": latency_samples_by_config,
        "execution_order": [str(config["config_id"]) for config in execution_order],
    }


def evaluate_hybrid_weight_configurations(
    *,
    train_dataset: BookDataset,
    holdouts: pd.DataFrame,
    configs: list[dict[str, object]],
    args: argparse.Namespace,
    split_id: str,
    split_seed: int,
    partition: str,
    collect_resources: bool = False,
    preparation_seconds: float = 0.0,
) -> dict[str, object]:
    """Fit once, score every requested weight configuration, and retain paired rows."""

    def fit_recommender() -> HybridRecommender:
        return HybridRecommender(
            n_neighbors=args.neighbors,
            dynamic_weights=not args.disable_hybrid_dynamic_weights,
            positive_rating_threshold=args.positive_rating_threshold,
        ).fit(train_dataset.books, train_dataset.ratings)

    if collect_resources:
        with threadpool_limits(limits=1):
            hybrid_recommender, fit_elapsed_ms = timed_call(fit_recommender)
        model_proxy, model_structure = catalog_model_resource_profile(hybrid_recommender)
    else:
        hybrid_recommender = fit_recommender()
        fit_elapsed_ms = 0.0
        model_proxy = {}
        model_structure = {}
    work_context = build_work_evaluation_context(train_dataset, hybrid_recommender.content_model)
    catalog_work_set = set(work_context["catalog_items"])
    history_user_set = set(train_dataset.ratings["user_id"].astype(int))
    positive_user_counts = positive_work_counts(
        train_dataset.ratings,
        work_context["isbn_to_work"],
        threshold=args.positive_rating_threshold,
    )
    positive_user_set = set(positive_user_counts.index.astype(int))
    availability = summarize_holdout_availability(
        holdouts,
        candidate_items=catalog_work_set,
        history_users=history_user_set,
        item_column="work_key",
    )
    availability_payload = availability_with_personalization(
        availability,
        holdouts,
        catalog_work_set=catalog_work_set,
        positive_user_set=positive_user_set,
    )
    train_user_counts = train_dataset.ratings["user_id"].value_counts()
    all_rows: list[dict[str, object]] = []
    configurations = []
    recommendation_lists_by_config: dict[str, list[list[str]]] = {}
    latency_samples_by_config: dict[str, list[float]] = {}
    cohort_hash = benchmark_holdout_cohort_hash(holdouts, split_id=split_id)
    for config in configs:
        config_id = str(config["config_id"])
        weights = {
            "collaborative_weight": float(config["collaborative_weight"]),
            "content_weight": float(config["content_weight"]),
            "popularity_weight": float(config["popularity_weight"]),
        }
        metric_rows: list[dict[str, object]] = []
        recommendation_lists = []
        latency_samples: list[float] = []

        def recommend(user_id: int):
            return recommend_hybrid_with_weights(
                hybrid_recommender,
                user_id=user_id,
                top_n=args.k,
                candidate_pool_size=args.hybrid_candidate_pool_size,
                dynamic_weights=not args.disable_hybrid_dynamic_weights,
                allow_popularity_fallback=True,
                **weights,
            )

        if collect_resources and len(holdouts):
            with threadpool_limits(limits=1):
                recommend(int(holdouts.iloc[0]["user_id"]))
        for holdout in holdouts.itertuples(index=False):
            user_id = int(holdout.user_id)
            candidate_available = str(holdout.work_key) in catalog_work_set
            history_available = user_id in history_user_set
            positive_seed_count = int(positive_user_counts.get(user_id, 0))
            positive_seed_available = user_id in positive_user_set
            if collect_resources:
                with threadpool_limits(limits=1):
                    recommendations, elapsed_ms = timed_call(recommend, user_id)
                latency_samples.append(elapsed_ms)
            else:
                recommendations = recommend(user_id)
            recommended = recommendation_work_keys(recommendations, work_context["isbn_to_work"])
            row: dict[str, object] = {
                "split_id": split_id,
                "split_seed": int(split_seed),
                "partition": partition,
                "config_id": config_id,
                "model": "hybrid",
                "user_id": user_id,
                "user_segment": interaction_segment(positive_seed_count),
                "train_interactions": int(train_user_counts.get(user_id, 0)),
                "positive_seed_count": positive_seed_count,
                "holdout_isbn": str(holdout.isbn),
                "holdout_work_key": str(holdout.work_key),
                "candidate_available": candidate_available,
                "history_available": history_available,
                "positive_seed_available": positive_seed_available,
                "evaluation_status": holdout_status(
                    candidate_available,
                    history_available,
                    positive_seed_available,
                ),
                "recommendation_count": int(len(recommended)),
                "list_fill_rate": float(min(len(recommended), args.k) / args.k),
                **ranking_metrics_at_k(recommended, [str(holdout.work_key)], args.k),
            }
            metric_rows.append(row)
            recommendation_lists.append(recommended)
        conditional_rows = [
            row
            for row in metric_rows
            if row["candidate_available"] and row["positive_seed_available"]
        ]
        confidence_intervals = {
            "end_to_end": {
                metric: interval.to_dict()
                for metric, interval in bootstrap_ranking_metrics_at_k(
                    metric_rows,
                    k=args.k,
                    sampled_holdouts=len(holdouts),
                    confidence_level=args.confidence_level,
                    n_resamples=args.bootstrap_resamples,
                    random_state=_stable_seed(split_seed, config_id),
                ).items()
            },
            "conditional": (
                {
                    metric: interval.to_dict()
                    for metric, interval in bootstrap_ranking_metrics_at_k(
                        conditional_rows,
                        k=args.k,
                        confidence_level=args.confidence_level,
                        n_resamples=args.bootstrap_resamples,
                        random_state=_stable_seed(split_seed, f"{config_id}:conditional"),
                    ).items()
                }
                if conditional_rows
                else {}
            ),
        }
        configuration = {
                "config_id": config_id,
                **weights,
                "dynamic_weights": not args.disable_hybrid_dynamic_weights,
                "sampled_holdouts": int(len(metric_rows)),
                "conditional_holdouts": int(len(conditional_rows)),
                "end_to_end_metrics": mean_ranking_metrics_at_k(metric_rows, k=args.k),
                "conditional_metrics": mean_ranking_metrics_at_k(conditional_rows, k=args.k),
                "confidence_intervals": confidence_intervals,
                "diagnostics": recommendation_diagnostics_at_k(
                    recommendation_lists,
                    catalog_items=work_context["catalog_items"],
                    popularity_counts=work_context["popularity_counts"],
                    k=args.k,
                    item_features=work_context["item_features"],
                    item_to_index=work_context["item_to_index"],
                ),
            }
        if collect_resources:
            configuration["resources"] = catalog_resource_metrics(
                preparation_seconds=preparation_seconds,
                fit_seconds=fit_elapsed_ms / 1_000.0,
                latency_samples_ms=latency_samples,
                cohort_hash=cohort_hash,
                model_proxy=model_proxy,
            )
            configuration["model_structure"] = model_structure
        configurations.append(configuration)
        recommendation_lists_by_config[config_id] = recommendation_lists
        latency_samples_by_config[config_id] = latency_samples
        all_rows.extend(metric_rows)
    return {
        "rows": all_rows,
        "configurations": configurations,
        "availability": availability_payload,
        "recommendation_lists_by_config": recommendation_lists_by_config,
        "latency_samples_by_config": latency_samples_by_config,
        "work_context": work_context,
    }


def enrich_catalog_evaluation(
    evaluated: dict[str, object],
    *,
    train_dataset: BookDataset,
    config: dict[str, object],
    split_id: str,
    k: int,
) -> None:
    """Attach catalog reach and actual-size evidence to a single-config evaluation."""

    work_context = evaluated["work_context"]
    isbn_to_work = work_context["isbn_to_work"]
    interaction_work_set = {
        str(work_key)
        for work_key in train_dataset.ratings["isbn"].astype(str).map(isbn_to_work).dropna()
    }
    content_work_set = set(work_context["catalog_items"])
    for row in evaluated["rows"]:
        target_work = str(row["holdout_work_key"])
        row["target_in_interaction_catalog"] = target_work in interaction_work_set
        row["target_in_content_catalog"] = target_work in content_work_set

    configuration = evaluated["configurations"][0]
    configuration.update(
        {
            "split_id": split_id,
            "max_interaction_books": int(config["max_interaction_books"]),
            "max_content_books": int(config["max_content_books"]),
            "actual_catalog": {
                "interaction_books": int(train_dataset.ratings["isbn"].nunique()),
                "interaction_works": int(len(interaction_work_set)),
                "content_books": int(len(train_dataset.books)),
                "content_works": int(len(content_work_set)),
                "train_ratings": int(len(train_dataset.ratings)),
            },
            "availability": catalog_availability_from_rows(
                evaluated["rows"],
                k=k,
            ),
            "training_fingerprint": fingerprint_dataset(
                train_dataset.books,
                train_dataset.ratings,
                train_dataset.users,
            ).fingerprint,
        }
    )


def catalog_availability_from_rows(
    rows: list[dict[str, object]],
    *,
    k: int,
) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    sampled = int(len(frame))
    if sampled == 0:
        raise ValueError("Catalog evaluation requires at least one sampled holdout.")
    interaction = frame["target_in_interaction_catalog"].astype(bool)
    content = frame["target_in_content_catalog"].astype(bool)
    history = frame["history_available"].astype(bool)
    positive = frame["positive_seed_available"].astype(bool)
    personalized = content & positive
    content_only = content & ~interaction
    recommendation_counts = frame["recommendation_count"].astype(int)

    def count_rate(mask: pd.Series) -> tuple[int, float]:
        count = int(mask.sum())
        return count, float(count / sampled)

    interaction_count, interaction_rate = count_rate(interaction)
    content_count, content_rate = count_rate(content)
    content_only_count, content_only_rate = count_rate(content_only)
    history_count, history_rate = count_rate(history)
    positive_count, positive_rate = count_rate(positive)
    personalized_count, personalized_rate = count_rate(personalized)
    full_count, full_rate = count_rate(recommendation_counts >= k)
    empty_count, empty_rate = count_rate(recommendation_counts == 0)
    return {
        "sampled_holdouts": sampled,
        "target_in_interaction_catalog": interaction_count,
        "interaction_retention_rate": interaction_rate,
        "target_in_content_catalog": content_count,
        "content_retention_rate": content_rate,
        "content_only_targets": content_only_count,
        "content_only_retention_rate": content_only_rate,
        "history_available_holdouts": history_count,
        "history_retention_rate": history_rate,
        "positive_seed_available_holdouts": positive_count,
        "positive_seed_retention_rate": positive_rate,
        "personalization_evaluable_holdouts": personalized_count,
        "personalization_retention_rate": personalized_rate,
        "full_recommendation_lists": full_count,
        "full_list_rate": full_rate,
        "empty_recommendation_lists": empty_count,
        "empty_list_rate": empty_rate,
        "mean_list_fill_rate": float(recommendation_counts.clip(upper=k).mean() / k),
        "status_counts": {
            str(status): int(count)
            for status, count in frame["evaluation_status"].value_counts().sort_index().items()
        },
    }


def build_common_catalog_reference_context(
    *,
    clean_dataset: BookDataset,
    train_ratings: pd.DataFrame,
    datasets: list[BookDataset],
) -> dict[str, object]:
    """Fit one train-only metadata reference so discovery metrics share a denominator."""

    union_isbns = {
        str(isbn)
        for dataset in datasets
        for isbn in dataset.books["isbn"].astype(str)
    }
    reference_books = clean_dataset.books.loc[
        clean_dataset.books["isbn"].astype(str).isin(union_isbns)
    ].reset_index(drop=True)
    reference_content = ContentBasedRecommender().fit(reference_books)
    isbn_to_work = {
        str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
        for row in clean_dataset.books[
            ["isbn", "book_title", "book_author"]
        ].itertuples(index=False)
    }
    catalog_items: list[str] = []
    representative_indices: list[int] = []
    seen: set[str] = set()
    for index, row in enumerate(
        reference_books[["isbn", "book_title", "book_author"]].itertuples(index=False)
    ):
        work_key = book_work_key(row.book_title, row.book_author, row.isbn)
        if work_key in seen:
            continue
        seen.add(work_key)
        catalog_items.append(work_key)
        representative_indices.append(index)
    rating_work_keys = train_ratings["isbn"].astype(str).map(isbn_to_work).dropna()
    return {
        "catalog_items": catalog_items,
        "popularity_counts": rating_work_keys.value_counts(),
        "item_features": reference_content.matrix[representative_indices],
        "item_to_index": {
            work_key: index for index, work_key in enumerate(catalog_items)
        },
    }


def benchmark_holdout_cohort_hash(
    holdouts: pd.DataFrame,
    *,
    split_id: str | None,
) -> str:
    columns = ["user_id", "holdout_work_key"]
    frame = holdouts.copy()
    if "holdout_work_key" not in frame.columns and "work_key" in frame.columns:
        frame = frame.rename(columns={"work_key": "holdout_work_key"})
    if split_id is None:
        if "split_id" not in frame.columns:
            raise ValueError("A split_id column is required for a multi-split cohort hash.")
        columns.insert(0, "split_id")
    else:
        frame["split_id"] = split_id
        columns.insert(0, "split_id")
    records = (
        frame[columns]
        .drop_duplicates()
        .sort_values(columns, kind="stable")
        .to_dict("records")
    )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def catalog_model_resource_profile(
    recommender: HybridRecommender,
) -> tuple[dict[str, int], dict[str, object]]:
    structure = hybrid_sparse_payload(recommender)
    dense_array_bytes = int(getattr(recommender.popularity_scores, "nbytes", 0))
    content_bytes = int(structure["content"]["payload_bytes"])
    collaborative_bytes = int(structure["collaborative"]["payload_bytes"])
    proxy = {
        "serialized_model_bytes": model_serialized_size_bytes(recommender),
        "matrix_payload_bytes": content_bytes + collaborative_bytes + dense_array_bytes,
        "content_matrix_bytes": content_bytes,
        "collaborative_matrix_bytes": collaborative_bytes,
        "dense_array_bytes": dense_array_bytes,
    }
    structure = {**structure, "dense_array_bytes": dense_array_bytes}
    return proxy, structure


def _timing_resource(samples: list[float]) -> dict[str, object]:
    if not samples:
        raise ValueError("Timing resources require at least one sample.")
    values = pd.Series(samples, dtype="float64")
    return {
        "trial_count": int(len(values)),
        "seconds": [float(value) for value in values],
        "median_seconds": float(values.quantile(0.50)),
        "p95_seconds": float(values.quantile(0.95)),
    }


def catalog_resource_metrics(
    *,
    preparation_seconds: float,
    fit_seconds: float,
    latency_samples_ms: list[float],
    cohort_hash: str,
    model_proxy: dict[str, int],
    warmup_calls: int = 1,
) -> dict[str, object]:
    latency = summarize_latency_ms(latency_samples_ms)
    payload = {
        "preparation": _timing_resource([float(preparation_seconds)]),
        "fit": _timing_resource([float(fit_seconds)]),
        "warm_inference": {
            "warmup_calls": int(warmup_calls),
            "measured_calls": int(latency["sample_count"]),
            "cohort_hash": cohort_hash,
            "mean_ms": float(latency["mean_ms"]),
            "p50_ms": float(latency["p50_ms"]),
            "p95_ms": float(latency["p95_ms"]),
            "max_ms": float(latency["max_ms"]),
        },
        "model_proxy": {key: int(value) for key, value in model_proxy.items()},
    }
    return validate_resource_metrics(payload)


def aggregate_catalog_configurations(
    rows: list[dict[str, object]],
    *,
    configs: list[dict[str, object]],
    configuration_records: list[dict[str, object]],
    latency_samples_by_config: dict[str, list[float]],
    k: int,
) -> list[dict[str, object]]:
    base_records = aggregate_benchmark_configurations(rows, configs=configs, k=k)
    row_frame = pd.DataFrame(rows)
    by_config = {
        str(config["config_id"]): config
        for config in configs
    }
    result = []
    for base in base_records:
        config_id = str(base["config_id"])
        config = by_config[config_id]
        records = [
            record
            for record in configuration_records
            if str(record["config_id"]) == config_id
        ]
        config_rows = row_frame.loc[row_frame["config_id"] == config_id].copy()
        preparation = [
            float(record["resources"]["preparation"]["seconds"][0])
            for record in records
        ]
        fit = [
            float(record["resources"]["fit"]["seconds"][0])
            for record in records
        ]
        proxy_fields = records[0]["resources"]["model_proxy"].keys()
        model_proxy = {
            field: max(int(record["resources"]["model_proxy"][field]) for record in records)
            for field in proxy_fields
        }
        resources = catalog_resource_metrics(
            preparation_seconds=preparation[0],
            fit_seconds=fit[0],
            latency_samples_ms=latency_samples_by_config[config_id],
            cohort_hash=benchmark_holdout_cohort_hash(config_rows, split_id=None),
            model_proxy=model_proxy,
            warmup_calls=len(records),
        )
        resources["preparation"] = _timing_resource(preparation)
        resources["fit"] = _timing_resource(fit)
        resources = validate_resource_metrics(resources)
        base.update(
            {
                "max_interaction_books": int(config["max_interaction_books"]),
                "max_content_books": int(config["max_content_books"]),
                "availability": catalog_availability_from_rows(
                    config_rows.to_dict("records"),
                    k=k,
                ),
                "resources": resources,
                "actual_catalog_trials": [record["actual_catalog"] for record in records],
                "config_relative_diagnostics": mean_numeric_mappings(
                    [record["diagnostics"] for record in records]
                ),
                "common_reference_diagnostics": mean_numeric_mappings(
                    [record["common_reference_diagnostics"] for record in records]
                ),
                "model_structure_trials": [record["model_structure"] for record in records],
            }
        )
        result.append(base)
    return result


def mean_numeric_mappings(values: list[dict[str, object]]) -> dict[str, float]:
    if not values:
        return {}
    common_keys = set.intersection(*(set(value) for value in values))
    return {
        key: float(sum(float(value[key]) for value in values) / len(values))
        for key in sorted(common_keys)
        if all(
            isinstance(value[key], (int, float)) and not isinstance(value[key], bool)
            for value in values
        )
    }


def select_catalog_configuration(
    aggregate: list[dict[str, object]],
    *,
    validation_deltas: dict[str, dict[str, object]],
    baseline_config_id: str,
    selection_metric: str,
    args: argparse.Namespace,
    confidence_level: float,
    multiplicity_adjustment: str,
) -> dict[str, object]:
    """Promote only when quality, reach, and validation resource guardrails all pass."""

    by_config = {str(record["config_id"]): record for record in aggregate}
    baseline = by_config[baseline_config_id]
    ranked_ids = sorted(
        by_config,
        key=lambda config_id: (
            -float(by_config[config_id]["end_to_end_metrics"][selection_metric]),
            int(by_config[config_id]["max_content_books"]),
            config_id,
        ),
    )
    evidence_rows = []
    selected_config_id = baseline_config_id
    for config_id in ranked_ids:
        if config_id == baseline_config_id:
            continue
        candidate = by_config[config_id]
        delta_evidence = validation_deltas[config_id]
        interval = delta_evidence["end_to_end"][selection_metric]
        quality_lower = float(interval["lower"])
        selection_rate = float(delta_evidence["selection_rate"])
        content_delta = float(
            candidate["availability"]["content_retention_rate"]
            - baseline["availability"]["content_retention_rate"]
        )
        personalization_delta = float(
            candidate["availability"]["personalization_retention_rate"]
            - baseline["availability"]["personalization_retention_rate"]
        )
        latency_ratio = positive_resource_ratio(
            candidate["resources"]["warm_inference"]["p95_ms"],
            baseline["resources"]["warm_inference"]["p95_ms"],
        )
        model_ratio = positive_resource_ratio(
            candidate["resources"]["model_proxy"]["serialized_model_bytes"],
            baseline["resources"]["model_proxy"]["serialized_model_bytes"],
        )
        passes_quality = quality_lower > args.min_delta and selection_rate >= args.min_selection_rate
        passes_reach = (
            content_delta >= args.min_content_retention_delta
            and personalization_delta >= args.min_personalization_retention_delta
        )
        passes_resources = (
            latency_ratio <= args.max_latency_ratio
            and model_ratio <= args.max_serialized_model_ratio
        )
        passes = passes_quality and passes_reach and passes_resources
        evidence_rows.append(
            {
                "config_id": config_id,
                "primary_metric": float(candidate["end_to_end_metrics"][selection_metric]),
                "paired_delta_lower": quality_lower,
                "selection_rate": selection_rate,
                "content_retention_delta": content_delta,
                "personalization_retention_delta": personalization_delta,
                "p95_latency_ratio": latency_ratio,
                "serialized_model_ratio": model_ratio,
                "passes_quality": passes_quality,
                "passes_reach": passes_reach,
                "passes_resources": passes_resources,
                "passes": passes,
            }
        )
        if selected_config_id == baseline_config_id and passes:
            selected_config_id = config_id

    return {
        "partition": "validation",
        "scope": "end_to_end",
        "metric": selection_metric,
        "baseline_config_id": baseline_config_id,
        "selected_config_id": selected_config_id,
        "decision": (
            "promote_to_fixed_test"
            if selected_config_id != baseline_config_id
            else "retain_baseline"
        ),
        "min_delta": float(args.min_delta),
        "min_selection_rate": float(args.min_selection_rate),
        "min_content_retention_delta": float(args.min_content_retention_delta),
        "min_personalization_retention_delta": float(
            args.min_personalization_retention_delta
        ),
        "max_latency_ratio": float(args.max_latency_ratio),
        "max_serialized_model_ratio": float(args.max_serialized_model_ratio),
        "confidence_level": float(confidence_level),
        "multiplicity_adjustment": multiplicity_adjustment,
        "ranked_validation_config_ids": ranked_ids,
        "candidate_evidence": evidence_rows,
        "decision_rule": (
            "paired lower CI and repeat selection rate, non-decreasing reach, and validation "
            "latency/model-size ratios must all pass"
        ),
    }


def positive_resource_ratio(candidate: object, baseline: object) -> float:
    candidate_value = float(candidate)
    baseline_value = float(baseline)
    if baseline_value <= 0.0:
        return 1.0 if candidate_value <= 0.0 else 1e308
    return float(candidate_value / baseline_value)


def catalog_deployment_decision(
    *,
    selection: dict[str, object],
    final_delta: dict[str, object] | None,
    baseline_config_id: str,
    selection_metric: str,
    min_delta: float,
) -> dict[str, object]:
    selected_config_id = str(selection["selected_config_id"])
    if selected_config_id == baseline_config_id:
        return {
            "partition": "fixed_test",
            "deployment_config_id": baseline_config_id,
            "decision": "retain_baseline",
            "reason": "No challenger passed all validation promotion gates.",
        }
    interval = final_delta["end_to_end"][selection_metric]
    passes = float(interval["lower"]) > float(min_delta)
    return {
        "partition": "fixed_test",
        "validation_selected_config_id": selected_config_id,
        "deployment_config_id": selected_config_id if passes else baseline_config_id,
        "decision": "promote" if passes else "retain_baseline_after_test",
        "metric": selection_metric,
        "paired_delta_mean": float(interval["mean"]),
        "paired_delta_lower": float(interval["lower"]),
        "min_delta": float(min_delta),
        "passes": passes,
        "reason": (
            "Fixed-test paired lower confidence bound cleared the deployment threshold."
            if passes
            else "Fixed-test paired lower confidence bound did not clear the threshold."
        ),
    }


def aggregate_benchmark_configurations(
    rows: list[dict[str, object]],
    *,
    configs: list[dict[str, object]],
    k: int,
) -> list[dict[str, object]]:
    """Aggregate each configuration as a macro average over repeated split means."""

    metric_names = list(ranking_metrics_at_k([], [], k).keys())
    frame = pd.DataFrame(rows)
    results = []
    for config in configs:
        config_id = str(config["config_id"])
        config_rows = frame.loc[frame["config_id"] == config_id].copy()
        split_ids = sorted(config_rows["split_id"].drop_duplicates().tolist())
        per_split = []
        for split_id in split_ids:
            split_rows = config_rows.loc[config_rows["split_id"] == split_id]
            conditional = split_rows.loc[
                split_rows["candidate_available"] & split_rows["positive_seed_available"]
            ]
            per_split.append(
                {
                    "split_id": str(split_id),
                    "sampled_holdouts": int(len(split_rows)),
                    "conditional_holdouts": int(len(conditional)),
                    "end_to_end_metrics": {
                        metric: float(split_rows[metric].mean()) for metric in metric_names
                    },
                    "conditional_metrics": {
                        metric: float(conditional[metric].mean()) if len(conditional) else 0.0
                        for metric in metric_names
                    },
                }
            )
        results.append(
            {
                "config_id": config_id,
                "collaborative_weight": float(config["collaborative_weight"]),
                "content_weight": float(config["content_weight"]),
                "popularity_weight": float(config["popularity_weight"]),
                "repetitions": int(len(per_split)),
                "sampled_holdouts": int(len(config_rows)),
                "conditional_holdouts": int(
                    (config_rows["candidate_available"] & config_rows["positive_seed_available"]).sum()
                ),
                "end_to_end_metrics": {
                    metric: float(
                        sum(split["end_to_end_metrics"][metric] for split in per_split)
                        / len(per_split)
                    )
                    for metric in metric_names
                },
                "conditional_metrics": {
                    metric: float(
                        sum(split["conditional_metrics"][metric] for split in per_split)
                        / len(per_split)
                    )
                    for metric in metric_names
                },
                "per_split": per_split,
            }
        )
    return results


def paired_benchmark_deltas(
    rows: list[dict[str, object]],
    *,
    configs: list[dict[str, object]],
    baseline_config_id: str,
    k: int,
    selection_metric: str,
    min_delta: float,
    confidence_level: float,
    n_resamples: int,
    random_state: int,
    cluster_column: str | None,
    require_matching_status: bool = True,
) -> dict[str, dict[str, object]]:
    """Return paired baseline deltas and repeat-level win evidence for every challenger."""

    frame = pd.DataFrame(rows)
    baseline_rows = frame.loc[frame["config_id"] == baseline_config_id].to_dict("records")
    result: dict[str, dict[str, object]] = {}
    for config in configs:
        config_id = str(config["config_id"])
        if config_id == baseline_config_id:
            continue
        challenger_rows = frame.loc[frame["config_id"] == config_id].to_dict("records")
        end_to_end = bootstrap_paired_ranking_deltas_at_k(
            baseline_rows,
            challenger_rows,
            k=k,
            scope="end_to_end",
            cluster_column=cluster_column,
            require_matching_status=require_matching_status,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=_stable_seed(random_state, f"{baseline_config_id}:{config_id}:end_to_end"),
        )
        conditional = {}
        try:
            conditional = {
                metric: interval.to_dict()
                for metric, interval in bootstrap_paired_ranking_deltas_at_k(
                    baseline_rows,
                    challenger_rows,
                    k=k,
                    scope="conditional",
                    cluster_column=cluster_column,
                    require_matching_status=require_matching_status,
                    confidence_level=confidence_level,
                    n_resamples=n_resamples,
                    random_state=_stable_seed(
                        random_state,
                        f"{baseline_config_id}:{config_id}:conditional",
                    ),
                ).items()
            }
        except ValueError as error:
            if "No paired rows" not in str(error):
                raise
        per_repeat = paired_repeat_deltas(
            baseline_rows,
            challenger_rows,
            metric=selection_metric,
        )
        result[config_id] = {
            "config_id": config_id,
            "end_to_end": {
                metric: interval.to_dict() for metric, interval in end_to_end.items()
            },
            "conditional": conditional,
            "primary_metric": selection_metric,
            "per_repeat_primary_delta": per_repeat,
            "selection_rate": float(
                sum(record["delta"] > min_delta for record in per_repeat) / len(per_repeat)
            ),
        }
    return result


def paired_repeat_deltas(
    baseline_rows: list[dict[str, object]],
    challenger_rows: list[dict[str, object]],
    *,
    metric: str,
) -> list[dict[str, object]]:
    baseline = pd.DataFrame(baseline_rows)
    challenger = pd.DataFrame(challenger_rows)
    grouped_baseline = baseline.groupby("split_id", sort=True)[metric].mean()
    grouped_challenger = challenger.groupby("split_id", sort=True)[metric].mean()
    if not grouped_baseline.index.equals(grouped_challenger.index):
        raise ValueError("Paired benchmark configurations do not share the same split IDs.")
    return [
        {
            "split_id": str(split_id),
            "delta": float(grouped_challenger.loc[split_id] - grouped_baseline.loc[split_id]),
        }
        for split_id in grouped_baseline.index
    ]


def select_benchmark_configuration(
    aggregate: list[dict[str, object]],
    *,
    validation_deltas: dict[str, dict[str, object]],
    baseline_config_id: str,
    selection_metric: str,
    min_delta: float,
    min_selection_rate: float,
    confidence_level: float,
    multiplicity_adjustment: str,
) -> dict[str, object]:
    """Promote only a validation config whose paired evidence clears both safeguards."""

    by_config = {str(record["config_id"]): record for record in aggregate}
    if baseline_config_id not in by_config:
        raise ValueError("The baseline configuration is missing from validation aggregates.")
    ranked_config_ids = sorted(
        by_config,
        key=lambda config_id: (
            -float(by_config[config_id]["end_to_end_metrics"][selection_metric]),
            config_id,
        ),
    )
    candidates = []
    for config_id in ranked_config_ids:
        if config_id == baseline_config_id:
            continue
        evidence = validation_deltas[config_id]
        interval = evidence["end_to_end"][selection_metric]
        lower = float(interval["lower"])
        selection_rate = float(evidence["selection_rate"])
        passes = lower > min_delta and selection_rate >= min_selection_rate
        candidates.append(
            {
                "config_id": config_id,
                "primary_metric": float(
                    by_config[config_id]["end_to_end_metrics"][selection_metric]
                ),
                "paired_delta_lower": lower,
                "selection_rate": selection_rate,
                "passes": passes,
            }
        )
        if passes:
            return {
                "partition": "validation",
                "scope": "end_to_end",
                "metric": selection_metric,
                "baseline_config_id": baseline_config_id,
                "selected_config_id": config_id,
                "decision": "promote",
                "min_delta": float(min_delta),
                "min_selection_rate": float(min_selection_rate),
                "confidence_level": float(confidence_level),
                "multiplicity_adjustment": multiplicity_adjustment,
                "decision_rule": (
                    "paired bootstrap lower bound > min_delta and validation repeat "
                    "selection_rate >= min_selection_rate"
                ),
                "ranked_validation_config_ids": ranked_config_ids,
                "candidate_evidence": candidates,
            }
    return {
        "partition": "validation",
        "scope": "end_to_end",
        "metric": selection_metric,
        "baseline_config_id": baseline_config_id,
        "selected_config_id": baseline_config_id,
        "decision": "retain_baseline",
        "min_delta": float(min_delta),
        "min_selection_rate": float(min_selection_rate),
        "confidence_level": float(confidence_level),
        "multiplicity_adjustment": multiplicity_adjustment,
        "decision_rule": (
            "paired bootstrap lower bound > min_delta and validation repeat "
            "selection_rate >= min_selection_rate"
        ),
        "ranked_validation_config_ids": ranked_config_ids,
        "candidate_evidence": candidates,
    }


def load_prepared_dataset(args: argparse.Namespace):
    return prepare_dataset(
        data_dir=args.data_dir,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
        max_books=args.max_books,
        max_ratings=args.max_ratings,
    )


def load_content_catalog(args: argparse.Namespace):
    dataset = load_clean_dataset(args.data_dir)
    max_content_books = getattr(args, "max_content_books", None)
    if max_content_books is None or len(dataset.books) <= max_content_books:
        books = dataset.books
    else:
        popular_slots = max(1, int(max_content_books * 0.75))
        popular_isbns = (
            dataset.ratings["isbn"]
            .astype(str)
            .value_counts()
            .rename_axis("isbn")
            .reset_index(name="count")
            .sort_values(["count", "isbn"], ascending=[False, True], kind="stable")
            .head(popular_slots)["isbn"]
        )
        required = dataset.books[dataset.books["isbn"].isin(popular_isbns)]
        query_books = content_query_books(
            dataset.books,
            title=getattr(args, "title", None),
            isbn=getattr(args, "isbn", None),
        )
        required = pd.concat([required, query_books], ignore_index=True).drop_duplicates("isbn")
        books = select_content_catalog(
            dataset.books,
            required_books=required,
            max_content_books=max_content_books,
        )
    return BookDataset(
        books=books.reset_index(drop=True),
        ratings=dataset.ratings.reset_index(drop=True),
        users=dataset.users.reset_index(drop=True),
    )


def content_query_books(
    books,
    *,
    title: str | None,
    isbn: str | None,
):
    if isbn:
        return books[books["isbn"].astype(str).str.casefold() == str(isbn).casefold()].head(1)
    if not title:
        return books.iloc[0:0]
    normalized = title.casefold()
    exact = books[books["book_title"].astype(str).str.casefold() == normalized]
    if not exact.empty:
        return exact.head(1)
    return books[
        books["book_title"].astype(str).str.casefold().str.contains(normalized, regex=False)
    ].head(1)


def load_hybrid_prepared_dataset(args: argparse.Namespace):
    return prepare_hybrid_dataset(
        data_dir=args.data_dir,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
        max_books=args.max_books,
        max_ratings=args.max_ratings,
        max_content_books=getattr(args, "max_content_books", None),
        random_state=getattr(args, "random_state", 42),
    )


def save_training_bundle(
    recommender,
    output: Path,
    *,
    model_type: str,
    dataset,
    args: argparse.Namespace,
    metrics: dict | None = None,
):
    config_keys = (
        "min_user_ratings",
        "min_book_ratings",
        "max_books",
        "max_ratings",
        "max_content_books",
        "neighbors",
        "collaborative_weight",
        "content_weight",
        "popularity_weight",
        "positive_rating_threshold",
        "disable_dynamic_weights",
        "random_state",
    )
    training_config = {
        key: getattr(args, key)
        for key in config_keys
        if hasattr(args, key)
    }
    return save_model_bundle(
        recommender,
        output,
        model_type=model_type,
        dataset_fingerprint=fingerprint_dataset(
            dataset.books,
            dataset.ratings,
            dataset.users,
        ),
        training_config=training_config,
        metrics=metrics,
    )


def read_evaluation_metrics(
    path: Path | None,
    *,
    expected_source_dataset_fingerprint: str | None = None,
    expected_parameters: dict[str, object] | None = None,
) -> dict:
    if path is None:
        return {}
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read evaluation artifact {path}: {exc}") from exc

    _validate_finite_json_tree(payload, "evaluation artifact")
    if not isinstance(payload, dict):
        raise ValueError("Evaluation artifact root must be a JSON object.")
    required_contract = {
        "schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "evaluation",
        "protocol": EVALUATION_PROTOCOL,
        "evaluation_partition": "test",
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
    }
    for key, expected in required_contract.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Evaluation artifact {key!r} is {payload.get(key)!r}; expected {expected!r}."
            )

    parameters = payload.get("parameters")
    algorithms = payload.get("algorithms")
    dataset = payload.get("dataset")
    availability = payload.get("availability")
    summary = payload.get("summary")
    intervals = payload.get("confidence_intervals")
    per_user_rows = payload.get("per_user_rows")
    if not isinstance(parameters, dict):
        raise ValueError("Evaluation artifact parameters must be an object.")
    expected_algorithms = evaluation_algorithm_contract()
    if algorithms != expected_algorithms:
        raise ValueError("Evaluation artifact algorithm contract is incompatible.")
    if not isinstance(dataset, dict):
        raise ValueError("Evaluation artifact dataset must be an object.")
    if not isinstance(availability, dict):
        raise ValueError("Evaluation artifact availability must be an object.")
    if not isinstance(summary, dict):
        raise ValueError("Evaluation artifact summary must be an object.")
    if not isinstance(intervals, dict):
        raise ValueError("Evaluation artifact confidence_intervals must be an object.")
    if not isinstance(per_user_rows, list):
        raise ValueError("Evaluation artifact per_user_rows must be an array.")

    source_fingerprint = dataset.get("source_fingerprint")
    training_fingerprint = dataset.get("training_fingerprint")
    for name, fingerprint in (
        ("source_fingerprint", source_fingerprint),
        ("training_fingerprint", training_fingerprint),
    ):
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint.lower())
        ):
            raise ValueError(f"Evaluation artifact dataset.{name} is not a SHA-256 digest.")
    if (
        expected_source_dataset_fingerprint is not None
        and source_fingerprint != expected_source_dataset_fingerprint
    ):
        raise ValueError("Evaluation artifact was produced from a different source dataset.")
    for key, expected in (expected_parameters or {}).items():
        if key not in parameters or parameters[key] != expected:
            raise ValueError(
                f"Evaluation artifact parameter {key!r} is {parameters.get(key)!r}; "
                f"expected {expected!r}."
            )

    if not isinstance(summary.get("conditional"), list):
        raise ValueError("Evaluation artifact summary.conditional must be an array.")
    if not isinstance(summary.get("end_to_end"), list):
        raise ValueError("Evaluation artifact summary.end_to_end must be an array.")
    by_model = summary.get("by_model")
    if not isinstance(by_model, dict) or not isinstance(by_model.get("hybrid"), dict):
        raise ValueError("Evaluation artifact must contain a hybrid model summary.")
    hybrid_summary = by_model["hybrid"]
    if not isinstance(hybrid_summary.get("conditional_metrics"), dict) or not isinstance(
        hybrid_summary.get("end_to_end_metrics"), dict
    ):
        raise ValueError(
            "Evaluation artifact hybrid summary must contain conditional and end-to-end metrics."
        )
    _validate_ranking_metric_mapping(
        hybrid_summary["conditional_metrics"],
        "summary.by_model.hybrid.conditional_metrics",
    )
    _validate_ranking_metric_mapping(
        hybrid_summary["end_to_end_metrics"],
        "summary.by_model.hybrid.end_to_end_metrics",
    )

    return {
        "schema_version": payload["schema_version"],
        "protocol": payload["protocol"],
        "evaluation_partition": payload["evaluation_partition"],
        "relevance_unit": payload["relevance_unit"],
        "split_unit": payload["split_unit"],
        "source_dataset_fingerprint": source_fingerprint,
        "availability": availability,
        "summary": summary,
        "confidence_intervals": intervals,
    }


def read_benchmark_metrics(
    path: Path | None,
    *,
    expected_source_dataset_fingerprint: str | None = None,
    expected_parameters: dict[str, object] | None = None,
    expected_weights: dict[str, float] | None = None,
) -> dict:
    """Read a repeated-split benchmark only when its selected model is compatible."""

    if path is None:
        return {}
    try:
        raw_payload = path.read_bytes()
        payload = json.loads(raw_payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read benchmark artifact {path}: {exc}") from exc

    _validate_finite_json_tree(payload, "benchmark artifact")
    if not isinstance(payload, dict):
        raise ValueError("Benchmark artifact root must be a JSON object.")
    required_contract = {
        "schema_version": BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "repeated_split_benchmark",
        "protocol": BENCHMARK_PROTOCOL,
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
    }
    for key, expected in required_contract.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Benchmark artifact {key!r} is {payload.get(key)!r}; expected {expected!r}."
            )
    if payload.get("algorithms") != evaluation_algorithm_contract():
        raise ValueError("Benchmark artifact algorithm contract is incompatible.")

    parameters = payload.get("parameters")
    dataset = payload.get("dataset")
    selection = payload.get("selection")
    validation = payload.get("validation")
    fixed_test = payload.get("fixed_test")
    if not isinstance(parameters, dict):
        raise ValueError("Benchmark artifact parameters must be an object.")
    if not isinstance(dataset, dict):
        raise ValueError("Benchmark artifact dataset must be an object.")
    if not isinstance(selection, dict):
        raise ValueError("Benchmark artifact selection must be an object.")
    if not isinstance(validation, dict):
        raise ValueError("Benchmark artifact validation must be an object.")
    if not isinstance(fixed_test, dict):
        raise ValueError("Benchmark artifact fixed_test must be an object.")

    source_fingerprint = dataset.get("source_fingerprint")
    if not _is_sha256_digest(source_fingerprint):
        raise ValueError("Benchmark artifact dataset.source_fingerprint is not a SHA-256 digest.")
    if (
        expected_source_dataset_fingerprint is not None
        and source_fingerprint != expected_source_dataset_fingerprint
    ):
        raise ValueError("Benchmark artifact was produced from a different source dataset.")
    for key, expected in (expected_parameters or {}).items():
        if not _benchmark_parameter_matches(parameters.get(key), expected):
            raise ValueError(
                f"Benchmark artifact parameter {key!r} is {parameters.get(key)!r}; "
                f"expected {expected!r}."
            )

    k = _benchmark_positive_integer(parameters.get("k"), "parameters.k")
    repetitions = _benchmark_positive_integer(
        parameters.get("repetitions"),
        "parameters.repetitions",
    )
    if repetitions < 2:
        raise ValueError("Benchmark artifact parameters.repetitions must be at least 2.")
    selection_metric_name = parameters.get("selection_metric")
    if selection_metric_name not in {"precision", "recall", "hit_rate", "map", "mrr", "ndcg"}:
        raise ValueError("Benchmark artifact parameters.selection_metric is invalid.")
    selection_metric = f"{selection_metric_name}_at_{k}"
    min_delta = _benchmark_number(parameters.get("min_delta"), "parameters.min_delta")
    min_selection_rate = _benchmark_number(
        parameters.get("min_selection_rate"),
        "parameters.min_selection_rate",
    )
    if min_delta < 0.0 or not 0.0 <= min_selection_rate <= 1.0:
        raise ValueError("Benchmark artifact selection thresholds are invalid.")
    selection_confidence_level = _benchmark_number(
        parameters.get("selection_confidence_level"),
        "parameters.selection_confidence_level",
    )
    if not 0.0 < selection_confidence_level < 1.0:
        raise ValueError("Benchmark artifact selection confidence level is invalid.")
    multiplicity_adjustment = parameters.get("selection_multiplicity_adjustment")
    if multiplicity_adjustment not in {"none", "bonferroni_familywise"}:
        raise ValueError("Benchmark artifact selection multiplicity adjustment is invalid.")
    confidence_level = _benchmark_number(
        parameters.get("confidence_level"),
        "parameters.confidence_level",
    )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Benchmark artifact confidence level is invalid.")

    configs_by_id = _benchmark_config_registry(parameters.get("weight_configs"))
    baseline_config_id = selection.get("baseline_config_id")
    selected_config_id = selection.get("selected_config_id")
    if (
        selection.get("partition") != "validation"
        or selection.get("scope") != "end_to_end"
        or selection.get("metric") != selection_metric
        or not isinstance(baseline_config_id, str)
        or not isinstance(selected_config_id, str)
        or baseline_config_id not in configs_by_id
        or selected_config_id not in configs_by_id
    ):
        raise ValueError("Benchmark artifact selection contract is incompatible.")
    if parameters.get("baseline_config_id") != baseline_config_id:
        raise ValueError("Benchmark artifact baseline configuration is inconsistent.")
    decision = selection.get("decision")
    expected_decision = "retain_baseline" if selected_config_id == baseline_config_id else "promote"
    if decision != expected_decision:
        raise ValueError("Benchmark artifact selection decision is inconsistent.")
    if (
        not _benchmark_numbers_match(selection.get("min_delta"), min_delta)
        or not _benchmark_numbers_match(
            selection.get("min_selection_rate"),
            min_selection_rate,
        )
        or not _benchmark_numbers_match(
            selection.get("confidence_level"),
            selection_confidence_level,
        )
        or selection.get("multiplicity_adjustment") != multiplicity_adjustment
    ):
        raise ValueError("Benchmark artifact selection thresholds are inconsistent.")

    expected_config_ids = set(configs_by_id)
    validation_user_ids = _validate_benchmark_validation(
        validation,
        config_registry=configs_by_id,
        expected_config_ids=expected_config_ids,
        repetitions=repetitions,
        k=k,
    )
    deltas = validation["paired_deltas_vs_baseline"]
    challenger_ids = expected_config_ids.difference({baseline_config_id})
    if not isinstance(deltas, dict) or set(deltas) != challenger_ids:
        raise ValueError("Benchmark artifact validation paired deltas are incomplete.")
    passing_challengers = set()
    for challenger_id in challenger_ids:
        lower, selection_rate = _validate_benchmark_delta_evidence(
            deltas[challenger_id],
            config_id=challenger_id,
            metric=selection_metric,
            expected_confidence_level=selection_confidence_level,
        )
        if lower > min_delta and selection_rate >= min_selection_rate:
            passing_challengers.add(challenger_id)
    if selected_config_id == baseline_config_id:
        if passing_challengers:
            raise ValueError("Benchmark artifact retained a baseline despite promotable evidence.")
    elif selected_config_id not in passing_challengers:
        raise ValueError("Benchmark artifact selected a configuration without sufficient evidence.")

    selected = configs_by_id[selected_config_id]
    if expected_weights is not None:
        for key, expected in expected_weights.items():
            actual = selected.get(key)
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Benchmark artifact selected {key!r} is {actual!r}; expected {expected!r}."
                )

    expected_test_config_ids = [baseline_config_id]
    if selected_config_id != baseline_config_id:
        expected_test_config_ids.append(selected_config_id)
    test_user_ids = _validate_benchmark_fixed_test(
        fixed_test,
        config_registry=configs_by_id,
        expected_config_ids=expected_test_config_ids,
        k=k,
    )
    if not validation_user_ids.isdisjoint(test_user_ids):
        raise ValueError("Benchmark artifact reuses fixed-test users in validation evidence.")

    evaluated_config_ids = fixed_test.get("evaluated_config_ids")
    if (
        fixed_test.get("evaluated_after_selection") is not True
        or not isinstance(evaluated_config_ids, list)
        or evaluated_config_ids != expected_test_config_ids
    ):
        raise ValueError("Benchmark artifact fixed test is not restricted to baseline and selection.")
    final_delta = fixed_test.get("paired_delta_selected_vs_baseline")
    if selected_config_id == baseline_config_id:
        if final_delta is not None:
            raise ValueError("Benchmark artifact has an unnecessary fixed-test paired delta.")
    else:
        _validate_benchmark_delta_evidence(
            final_delta,
            config_id=selected_config_id,
            metric=selection_metric,
            expected_confidence_level=confidence_level,
        )

    return {
        "schema_version": payload["schema_version"],
        "protocol": payload["protocol"],
        "relevance_unit": payload["relevance_unit"],
        "split_unit": payload["split_unit"],
        "source_dataset_fingerprint": source_fingerprint,
        "artifact_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "selection": selection,
        "fixed_test": {
            "aggregate_by_config": fixed_test["aggregate_by_config"],
            "paired_delta_selected_vs_baseline": fixed_test.get(
                "paired_delta_selected_vs_baseline"
            ),
        },
    }


def read_catalog_benchmark_metrics(
    path: Path | None,
    *,
    expected_source_dataset_fingerprint: str | None = None,
    expected_parameters: dict[str, object] | None = None,
    expected_catalog: dict[str, int] | None = None,
    expected_weights: dict[str, float] | None = None,
) -> dict:
    """Verify catalog benchmark lineage before it can authorize a serving bundle."""

    if path is None:
        return {}
    try:
        raw_payload = path.read_bytes()
        payload = json.loads(raw_payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read catalog benchmark artifact {path}: {exc}") from exc
    _validate_finite_json_tree(payload, "catalog benchmark artifact")
    if not isinstance(payload, dict):
        raise ValueError("Catalog benchmark artifact root must be a JSON object.")
    required = {
        "schema_version": CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": CATALOG_BENCHMARK_ARTIFACT_TYPE,
        "protocol": CATALOG_BENCHMARK_PROTOCOL,
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Catalog benchmark artifact {key!r} is {payload.get(key)!r}; "
                f"expected {expected!r}."
            )
    if payload.get("algorithms") != evaluation_algorithm_contract():
        raise ValueError("Catalog benchmark algorithm contract is incompatible.")

    parameters = payload.get("parameters")
    dataset = payload.get("dataset")
    selection = payload.get("selection")
    deployment = payload.get("deployment")
    validation = payload.get("validation")
    fixed_test = payload.get("fixed_test")
    for name, value in (
        ("parameters", parameters),
        ("dataset", dataset),
        ("selection", selection),
        ("deployment", deployment),
        ("validation", validation),
        ("fixed_test", fixed_test),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"Catalog benchmark artifact {name} must be an object.")

    source_fingerprint = dataset.get("source_fingerprint")
    if not _is_sha256_digest(source_fingerprint):
        raise ValueError("Catalog benchmark source fingerprint is not a SHA-256 digest.")
    if (
        expected_source_dataset_fingerprint is not None
        and source_fingerprint != expected_source_dataset_fingerprint
    ):
        raise ValueError("Catalog benchmark was produced from a different source dataset.")
    for key, expected in (expected_parameters or {}).items():
        if not _benchmark_parameter_matches(parameters.get(key), expected):
            raise ValueError(
                f"Catalog benchmark parameter {key!r} is {parameters.get(key)!r}; "
                f"expected {expected!r}."
            )

    configs_by_id = catalog_config_registry(parameters.get("catalog_configs"))
    expected_config_ids = set(configs_by_id)
    baseline_config_id = selection.get("baseline_config_id")
    selected_config_id = selection.get("selected_config_id")
    if (
        not isinstance(baseline_config_id, str)
        or not isinstance(selected_config_id, str)
        or baseline_config_id not in configs_by_id
        or selected_config_id not in configs_by_id
        or parameters.get("baseline_config_id") != baseline_config_id
    ):
        raise ValueError("Catalog benchmark selection references an invalid configuration.")
    k = _benchmark_positive_integer(parameters.get("k"), "catalog parameters.k")
    repetitions = _benchmark_positive_integer(
        parameters.get("repetitions"),
        "catalog parameters.repetitions",
    )
    if repetitions < 2:
        raise ValueError("Catalog benchmark requires at least two validation repetitions.")
    selection_metric_name = parameters.get("selection_metric")
    if selection_metric_name not in {"precision", "recall", "hit_rate", "map", "mrr", "ndcg"}:
        raise ValueError("Catalog benchmark selection metric is invalid.")
    selection_metric = f"{selection_metric_name}_at_{k}"
    if (
        selection.get("partition") != "validation"
        or selection.get("scope") != "end_to_end"
        or selection.get("metric") != selection_metric
    ):
        raise ValueError("Catalog benchmark selection must be validation-only end-to-end.")

    raw_weights = parameters.get("hybrid_weights")
    if not isinstance(raw_weights, dict) or set(raw_weights) != {
        "collaborative_weight",
        "content_weight",
        "popularity_weight",
    }:
        raise ValueError("Catalog benchmark fixed hybrid weights are invalid.")
    weights = normalize_catalog_hybrid_weights(
        raw_weights["collaborative_weight"],
        raw_weights["content_weight"],
        raw_weights["popularity_weight"],
    )
    if any(
        not _benchmark_numbers_match(raw_weights[key], value)
        for key, value in weights.items()
    ):
        raise ValueError("Catalog benchmark hybrid weights must already be normalized.")
    if expected_weights is not None:
        for key, expected in expected_weights.items():
            if key not in weights or not _benchmark_numbers_match(weights[key], float(expected)):
                raise ValueError(
                    f"Catalog benchmark fixed {key!r} is {weights.get(key)!r}; "
                    f"expected {expected!r}."
                )

    validation_users = _validate_catalog_benchmark_partition(
        validation,
        name="validation",
        config_registry=configs_by_id,
        expected_config_ids=expected_config_ids,
        expected_partition="validation",
        expected_repetitions=repetitions,
        k=k,
        expected_weights=weights,
    )
    expected_fixed_ids = [baseline_config_id]
    if selected_config_id != baseline_config_id:
        expected_fixed_ids.append(selected_config_id)
    fixed_users = _validate_catalog_benchmark_partition(
        fixed_test,
        name="fixed_test",
        config_registry=configs_by_id,
        expected_config_ids=set(expected_fixed_ids),
        expected_partition="test",
        expected_repetitions=1,
        k=k,
        expected_weights=weights,
    )
    if not validation_users.isdisjoint(fixed_users):
        raise ValueError("Catalog benchmark reuses fixed-test users in validation evidence.")
    if fixed_test.get("evaluated_after_selection") is not True or fixed_test.get(
        "evaluated_config_ids"
    ) != expected_fixed_ids:
        raise ValueError("Catalog fixed test is not restricted to baseline and selection.")
    fixed_hash = benchmark_holdout_cohort_hash(
        pd.DataFrame(fixed_test["per_user_rows"]),
        split_id="fixed-test",
    )
    if dataset.get("fixed_test_cohort_hash") != fixed_hash:
        raise ValueError("Catalog benchmark fixed-test cohort fingerprint is inconsistent.")

    candidate_evidence = selection.get("candidate_evidence")
    if not isinstance(candidate_evidence, list):
        raise ValueError("Catalog benchmark selection evidence must be an array.")
    evidence_by_id = {
        str(record.get("config_id")): record
        for record in candidate_evidence
        if isinstance(record, dict)
    }
    challenger_ids = expected_config_ids.difference({baseline_config_id})
    if set(evidence_by_id) != challenger_ids:
        raise ValueError("Catalog benchmark selection evidence is incomplete.")
    validation_deltas = validation.get("paired_deltas_vs_baseline")
    if not isinstance(validation_deltas, dict) or set(validation_deltas) != challenger_ids:
        raise ValueError("Catalog benchmark validation paired deltas are incomplete.")
    selection_confidence = _benchmark_number(
        parameters.get("selection_confidence_level"),
        "catalog parameters.selection_confidence_level",
    )
    multiplicity_adjustment = parameters.get("selection_multiplicity_adjustment")
    if multiplicity_adjustment not in {"none", "bonferroni_familywise"}:
        raise ValueError("Catalog benchmark multiplicity adjustment is invalid.")
    for config_id in challenger_ids:
        _validate_benchmark_delta_evidence(
            validation_deltas[config_id],
            config_id=config_id,
            metric=selection_metric,
            expected_confidence_level=selection_confidence,
        )
    selection_args = argparse.Namespace(
        min_delta=_benchmark_number(parameters.get("min_delta"), "catalog parameters.min_delta"),
        min_selection_rate=_benchmark_number(
            parameters.get("min_selection_rate"),
            "catalog parameters.min_selection_rate",
        ),
        min_content_retention_delta=_benchmark_number(
            parameters.get("min_content_retention_delta"),
            "catalog parameters.min_content_retention_delta",
        ),
        min_personalization_retention_delta=_benchmark_number(
            parameters.get("min_personalization_retention_delta"),
            "catalog parameters.min_personalization_retention_delta",
        ),
        max_latency_ratio=_benchmark_number(
            parameters.get("max_latency_ratio"),
            "catalog parameters.max_latency_ratio",
        ),
        max_serialized_model_ratio=_benchmark_number(
            parameters.get("max_serialized_model_ratio"),
            "catalog parameters.max_serialized_model_ratio",
        ),
    )
    recomputed_selection = select_catalog_configuration(
        validation["aggregate_by_config"],
        validation_deltas=validation_deltas,
        baseline_config_id=baseline_config_id,
        selection_metric=selection_metric,
        args=selection_args,
        confidence_level=selection_confidence,
        multiplicity_adjustment=str(multiplicity_adjustment),
    )
    if selection != recomputed_selection:
        raise ValueError("Catalog benchmark selection is inconsistent with validation evidence.")
    passing_ids = {
        config_id
        for config_id, evidence in evidence_by_id.items()
        if evidence.get("passes") is True
    }
    if selected_config_id == baseline_config_id:
        if passing_ids or selection.get("decision") != "retain_baseline":
            raise ValueError("Catalog benchmark retained a baseline despite passing evidence.")
    elif (
        selected_config_id not in passing_ids
        or selection.get("decision") != "promote_to_fixed_test"
    ):
        raise ValueError("Catalog benchmark selected a challenger without passing evidence.")

    final_delta = fixed_test.get("paired_delta_selected_vs_baseline")
    min_delta = _benchmark_number(parameters.get("min_delta"), "catalog parameters.min_delta")
    expected_deployment = catalog_deployment_decision(
        selection=selection,
        final_delta=final_delta,
        baseline_config_id=baseline_config_id,
        selection_metric=selection_metric,
        min_delta=min_delta,
    )
    if deployment != expected_deployment:
        raise ValueError("Catalog benchmark deployment decision is inconsistent.")
    deployment_config_id = deployment.get("deployment_config_id")
    if expected_catalog is not None:
        deployment_config = configs_by_id.get(str(deployment_config_id))
        if deployment_config is None or any(
            int(deployment_config[key]) != int(expected)
            for key, expected in expected_catalog.items()
        ):
            raise ValueError(
                "Catalog benchmark deployment configuration does not match the serving catalog."
            )

    return {
        "schema_version": payload["schema_version"],
        "protocol": payload["protocol"],
        "relevance_unit": payload["relevance_unit"],
        "split_unit": payload["split_unit"],
        "source_dataset_fingerprint": source_fingerprint,
        "artifact_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "selection": selection,
        "deployment": deployment,
        "hybrid_weights": weights,
        "fixed_test": {
            "aggregate_by_config": fixed_test["aggregate_by_config"],
            "paired_delta_selected_vs_baseline": final_delta,
        },
    }


def _validate_catalog_benchmark_partition(
    partition_payload: dict,
    *,
    name: str,
    config_registry: dict[str, dict[str, object]],
    expected_config_ids: set[str],
    expected_partition: str,
    expected_repetitions: int,
    k: int,
    expected_weights: dict[str, float],
) -> set[int]:
    if partition_payload.get("partition") != expected_partition:
        raise ValueError(f"Catalog benchmark {name} partition is incompatible.")
    if name == "validation":
        repeats = partition_payload.get("repeats")
        if not isinstance(repeats, list) or len(repeats) != expected_repetitions:
            raise ValueError("Catalog benchmark validation repeats are incomplete.")
        split_ids = set()
        for index, repeat in enumerate(repeats):
            if not isinstance(repeat, dict):
                raise ValueError("Catalog benchmark validation repeat must be an object.")
            split_id = repeat.get("split_id")
            if not isinstance(split_id, str) or not split_id or split_id in split_ids:
                raise ValueError("Catalog benchmark validation split IDs must be unique.")
            split_ids.add(split_id)
            _validate_catalog_configuration_records(
                repeat.get("configurations"),
                name=f"validation.repeats[{index}].configurations",
                config_registry=config_registry,
                expected_config_ids=expected_config_ids,
                k=k,
                expected_weights=expected_weights,
            )
    else:
        split_ids = {"fixed-test"}
        _validate_catalog_configuration_records(
            partition_payload.get("configurations"),
            name="fixed_test.configurations",
            config_registry=config_registry,
            expected_config_ids=expected_config_ids,
            k=k,
            expected_weights=expected_weights,
        )
    _validate_catalog_configuration_records(
        partition_payload.get("aggregate_by_config"),
        name=f"{name}.aggregate_by_config",
        config_registry=config_registry,
        expected_config_ids=expected_config_ids,
        k=k,
        expected_weights=expected_weights,
    )
    rows = partition_payload.get("per_user_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Catalog benchmark {name} rows must be a non-empty array.")
    frame = pd.DataFrame(rows)
    required_columns = {
        "partition",
        "config_id",
        "split_id",
        "user_id",
        "holdout_work_key",
        "candidate_available",
        "positive_seed_available",
        "target_in_interaction_catalog",
        "target_in_content_catalog",
    }
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Catalog benchmark {name} rows have an invalid schema.")
    if set(frame["config_id"].astype(str)) != expected_config_ids:
        raise ValueError(f"Catalog benchmark {name} rows omit a configuration.")
    if set(frame["split_id"].astype(str)) != split_ids:
        raise ValueError(f"Catalog benchmark {name} rows reference invalid split IDs.")
    if not frame["partition"].eq(expected_partition).all():
        raise ValueError(f"Catalog benchmark {name} rows have an invalid partition.")
    if frame[["config_id", "split_id", "user_id", "holdout_work_key"]].duplicated().any():
        raise ValueError(f"Catalog benchmark {name} rows contain duplicate pairs.")
    for split_id, split_rows in frame.groupby("split_id", sort=True):
        pairs_by_config = []
        for config_id in sorted(expected_config_ids):
            config_rows = split_rows.loc[split_rows["config_id"] == config_id]
            pairs_by_config.append(
                set(
                    zip(
                        config_rows["user_id"].astype(int),
                        config_rows["holdout_work_key"].astype(str),
                    )
                )
            )
        if any(pairs != pairs_by_config[0] for pairs in pairs_by_config[1:]):
            raise ValueError(
                f"Catalog benchmark {name} configurations do not share cohort {split_id}."
            )
    return set(frame["user_id"].astype(int))


def _validate_catalog_configuration_records(
    records: object,
    *,
    name: str,
    config_registry: dict[str, dict[str, object]],
    expected_config_ids: set[str],
    k: int,
    expected_weights: dict[str, float],
) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError(f"Catalog benchmark {name} must be a non-empty array.")
    by_id = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Catalog benchmark {name}[{index}] must be an object.")
        config_id = record.get("config_id")
        if not isinstance(config_id, str) or config_id in by_id or config_id not in expected_config_ids:
            raise ValueError(f"Catalog benchmark {name} has invalid configuration IDs.")
        registry = config_registry[config_id]
        for key in ("max_interaction_books", "max_content_books"):
            if record.get(key) != registry[key]:
                raise ValueError(f"Catalog benchmark {name} catalog sizes are inconsistent.")
        for key, expected_weight in expected_weights.items():
            if not _benchmark_numbers_match(record.get(key), expected_weight):
                raise ValueError(f"Catalog benchmark {name} hybrid weights are inconsistent.")
        _validate_benchmark_metric_mapping(
            record.get("end_to_end_metrics"),
            k=k,
            name=f"{name}[{index}].end_to_end_metrics",
        )
        _validate_benchmark_metric_mapping(
            record.get("conditional_metrics"),
            k=k,
            name=f"{name}[{index}].conditional_metrics",
        )
        validate_resource_metrics(record.get("resources"))
        by_id[config_id] = record
    if set(by_id) != expected_config_ids:
        raise ValueError(f"Catalog benchmark {name} does not cover all configurations.")


def _benchmark_parameter_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if expected is None:
        return actual is None
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == float(expected)
        )
    return actual == expected


def _benchmark_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Benchmark artifact {name} must be a positive integer.")
    return int(value)


def _benchmark_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Benchmark artifact {name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Benchmark artifact {name} must be a finite number.")
    return numeric


def _benchmark_numbers_match(value: object, expected: float) -> bool:
    try:
        return math.isclose(
            _benchmark_number(value, "number"),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except ValueError:
        return False


def _benchmark_config_registry(value: object) -> dict[str, dict]:
    if not isinstance(value, list) or not value:
        raise ValueError("Benchmark artifact parameters.weight_configs must be a non-empty array.")
    registry: dict[str, dict] = {}
    weight_keys = ("collaborative_weight", "content_weight", "popularity_weight")
    for index, config in enumerate(value):
        if not isinstance(config, dict):
            raise ValueError(f"Benchmark artifact weight_configs[{index}] must be an object.")
        config_id = config.get("config_id")
        if not isinstance(config_id, str) or not config_id or config_id in registry:
            raise ValueError("Benchmark artifact configuration IDs must be unique non-empty strings.")
        weights = {
            key: _benchmark_number(config.get(key), f"weight_configs[{index}].{key}")
            for key in weight_keys
        }
        if any(weight < 0.0 for weight in weights.values()) or not math.isclose(
            sum(weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Benchmark artifact configurations must contain normalized weights.")
        if config_id != weight_config_id(weights):
            raise ValueError("Benchmark artifact configuration ID does not match its weights.")
        registry[config_id] = {"config_id": config_id, **weights}
    return registry


def _validate_benchmark_metric_mapping(metrics: object, *, k: int, name: str) -> None:
    if not isinstance(metrics, dict):
        raise ValueError(f"Benchmark artifact {name} must be an object.")
    expected = set(ranking_metrics_at_k([], [], k))
    if set(metrics) != expected:
        raise ValueError(f"Benchmark artifact {name} has an incompatible metric set.")
    for metric, value in metrics.items():
        numeric = _benchmark_number(value, f"{name}.{metric}")
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"Benchmark artifact {name}.{metric} must be between 0 and 1.")


def _validate_benchmark_config_records(
    records: object,
    *,
    name: str,
    config_registry: dict[str, dict],
    expected_config_ids: set[str],
    k: int,
    repetitions: int | None,
    require_metrics: bool,
) -> dict[str, dict]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"Benchmark artifact {name} must be a non-empty array.")
    by_config: dict[str, dict] = {}
    weight_keys = ("collaborative_weight", "content_weight", "popularity_weight")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Benchmark artifact {name}[{index}] must be an object.")
        config_id = record.get("config_id")
        if not isinstance(config_id, str) or config_id in by_config:
            raise ValueError(f"Benchmark artifact {name} has duplicate or invalid config IDs.")
        if config_id not in expected_config_ids:
            raise ValueError(f"Benchmark artifact {name} references an unexpected configuration.")
        for key in weight_keys:
            if not _benchmark_numbers_match(record.get(key), config_registry[config_id][key]):
                raise ValueError(f"Benchmark artifact {name} weights do not match its registry.")
        if repetitions is not None and _benchmark_positive_integer(
            record.get("repetitions"),
            f"{name}[{index}].repetitions",
        ) != repetitions:
            raise ValueError(f"Benchmark artifact {name} has an inconsistent repetition count.")
        if require_metrics:
            _validate_benchmark_metric_mapping(
                record.get("end_to_end_metrics"),
                k=k,
                name=f"{name}[{index}].end_to_end_metrics",
            )
            _validate_benchmark_metric_mapping(
                record.get("conditional_metrics"),
                k=k,
                name=f"{name}[{index}].conditional_metrics",
            )
        by_config[config_id] = record
    if set(by_config) != expected_config_ids:
        raise ValueError(f"Benchmark artifact {name} does not cover the expected configurations.")
    return by_config


def _validate_benchmark_rows(
    rows: object,
    *,
    name: str,
    expected_config_ids: set[str],
    partition: str,
    expected_split_ids: set[str],
) -> set[int]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Benchmark artifact {name} must be a non-empty array.")
    seen_configs = set()
    user_ids = set()
    seen_pairs = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Benchmark artifact {name}[{index}] must be an object.")
        if row.get("partition") != partition:
            raise ValueError(f"Benchmark artifact {name} has an invalid partition.")
        config_id = row.get("config_id")
        split_id = row.get("split_id")
        user_id = row.get("user_id")
        work_key = row.get("holdout_work_key")
        if (
            not isinstance(config_id, str)
            or config_id not in expected_config_ids
            or not isinstance(split_id, str)
            or split_id not in expected_split_ids
            or isinstance(user_id, bool)
            or not isinstance(user_id, int)
            or user_id <= 0
            or not isinstance(work_key, str)
            or not work_key
        ):
            raise ValueError(f"Benchmark artifact {name} has an invalid per-user row.")
        pair = (config_id, split_id, user_id, work_key)
        if pair in seen_pairs:
            raise ValueError(f"Benchmark artifact {name} has duplicate per-user pairs.")
        seen_pairs.add(pair)
        seen_configs.add(config_id)
        user_ids.add(user_id)
    if seen_configs != expected_config_ids:
        raise ValueError(f"Benchmark artifact {name} does not cover the expected configurations.")
    return user_ids


def _validate_benchmark_validation(
    validation: dict,
    *,
    config_registry: dict[str, dict],
    expected_config_ids: set[str],
    repetitions: int,
    k: int,
) -> set[int]:
    if validation.get("partition") != "validation":
        raise ValueError("Benchmark artifact validation partition is incompatible.")
    if _benchmark_positive_integer(validation.get("repetitions"), "validation.repetitions") != repetitions:
        raise ValueError("Benchmark artifact validation repetition count is inconsistent.")
    repeats = validation.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != repetitions:
        raise ValueError("Benchmark artifact validation repeats are incomplete.")
    split_ids = set()
    for index, repeat in enumerate(repeats):
        if not isinstance(repeat, dict):
            raise ValueError(f"Benchmark artifact validation.repeats[{index}] must be an object.")
        split_id = repeat.get("split_id")
        if not isinstance(split_id, str) or not split_id or split_id in split_ids:
            raise ValueError("Benchmark artifact validation repeat IDs must be unique strings.")
        split_ids.add(split_id)
        _validate_benchmark_config_records(
            repeat.get("configurations"),
            name=f"validation.repeats[{index}].configurations",
            config_registry=config_registry,
            expected_config_ids=expected_config_ids,
            k=k,
            repetitions=None,
            require_metrics=True,
        )
    _validate_benchmark_config_records(
        validation.get("aggregate_by_config"),
        name="validation.aggregate_by_config",
        config_registry=config_registry,
        expected_config_ids=expected_config_ids,
        k=k,
        repetitions=repetitions,
        require_metrics=True,
    )
    if "paired_deltas_vs_baseline" not in validation:
        raise ValueError("Benchmark artifact validation paired deltas are missing.")
    return _validate_benchmark_rows(
        validation.get("per_user_rows"),
        name="validation.per_user_rows",
        expected_config_ids=expected_config_ids,
        partition="validation",
        expected_split_ids=split_ids,
    )


def _validate_benchmark_interval(
    interval: object,
    *,
    name: str,
    expected_confidence_level: float | None = None,
) -> float:
    if not isinstance(interval, dict):
        raise ValueError(f"Benchmark artifact {name} must be an interval object.")
    mean = _benchmark_number(interval.get("mean"), f"{name}.mean")
    lower = _benchmark_number(interval.get("lower"), f"{name}.lower")
    upper = _benchmark_number(interval.get("upper"), f"{name}.upper")
    confidence = _benchmark_number(interval.get("confidence_level"), f"{name}.confidence_level")
    if lower > upper or not 0.0 < confidence < 1.0:
        raise ValueError(f"Benchmark artifact {name} is an invalid interval.")
    _benchmark_positive_integer(interval.get("sample_size"), f"{name}.sample_size")
    _benchmark_positive_integer(interval.get("n_resamples"), f"{name}.n_resamples")
    if expected_confidence_level is not None and not math.isclose(
        confidence,
        expected_confidence_level,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Benchmark artifact {name} has an incompatible confidence level.")
    _ = mean
    return lower


def _validate_benchmark_delta_evidence(
    evidence: object,
    *,
    config_id: str,
    metric: str,
    expected_confidence_level: float | None = None,
) -> tuple[float, float]:
    if not isinstance(evidence, dict) or evidence.get("config_id") != config_id:
        raise ValueError("Benchmark artifact paired delta evidence is incompatible.")
    end_to_end = evidence.get("end_to_end")
    if not isinstance(end_to_end, dict):
        raise ValueError("Benchmark artifact paired delta end-to-end evidence is missing.")
    lower = _validate_benchmark_interval(
        end_to_end.get(metric),
        name=f"paired_delta.{config_id}.{metric}",
        expected_confidence_level=expected_confidence_level,
    )
    selection_rate = _benchmark_number(evidence.get("selection_rate"), "paired_delta.selection_rate")
    if not 0.0 <= selection_rate <= 1.0:
        raise ValueError("Benchmark artifact paired delta selection rate is invalid.")
    return lower, selection_rate


def _validate_benchmark_fixed_test(
    fixed_test: dict,
    *,
    config_registry: dict[str, dict],
    expected_config_ids: list[str],
    k: int,
) -> set[int]:
    if fixed_test.get("partition") != "test":
        raise ValueError("Benchmark artifact fixed-test partition is incompatible.")
    _validate_benchmark_config_records(
        fixed_test.get("configurations"),
        name="fixed_test.configurations",
        config_registry=config_registry,
        expected_config_ids=set(expected_config_ids),
        k=k,
        repetitions=None,
        require_metrics=True,
    )
    _validate_benchmark_config_records(
        fixed_test.get("aggregate_by_config"),
        name="fixed_test.aggregate_by_config",
        config_registry=config_registry,
        expected_config_ids=set(expected_config_ids),
        k=k,
        repetitions=1,
        require_metrics=True,
    )
    return _validate_benchmark_rows(
        fixed_test.get("per_user_rows"),
        name="fixed_test.per_user_rows",
        expected_config_ids=set(expected_config_ids),
        partition="test",
        expected_split_ids={"fixed-test"},
    )


def evaluation_parameters_for_training(args: argparse.Namespace) -> dict[str, object]:
    """Map serving model settings to the corresponding evaluation parameters."""

    return {
        "min_user_ratings": args.min_user_ratings,
        "min_book_ratings": args.min_book_ratings,
        "max_books": args.max_books,
        "max_ratings": args.max_ratings,
        "max_content_books": args.max_content_books,
        "neighbors": args.neighbors,
        "hybrid_candidate_pool_size": DEFAULT_HYBRID_CANDIDATE_POOL_SIZE,
        "hybrid_collaborative_weight": args.collaborative_weight,
        "hybrid_content_weight": args.content_weight,
        "hybrid_popularity_weight": args.popularity_weight,
        "hybrid_dynamic_weights": not args.disable_dynamic_weights,
        "positive_rating_threshold": args.positive_rating_threshold,
        "random_state": args.random_state,
    }


def benchmark_parameters_for_training(args: argparse.Namespace) -> dict[str, object]:
    """Map serving settings to the subset that must match a benchmarked model."""

    parameters = {
        "min_user_ratings": args.min_user_ratings,
        "min_book_ratings": args.min_book_ratings,
        "max_books": args.max_books,
        "max_ratings": args.max_ratings,
        "max_content_books": args.max_content_books,
        "neighbors": args.neighbors,
        "hybrid_candidate_pool_size": DEFAULT_HYBRID_CANDIDATE_POOL_SIZE,
        "hybrid_dynamic_weights": not args.disable_dynamic_weights,
        "positive_rating_threshold": args.positive_rating_threshold,
    }
    if args.max_ratings is not None:
        parameters["test_random_state"] = getattr(args, "random_state", 42)
    return parameters


def catalog_benchmark_parameters_for_training(args: argparse.Namespace) -> dict[str, object]:
    """Map serving settings to catalog-benchmark invariants other than catalog size."""

    parameters = {
        "min_user_ratings": args.min_user_ratings,
        "min_book_ratings": args.min_book_ratings,
        "max_ratings": args.max_ratings,
        "neighbors": args.neighbors,
        "hybrid_candidate_pool_size": DEFAULT_HYBRID_CANDIDATE_POOL_SIZE,
        "hybrid_dynamic_weights": not args.disable_dynamic_weights,
        "positive_rating_threshold": args.positive_rating_threshold,
    }
    if args.max_ratings is not None:
        parameters["test_random_state"] = getattr(args, "random_state", 42)
    return parameters


def normalized_hybrid_weights(
    collaborative_weight: float,
    content_weight: float,
    popularity_weight: float,
) -> dict[str, float]:
    values = (float(collaborative_weight), float(content_weight), float(popularity_weight))
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Hybrid weights must be finite and non-negative.")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("Hybrid weights must sum to a positive value.")
    return {
        "collaborative_weight": values[0] / total,
        "content_weight": values[1] / total,
        "popularity_weight": values[2] / total,
    }


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def evaluation_algorithm_contract() -> dict[str, str | int]:
    return {
        "hybrid_algorithm_id": HybridRecommender.ALGORITHM_ID,
        "hybrid_model_format_version": HybridRecommender.MODEL_FORMAT_VERSION,
        "collaborative_algorithm_id": ItemBasedCollaborativeRecommender.ALGORITHM_ID,
        "collaborative_model_format_version": (
            ItemBasedCollaborativeRecommender.MODEL_FORMAT_VERSION
        ),
        "content_algorithm_id": ContentBasedRecommender.ALGORITHM_ID,
        "content_model_format_version": ContentBasedRecommender.MODEL_FORMAT_VERSION,
    }


def print_training_summary(dataset) -> None:
    for key, value in dataset_summary(dataset).items():
        print(f"{key}: {value:,}")


def print_recommendations(recommendations) -> None:
    if recommendations.empty:
        print("No recommendations found.")
        return

    columns = [
        column
        for column in [
            "score",
            "collaborative_score",
            "content_score",
            "popularity_score",
            "source",
            "reason",
            "isbn",
            "book_title",
            "book_author",
            "publisher",
            "mean_rating",
            "rating_count",
        ]
        if column in recommendations.columns
    ]
    printable = recommendations[columns].copy()
    if "score" in printable.columns:
        printable["score"] = printable["score"].map(lambda value: f"{float(value):.4f}")
    print(printable.to_string(index=False))


def interaction_segment(interactions: int) -> str:
    if interactions <= 0:
        return "0-cold"
    if interactions <= 2:
        return "1-2"
    if interactions <= 5:
        return "3-5"
    if interactions <= 10:
        return "6-10"
    return ">10"


def holdout_status(
    candidate_available: bool,
    history_available: bool,
    positive_seed_available: bool,
) -> str:
    if candidate_available and positive_seed_available:
        return "personalized_rankable"
    if candidate_available and history_available:
        return "no_positive_seed_fallback"
    if candidate_available:
        return "cold_user_fallback"
    if positive_seed_available:
        return "target_not_in_catalog"
    if history_available:
        return "target_not_in_catalog_and_no_positive_seed"
    return "target_not_in_catalog_and_cold_user"


def availability_with_personalization(
    availability,
    holdouts,
    *,
    catalog_work_set: set[str],
    positive_user_set: set[int],
) -> dict[str, int | float]:
    payload = availability.to_dict()
    payload["history_evaluable_holdouts"] = payload.pop("evaluable_holdouts")
    payload["history_dropped_holdouts"] = payload.pop("dropped_holdouts")
    payload["history_evaluation_retention_rate"] = payload.pop(
        "evaluation_retention_rate"
    )
    candidate_mask = holdouts["work_key"].astype(str).isin(catalog_work_set)
    positive_mask = holdouts["user_id"].astype(int).isin(positive_user_set)
    sampled = int(len(holdouts))
    positive_count = int(positive_mask.sum())
    personalization_count = int((candidate_mask & positive_mask).sum())
    payload.update(
        {
            "positive_seed_available_holdouts": positive_count,
            "personalization_evaluable_holdouts": personalization_count,
            "personalization_dropped_holdouts": sampled - personalization_count,
            "positive_seed_retention_rate": positive_count / sampled if sampled else 0.0,
            "personalization_retention_rate": (
                personalization_count / sampled if sampled else 0.0
            ),
        }
    )
    return payload


def summarize_evaluation_rows(rows: list[dict[str, float | str]], group_columns: list[str]):
    if not rows:
        return pd.DataFrame(columns=[*group_columns, "sample_size"])
    results = pd.DataFrame(rows)
    metric_prefixes = (
        "precision_at_",
        "recall_at_",
        "hit_rate_at_",
        "map_at_",
        "mrr_at_",
        "ndcg_at_",
    )
    metrics = [column for column in results.columns if column.startswith(metric_prefixes)]
    means = results.groupby(group_columns)[metrics].mean().reset_index()
    counts = results.groupby(group_columns).size().rename("sample_size").reset_index()
    return means.merge(counts, on=group_columns, how="left")


def evaluation_summary_dataframe(metric_summaries: dict[str, dict], metric_key: str):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "model": model,
                **summary[metric_key],
                "sample_size": summary["sampled_holdouts"],
            }
            for model, summary in metric_summaries.items()
        ]
    )


def recommendation_work_keys(recommendations, isbn_to_work: dict[str, str]) -> list[str]:
    keys = []
    seen = set()
    for isbn in recommendations["isbn"].astype(str):
        work_key = isbn_to_work.get(isbn)
        if work_key is None or work_key in seen:
            continue
        seen.add(work_key)
        keys.append(work_key)
    return keys


def build_work_evaluation_context(dataset, content_model) -> dict:
    books = dataset.books.reset_index(drop=True)
    isbn_to_work = {
        str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
        for row in books[["isbn", "book_title", "book_author"]].itertuples(index=False)
    }
    representative_indices = []
    catalog_items = []
    seen = set()
    for index, isbn in enumerate(books["isbn"].astype(str)):
        work_key = isbn_to_work[isbn]
        if work_key in seen:
            continue
        seen.add(work_key)
        catalog_items.append(work_key)
        representative_indices.append(index)

    rating_work_keys = dataset.ratings["isbn"].astype(str).map(isbn_to_work)
    if rating_work_keys.isna().any():
        raise ValueError("Evaluation ratings contain ISBNs missing from the work catalog.")
    popularity_counts = rating_work_keys.value_counts()
    return {
        "isbn_to_work": isbn_to_work,
        "catalog_items": catalog_items,
        "popularity_counts": popularity_counts,
        "item_features": content_model.matrix[representative_indices],
        "item_to_index": {work_key: index for index, work_key in enumerate(catalog_items)},
    }


def recommend_hybrid_with_weights(
    recommender: HybridRecommender,
    *,
    user_id: int,
    top_n: int,
    candidate_pool_size: int,
    collaborative_weight: float,
    content_weight: float,
    popularity_weight: float,
    dynamic_weights: bool,
    allow_popularity_fallback: bool,
):
    original = (
        recommender.collaborative_weight,
        recommender.content_weight,
        recommender.popularity_weight,
        recommender.dynamic_weights,
    )
    try:
        recommender.collaborative_weight = collaborative_weight
        recommender.content_weight = content_weight
        recommender.popularity_weight = popularity_weight
        recommender.dynamic_weights = dynamic_weights
        return recommender.recommend_for_user(
            user_id,
            top_n=top_n,
            candidate_pool_size=candidate_pool_size,
            allow_popularity_fallback=allow_popularity_fallback,
        )
    finally:
        (
            recommender.collaborative_weight,
            recommender.content_weight,
            recommender.popularity_weight,
            recommender.dynamic_weights,
        ) = original


def print_evaluation_summary(summary) -> None:
    summary = summary.copy()
    metrics = [
        column
        for column in summary.columns
        if column.startswith(
            (
                "precision_at_",
                "recall_at_",
                "hit_rate_at_",
                "map_at_",
                "mrr_at_",
                "ndcg_at_",
            )
        )
    ]
    for column in metrics:
        summary[column] = summary[column].map(lambda value: f"{float(value):.4f}")
    print(summary.to_string(index=False))


def print_diagnostics(diagnostics: dict[str, dict[str, float]]) -> None:
    import pandas as pd

    rows = [{"model": model, **metrics} for model, metrics in diagnostics.items()]
    results = pd.DataFrame(rows)
    for column in results.columns:
        if column != "model":
            results[column] = results[column].map(lambda value: f"{float(value):.4f}")
    print(results.to_string(index=False))


def positive_int(value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def at_least_two_int(value: str | int) -> int:
    parsed = positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("value must be an integer of at least 2")
    return parsed


def non_negative_int(value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def random_state_int(value: str | int) -> int:
    parsed = non_negative_int(value)
    if parsed >= 2**32:
        raise argparse.ArgumentTypeError("random state must be between 0 and 2**32 - 1")
    return parsed


def non_negative_float(value: str | float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return parsed


def positive_float(value: str | float) -> float:
    parsed = non_negative_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def fraction(value: str | float) -> float:
    parsed = non_negative_float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than 0 and less than 1")
    return parsed


def closed_fraction(value: str | float) -> float:
    parsed = non_negative_float(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def familywise_confidence_level(confidence_level: float, challenger_count: int) -> float:
    if challenger_count <= 0:
        return float(confidence_level)
    if challenger_count == 1:
        return float(confidence_level)
    return 1.0 - (1.0 - float(confidence_level)) / challenger_count


def rating_threshold(value: str | float) -> float:
    parsed = non_negative_float(value)
    if not 1 <= parsed <= 10:
        raise argparse.ArgumentTypeError("rating threshold must be between 1 and 10")
    return parsed


def parse_weight_grid(value: str | None) -> list[dict[str, float]]:
    if value is None or not value.strip():
        raw_configs = DEFAULT_HYBRID_WEIGHT_GRID
    else:
        raw_configs = []
        for config in value.split(";"):
            parts = [part.strip() for part in config.split(",") if part.strip()]
            if len(parts) != 3:
                raise ValueError("Each weight config must contain exactly 3 comma-separated numbers.")
            try:
                raw_configs.append(tuple(float(part) for part in parts))
            except ValueError as exc:
                raise ValueError("Hybrid weights must contain only numbers.") from exc

    parsed = []
    for collaborative, content, popularity in raw_configs:
        if not all(math.isfinite(weight) for weight in (collaborative, content, popularity)):
            raise ValueError("Hybrid weights must be finite.")
        if collaborative < 0 or content < 0 or popularity < 0:
            raise ValueError("Hybrid weights must be non-negative.")
        total = collaborative + content + popularity
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Hybrid weights must sum to a positive value.")
        parsed.append(
            {
                "collaborative_weight": collaborative / total,
                "content_weight": content / total,
                "popularity_weight": popularity / total,
            }
        )
    return parsed


def benchmark_weight_configs(
    *,
    weight_grid: str | None,
    baseline_weights: str,
) -> tuple[list[dict[str, object]], str]:
    """Return normalized, deduplicated configurations with order-independent IDs."""

    baseline = parse_weight_grid(baseline_weights)
    if len(baseline) != 1:
        raise ValueError("--baseline-weights must contain exactly one weight configuration.")
    by_config_id: dict[str, dict[str, object]] = {}
    for weights in [*parse_weight_grid(weight_grid), baseline[0]]:
        config_id = weight_config_id(weights)
        by_config_id[config_id] = {"config_id": config_id, **weights}
    return [by_config_id[config_id] for config_id in sorted(by_config_id)], weight_config_id(
        baseline[0]
    )


def weight_config_id(weights: dict[str, float]) -> str:
    """Use normalized weights rather than input order as a stable configuration identity."""

    return (
        f"cw{float(weights['collaborative_weight']):.12f}_"
        f"ct{float(weights['content_weight']):.12f}_"
        f"pw{float(weights['popularity_weight']):.12f}"
    )


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return (int(base_seed) + offset) % (2**32)


def _validate_benchmark_seed_range(args: argparse.Namespace) -> None:
    last_seed = int(args.random_state) + (int(args.repetitions) - 1) * int(args.seed_stride)
    if last_seed >= 2**32:
        raise ValueError("random_state plus repeated seed stride must stay below 2**32.")


def tuning_result_identity(result: dict) -> str:
    weight_keys = ("collaborative_weight", "content_weight", "popularity_weight")
    if all(key in result for key in weight_keys):
        return weight_config_id(result)
    return str(result.get("config_id", ""))


def rank_tuning_results(results: list[dict], k: int) -> list[dict]:
    return sorted(
        results,
        key=lambda row: (
            -float(row[f"hit_rate_at_{k}"]),
            -float(row[f"ndcg_at_{k}"]),
            -float(row[f"catalog_coverage_at_{k}"]),
            float(row[f"top_10pct_popularity_share_at_{k}"]),
            tuning_result_identity(row),
        ),
    )


def print_tuning_summary(results: list[dict], k: int) -> None:
    import pandas as pd

    columns = [
        "config_id",
        "collaborative_weight",
        "content_weight",
        "popularity_weight",
        f"hit_rate_at_{k}",
        f"map_at_{k}",
        f"mrr_at_{k}",
        f"ndcg_at_{k}",
        f"catalog_coverage_at_{k}",
        f"novelty_at_{k}",
        f"top_10pct_popularity_share_at_{k}",
    ]
    table = pd.DataFrame(results)[columns].copy()
    for column in table.columns:
        if column != "config_id":
            table[column] = table[column].map(lambda value: f"{float(value):.4f}")
    print(table.to_string(index=False))


def save_evaluation_artifact(
    output: Path,
    *,
    clean_dataset,
    evaluation_data,
    rows: list[dict[str, float | str]],
    summary,
    end_to_end_summary,
    segment_summary,
    conditional_segment_summary,
    diagnostics: dict[str, dict[str, float]],
    metric_summaries: dict[str, dict],
    confidence_intervals: dict[str, dict],
    availability,
    args: argparse.Namespace,
) -> None:
    train_fingerprint = fingerprint_dataset(
        evaluation_data.train_dataset.books,
        evaluation_data.train_dataset.ratings,
        evaluation_data.train_dataset.users,
    )
    source_fingerprint = fingerprint_dataset(
        clean_dataset.books,
        clean_dataset.ratings,
        clean_dataset.users,
    )
    payload = {
        "schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": EVALUATION_PROTOCOL,
        "evaluation_partition": "test",
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
        "algorithms": evaluation_algorithm_contract(),
        "parameters": {
            "sample_users": args.sample_users,
            "validation_fraction": args.validation_fraction,
            "min_interactions": args.min_interactions,
            "min_relevant_rating": args.min_relevant_rating,
            "k": args.k,
            "neighbors": args.neighbors,
            "random_state": args.random_state,
            "min_user_ratings": args.min_user_ratings,
            "min_book_ratings": args.min_book_ratings,
            "max_books": args.max_books,
            "max_ratings": args.max_ratings,
            "max_content_books": args.max_content_books,
            "hybrid_candidate_pool_size": getattr(args, "hybrid_candidate_pool_size", None),
            "hybrid_collaborative_weight": getattr(args, "hybrid_collaborative_weight", None),
            "hybrid_content_weight": getattr(args, "hybrid_content_weight", None),
            "hybrid_popularity_weight": getattr(args, "hybrid_popularity_weight", None),
            "hybrid_dynamic_weights": not getattr(args, "disable_hybrid_dynamic_weights", False),
            "positive_rating_threshold": args.positive_rating_threshold,
            "confidence_level": args.confidence_level,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "dataset": {
            "clean_books": int(len(clean_dataset.books)),
            "clean_ratings": int(len(clean_dataset.ratings)),
            "clean_users": int(len(clean_dataset.users)),
            "validation_sampled_holdouts": int(len(evaluation_data.validation_holdout_ratings)),
            "validation_history_evaluable_holdouts": int(
                len(evaluation_data.validation_evaluable_holdout_ratings)
            ),
            "test_sampled_holdouts": int(len(evaluation_data.test_holdout_ratings)),
            "test_history_evaluable_holdouts": int(
                len(evaluation_data.test_evaluable_holdout_ratings)
            ),
            "train_ratings_before_filter": int(len(evaluation_data.train_ratings_before_filter)),
            "train_ratings": int(len(evaluation_data.train_dataset.ratings)),
            "candidate_books": int(len(evaluation_data.train_dataset.books)),
            "candidate_works": candidate_work_count(evaluation_data.train_dataset.books),
            "source_fingerprint": source_fingerprint.fingerprint,
            "training_fingerprint": train_fingerprint.fingerprint,
        },
        "availability": availability,
        "summary": {
            "conditional": dataframe_records(summary),
            "end_to_end": dataframe_records(end_to_end_summary),
            "by_model": metric_summaries,
        },
        "segment_summary": {
            "conditional": dataframe_records(conditional_segment_summary),
            "end_to_end": dataframe_records(segment_summary),
        },
        "confidence_intervals": confidence_intervals,
        "diagnostics": {
            model: {metric: float(value) for metric, value in metrics.items()}
            for model, metrics in diagnostics.items()
        },
        "per_user_rows": rows,
    }
    write_json_atomic(output, payload)


def save_tuning_artifact(
    output: Path,
    *,
    clean_dataset,
    evaluation_data,
    results: list[dict],
    availability,
    args: argparse.Namespace,
) -> None:
    train_fingerprint = fingerprint_dataset(
        evaluation_data.train_dataset.books,
        evaluation_data.train_dataset.ratings,
        evaluation_data.train_dataset.users,
    )
    source_fingerprint = fingerprint_dataset(
        clean_dataset.books,
        clean_dataset.ratings,
        clean_dataset.users,
    )
    payload = {
        "schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "tuning",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": TUNING_PROTOCOL,
        "evaluation_partition": "validation",
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
        "algorithms": evaluation_algorithm_contract(),
        "parameters": {
            "sample_users": args.sample_users,
            "validation_fraction": args.validation_fraction,
            "min_interactions": args.min_interactions,
            "min_relevant_rating": args.min_relevant_rating,
            "k": args.k,
            "neighbors": args.neighbors,
            "random_state": args.random_state,
            "min_user_ratings": args.min_user_ratings,
            "min_book_ratings": args.min_book_ratings,
            "max_books": args.max_books,
            "max_ratings": args.max_ratings,
            "max_content_books": args.max_content_books,
            "hybrid_candidate_pool_size": args.hybrid_candidate_pool_size,
            "hybrid_dynamic_weights": not args.disable_hybrid_dynamic_weights,
            "positive_rating_threshold": args.positive_rating_threshold,
            "confidence_level": args.confidence_level,
            "bootstrap_resamples": args.bootstrap_resamples,
            "weight_grid": parse_weight_grid(args.weight_grid),
        },
        "dataset": {
            "clean_books": int(len(clean_dataset.books)),
            "clean_ratings": int(len(clean_dataset.ratings)),
            "clean_users": int(len(clean_dataset.users)),
            "validation_sampled_holdouts": int(len(evaluation_data.validation_holdout_ratings)),
            "validation_history_evaluable_holdouts": int(
                len(evaluation_data.validation_evaluable_holdout_ratings)
            ),
            "test_sampled_holdouts_reserved": int(len(evaluation_data.test_holdout_ratings)),
            "train_ratings_before_filter": int(len(evaluation_data.train_ratings_before_filter)),
            "train_ratings": int(len(evaluation_data.train_dataset.ratings)),
            "candidate_books": int(len(evaluation_data.train_dataset.books)),
            "candidate_works": candidate_work_count(evaluation_data.train_dataset.books),
            "source_fingerprint": source_fingerprint.fingerprint,
            "training_fingerprint": train_fingerprint.fingerprint,
        },
        "availability": availability,
        "results": results,
        "best_config": results[0] if results else None,
    }
    write_json_atomic(output, payload)


def save_benchmark_artifact(
    output: Path,
    *,
    clean_dataset: BookDataset,
    fixed_test_holdouts: pd.DataFrame,
    final_train_dataset: BookDataset,
    configs: list[dict[str, object]],
    baseline_config_id: str,
    validation_repeats: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    validation_aggregate: list[dict[str, object]],
    validation_deltas: dict[str, dict[str, object]],
    selection: dict[str, object],
    final_evaluated: dict[str, object],
    final_rows: list[dict[str, object]],
    final_aggregate: list[dict[str, object]],
    final_delta: dict[str, object] | None,
    args: argparse.Namespace,
) -> None:
    """Persist a self-contained benchmark record with a validation-only decision."""

    source_fingerprint = fingerprint_dataset(
        clean_dataset.books,
        clean_dataset.ratings,
        clean_dataset.users,
    )
    final_training_fingerprint = fingerprint_dataset(
        final_train_dataset.books,
        final_train_dataset.ratings,
        final_train_dataset.users,
    )
    payload = {
        "schema_version": BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "repeated_split_benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": BENCHMARK_PROTOCOL,
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
        "algorithms": evaluation_algorithm_contract(),
        "parameters": {
            "validation_users": int(args.validation_users),
            "test_users": int(args.test_users),
            "repetitions": int(args.repetitions),
            "seed_stride": int(args.seed_stride),
            "random_state": int(args.random_state),
            "test_random_state": int(args.test_random_state),
            "min_interactions": int(args.min_interactions),
            "min_relevant_rating": float(args.min_relevant_rating),
            "k": int(args.k),
            "neighbors": int(args.neighbors),
            "min_user_ratings": int(args.min_user_ratings),
            "min_book_ratings": int(args.min_book_ratings),
            "max_books": args.max_books,
            "max_ratings": args.max_ratings,
            "max_content_books": args.max_content_books,
            "hybrid_candidate_pool_size": int(args.hybrid_candidate_pool_size),
            "hybrid_dynamic_weights": not args.disable_hybrid_dynamic_weights,
            "positive_rating_threshold": float(args.positive_rating_threshold),
            "weight_configs": configs,
            "baseline_config_id": baseline_config_id,
            "selection_metric": str(args.selection_metric),
            "min_delta": float(args.min_delta),
            "min_selection_rate": float(args.min_selection_rate),
            "confidence_level": float(args.confidence_level),
            "selection_confidence_level": float(selection["confidence_level"]),
            "selection_multiplicity_adjustment": str(selection["multiplicity_adjustment"]),
            "bootstrap_resamples": int(args.bootstrap_resamples),
        },
        "dataset": {
            "clean_books": int(len(clean_dataset.books)),
            "clean_ratings": int(len(clean_dataset.ratings)),
            "clean_users": int(len(clean_dataset.users)),
            "source_fingerprint": source_fingerprint.fingerprint,
            "fixed_test_sampled_holdouts": int(len(fixed_test_holdouts)),
            "fixed_test_users": int(fixed_test_holdouts["user_id"].nunique()),
            "final_train_ratings": int(len(final_train_dataset.ratings)),
            "final_candidate_books": int(len(final_train_dataset.books)),
            "final_candidate_works": candidate_work_count(final_train_dataset.books),
            "final_training_fingerprint": final_training_fingerprint.fingerprint,
        },
        "selection": selection,
        "validation": {
            "partition": "validation",
            "repetitions": int(args.repetitions),
            "repeats": validation_repeats,
            "aggregate_by_config": validation_aggregate,
            "paired_deltas_vs_baseline": validation_deltas,
            "per_user_rows": validation_rows,
        },
        "fixed_test": {
            "partition": "test",
            "evaluated_after_selection": True,
            "evaluated_config_ids": [
                str(configuration["config_id"])
                for configuration in final_evaluated["configurations"]
            ],
            "availability": final_evaluated["availability"],
            "configurations": final_evaluated["configurations"],
            "aggregate_by_config": final_aggregate,
            "paired_delta_selected_vs_baseline": final_delta,
            "per_user_rows": final_rows,
        },
    }
    write_json_atomic(output, payload)


def save_catalog_benchmark_artifact(
    output: Path,
    *,
    clean_dataset: BookDataset,
    fixed_test_holdouts: pd.DataFrame,
    configs: list[dict[str, object]],
    baseline_config_id: str,
    weights: dict[str, float],
    validation_repeats: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    validation_aggregate: list[dict[str, object]],
    validation_deltas: dict[str, dict[str, object]],
    selection: dict[str, object],
    final_evaluated: dict[str, object],
    final_rows: list[dict[str, object]],
    final_aggregate: list[dict[str, object]],
    final_delta: dict[str, object] | None,
    deployment: dict[str, object],
    args: argparse.Namespace,
) -> None:
    source_fingerprint = fingerprint_dataset(
        clean_dataset.books,
        clean_dataset.ratings,
        clean_dataset.users,
    )
    catalog_registry = [
        {
            "config_id": str(config["config_id"]),
            "max_interaction_books": int(config["max_interaction_books"]),
            "max_content_books": int(config["max_content_books"]),
        }
        for config in configs
    ]
    payload = {
        "schema_version": CATALOG_BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": CATALOG_BENCHMARK_ARTIFACT_TYPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": CATALOG_BENCHMARK_PROTOCOL,
        "relevance_unit": "canonical_work",
        "split_unit": "user_work",
        "algorithms": evaluation_algorithm_contract(),
        "parameters": {
            "validation_users": int(args.validation_users),
            "test_users": int(args.test_users),
            "repetitions": int(args.repetitions),
            "seed_stride": int(args.seed_stride),
            "random_state": int(args.random_state),
            "test_random_state": int(args.test_random_state),
            "min_interactions": int(args.min_interactions),
            "min_relevant_rating": float(args.min_relevant_rating),
            "min_user_ratings": int(args.min_user_ratings),
            "min_book_ratings": int(args.min_book_ratings),
            "max_ratings": args.max_ratings,
            "k": int(args.k),
            "neighbors": int(args.neighbors),
            "hybrid_candidate_pool_size": int(args.hybrid_candidate_pool_size),
            "hybrid_dynamic_weights": not args.disable_hybrid_dynamic_weights,
            "positive_rating_threshold": float(args.positive_rating_threshold),
            "hybrid_weights": weights,
            "catalog_configs": catalog_registry,
            "baseline_config_id": baseline_config_id,
            "selection_metric": str(args.selection_metric),
            "min_delta": float(args.min_delta),
            "min_selection_rate": float(args.min_selection_rate),
            "min_content_retention_delta": float(args.min_content_retention_delta),
            "min_personalization_retention_delta": float(
                args.min_personalization_retention_delta
            ),
            "max_latency_ratio": float(args.max_latency_ratio),
            "max_serialized_model_ratio": float(args.max_serialized_model_ratio),
            "confidence_level": float(args.confidence_level),
            "selection_confidence_level": float(selection["confidence_level"]),
            "selection_multiplicity_adjustment": str(
                selection["multiplicity_adjustment"]
            ),
            "bootstrap_resamples": int(args.bootstrap_resamples),
        },
        "dataset": {
            "clean_books": int(len(clean_dataset.books)),
            "clean_ratings": int(len(clean_dataset.ratings)),
            "clean_users": int(len(clean_dataset.users)),
            "source_fingerprint": source_fingerprint.fingerprint,
            "fixed_test_sampled_holdouts": int(len(fixed_test_holdouts)),
            "fixed_test_users": int(fixed_test_holdouts["user_id"].nunique()),
            "fixed_test_cohort_hash": benchmark_holdout_cohort_hash(
                fixed_test_holdouts,
                split_id="fixed-test",
            ),
        },
        "selection": selection,
        "deployment": deployment,
        "validation": {
            "partition": "validation",
            "repetitions": int(args.repetitions),
            "repeats": validation_repeats,
            "aggregate_by_config": validation_aggregate,
            "paired_deltas_vs_baseline": validation_deltas,
            "per_user_rows": validation_rows,
        },
        "fixed_test": {
            "partition": "test",
            "evaluated_after_selection": True,
            "evaluated_config_ids": [
                str(configuration["config_id"])
                for configuration in final_evaluated["configurations"]
            ],
            "configurations": final_evaluated["configurations"],
            "aggregate_by_config": final_aggregate,
            "paired_delta_selected_vs_baseline": final_delta,
            "per_user_rows": final_rows,
        },
    }
    write_json_atomic(output, payload)


def write_json_atomic(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def dataframe_records(dataframe) -> list[dict]:
    return json.loads(dataframe.to_json(orient="records"))


def candidate_work_count(books) -> int:
    return len(
        {
            book_work_key(row.book_title, row.book_author, row.isbn)
            for row in books[["isbn", "book_title", "book_author"]].itertuples(index=False)
        }
    )


def positive_work_counts(
    ratings,
    isbn_to_work: dict[str, str],
    *,
    threshold: float,
):
    positive = ratings.loc[
        ratings["book_rating"] >= threshold,
        ["user_id", "isbn"],
    ].copy()
    if positive.empty:
        return pd.Series(dtype="int64")
    positive["work_key"] = positive["isbn"].astype(str).map(isbn_to_work)
    if positive["work_key"].isna().any():
        raise ValueError("Positive rating ISBN is missing from the canonical-work catalog.")
    return positive.drop_duplicates(["user_id", "work_key"])["user_id"].value_counts()


def _reject_json_constant(value: str):
    raise ValueError(f"Invalid non-finite JSON constant: {value}")


def _validate_finite_json_tree(value, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json_tree(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} contains a non-string object key.")
            _validate_finite_json_tree(item, f"{name}.{key}")
        return
    raise ValueError(f"{name} contains a non-JSON value.")


def _validate_ranking_metric_mapping(metrics: dict, name: str) -> None:
    if not metrics:
        raise ValueError(f"Evaluation artifact {name} must not be empty.")
    allowed_prefixes = (
        "precision_at_",
        "recall_at_",
        "hit_rate_at_",
        "map_at_",
        "mrr_at_",
        "ndcg_at_",
    )
    for metric, value in metrics.items():
        if not isinstance(metric, str) or not metric.startswith(allowed_prefixes):
            raise ValueError(f"Evaluation artifact {name} contains an unknown metric.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Evaluation artifact {name}.{metric} must be numeric.")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Evaluation artifact {name}.{metric} must be between 0 and 1.")
