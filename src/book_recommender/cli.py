from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from book_recommender.artifacts import (
    ArtifactError,
    fingerprint_dataset,
    load_model_bundle,
    save_model_bundle,
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
    bootstrap_ranking_metrics_at_k,
    mean_ranking_metrics_at_k,
    prepare_validation_test_evaluation,
    ranking_metrics_at_k,
    recommendation_diagnostics_at_k,
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
    dataset = load_hybrid_prepared_dataset(args)
    recommender = HybridRecommender(
        n_neighbors=args.neighbors,
        collaborative_weight=args.collaborative_weight,
        content_weight=args.content_weight,
        popularity_weight=args.popularity_weight,
        dynamic_weights=not args.disable_dynamic_weights,
        positive_rating_threshold=args.positive_rating_threshold,
    ).fit(dataset.books, dataset.ratings)
    metrics = {}
    if args.evaluation_artifact is not None:
        clean_dataset = load_clean_dataset(args.data_dir)
        source_fingerprint = fingerprint_dataset(
            clean_dataset.books,
            clean_dataset.ratings,
            clean_dataset.users,
        ).fingerprint
        metrics = read_evaluation_metrics(
            args.evaluation_artifact,
            expected_source_dataset_fingerprint=source_fingerprint,
            expected_parameters=evaluation_parameters_for_training(args),
        )
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


def fraction(value: str | float) -> float:
    parsed = non_negative_float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than 0 and less than 1")
    return parsed


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


def rank_tuning_results(results: list[dict], k: int) -> list[dict]:
    return sorted(
        results,
        key=lambda row: (
            row[f"hit_rate_at_{k}"],
            row[f"ndcg_at_{k}"],
            row[f"catalog_coverage_at_{k}"],
            -row[f"top_10pct_popularity_share_at_{k}"],
        ),
        reverse=True,
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
