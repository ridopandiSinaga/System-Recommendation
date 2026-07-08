from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender
from book_recommender.data import (
    DatasetNotFoundError,
    dataset_summary,
    load_clean_dataset,
    prepare_dataset,
    validate_data_dir,
)
from book_recommender.evaluation import (
    prepare_leave_one_out_evaluation,
    ranking_metrics_at_k,
    recommendation_diagnostics_at_k,
)
from book_recommender.hybrid import HybridRecommender


DEFAULT_MAX_BOOKS = 5_000


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except DatasetNotFoundError as exc:
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
    add_common_dataset_args(content)
    content.add_argument("--title", help="Book title to use as query.")
    content.add_argument("--isbn", help="ISBN to use as query.")
    content.add_argument("--top-n", type=int, default=10)
    content.set_defaults(func=recommend_content)

    collab = subparsers.add_parser("recommend-collab", help="Recommend books for a user from rating history.")
    add_common_dataset_args(collab)
    collab.add_argument("--user-id", type=int, required=True)
    collab.add_argument("--top-n", type=int, default=10)
    collab.add_argument("--neighbors", type=int, default=50)
    collab.set_defaults(func=recommend_collab)

    hybrid = subparsers.add_parser("recommend-hybrid", help="Recommend books using hybrid scoring.")
    add_common_dataset_args(hybrid)
    hybrid.add_argument("--user-id", type=int, required=True)
    hybrid.add_argument("--top-n", type=int, default=10)
    hybrid.add_argument("--neighbors", type=int, default=50)
    hybrid.add_argument("--candidate-pool-size", type=int, default=500)
    hybrid.set_defaults(func=recommend_hybrid)

    evaluate_collab = subparsers.add_parser(
        "evaluate-collab",
        help="Evaluate collaborative, hybrid, and popularity ranking metrics.",
    )
    add_common_dataset_args(evaluate_collab)
    evaluate_collab.add_argument("--sample-users", type=int, default=100)
    evaluate_collab.add_argument("--min-interactions", type=int, default=3)
    evaluate_collab.add_argument("--min-relevant-rating", type=float, default=8.0)
    evaluate_collab.add_argument("--k", type=int, default=50)
    evaluate_collab.add_argument("--neighbors", type=int, default=100)
    evaluate_collab.add_argument("--hybrid-candidate-pool-size", type=int, default=500)
    evaluate_collab.add_argument("--hybrid-collaborative-weight", type=float, default=0.45)
    evaluate_collab.add_argument("--hybrid-content-weight", type=float, default=0.45)
    evaluate_collab.add_argument("--hybrid-popularity-weight", type=float, default=0.10)
    evaluate_collab.add_argument("--hybrid-min-content-rating", type=float, default=8.0)
    evaluate_collab.add_argument("--disable-hybrid-dynamic-weights", action="store_true")
    evaluate_collab.add_argument("--random-state", type=int, default=42)
    evaluate_collab.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/recommender_evaluation.json"),
        help="Path to save evaluation metrics and parameters as JSON.",
    )
    evaluate_collab.set_defaults(func=evaluate_collab_model)

    train_content = subparsers.add_parser("train-content", help="Build and save content model artifact.")
    add_common_dataset_args(train_content)
    train_content.add_argument("--output", type=Path, default=Path("models/content.joblib"))
    train_content.set_defaults(func=train_content_model)

    train_collab = subparsers.add_parser("train-collab", help="Build and save collaborative model artifact.")
    add_common_dataset_args(train_collab)
    train_collab.add_argument("--neighbors", type=int, default=50)
    train_collab.add_argument("--output", type=Path, default=Path("models/collab.joblib"))
    train_collab.set_defaults(func=train_collab_model)

    return parser


def add_data_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data/raw", type=Path, help="Folder containing Kaggle CSV files.")


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    add_data_dir_arg(parser)
    parser.add_argument("--min-user-ratings", type=int, default=2)
    parser.add_argument("--min-book-ratings", type=int, default=2)
    parser.add_argument("--max-books", type=int, default=DEFAULT_MAX_BOOKS)
    parser.add_argument("--max-ratings", type=int, default=None)


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
    dataset = load_prepared_dataset(args)
    recommender = ContentBasedRecommender().fit(dataset.books)
    recommendations = recommender.recommend_similar(title=args.title, isbn=args.isbn, top_n=args.top_n)
    print_recommendations(recommendations)


def recommend_collab(args: argparse.Namespace) -> None:
    dataset = load_prepared_dataset(args)
    recommender = ItemBasedCollaborativeRecommender(n_neighbors=args.neighbors).fit(
        dataset.books,
        dataset.ratings,
    )
    recommendations = recommender.recommend_for_user(user_id=args.user_id, top_n=args.top_n)
    print_recommendations(recommendations)


def recommend_hybrid(args: argparse.Namespace) -> None:
    dataset = load_prepared_dataset(args)
    recommender = HybridRecommender(n_neighbors=args.neighbors).fit(
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
    dataset = load_prepared_dataset(args)
    recommender = ContentBasedRecommender().fit(dataset.books)
    save_artifact(recommender, args.output)
    print(f"Saved content model to {args.output.resolve()}")
    print_training_summary(dataset)


def train_collab_model(args: argparse.Namespace) -> None:
    dataset = load_prepared_dataset(args)
    recommender = ItemBasedCollaborativeRecommender(n_neighbors=args.neighbors).fit(
        dataset.books,
        dataset.ratings,
    )
    save_artifact(recommender, args.output)
    print(f"Saved collaborative model to {args.output.resolve()}")
    print_training_summary(dataset)


def evaluate_collab_model(args: argparse.Namespace) -> None:
    clean_dataset = load_clean_dataset(args.data_dir)
    evaluation_data = prepare_leave_one_out_evaluation(
        clean_dataset,
        sample_users=args.sample_users,
        min_interactions=args.min_interactions,
        min_relevant_rating=args.min_relevant_rating,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
        max_books=args.max_books,
        max_ratings=args.max_ratings,
        random_state=args.random_state,
    )
    if evaluation_data.evaluable_holdout_ratings.empty:
        raise SystemExit(
            "No evaluable holdout ratings remain after train-only filtering. "
            "Try increasing --max-books or lowering min rating filters."
        )

    train_dataset = evaluation_data.train_dataset
    hybrid_recommender = HybridRecommender(
        n_neighbors=args.neighbors,
        collaborative_weight=args.hybrid_collaborative_weight,
        content_weight=args.hybrid_content_weight,
        popularity_weight=args.hybrid_popularity_weight,
        dynamic_weights=not args.disable_hybrid_dynamic_weights,
        min_content_rating=args.hybrid_min_content_rating,
    ).fit(
        train_dataset.books,
        train_dataset.ratings,
    )
    recommender = hybrid_recommender.collaborative_model
    content_model = hybrid_recommender.content_model
    item_to_index = {
        str(isbn): index
        for index, isbn in enumerate(train_dataset.books["isbn"].astype(str).tolist())
    }

    rows = []
    recommendation_lists = {"hybrid": [], "collaborative": [], "popularity": []}
    train_user_counts = train_dataset.ratings["user_id"].value_counts()
    for holdout in evaluation_data.evaluable_holdout_ratings.itertuples(index=False):
        user_id = int(holdout.user_id)
        relevant = [str(holdout.isbn)]
        hybrid_recommended = [
            str(isbn)
            for isbn in hybrid_recommender.recommend_for_user(
                user_id,
                top_n=args.k,
                candidate_pool_size=args.hybrid_candidate_pool_size,
            )["isbn"].tolist()
        ]
        collab_recommended = [
            str(isbn) for isbn in recommender.recommend_for_user(user_id, top_n=args.k)["isbn"].tolist()
        ]
        train_history = train_dataset.ratings[train_dataset.ratings["user_id"] == user_id]
        read_isbns = set(train_history["isbn"])
        popularity_recommended = recommender.recommend_popular(
            top_n=args.k,
            exclude_isbns=read_isbns,
        )["isbn"].tolist()
        popularity_recommended = [str(isbn) for isbn in popularity_recommended]

        train_interactions = int(train_user_counts.get(user_id, 0))
        segment = interaction_segment(train_interactions)

        rows.append(
            {
                "model": "hybrid",
                "user_id": user_id,
                "user_segment": segment,
                "train_interactions": train_interactions,
                "holdout_isbn": str(holdout.isbn),
                **ranking_metrics_at_k(hybrid_recommended, relevant, args.k),
            }
        )
        rows.append(
            {
                "model": "collaborative",
                "user_id": user_id,
                "user_segment": segment,
                "train_interactions": train_interactions,
                "holdout_isbn": str(holdout.isbn),
                **ranking_metrics_at_k(collab_recommended, relevant, args.k),
            }
        )
        rows.append(
            {
                "model": "popularity",
                "user_id": user_id,
                "user_segment": segment,
                "train_interactions": train_interactions,
                "holdout_isbn": str(holdout.isbn),
                **ranking_metrics_at_k(popularity_recommended, relevant, args.k),
            }
        )
        recommendation_lists["hybrid"].append(hybrid_recommended)
        recommendation_lists["collaborative"].append(collab_recommended)
        recommendation_lists["popularity"].append(popularity_recommended)

    popularity_counts = train_dataset.ratings["isbn"].astype(str).value_counts()
    catalog_items = train_dataset.books["isbn"].astype(str).tolist()
    diagnostics = {
        model: recommendation_diagnostics_at_k(
            recommendations,
            catalog_items=catalog_items,
            popularity_counts=popularity_counts,
            k=args.k,
            item_features=content_model.matrix,
            item_to_index=item_to_index,
        )
        for model, recommendations in recommendation_lists.items()
    }

    summary = summarize_evaluation_rows(rows, ["model"])
    segment_summary = summarize_evaluation_rows(rows, ["model", "user_segment"])

    print("protocol: leave-one-out with train-only filtering/top-books")
    print(f"sampled_holdouts: {len(evaluation_data.holdout_ratings):,}")
    print(f"evaluable_holdouts: {len(evaluation_data.evaluable_holdout_ratings):,}")
    print(f"dropped_holdouts: {evaluation_data.dropped_holdouts:,}")
    print(f"k: {args.k}")
    print(f"train_ratings_before_filter: {len(evaluation_data.train_ratings_before_filter):,}")
    print(f"train_ratings: {len(train_dataset.ratings):,}")
    print(f"candidate_books: {len(train_dataset.books):,}")
    print()
    print("Ranking metrics:")
    print_evaluation_summary(summary)
    print()
    print("Segment metrics:")
    print_evaluation_summary(segment_summary)
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
            segment_summary=segment_summary,
            diagnostics=diagnostics,
            args=args,
        )
        print()
        print(f"saved_evaluation: {args.output.resolve()}")


def load_prepared_dataset(args: argparse.Namespace):
    return prepare_dataset(
        data_dir=args.data_dir,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
        max_books=args.max_books,
        max_ratings=args.max_ratings,
    )


def save_artifact(recommender, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    recommender.save(output)


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
    if interactions <= 2:
        return "1-2"
    if interactions <= 5:
        return "3-5"
    if interactions <= 10:
        return "6-10"
    return ">10"


def summarize_evaluation_rows(rows: list[dict[str, float | str]], group_columns: list[str]):
    import pandas as pd

    results = pd.DataFrame(rows)
    metric_prefixes = ("precision_at_", "recall_at_", "hit_rate_at_", "map_at_", "mrr_at_", "ndcg_at_")
    metrics = [column for column in results.columns if column.startswith(metric_prefixes)]
    return results.groupby(group_columns)[metrics].mean().reset_index()


def print_evaluation_summary(summary) -> None:
    summary = summary.copy()
    metrics = [
        column
        for column in summary.columns
        if column.startswith(("precision_at_", "recall_at_", "hit_rate_at_", "map_at_", "mrr_at_", "ndcg_at_"))
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


def save_evaluation_artifact(
    output: Path,
    *,
    clean_dataset,
    evaluation_data,
    rows: list[dict[str, float | str]],
    summary,
    segment_summary,
    diagnostics: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "leave_one_out_train_only_filtering",
        "parameters": {
            "sample_users": args.sample_users,
            "min_interactions": args.min_interactions,
            "min_relevant_rating": args.min_relevant_rating,
            "k": args.k,
            "neighbors": args.neighbors,
            "random_state": args.random_state,
            "min_user_ratings": args.min_user_ratings,
            "min_book_ratings": args.min_book_ratings,
            "max_books": args.max_books,
            "max_ratings": args.max_ratings,
            "hybrid_candidate_pool_size": getattr(args, "hybrid_candidate_pool_size", None),
            "hybrid_collaborative_weight": getattr(args, "hybrid_collaborative_weight", None),
            "hybrid_content_weight": getattr(args, "hybrid_content_weight", None),
            "hybrid_popularity_weight": getattr(args, "hybrid_popularity_weight", None),
            "hybrid_min_content_rating": getattr(args, "hybrid_min_content_rating", None),
            "hybrid_dynamic_weights": not getattr(args, "disable_hybrid_dynamic_weights", False),
        },
        "dataset": {
            "clean_books": int(len(clean_dataset.books)),
            "clean_ratings": int(len(clean_dataset.ratings)),
            "clean_users": int(len(clean_dataset.users)),
            "sampled_holdouts": int(len(evaluation_data.holdout_ratings)),
            "evaluable_holdouts": int(len(evaluation_data.evaluable_holdout_ratings)),
            "dropped_holdouts": int(evaluation_data.dropped_holdouts),
            "train_ratings_before_filter": int(len(evaluation_data.train_ratings_before_filter)),
            "train_ratings": int(len(evaluation_data.train_dataset.ratings)),
            "candidate_books": int(len(evaluation_data.train_dataset.books)),
        },
        "summary": dataframe_records(summary),
        "segment_summary": dataframe_records(segment_summary),
        "diagnostics": {
            model: {metric: float(value) for metric, value in metrics.items()}
            for model, metrics in diagnostics.items()
        },
        "per_user_rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dataframe_records(dataframe) -> list[dict]:
    return json.loads(dataframe.to_json(orient="records"))
