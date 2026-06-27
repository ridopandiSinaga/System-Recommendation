from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender
from book_recommender.data import DatasetNotFoundError, dataset_summary, prepare_dataset, validate_data_dir
from book_recommender.evaluation import build_leave_one_out_split, ranking_metrics_at_k


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

    evaluate_collab = subparsers.add_parser(
        "evaluate-collab",
        help="Evaluate collaborative filtering with leave-one-out ranking metrics.",
    )
    add_common_dataset_args(evaluate_collab)
    evaluate_collab.add_argument("--sample-users", type=int, default=100)
    evaluate_collab.add_argument("--min-interactions", type=int, default=3)
    evaluate_collab.add_argument("--min-relevant-rating", type=float, default=8.0)
    evaluate_collab.add_argument("--k", type=int, default=50)
    evaluate_collab.add_argument("--neighbors", type=int, default=100)
    evaluate_collab.add_argument("--random-state", type=int, default=42)
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
    dataset = load_prepared_dataset(args)
    train_ratings, holdout_ratings = build_leave_one_out_split(
        dataset.ratings,
        sample_users=args.sample_users,
        min_interactions=args.min_interactions,
        min_relevant_rating=args.min_relevant_rating,
        random_state=args.random_state,
    )
    recommender = ItemBasedCollaborativeRecommender(n_neighbors=args.neighbors).fit(
        dataset.books,
        train_ratings,
    )

    rows = []
    for holdout in holdout_ratings.itertuples(index=False):
        user_id = int(holdout.user_id)
        relevant = [holdout.isbn]
        collab_recommended = recommender.recommend_for_user(user_id, top_n=args.k)["isbn"].tolist()
        train_history = train_ratings[train_ratings["user_id"] == user_id]
        read_isbns = set(train_history["isbn"])
        popularity_recommended = recommender.recommend_popular(
            top_n=args.k,
            exclude_isbns=read_isbns,
        )["isbn"].tolist()

        rows.append({"model": "collaborative", **ranking_metrics_at_k(collab_recommended, relevant, args.k)})
        rows.append({"model": "popularity", **ranking_metrics_at_k(popularity_recommended, relevant, args.k)})

    print(f"evaluated_users: {holdout_ratings['user_id'].nunique():,}")
    print(f"holdout_items: {len(holdout_ratings):,}")
    print(f"k: {args.k}")
    print(f"train_ratings: {len(train_ratings):,}")
    print()
    print_evaluation_summary(rows)


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
        for column in ["score", "isbn", "book_title", "book_author", "publisher", "mean_rating", "rating_count"]
        if column in recommendations.columns
    ]
    printable = recommendations[columns].copy()
    if "score" in printable.columns:
        printable["score"] = printable["score"].map(lambda value: f"{float(value):.4f}")
    print(printable.to_string(index=False))


def print_evaluation_summary(rows: list[dict[str, float | str]]) -> None:
    import pandas as pd

    results = pd.DataFrame(rows)
    metrics = [column for column in results.columns if column != "model"]
    summary = results.groupby("model")[metrics].mean().reset_index()
    for column in metrics:
        summary[column] = summary[column].map(lambda value: f"{float(value):.4f}")
    print(summary.to_string(index=False))
