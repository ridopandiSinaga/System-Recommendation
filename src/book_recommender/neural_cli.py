"""Optional neural commands; parser registration does not import TensorFlow."""
from __future__ import annotations

import json
from pathlib import Path

from book_recommender.content_based import book_work_key
from book_recommender.data import load_clean_dataset
from book_recommender.evaluation import (
    mean_ranking_metrics_at_k,
    prepare_validation_test_evaluation,
    ranking_metrics_at_k,
)


def register_neural_commands(subparsers):
    from book_recommender.cli import (
        add_common_dataset_args, at_least_two_int, non_negative_float,
        positive_float, positive_int, random_state_int,
    )

    train = subparsers.add_parser("train-neural", help="Train embeddings + MLP and trace Adam.")
    add_common_dataset_args(train)
    train.add_argument("--sample-users", type=at_least_two_int, default=200)
    train.add_argument("--epochs", type=positive_int, default=10)
    train.add_argument("--batch-size", type=positive_int, default=256)
    train.add_argument("--embedding-size", type=positive_int, default=16)
    train.add_argument("--learning-rate", type=positive_float, default=0.001)
    train.add_argument("--architecture", choices=("mlp", "residual"), default="mlp",
                       help="Use the original MLP or train-derived biases plus a neural correction.")
    train.add_argument("--user-shrinkage", type=non_negative_float, default=0.0)
    train.add_argument("--item-shrinkage", type=non_negative_float, default=20.0)
    train.add_argument("--residual-scale", type=non_negative_float, default=0.25)
    train.add_argument("--random-state", type=random_state_int, default=42)
    train.add_argument("--k", type=positive_int, default=10)
    train.add_argument("--output", type=Path, default=Path("models/neural"))
    train.set_defaults(func=train_neural)
    recommend = subparsers.add_parser("recommend-neural", help="Recommend from a neural artifact.")
    recommend.add_argument("--model-dir", type=Path, default=Path("models/neural"))
    recommend.add_argument("--user-id", type=positive_int, required=True)
    recommend.add_argument("--top-n", type=positive_int, default=10)
    recommend.set_defaults(func=recommend_neural)


def neural_recommender_class(architecture="mlp"):
    try:
        if architecture == "residual":
            from book_recommender.residual_neural import ResidualNeuralRecommender
            return ResidualNeuralRecommender
        if architecture != "mlp":
            raise ValueError(f"Unsupported neural architecture: {architecture}")
        from book_recommender.neural import NeuralRecommender
    except ModuleNotFoundError as exc:
        if exc.name != "tensorflow":
            raise
        raise SystemExit("Install neural dependencies first: uv sync --dev --extra neural") from exc
    return NeuralRecommender


def evaluate_neural_ranking(recommender, books, holdouts, k=10):
    """All sampled holdouts count; unavailable targets remain misses."""
    if holdouts.empty:
        return None
    work_map = {
        str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
        for row in books.itertuples(index=False)
    }
    rows = []
    for row in holdouts.itertuples(index=False):
        recommendations = recommender.recommend_for_user(int(row.user_id), top_n=k)
        works = [work_map[str(isbn)] for isbn in recommendations.isbn]
        rows.append(ranking_metrics_at_k(works, {row.work_key}, k))
    return mean_ranking_metrics_at_k(rows, k=k)


def train_neural(args):
    recommender_class = neural_recommender_class(args.architecture)
    dataset = load_clean_dataset(args.data_dir)
    # Split before catalog selection/filtering; remove all editions of holdout works.
    split = prepare_validation_test_evaluation(
        dataset, sample_users=args.sample_users, min_relevant_rating=1.0,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings, max_books=args.max_books,
        max_ratings=args.max_ratings, max_content_books=args.max_books,
        random_state=args.random_state,
    )
    train = split.train_dataset
    model_options = {}
    if args.architecture == "residual":
        model_options.update(user_shrinkage=args.user_shrinkage,
                             item_shrinkage=args.item_shrinkage,
                             residual_scale=args.residual_scale)
    recommender = recommender_class(
        embedding_size=args.embedding_size, learning_rate=args.learning_rate,
        random_state=args.random_state, **model_options,
    ).fit(
        train.books, train.ratings, validation_ratings=split.validation_holdout_ratings,
        epochs=args.epochs, batch_size=args.batch_size,
    )
    # Fixed test is read only after validation-based early stopping has completed.
    relevant_test = split.test_holdout_ratings[
        split.test_holdout_ratings.book_rating >= 8.0
    ]
    report = {
        "architecture": args.architecture,
        "protocol": "canonical_work_holdout_disjoint_validation_test_train_only_filtering",
        "train_users": len(recommender.user_to_index),
        "train_books": len(recommender.isbn_to_index), "train_pairs": len(recommender.ratings),
        "best_epoch": recommender.best_epoch,
        "validation_rating_metrics": recommender.evaluate_ratings(split.validation_holdout_ratings),
        "test_rating_metrics": recommender.evaluate_ratings(split.test_holdout_ratings),
        "test_holdouts": len(split.test_holdout_ratings),
        "test_relevant_holdouts": len(relevant_test),
        "ranking_min_relevant_rating": 8.0,
        "ranking_status": "evaluated" if len(relevant_test) else "no_relevant_holdouts",
        "test_end_to_end_ranking": evaluate_neural_ranking(
            recommender, dataset.books, relevant_test, k=args.k,
        ),
        "first_step_trace": recommender.first_step_trace,
        "history": recommender.history,
    }
    recommender.save(args.output)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def recommend_neural(args):
    try:
        metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
        recommender = neural_recommender_class(metadata.get("architecture", "mlp")).load(
            args.model_dir,
        )
    except OSError as exc:
        raise SystemExit(f"Cannot read neural model in {args.model_dir}: {exc}") from exc
    recommendations = recommender.recommend_for_user(args.user_id, top_n=args.top_n)
    columns = ["isbn", "book_title", "book_author", "score", "source"]
    print(recommendations[columns].to_string(index=False))
