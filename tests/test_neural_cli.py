"""CLI integration and ranking coverage for the optional neural workflow."""
import builtins
import json
from pathlib import Path

import pandas as pd
import pytest

from book_recommender.cli import build_parser
from book_recommender.neural_cli import evaluate_neural_ranking, recommend_neural, train_neural


def test_neural_parser_is_available_without_importing_tensorflow(monkeypatch):
    original_import = builtins.__import__

    def reject_tensorflow(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise AssertionError("CLI registration must not import TensorFlow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_tensorflow)
    args = build_parser().parse_args([
        "train-neural", "--epochs", "2", "--random-state", "7",
        "--architecture", "residual", "--residual-scale", "0.25",
    ])
    assert args.epochs == 2
    assert args.random_state == 7
    assert args.architecture == "residual"
    assert args.residual_scale == 0.25
    assert args.func is train_neural


def test_ranking_counts_unavailable_targets_as_misses():
    from book_recommender.content_based import book_work_key

    books = pd.DataFrame([
        {"isbn": "A", "book_title": "Alpha", "book_author": "Ada"},
    ])

    class Recommender:
        def recommend_for_user(self, user_id, top_n):
            return pd.DataFrame({"isbn": ["A"]})

    holdouts = pd.DataFrame([
        {"user_id": 1, "work_key": book_work_key("Alpha", "Ada", "A")},
        {"user_id": 2, "work_key": "unavailable"},
    ])
    metrics = evaluate_neural_ranking(Recommender(), books, holdouts, k=1)
    assert metrics["hit_rate_at_1"] == 0.5
    assert metrics["recall_at_1"] == 0.5


def test_no_relevant_holdouts_has_no_measured_ranking_metric():
    assert evaluate_neural_ranking(None, pd.DataFrame(), pd.DataFrame(), k=1) is None


@pytest.mark.parametrize("architecture", ["mlp", "residual"])
def test_train_save_recommend_cli_round_trip(tmp_path, capsys, architecture):
    pytest.importorskip("tensorflow")
    fixture = Path(__file__).parent / "fixtures" / "notebook_data"
    model_dir = tmp_path / "neural"
    parser = build_parser()
    train_args = parser.parse_args([
        "train-neural", "--data-dir", str(fixture), "--epochs", "2",
        "--embedding-size", "4", "--sample-users", "6", "--output", str(model_dir),
        "--architecture", architecture,
    ])
    train_neural(train_args)
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads((model_dir / "evaluation.json").read_text())
    assert report["architecture"] == architecture
    assert (model_dir / "network.keras").is_file()
    assert report["train_pairs"] > 0
    assert report["test_rating_metrics"]["known_pairs"] > 0
    assert report["first_step_trace"]["embedding_delta_norm"] > 0
    assert report["test_relevant_holdouts"] <= report["test_holdouts"]
    recommend_args = parser.parse_args([
        "recommend-neural", "--model-dir", str(model_dir), "--user-id", "5709",
        "--top-n", "2",
    ])
    recommend_neural(recommend_args)
    output = capsys.readouterr().out
    assert "book_title" in output
    assert "neural" in output
