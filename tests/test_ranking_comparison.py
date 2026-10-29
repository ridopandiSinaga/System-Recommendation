import json

import numpy as np
import pandas as pd
import pytest

from scripts.ranking_comparison import (
    paired_mean_bootstrap, prepare_ranking_data, run_ranking_comparison,
)


def frames():
    books = pd.DataFrame([{"isbn": isbn, "book_title": title, "book_author": "Author"}
                          for isbn, title in [("A1", "A"), ("A2", "A"), ("B", "B"),
                                              ("C", "C"), ("D", "D")]])
    fit = pd.DataFrame([(1, "A1", 8), (1, "A2", 10), (2, "B", 9), (3, "B", 7)],
                       columns=["user_id", "isbn", "book_rating"])
    assessment = pd.DataFrame([(1, "B", 8), (2, "A1", 3), (2, "A2", 9), (3, "C", 7)],
                              columns=fit.columns)
    return books, fit, assessment


def test_relevance_and_popularity_count_work_once():
    data = prepare_ranking_data(*frames(), np, pd)
    _, works, _, _, _, targets, popularity = data
    assert targets.user_id.tolist() == [1, 2]
    assert targets.book_rating.tolist() == [8, 9]
    assert dict(zip(works, popularity)) == {
        f"title:{letter}|author:author": count for letter, count in zip("abcd", [1, 1, 0, 0])}


def test_rank_max_edition_exclude_all_read_editions_and_break_ties(tmp_path):
    books, fit, assessment = frames()
    def predict(users, isbns):
        return np.asarray([{"A1": 0, "A2": 100, "B": 2, "C": 2, "D": 1}[i] for i in isbns])
    result = run_ranking_comparison(split_seed=17, books=books, fit=fit, assessment=assessment,
                                   predictors={"mlp": predict, "mf": predict, "bias": predict},
                                   primary_mlp="mlp", primary_mf="mf", bias_name="bias",
                                   output_dir=tmp_path, np=np, pd=pd)
    assert result["relevant_users"] == 2
    assert result["metrics"]["mlp"]["hits_at_10"] == 2
    assert result["metrics"]["mlp"]["ndcg_at_10"] == 1
    recommendations = pd.read_csv(tmp_path / "ranking_top10.csv")
    recommendations = recommendations[recommendations.model == "mlp"]
    first, second = recommendations.itertuples(index=False)
    assert json.loads(first.work_keys) == [f"title:{c}|author:author" for c in "bcd"]
    assert json.loads(second.work_keys) == [f"title:{c}|author:author" for c in "acd"]
    assert json.loads(second.raw_scores)[0] == 100
    rows = pd.read_csv(tmp_path / "ranking_user_outcomes.csv")
    assert rows.groupby("model").user_id.agg(list).tolist() == [[1, 2]] * 4
    assert (rows.candidate_works == 3).all()


def test_unavailable_relevant_target_remains_miss(tmp_path):
    books, fit, assessment = frames()
    assessment = assessment.iloc[:1].copy()
    assessment["isbn"] = "missing"
    assessment["work_key"] = "title:missing|author:author"
    def predictor(users, isbns):
        return np.ones(len(users))
    result = run_ranking_comparison(split_seed=17, books=books, fit=fit, assessment=assessment,
                                   predictors={"one": predictor}, primary_mlp="one",
                                   primary_mf="one", bias_name="one", output_dir=tmp_path, np=np, pd=pd)
    assert result["target_coverage"] == 0
    assert result["metrics"]["one"]["users"] == 1
    assert result["metrics"]["one"]["hits_at_10"] == 0


def test_work_leakage_is_rejected():
    books, fit, assessment = frames()
    assessment.loc[0, "isbn"] = "A2"
    with pytest.raises(ValueError, match="overlaps fit"):
        prepare_ranking_data(books, fit, assessment, np, pd)


def test_rank_sensitive_metrics_and_finite_predictions(tmp_path):
    books, fit, assessment = frames()
    def predictor(users, isbns):
        return np.asarray([0 if i == "B" else 1 for i in isbns])
    result = run_ranking_comparison(split_seed=17, books=books, fit=fit,
                                   assessment=assessment.iloc[:1], predictors={"one": predictor},
                                   primary_mlp="one", primary_mf="one", bias_name="one",
                                   output_dir=tmp_path, np=np, pd=pd)
    assert result["metrics"]["one"]["mrr_at_10"] == pytest.approx(1 / 3)
    assert result["metrics"]["one"]["ndcg_at_10"] == .5
    with pytest.raises(ValueError, match="Invalid raw"):
        run_ranking_comparison(split_seed=17, books=books, fit=fit, assessment=assessment,
                               predictors={"one": lambda u, i: np.full(len(u), np.nan)},
                               primary_mlp="one", primary_mf="one", bias_name="one",
                               output_dir=tmp_path, np=np, pd=pd)


def test_bootstrap_matches_independent_paired_oracle():
    candidate = np.array([1, 0, .5])
    reference = np.array([0, 1, .5])
    result = paired_mean_bootstrap(candidate, reference, np, draws=113, seed=9)
    sample = np.random.default_rng(9).integers(3, size=(113, 3))
    means = (candidate - reference)[sample].mean(axis=1)
    np.testing.assert_allclose(result["ci95"], np.quantile(means, [.025, .975]))
    assert result["users"] == 3
    assert result["difference"] == 0
