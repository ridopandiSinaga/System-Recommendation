"""Shared fallback, work exclusions, cohorts, and paired reuse invariants."""
import json

import numpy as np
import pandas as pd
import pytest

from book_recommender.ranking_neural import prepare_work_data
from scripts.ranking_objective_evaluation import (
    assemble_assessment_report, evaluate_catalogue_scores, prepare_evaluation,
)


def fixture_context():
    books = pd.DataFrame([{"isbn": value, "book_title": f"Work {value}",
                           "book_author": "Author", "publisher": "Publisher"}
                          for value in "ABCD"])
    fit = pd.DataFrame([(1, "A", 9), (1, "B", 1), (2, "B", 4),
                        (2, "C", 6), (3, "C", 8), (3, "D", 9)],
                       columns=["user_id", "isbn", "book_rating"])
    heldout = pd.DataFrame([(1, "C", 8), (2, "D", 9), (3, "A", 8)], columns=fit.columns)
    data = prepare_work_data(books, fit, max_features=8)
    context = prepare_evaluation(books, fit, heldout, data)
    # u1 targets C behind D, u2 model strongly favors D but fallback ties A/D,
    # u3 targets A behind B. Fit exclusions remove all previously read works.
    scores = np.asarray([[20, 20, 2, 3], [0, 0, 0, 100], [0, 10, 20, 20]], dtype=float)
    return context, lambda users: scores[users]


def test_fallback_applies_to_every_new_arm_and_cohorts_share_exact_users(tmp_path):
    context, scorer = fixture_context()
    result = evaluate_catalogue_scores(context, scorer, model_name="neural", split_seed=17,
                                       output_dir=tmp_path)
    outcomes = result["outcomes"]
    assert outcomes.user_id.tolist() == [1, 2, 3]
    assert outcomes.rank_at_10.tolist() == [2, 2, 2]
    assert outcomes.used_popularity_fallback.tolist() == [False, True, False]
    assert outcomes.has_positive_fit_history.tolist() == [True, False, True]
    assert outcomes.candidate_works.tolist() == [2, 2, 2]
    assert result["metrics"]["all_relevant"]["users"] == 3
    assert result["metrics"]["fit_positive_history"]["users"] == 2
    assert result["metrics"]["no_fit_positive_history"]["users"] == 1
    assert result["metrics"]["all_relevant"]["ndcg_at_10"] == pytest.approx(1 / np.log2(3))
    for row_index, row in result["top10"].iterrows():
        recommendations = json.loads(row.work_keys)
        excluded = {context["works"][index] for index in context["excluded"][row_index]}
        assert not excluded.intersection(recommendations)
    # Stable lexical tie-breaking puts A before D in user 2's popularity fallback.
    assert json.loads(result["top10"].iloc[1].work_keys) == context["works"][[0, 3]].tolist()
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "neural_outcomes.csv"), outcomes)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "neural_top10.csv"), result["top10"])


def test_popularity_is_control_and_not_counted_as_neural_fallback():
    context, _ = fixture_context()
    result = evaluate_catalogue_scores(context, None, model_name="popularity", split_seed=17)
    assert not result["outcomes"].used_popularity_fallback.any()
    assert result["outcomes"].rank_at_10.tolist() == [1, 2, 1]


@pytest.mark.parametrize("scores", [np.zeros((3, 3)), np.full((3, 4), np.nan)])
def test_invalid_catalogue_scores_are_rejected(scores):
    context, _ = fixture_context()
    with pytest.raises(ValueError, match="Invalid catalogue score"):
        evaluate_catalogue_scores(context, lambda users: scores, model_name="broken", split_seed=17)


def assessment_inputs():
    context, scorer = fixture_context()
    names = {arm: f"split17_{arm}_seed42" for arm in ("mse_id", "bpr_id", "bpr_metadata")}
    evaluations = {name: evaluate_catalogue_scores(context, scorer, model_name=name, split_seed=17)
                   for name in names.values()}
    pop = evaluate_catalogue_scores(context, None, model_name="popularity", split_seed=17)
    old_name = "split17_mlp_seed42"
    old_outcomes = pop["outcomes"].assign(model=old_name)
    old_top = pop["top10"].assign(model=old_name)
    previous_outcomes = pd.concat([pop["outcomes"], old_outcomes], ignore_index=True)
    previous_top = pd.concat([pop["top10"], old_top], ignore_index=True)
    return context, evaluations, previous_outcomes, previous_top, names, old_name


def test_assessment_reuses_frozen_outcomes_and_paired_comparisons_after_csv_roundtrip(tmp_path):
    context, evaluations, previous, top, names, old = assessment_inputs()
    # CSV serialization can change the last bit of NDCG. Reuse should tolerate
    # representational error, while requiring the same users/targets/ranks.
    previous.to_csv(tmp_path / "prior.csv", index=False)
    previous = pd.read_csv(tmp_path / "prior.csv")
    result = assemble_assessment_report(context, evaluations, previous, top,
        primary_models=names, old_primary_mlp=old, split_seed=17, output_dir=tmp_path / "report")
    assert result["cohort_counts"] == {"all_relevant": 3, "fit_positive_history": 2,
                                      "no_fit_positive_history": 1}
    assert len(result["metrics"]) == 5
    assert result["reused_models"] == [old]
    assert set(result["new_models"]) == set(names.values())
    assert len(result["paired_bootstrap"]) == 60
    for comparison in result["paired_bootstrap"]:
        assert comparison["users"] == (3 if comparison["cohort"] == "all_relevant" else 2)
        if comparison["reference"] in names.values():
            assert comparison["difference"] == 0
            assert comparison["ci95"] == [0, 0]
    combined = pd.read_csv(tmp_path / "report/ranking_user_outcomes.csv")
    assert len(combined) == 5 * 3
    assert combined.groupby("model").user_id.nunique().eq(3).all()


def test_assessment_refuses_misaligned_prior_user_outcomes(tmp_path):
    context, evaluations, previous, top, names, old = assessment_inputs()
    previous = previous[~((previous.model == old) & (previous.user_id == 3))]
    with pytest.raises(AssertionError):
        assemble_assessment_report(context, evaluations, previous, top,
            primary_models=names, old_primary_mlp=old, split_seed=17, output_dir=tmp_path)
