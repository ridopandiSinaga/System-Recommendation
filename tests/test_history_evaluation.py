"""Recipient history, sparse ties, cohorts, and prior evidence reuse."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from book_recommender.ranking_neural import prepare_work_data
from scripts.history_objective_evaluation import (
    assemble_history_assessment_report, compare_history_outcomes,
    evaluate_history_scores, prepare_history_evaluation,
)


def fixture_context():
    books = pd.DataFrame([{'isbn': k, 'book_title': 'Book '+k, 'book_author': 'Writer'} for k in 'ABCD'])
    fit = pd.DataFrame([(1, 'A', 9), (1, 'B', 2), (2, 'B', 3), (2, 'C', 5),
                        (3, 'C', 8), (3, 'D', 9), (4, 'D', 8)],
                       columns=['user_id', 'isbn', 'book_rating'])
    heldout = pd.DataFrame([(1, 'C', 9), (2, 'D', 8), (3, 'A', 10)], columns=fit.columns)
    work = prepare_work_data(books, fit, max_features=8)
    return prepare_history_evaluation(books, fit, heldout, work,
                                      SimpleNamespace(positive_counts=np.array([1, 0, 2, 1])))


def test_history_cohorts_and_explicit_classical_secondary_ties(tmp_path):
    context = fixture_context()
    callback = lambda users: np.zeros((len(users), 4))  # noqa: E731
    neural = evaluate_history_scores(context, callback, model_name='neural', split_seed=17, output_dir=tmp_path)
    classic = evaluate_history_scores(context, callback, model_name='cosine', split_seed=17, popularity_tiebreak=True)
    assert neural['outcomes'].rank_at_10.tolist() == [1, 1, 1]
    assert classic['outcomes'].rank_at_10.tolist() == [2, 1, 1]
    assert neural['outcomes'].used_popularity_fallback.tolist() == [False, True, False]
    assert {k: v['users'] for k, v in neural['metrics'].items()} == {
        'all_relevant': 3, 'fit_positive_history': 2, 'no_fit_positive_history': 1,
        'single_fit_positive': 1, 'multiple_fit_positive': 1}
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'neural_outcomes.csv'), neural['outcomes'])


def test_nonfinite_or_wrong_scores_refused():
    context = fixture_context()
    for bad in [np.zeros((3, 3)), np.full((3, 4), np.inf)]:
        with pytest.raises(ValueError, match='Invalid history catalogue scores'):
            evaluate_history_scores(context, lambda users: bad, model_name='broken', split_seed=17)


def test_paired_intervals_require_same_users_targets_and_history():
    context = fixture_context()
    result = evaluate_history_scores(context, None, model_name='popularity', split_seed=17)
    comparisons = compare_history_outcomes(result['outcomes'], result['outcomes'], model='x', reference='x')
    assert len(comparisons) == 9
    assert all(row['difference'] == 0 and row['ci95'] == [0, 0] for row in comparisons)
    changed = result['outcomes'].copy()
    changed.loc[0, 'positive_fit_count'] += 1
    with pytest.raises(AssertionError):
        compare_history_outcomes(result['outcomes'], changed, model='x', reference='y')


def test_reuses_prior_fallback_flags_and_recomputes_all_five_cohorts(tmp_path):
    context = fixture_context()
    names = {arm: f'split17_{arm}_seed42' for arm in ['history_bpr', 'history_softmax']}
    baseline = 'split17_cosine_k50_shrink0'
    evaluations = {name: evaluate_history_scores(context, lambda users: np.zeros((len(users), 4)),
        model_name=name, split_seed=17, popularity_tiebreak=name == baseline)
        for name in [*names.values(), baseline]}
    pop = evaluate_history_scores(context, None, model_name='popularity', split_seed=17)
    previous_names = {arm: f'split17_{arm}_seed42' for arm in ['mse_id', 'bpr_id', 'bpr_metadata']}
    frames, tops = [pop['outcomes']], [pop['top10']]
    for name in previous_names.values():
        frames.append(pop['outcomes'].assign(model=name, used_popularity_fallback=[False, True, False]))
        tops.append(pop['top10'].assign(model=name))
    report = assemble_history_assessment_report(context, evaluations, pd.concat(frames), pd.concat(tops),
        primary_models=names, baseline_name=baseline, previous_primary_models=previous_names,
        split_seed=17, output_dir=tmp_path)
    assert len(report['metrics']) == 7
    assert len(report['paired_bootstrap']) == 90
    assert report['metrics'][previous_names['bpr_id']]['no_fit_positive_history']['popularity_fallback_users'] == 1
    assert report['cohort_counts']['multiple_fit_positive'] == 1
    assert len(pd.read_csv(tmp_path/'ranking_user_outcomes.csv')) == 21
