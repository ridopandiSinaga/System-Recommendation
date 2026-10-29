"""Independent score ordering and recipient/donor semantics in the local audit."""
import numpy as np

from scripts.audit_history_assessment import cohort_masks, expected_donors, expected_top10


def test_independent_top10_preserves_raw_scores_and_declared_tie_policy():
    works = ["a", "b", "c", "d"]
    scores = [.5, .5, .50000000000001, 9.]
    popularity = {"a": 1, "b": 1000, "c": 0, "d": 10000}
    excluded = {"d"}  # A donor may contain d, but recipient exclusion still applies.
    assert [item for item, _ in expected_top10(scores, works, excluded, popularity)] == ["c", "a", "b"]
    assert [item for item, _ in expected_top10(scores, works, excluded, popularity,
                                              popularity_tiebreak=True)] == ["c", "b", "a"]
    # Popularity only breaks exact ties and never erases a small raw-score lead.
    assert expected_top10(scores, works, excluded, popularity, popularity_tiebreak=True)[0][1] == scores[2]


def test_independent_donor_cycle_uses_fit_positive_users_and_cohorts_are_nested():
    counts = np.array([0, 1, 0, 4, 2, 0])
    np.testing.assert_array_equal(expected_donors(counts), [0, 3, 2, 4, 1, 5])
    masks = cohort_masks(counts)
    np.testing.assert_array_equal(masks["fit_positive_history"], masks["single_fit_positive"] | masks["multiple_fit_positive"])
    assert not np.any(masks["single_fit_positive"] & masks["multiple_fit_positive"])
    assert not np.any(masks["fit_positive_history"] & masks["no_fit_positive_history"])
    assert masks["all_relevant"].all()
