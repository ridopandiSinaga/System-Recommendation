"""History-size cohorts, classical tie policy, and locked development comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .ranking_comparison import paired_mean_bootstrap
    from .ranking_objective_evaluation import prepare_evaluation, summarize_outcomes
except ImportError:
    from ranking_comparison import paired_mean_bootstrap
    from ranking_objective_evaluation import prepare_evaluation, summarize_outcomes

METRICS = ("recall_at_10", "ndcg_at_10", "mrr_at_10")
COHORTS = ("all_relevant", "fit_positive_history", "no_fit_positive_history",
           "single_fit_positive", "multiple_fit_positive")
COMPARISON_COHORTS = ("all_relevant", "fit_positive_history", "multiple_fit_positive")


def prepare_history_evaluation(books, fit, heldout, work_data, history_data):
    context = prepare_evaluation(books, fit, heldout, work_data)
    indices = context["encoded_users"]
    counts = np.asarray(history_data.positive_counts)
    context["positive_counts"] = np.asarray([counts[i] if i >= 0 else 0 for i in indices], dtype=int)
    np.testing.assert_array_equal(context["positive_counts"] > 0, context["positive_history"])
    return context


def history_cohort_mask(outcomes, name):
    counts = outcomes.positive_fit_count.to_numpy(int)
    if name == "all_relevant":
        return np.ones(len(outcomes), dtype=bool)
    if name == "fit_positive_history":
        return counts > 0
    if name == "no_fit_positive_history":
        return counts == 0
    if name == "single_fit_positive":
        return counts == 1
    if name == "multiple_fit_positive":
        return counts >= 2
    raise ValueError(f"Unknown history cohort: {name}")


def summarize_history_outcomes(outcomes, top10, works):
    result = summarize_outcomes(outcomes, top10, works)
    for name in COHORTS[3:]:
        selected = outcomes.loc[history_cohort_mask(outcomes, name)]
        top = top10.loc[top10.user_id.isin(selected.user_id)]
        result[name] = summarize_outcomes(selected, top, works)["all_relevant"]
    return result


def evaluate_history_scores(context, score_catalog, *, model_name, split_seed,
                            output_dir=None, popularity_tiebreak=False):
    """Scores must be finite; recipient exclusions and zero-history fallback are fixed."""
    targets, works = context["targets"], context["works"]
    rows, recommendations = [], []
    for start in range(0, len(targets), 128):
        stop = min(start + 128, len(targets))
        indices = context["encoded_users"][start:stop]
        if score_catalog is None:
            scores = np.broadcast_to(context["popularity"], (stop-start, len(works))).astype(float).copy()
        else:
            scores = np.array(score_catalog(np.maximum(indices, 0)), dtype=float, copy=True)
        if scores.shape != (stop-start, len(works)) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid history catalogue scores: {model_name}")
        fallback = context["positive_counts"][start:stop] == 0
        scores[fallback] = context["popularity"]
        for local, absolute in enumerate(range(start, stop)):
            scores[local, context["excluded"][absolute]] = -np.inf
        if popularity_tiebreak:
            # Classical sparse zero-score tails use a declared secondary prior.
            # Raw scores are preserved; no epsilon changes their ordering.
            secondary = np.broadcast_to(-context["popularity"], scores.shape)
            lexical = np.broadcast_to(np.arange(len(works)), scores.shape)
            ordered = np.lexsort((lexical, secondary, -scores), axis=1)[:, :10]
        else:
            ordered = np.argsort(-scores, axis=1, kind="stable")[:, :10]
        for local, absolute in enumerate(range(start, stop)):
            target = targets.iloc[absolute]
            top = ordered[local]
            top = top[np.isfinite(scores[local, top])]
            recs = works[top].tolist()
            rank = recs.index(target.work_key) + 1 if target.work_key in recs else 0
            count = int(context["positive_counts"][absolute])
            rows.append({"split_seed": split_seed, "model": model_name,
                         "user_id": int(target.user_id), "target_work": target.work_key,
                         "target_available": bool(target.target_available), "known_user": bool(target.known_user),
                         "positive_fit_count": count, "has_positive_fit_history": count > 0,
                         "used_popularity_fallback": bool(fallback[local] and score_catalog is not None),
                         "candidate_works": len(works)-len(context["excluded"][absolute]),
                         "rank_at_10": rank, "recall_at_10": float(rank > 0),
                         "ndcg_at_10": float(1/np.log2(rank+1)) if rank else 0.,
                         "mrr_at_10": 1/rank if rank else 0.})
            recommendations.append({"split_seed": split_seed, "model": model_name,
                                    "user_id": int(target.user_id),
                                    "work_keys": json.dumps(recs, ensure_ascii=False),
                                    "raw_scores": json.dumps(scores[local, top].tolist())})
    outcomes, top10 = pd.DataFrame(rows), pd.DataFrame(recommendations)
    result = {"metrics": summarize_history_outcomes(outcomes, top10, works),
              "outcomes": outcomes, "top10": top10,
              "tie_policy": "score/popularity/lexical work" if popularity_tiebreak else "score/lexical work"}
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        outcomes.to_csv(directory / f"{model_name}_outcomes.csv", index=False)
        top10.to_csv(directory / f"{model_name}_top10.csv", index=False)
    return result


def compare_history_outcomes(candidate, baseline, *, model, reference):
    np.testing.assert_array_equal(candidate.user_id, baseline.user_id)
    np.testing.assert_array_equal(candidate.target_work, baseline.target_work)
    np.testing.assert_array_equal(candidate.positive_fit_count, baseline.positive_fit_count)
    comparisons = []
    for cohort in COMPARISON_COHORTS:
        mask = history_cohort_mask(candidate, cohort)
        if not mask.any():
            continue
        for metric in METRICS:
            comparisons.append({"model": model, "reference": reference, "cohort": cohort,
                                "metric": metric, **paired_mean_bootstrap(
                                    candidate.loc[mask, metric], baseline.loc[mask, metric], np)})
    return comparisons


def assemble_history_assessment_report(context, evaluations, previous_outcomes, previous_top10, *,
                                       primary_models, baseline_name, previous_primary_models,
                                       split_seed, output_dir):
    results = dict(evaluations)
    if "popularity" in results:
        raise ValueError("Popularity is recomputed by the report assembler")
    results["popularity"] = evaluate_history_scores(context, None, model_name="popularity", split_seed=split_seed)
    count_map = dict(zip(context["users"], context["positive_counts"]))
    for name, frame in previous_outcomes.groupby("model", sort=False):
        frame = frame.sort_values("user_id").reset_index(drop=True).copy()
        top = previous_top10.loc[previous_top10.model == name].sort_values("user_id").reset_index(drop=True).copy()
        np.testing.assert_array_equal(frame.user_id, context["users"])
        np.testing.assert_array_equal(top.user_id, context["users"])
        np.testing.assert_array_equal(frame.target_work, context["targets"].work_key)
        frame["positive_fit_count"] = frame.user_id.map(count_map).astype(int)
        np.testing.assert_array_equal(frame.has_positive_fit_history, frame.positive_fit_count > 0)
        if name == "popularity":
            for metric in (*METRICS, "rank_at_10"):
                np.testing.assert_allclose(frame[metric], results[name]["outcomes"][metric], rtol=0, atol=1e-14)
            assert top.work_keys.tolist() == results[name]["top10"].work_keys.tolist()
        elif name in results:
            raise ValueError(f"New/prior model name collision: {name}")
        else:
            results[name] = {"outcomes": frame, "top10": top,
                             "metrics": summarize_history_outcomes(frame, top, context["works"])}
    required = [baseline_name, *primary_models.values(), *previous_primary_models.values()]
    if any(name not in results for name in required):
        raise ValueError("Missing preselected model")
    previous_bpr = previous_primary_models["bpr_id"]
    comparisons = []
    for name, evaluated in evaluations.items():
        references = ["popularity", previous_bpr]
        if "_history_" in name:
            suffix = name.rsplit("_seed", 1)[-1]
            other = "history_softmax" if "_history_bpr_" in name else "history_bpr"
            references.extend([baseline_name, f"split{split_seed}_{other}_seed{suffix}"])
        for reference in dict.fromkeys(references):
            if reference == name:
                continue
            if reference not in results:
                raise ValueError(f"Missing paired reference {reference}")
            comparisons.extend(compare_history_outcomes(evaluated["outcomes"], results[reference]["outcomes"],
                                                        model=name, reference=reference))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat([x["outcomes"] for x in results.values()], ignore_index=True).to_csv(
        directory / "ranking_user_outcomes.csv", index=False)
    pd.concat([x["top10"] for x in results.values()], ignore_index=True).to_csv(
        directory / "ranking_top10.csv", index=False)
    return {"split_seed": split_seed, "catalogue_works": len(context["works"]),
            "cohort_counts": {c: int(history_cohort_mask(results["popularity"]["outcomes"], c).sum()) for c in COHORTS},
            "primary_models": primary_models, "baseline_name": baseline_name,
            "previous_primary_models": previous_primary_models,
            "metrics": {name: value["metrics"] for name, value in results.items()},
            "paired_bootstrap": comparisons, "new_models": list(evaluations),
            "reused_models": [name for name in results if name not in evaluations and name != "popularity"],
            "fallback_policy": "All new models use fit-only popularity only for zero-positive-history recipients",
            "scope": "Adaptive reused internal development; no independent test confirmation; conditional unadjusted intervals",
            "tie_policy": "Neural and popularity lexical; sparse cosine uses popularity then lexical on exact raw-score ties"}
