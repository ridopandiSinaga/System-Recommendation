"""Work-level evaluation shared by selection and locked ranking assessment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .ranking_comparison import paired_mean_bootstrap, prepare_ranking_data
except ImportError:
    from ranking_comparison import paired_mean_bootstrap, prepare_ranking_data

METRICS = ("recall_at_10", "ndcg_at_10", "mrr_at_10")
COHORTS = ("all_relevant", "fit_positive_history", "no_fit_positive_history")


def prepare_evaluation(books, fit, heldout, work_data):
    _, works, _, work_index, seen, targets, popularity = prepare_ranking_data(
        books, fit, heldout, np, pd)
    if list(works) != list(work_data.work_keys):
        raise ValueError("Training and evaluation canonical catalogues differ")
    np.testing.assert_array_equal(popularity, work_data.popularity)
    users = targets.user_id.to_numpy()
    encoded = np.asarray([work_data.user_to_index.get(int(u), -1) for u in users], dtype=np.int32)
    positive = np.asarray([i >= 0 and bool(work_data.positive_user_mask[i]) for i in encoded])
    excluded = [np.asarray(sorted(work_index[w] for w in seen.get(u, set()) if w in work_index), dtype=int)
                for u in users]
    return {"works": works, "targets": targets, "users": users, "encoded_users": encoded,
            "positive_history": positive, "excluded": excluded, "popularity": popularity,
            "assessment_users": int(heldout.user_id.nunique())}


def cohort_mask(outcomes, name):
    if name == "all_relevant":
        return np.ones(len(outcomes), dtype=bool)
    if name == "fit_positive_history":
        return outcomes.has_positive_fit_history.to_numpy(bool)
    if name == "no_fit_positive_history":
        return ~outcomes.has_positive_fit_history.to_numpy(bool)
    raise ValueError(f"Unknown cohort: {name}")


def summarize_outcomes(outcomes, top10, works):
    summaries = {}
    for cohort in COHORTS:
        frame = outcomes.loc[cohort_mask(outcomes, cohort)]
        selected = top10.loc[top10.user_id.isin(frame.user_id)]
        counts = {}
        for value in selected.work_keys:
            for work in json.loads(value):
                counts[work] = counts.get(work, 0) + 1
        frequency = np.asarray(list(counts.values()), dtype=float)
        probability = frequency / frequency.sum() if frequency.size else frequency
        entropy = float(-np.sum(probability * np.log(probability)))
        summaries[cohort] = {"users": len(frame), "hits_at_10": int(frame.recall_at_10.sum()),
                            **{metric: float(frame[metric].mean()) if len(frame) else None for metric in METRICS},
                            "recommended_unique_works": len(counts),
                            "catalogue_coverage": len(counts) / len(works),
                            "frequency_entropy": entropy,
                            "normalized_frequency_entropy": entropy / float(np.log(len(works)))
                            if len(works) > 1 else 0.,
                            "popularity_fallback_users": int(frame.used_popularity_fallback.sum())}
    return summaries


def evaluate_catalogue_scores(context, score_catalog, *, model_name, split_seed, output_dir=None):
    """Apply identical fit-positive-history fallback and fit-work exclusions."""
    targets, works = context["targets"], context["works"]
    rows, recommendations = [], []
    for start in range(0, len(targets), 128):
        stop = min(start + 128, len(targets))
        indices = context["encoded_users"][start:stop]
        if score_catalog is None:
            scores = np.broadcast_to(context["popularity"], (stop-start, len(works))).copy()
        else:
            scores = np.array(score_catalog(np.maximum(indices, 0)), dtype=float, copy=True)
        if scores.shape != (stop-start, len(works)) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid catalogue score matrix: {model_name}")
        fallback = ~context["positive_history"][start:stop]
        scores[fallback] = context["popularity"]
        for local, absolute in enumerate(range(start, stop)):
            scores[local, context["excluded"][absolute]] = -np.inf
        ordered = np.argsort(-scores, axis=1, kind="stable")[:, :10]
        for local, absolute in enumerate(range(start, stop)):
            target = targets.iloc[absolute]
            top = ordered[local]
            top = top[np.isfinite(scores[local, top])]
            recs = works[top].tolist()
            rank = recs.index(target.work_key) + 1 if target.work_key in recs else 0
            rows.append({"split_seed": split_seed, "model": model_name,
                         "user_id": int(target.user_id), "target_work": target.work_key,
                         "target_available": bool(target.target_available), "known_user": bool(target.known_user),
                         "has_positive_fit_history": bool(context["positive_history"][absolute]),
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
    result = {"metrics": summarize_outcomes(outcomes, top10, works), "outcomes": outcomes, "top10": top10}
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        outcomes.to_csv(directory / f"{model_name}_outcomes.csv", index=False)
        top10.to_csv(directory / f"{model_name}_top10.csv", index=False)
    return result


def assemble_assessment_report(context, evaluations, previous_outcomes, previous_top10, *,
                               primary_models, old_primary_mlp, split_seed, output_dir):
    """Combine new fits and hash-verified prior outcomes without reselecting old seeds."""
    results = dict(evaluations)
    if "popularity" not in results:
        results["popularity"] = evaluate_catalogue_scores(context, None, model_name="popularity", split_seed=split_seed)
    history_map = dict(zip(context["users"], context["positive_history"]))
    for name, frame in previous_outcomes.groupby("model", sort=False):
        frame = frame.sort_values("user_id").reset_index(drop=True).copy()
        top = previous_top10.loc[previous_top10.model == name].sort_values("user_id").reset_index(drop=True).copy()
        np.testing.assert_array_equal(frame.user_id, context["users"])
        np.testing.assert_array_equal(top.user_id, context["users"])
        np.testing.assert_array_equal(frame.target_work, context["targets"].work_key)
        frame["has_positive_fit_history"] = frame.user_id.map(history_map).astype(bool)
        # Previously stored MLP/bias/MF predictions did not use this new fallback.
        frame["used_popularity_fallback"] = False
        if name == "popularity":
            for metric in METRICS:
                np.testing.assert_allclose(frame[metric].to_numpy(), results[name]["outcomes"][metric].to_numpy(), rtol=0, atol=1e-14)
            assert top.work_keys.tolist() == results[name]["top10"].work_keys.tolist()
        elif name in results:
            raise ValueError(f"New/prior model name collision: {name}")
        else:
            results[name] = {"outcomes": frame, "top10": top,
                             "metrics": summarize_outcomes(frame, top, context["works"])}
    if old_primary_mlp not in results or any(name not in results for name in primary_models.values()):
        raise ValueError("Missing preselected primary model")
    comparisons = []
    for name, evaluated in evaluations.items():
        # Every seed's matched MSE and BPR controls are predeclared, not selected on assessment.
        seed_suffix = name.rsplit("_seed", 1)[-1]
        references = ["popularity", old_primary_mlp,
                      f"split{split_seed}_mse_id_seed{seed_suffix}", f"split{split_seed}_bpr_id_seed{seed_suffix}"]
        for reference in dict.fromkeys(references):
            if reference == name:
                continue
            if reference not in results:
                raise ValueError(f"Missing paired control {reference}")
            candidate, baseline = evaluated["outcomes"], results[reference]["outcomes"]
            np.testing.assert_array_equal(candidate.user_id, baseline.user_id)
            for cohort in COHORTS[:2]:
                mask = cohort_mask(candidate, cohort)
                if not mask.any():
                    continue
                for metric in METRICS:
                    comparisons.append({"model": name, "reference": reference, "cohort": cohort,
                                        "metric": metric, **paired_mean_bootstrap(
                                            candidate.loc[mask, metric], baseline.loc[mask, metric], np)})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat([x["outcomes"] for x in results.values()], ignore_index=True).to_csv(
        directory / "ranking_user_outcomes.csv", index=False)
    pd.concat([x["top10"] for x in results.values()], ignore_index=True).to_csv(
        directory / "ranking_top10.csv", index=False)
    cohort_counts = {name: int(cohort_mask(results["popularity"]["outcomes"], name).sum()) for name in COHORTS}
    return {"split_seed": split_seed, "catalogue_works": len(context["works"]),
            "cohort_counts": cohort_counts, "primary_models": primary_models, "old_primary_mlp": old_primary_mlp,
            "metrics": {name: value["metrics"] for name, value in results.items()},
            "paired_bootstrap": comparisons, "reused_models": [n for n in results if n not in evaluations and n != "popularity"],
            "new_models": list(evaluations),
            "fallback_policy": "All new arms use fit-only popularity for users without fit rating>=8; conditional cohort excludes those users",
            "scope": "Previously inspected internal development assessment; conditional unadjusted bootstrap; no independent confirmation",
            "unobserved_relevance": "Other unobserved catalogue works are not known negatives; one held-out relevant work is the measured target"}
