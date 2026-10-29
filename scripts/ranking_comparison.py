"""Frozen, full-catalogue work-level ranking on reused development assessment.

No model is selected here. All predictors return raw rating-scale scores.
One relevant held-out work is one user outcome, including unavailable targets.
"""
from __future__ import annotations

import json
from pathlib import Path
import time


def paired_mean_bootstrap(values, reference, np, *, draws=1000, seed=123):
    values, reference = (np.asarray(x, dtype=float).reshape(-1) for x in (values, reference))
    if not len(values) or len(values) != len(reference):
        raise ValueError("Paired outcomes must be aligned and nonempty")
    differences = values - reference
    if not np.isfinite(differences).all() or draws < 1:
        raise ValueError("Finite outcomes and positive draws are required")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for start in range(0, draws, 100):
        chosen = rng.integers(len(values), size=(min(100, draws - start), len(values)))
        samples[start:start + len(chosen)] = differences[chosen].mean(axis=1)
    return {
        "difference": float(differences.mean()),
        "ci95": np.quantile(samples, [.025, .975]).tolist(),
        "users": len(values), "draws": draws, "seed": seed,
        "direction": "Positive favors candidate; conditional unadjusted user bootstrap",
    }


def prepare_ranking_data(books, fit, assessment, np, pd):
    from book_recommender.content_based import book_work_key

    catalogue = books.copy()
    catalogue["isbn"] = catalogue.isbn.astype(str)
    if catalogue.isbn.duplicated().any():
        raise ValueError("Catalogue ISBNs must be unique")
    catalogue["work_key"] = [book_work_key(r.book_title, r.book_author, r.isbn)
                             for r in catalogue.itertuples(index=False)]
    catalogue = catalogue.sort_values(["work_key", "isbn"]).reset_index(drop=True)
    work_map = dict(zip(catalogue.isbn, catalogue.work_key))
    works, starts = np.unique(catalogue.work_key.to_numpy(), return_index=True)
    work_index = {work: i for i, work in enumerate(works)}

    def attach(frame):
        frame = frame.copy()
        computed = frame.isbn.astype(str).map(work_map)
        # Preserve known work keys for unavailable targets, so they remain misses.
        if "work_key" in frame:
            mismatch = computed.notna() & (computed != frame.work_key)
            if mismatch.any():
                raise ValueError("Stored canonical work differs from frozen catalogue")
            computed = computed.fillna(frame.work_key)
        if computed.isna().any():
            raise ValueError("An unavailable ISBN needs an explicit canonical work key")
        frame["work_key"] = computed
        return frame

    fit, assessment = attach(fit), attach(assessment)
    if (assessment.groupby("user_id").work_key.nunique() != 1).any():
        raise ValueError("Ranking requires exactly one held-out canonical work per user")
    seen = fit.groupby("user_id").work_key.agg(set).to_dict()
    for row in assessment.itertuples(index=False):
        if row.work_key in seen.get(row.user_id, set()):
            raise ValueError("Held-out work overlaps fit history")
    targets = assessment.groupby(["user_id", "work_key"], sort=True).book_rating.max().reset_index()
    targets = targets.loc[targets.book_rating >= 8].sort_values("user_id").reset_index(drop=True)
    if targets.empty:
        raise ValueError("No relevant held-out work exists")
    targets["target_available"] = targets.work_key.isin(work_index)
    targets["known_user"] = targets.user_id.isin(set(fit.user_id))
    liked = fit.loc[fit.book_rating >= 8, ["user_id", "work_key"]].drop_duplicates()
    popularity = liked.groupby("work_key").size().reindex(works, fill_value=0).to_numpy(float)
    return catalogue, works, starts, work_index, seen, targets, popularity


def run_ranking_comparison(*, split_seed, books, fit, assessment, predictors,
                           primary_mlp, primary_mf, bias_name, output_dir, np, pd):
    started = time.monotonic()
    if "popularity" in predictors:
        raise ValueError("popularity is reserved for the fit-only control")
    if any(name not in predictors for name in (primary_mlp, primary_mf, bias_name)):
        raise ValueError("Preselected primary models must be present")
    catalogue, works, starts, work_index, seen, targets, popularity = prepare_ranking_data(
        books, fit, assessment, np, pd)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output / "ranking_targets.csv", index=False)
    pd.DataFrame({"work_key": works, "fit_positive_users": popularity.astype(int)}).to_csv(
        output / "ranking_popularity.csv", index=False)
    isbns = catalogue.isbn.to_numpy(str)
    users = targets.user_id.to_numpy()
    metrics, outcomes = {}, {}
    all_recommendations, all_outcomes = [], []
    for name, predictor in [*predictors.items(), ("popularity", None)]:
        counts = np.zeros(len(works), dtype=int)
        model_outcomes = []
        for start in range(0, len(users), 32):
            block_users = users[start:start + 32]
            if predictor is None:
                scores = np.broadcast_to(popularity, (len(block_users), len(works))).copy()
            else:
                raw = np.asarray(predictor(np.repeat(block_users, len(isbns)),
                                           np.tile(isbns, len(block_users))), dtype=float)
                if raw.size != len(block_users) * len(isbns) or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid raw predictions from {name}")
                scores = np.maximum.reduceat(raw.reshape(len(block_users), -1), starts, axis=1)
            for local_index, user in enumerate(block_users):
                row = targets.iloc[start + local_index]
                excluded = [work_index[w] for w in seen.get(user, set()) if w in work_index]
                scores[local_index, excluded] = -np.inf
                # Works are lexicographically sorted; stable sorting resolves ties.
                order = np.argsort(-scores[local_index], kind="stable")
                top = order[np.isfinite(scores[local_index, order])][:10]
                counts[top] += 1
                recommendations = works[top].tolist()
                rank = recommendations.index(row.work_key) + 1 if row.work_key in recommendations else 0
                result = {"split_seed": split_seed, "model": name, "user_id": int(user),
                          "target_work": row.work_key, "target_available": bool(row.target_available),
                          "known_user": bool(row.known_user), "candidate_works": len(works) - len(excluded),
                          "rank_at_10": rank, "recall_at_10": float(rank > 0),
                          "ndcg_at_10": float(1 / np.log2(rank + 1)) if rank else 0.,
                          "mrr_at_10": 1. / rank if rank else 0.}
                model_outcomes.append(result)
                all_recommendations.append({"split_seed": split_seed, "model": name,
                                            "user_id": int(user),
                                            "work_keys": json.dumps(recommendations, ensure_ascii=False),
                                            "raw_scores": json.dumps(scores[local_index, top].tolist())})
        frame = pd.DataFrame(model_outcomes)
        outcomes[name] = frame
        all_outcomes.extend(model_outcomes)
        frequency = counts[counts > 0] / counts.sum() if counts.sum() else np.asarray([])
        entropy = float(-(frequency * np.log(frequency)).sum())
        metrics[name] = {"users": len(frame), "hits_at_10": int(frame.recall_at_10.sum()),
                         **{key: float(frame[key].mean()) for key in
                            ("recall_at_10", "ndcg_at_10", "mrr_at_10")},
                         "recommended_unique_works": int((counts > 0).sum()),
                         "catalogue_coverage": float((counts > 0).mean()),
                         "recommendation_frequency_entropy": entropy,
                         "normalized_frequency_entropy": entropy / float(np.log(len(works)))
                         if len(works) > 1 else 0.}
        pd.DataFrame({"work_key": works, "recommendation_count": counts}).to_csv(
            output / f"ranking_frequency_{name}.csv", index=False)
    pd.DataFrame(all_outcomes).to_csv(output / "ranking_user_outcomes.csv", index=False)
    pd.DataFrame(all_recommendations).to_csv(output / "ranking_top10.csv", index=False)
    comparisons = []
    for name, frame in outcomes.items():
        for reference in dict.fromkeys([bias_name, "popularity", primary_mlp]):
            if name == reference:
                continue
            for metric in ("recall_at_10", "ndcg_at_10", "mrr_at_10"):
                comparisons.append({"model": name, "reference": reference, "metric": metric,
                                    **paired_mean_bootstrap(frame[metric], outcomes[reference][metric], np)})
    return {"split_seed": split_seed, "catalogue_isbns": len(catalogue),
            "catalogue_works": len(works), "assessment_users": int(assessment.user_id.nunique()),
            "relevant_users": len(targets), "available_targets": int(targets.target_available.sum()),
            "target_coverage": float(targets.target_available.mean()),
            "known_users": int(targets.known_user.sum()), "primary_mlp": primary_mlp,
            "primary_mf": primary_mf, "bias_name": bias_name,
            "protocol": "Full frozen catalogue; max raw ISBN score per canonical work; exclude all fit works; "
            "one held-out work/user; relevant if any edition rated >=8; unavailable targets count as misses; "
            "ties by lexicographic work key; Recall@10 equals Hit@10; no ranking-based tuning",
            "popularity_definition": "Distinct fit users with rating>=8 per canonical work",
            "dispersion_limitation": "Coverage and frequency entropy measure exposure dispersion, not semantic diversity",
            "evaluation_scope": "Reused internal development assessment; no fresh independent test",
            "metrics": metrics, "paired_bootstrap": comparisons,
            "elapsed_seconds": time.monotonic() - started}
