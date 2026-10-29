"""TensorFlow-free verification of downloaded history-encoder study evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from .audit_ranking_assessment import bootstrap_mean, verify_file_fingerprint
except ImportError:
    from audit_ranking_assessment import bootstrap_mean, verify_file_fingerprint

ARMS = ("history_bpr", "history_softmax")
SEEDS = (42, 2026, 7)
GRID = ((50, 0.), (50, 10.), (200, 0.), (200, 10.))
METRICS = ("recall_at_10", "ndcg_at_10", "mrr_at_10")
COHORTS = ("all_relevant", "fit_positive_history", "no_fit_positive_history",
           "single_fit_positive", "multiple_fit_positive")
COMPARISON_COHORTS = ("all_relevant", "fit_positive_history", "multiple_fit_positive")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def close(actual, expected, *, atol=1e-11):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def cohort_masks(counts):
    counts = np.asarray(counts, dtype=int)
    return {"all_relevant": np.ones(len(counts), dtype=bool), "fit_positive_history": counts > 0,
            "no_fit_positive_history": counts == 0, "single_fit_positive": counts == 1,
            "multiple_fit_positive": counts >= 2}


def expected_top10(scores, works, excluded, popularity, *, popularity_tiebreak=False):
    """Independent heap ordering; exclusions always belong to the recipient."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(works),) or not np.isfinite(scores).all():
        raise ValueError("Full-catalogue scores must be aligned and finite.")
    candidates = ((work, float(score)) for work, score in zip(works, scores) if work not in excluded)
    if popularity_tiebreak:
        def key(pair):
            return -pair[1], -popularity.get(pair[0], 0), pair[0]
    else:
        def key(pair):
            return -pair[1], pair[0]
    return heapq.nsmallest(10, candidates, key=key)


def expected_donors(positive_counts):
    counts = np.asarray(positive_counts)
    donors = np.arange(len(counts), dtype=np.int32)
    positive_users = [index for index, count in enumerate(counts) if count > 0]
    for position, user in enumerate(positive_users):
        donors[user] = positive_users[(position+1) % len(positive_users)]
    return donors


def verify(final, verification_dir):
    root = Path(__file__).resolve().parents[1]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from book_recommender.content_based import book_work_key
    from book_recommender.history_neural import build_item_cosine

    downloaded = read_json(final.parent / "backup_verification.json")["final_backup.tar.gz"]
    if (downloaded["kind"] != "completed_experiment"
            or Path(downloaded["extracted_directory"]).resolve() != final.resolve()):
        raise ValueError("Expected the monitor-verified final experiment extraction.")
    verify_file_fingerprint(final.parent / "final_backup.tar.gz", downloaded)
    for relative, expected_hash in downloaded["files"].items():
        path = final / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Downloaded history experiment file changed: {relative}")
    results = read_json(final / "results.json")
    locked = read_json(final / "weights_locked.json")
    manifest = read_json(final / "inputs/input_manifest.json")
    previous = read_json(final / "inputs/previous_results.json")
    protocol = read_json(verification_dir / "predeclared_protocol.json")
    configuration = results["configuration"]
    counts = {"fingerprinted_files": 0, "neural_selections": 0, "cosine_selections": 0,
              "selection_user_outcomes": 0, "assessment_user_outcomes": 0,
              "shuffle_user_outcomes": 0, "cohort_summaries": 0,
              "paired_bootstrap_intervals": 0, "shuffle_bootstrap_intervals": 0,
              "exact_popularity_or_fallback_lists": 0, "exact_cosine_lists": 0,
              "reused_model_tables": 0}

    def check_file(base, relative, expected):
        verify_file_fingerprint(base / relative, expected)
        counts["fingerprinted_files"] += 1

    assert results["experiment_kind"] == "history_encoder_ranking_reused_development"
    assert results["status"] == "completed" and results["all_candidates_completed"] and results["assessment_performed"]
    assert not results["original_test_used"] and not results["original_validation_used"]
    assert not results["model_promotion_to_application"]
    assert not locked["assessment_scored"] and locked["all_initializations_paired"]
    assert datetime.fromisoformat(protocol["declared_utc"]) < datetime.fromisoformat(configuration["started_utc"])
    assert datetime.fromisoformat(locked["locked_utc"]) < datetime.fromisoformat(results["completed_utc"])
    assert protocol["input_manifest_sha256"] == hashlib.sha256((final / "inputs/input_manifest.json").read_bytes()).hexdigest()
    for relative, expected in locked["files"].items():
        check_file(final, relative, expected)
    for relative, expected in manifest["files"].items():
        check_file(final / "inputs", relative, expected)
    assert {str(path.relative_to(final / "inputs")) for path in (final / "inputs").rglob("*") if path.is_file()} == set(manifest["files"]) | {"input_manifest.json"}
    assert manifest["files"]["previous_results.json"]["sha256"] == manifest["previous_results_sha256"]
    previous_local = root / "artifacts/colab/run-20261014-ranking/final"
    assert hashlib.sha256((previous_local / "results.json").read_bytes()).hexdigest() == manifest["previous_results_sha256"]
    for relative, expected in manifest["previous_ranking_fingerprints"].items():
        assert expected == manifest["files"][relative]
        check_file(previous_local, relative.replace("previous_ranking/", "ranking/", 1), expected)
    for relative, expected in read_json(final / "source_metadata.json")["source_files"].items():
        check_file(verification_dir / "executed_source", relative, expected)
    assert configuration["batch_size"] == 1024 and configuration["learning_rate"] == .001
    assert configuration["regularization_lambda"] == 1e-5
    assert configuration["max_neural_epochs_each"] == 20 and configuration["patience"] == 4
    candidates = results["candidate_validation_results"]
    baselines = results["baseline_grid_results"]
    expected_candidates = [f"split{split}_{arm}_seed{seed}" for split in (17, 29) for seed in SEEDS for arm in ARMS]
    assert [row["name"] for row in candidates] == expected_candidates
    assert len(candidates) == 12 and len(baselines) == 8
    assert all(row["status"] == "completed" and row["model_roundtrip_verified"] for row in candidates + baselines)
    assert all(row["assessment_scored"] for row in candidates)
    assert sum(row["epochs_completed"] for row in candidates) == results["training"]["total_neural_epochs"]
    assert results["primary_selection"] == locked["primary_selection"] == read_json(final / "primary_selection.json")
    assert results["ranking_results"] == read_json(final / "ranking_results.json")
    assert results["history_shuffle_results"] == read_json(final / "history_shuffle_results.json")
    books = read_csv(final / "inputs/train_books.csv", dtype={"isbn": str})
    book_work = {str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn) for row in books.itertuples(index=False)}
    works = sorted(set(book_work.values()))
    work_set = set(works)
    work_index = {work: index for index, work in enumerate(works)}
    assert len(books) == 5000 and len(works) == 4364

    def selection_key(row):
        metrics = row["selection"]["all_relevant"]
        return metrics["ndcg_at_10"], metrics["recall_at_10"]

    def audit_tables(frame, top, targets, *, name, summaries, positive_counts, seen,
                     popularity, new_fallback, partition, oracle=None, popularity_tiebreak=False):
        frame = frame.sort_values("user_id").reset_index(drop=True)
        top = top.sort_values("user_id").reset_index(drop=True)
        assert frame.user_id.is_unique and top.user_id.is_unique
        np.testing.assert_array_equal(frame.user_id, targets.user_id)
        np.testing.assert_array_equal(top.user_id, targets.user_id)
        assert set(frame.model) == set(top.model) == {name}
        actual_counts = frame.user_id.map(positive_counts).fillna(0).to_numpy(int)
        np.testing.assert_array_equal(frame.positive_fit_count, actual_counts)
        np.testing.assert_array_equal(frame.has_positive_fit_history, actual_counts > 0)
        if new_fallback:
            np.testing.assert_array_equal(frame.used_popularity_fallback, actual_counts == 0)
        if name == "popularity":
            assert not frame.used_popularity_fallback.any()
        masks = cohort_masks(actual_counts)
        frequency = {cohort: Counter() for cohort in COHORTS}
        global_pop_order = sorted(works, key=lambda work: (-popularity.get(work, 0), work))
        oracle_rows = None
        for index, row in enumerate(frame.itertuples(index=False)):
            target, recommendations_row = targets.iloc[index], top.iloc[index]
            recs = json.loads(recommendations_row.work_keys)
            scores = np.asarray(json.loads(recommendations_row.raw_scores), dtype=float)
            excluded = seen.get(row.user_id, set())
            assert len(recs) == len(set(recs)) == len(scores) == min(10, len(works)-len(excluded))
            assert set(recs).issubset(work_set) and not set(recs).intersection(excluded)
            assert np.isfinite(scores).all() and np.all(scores[:-1] >= scores[1:])
            for position in range(len(scores)-1):
                if scores[position] == scores[position+1]:
                    first, second = recs[position:position+2]
                    if popularity_tiebreak:
                        assert (-popularity.get(first, 0), first) <= (-popularity.get(second, 0), second)
                    else:
                        assert first <= second
            if name == "popularity" or (new_fallback and actual_counts[index] == 0):
                expected = [work for work in global_pop_order if work not in excluded][:10]
                assert recs == expected
                close(scores, [popularity.get(work, 0) for work in expected])
                counts["exact_popularity_or_fallback_lists"] += 1
            elif oracle is not None:
                if index % 128 == 0 or oracle_rows is None:
                    block_start = index
                    oracle_rows = oracle(frame.user_id.iloc[index:index+128].to_numpy())
                # A block may have begun after a skipped zero-history recipient.
                if index-block_start >= len(oracle_rows):
                    block_start = index
                    oracle_rows = oracle(frame.user_id.iloc[index:index+128].to_numpy())
                expected = expected_top10(oracle_rows[index-block_start], works, excluded,
                                          popularity, popularity_tiebreak=popularity_tiebreak)
                assert recs == [pair[0] for pair in expected]
                close(scores, [pair[1] for pair in expected], atol=1e-9)
                counts["exact_cosine_lists"] += 1
            rank = recs.index(target.work_key)+1 if target.work_key in recs else 0
            assert row.target_work == target.work_key and row.rank_at_10 == rank
            assert bool(row.target_available) == (target.work_key in work_set)
            assert bool(row.known_user) == (row.user_id in seen)
            assert row.candidate_works == len(works)-len(excluded)
            close([row.recall_at_10, row.ndcg_at_10, row.mrr_at_10],
                  [bool(rank), 1/np.log2(rank+1) if rank else 0, 1/rank if rank else 0])
            for cohort, mask in masks.items():
                if mask[index]:
                    frequency[cohort].update(recs)
            counts[f"{partition}_user_outcomes"] += 1
        for cohort, mask in masks.items():
            subset = frame.loc[mask]
            summary = summaries[cohort]
            assert summary["users"] == len(subset) and summary["hits_at_10"] == int(subset.recall_at_10.sum())
            assert summary["popularity_fallback_users"] == int(subset.used_popularity_fallback.sum())
            for metric in METRICS:
                if len(subset):
                    close(subset[metric].mean(), summary[metric])
                else:
                    assert summary[metric] is None
            values = np.asarray(list(frequency[cohort].values()), dtype=float)
            probability = values/values.sum() if len(values) else values
            entropy = float(-np.sum(probability*np.log(probability)))
            assert summary["recommended_unique_works"] == len(values)
            close([summary["catalogue_coverage"], summary["frequency_entropy"], summary["normalized_frequency_entropy"]],
                  [len(values)/len(works), entropy, entropy/np.log(len(works))])
            counts["cohort_summaries"] += 1
        return frame, top

    def audit_comparisons(comparisons, table_map, expected, *, shuffled=False):
        actual = {(row["model"], row["reference"], row["cohort"], row["metric"]) for row in comparisons}
        assert actual == expected and len(actual) == len(comparisons)
        for row in comparisons:
            candidate, reference = table_map[row["model"]], table_map[row["reference"]]
            np.testing.assert_array_equal(candidate.user_id, reference.user_id)
            mask = cohort_masks(candidate.positive_fit_count)[row["cohort"]]
            first, second = (frame.loc[mask, row["metric"]].to_numpy(float) for frame in (candidate, reference))
            assert row["draws"] == 1000 and row["seed"] == 123 and row["users"] == len(first)
            close((first-second).mean(), row["difference"])
            close(bootstrap_mean(first, second, 1000, 123), row["ci95"])
            counts["shuffle_bootstrap_intervals" if shuffled else "paired_bootstrap_intervals"] += 1

    donor_details = {}
    for split in (17, 29):
        directory = final / f"inputs/splits/split_{split}"
        frames = {part: read_csv(directory / f"{part}_ratings.csv", dtype={"isbn": str}) for part in ("fit", "selection", "assessment")}
        fit = frames["fit"]
        previous_split = next(row for row in previous["dataset"]["splits"] if row["split_seed"] == split)
        for part, frame in frames.items():
            check_file(directory, f"{part}_ratings.csv", previous_split["partition_file_fingerprints"][f"{part}_ratings.csv"])
            np.testing.assert_array_equal(frame.work_key, frame.isbn.map(book_work))
        source_rows = np.concatenate([frame.source_row.to_numpy() for frame in frames.values()])
        np.testing.assert_array_equal(np.sort(source_rows), np.arange(111875))
        assert set(frames["selection"].user_id).isdisjoint(frames["assessment"].user_id)
        seen = fit.groupby("user_id").work_key.agg(set).to_dict()
        for part in ("selection", "assessment"):
            assert not set(zip(frames[part].user_id, frames[part].work_key)).intersection(zip(fit.user_id, fit.work_key))
        users = sorted(fit.user_id.unique().tolist())
        user_index = {user: index for index, user in enumerate(users)}
        liked = fit.loc[fit.book_rating >= 8, ["user_id", "work_key"]].drop_duplicates().sort_values(["user_id", "work_key"])
        positive_count_map = liked.groupby("user_id").size().to_dict()
        popularity = liked.groupby("work_key").size().to_dict()
        positive_csr = sparse.csr_matrix((np.ones(len(liked), np.float32),
            (liked.user_id.map(user_index), liked.work_key.map(work_index))), shape=(len(users), len(works)))
        positive_csr.sort_indices()
        positive_counts = np.diff(positive_csr.indptr).astype(np.int32)
        all_pairs = np.column_stack([np.repeat(np.arange(len(users), dtype=np.int32), positive_counts), positive_csr.indices])
        eligible_pairs = all_pairs[positive_counts[all_pairs[:, 0]] >= 2].astype(np.int32)
        history_dir = final / f"histories/split_{split}"
        stored_csr = sparse.load_npz(history_dir / "positive_history.npz")
        assert stored_csr.shape == positive_csr.shape and (stored_csr != positive_csr).nnz == 0
        with np.load(history_dir / "training_data.npz") as stored:
            np.testing.assert_array_equal(stored["eligible_pairs"], eligible_pairs)
            np.testing.assert_array_equal(stored["positive_counts"], positive_counts)
            donors = expected_donors(positive_counts)
            np.testing.assert_array_equal(stored["history_donor_indices"], donors)
        vocabulary = read_json(final / f"vocabularies/split_{split}.json")
        assert vocabulary["user_ids"] == users and vocabulary["work_keys"] == works
        assert vocabulary["isbn_to_work"] == {isbn: book_work[isbn] for isbn in sorted(fit.isbn.unique())}
        summary = next(row for row in results["dataset"]["training_work_summaries"] if row["split_seed"] == split)
        assert summary["training_examples"] == len(eligible_pairs)
        assert summary["fit_positive_user_work_pairs"] == len(liked)
        assert summary["fit_no_positive_users"] == int((positive_counts == 0).sum())
        assert summary["fit_single_positive_users"] == int((positive_counts == 1).sum())
        assert summary["fit_multiple_positive_users"] == int((positive_counts >= 2).sum())
        assert summary["trained_positive_target_works"] == len(np.unique(eligible_pairs[:, 1]))
        targets = {}
        for part in ("selection", "assessment"):
            target = frames[part].groupby(["user_id", "work_key"]).book_rating.max().reset_index()
            targets[part] = target[target.book_rating >= 8].sort_values("user_id").reset_index(drop=True)
        common = {"positive_counts": positive_count_map, "seen": seen, "popularity": popularity}
        split_candidates = [row for row in candidates if row["split_seed"] == split]
        split_baselines = [row for row in baselines if row["split_seed"] == split]
        assert [(row["k"], row["shrink"]) for row in split_baselines] == list(GRID)
        baseline_objects = {}
        for row in split_baselines:
            name = row["name"]
            baseline_dir = final / f"baselines/{name}"
            for relative, expected in row["selected_model_fingerprints"].items():
                check_file(baseline_dir, relative, expected)
            baseline = build_item_cosine(SimpleNamespace(positive_csr=positive_csr), k=row["k"], shrink=row["shrink"])
            stored = sparse.load_npz(baseline_dir / "similarity.npz")
            assert stored.shape == baseline.similarity.shape
            difference = (stored-baseline.similarity).tocsr()
            assert not difference.nnz or np.max(abs(difference.data)) < 1e-12
            assert stored.nnz == row["similarity_nonzero"]
            baseline_objects[name] = baseline
            def oracle(original_users, baseline=baseline):
                return baseline.score_catalog(np.asarray([user_index[u] for u in original_users], dtype=np.int32))
            audit_tables(read_csv(baseline_dir / f"selection/{name}_outcomes.csv"),
                         read_csv(baseline_dir / f"selection/{name}_top10.csv"), targets["selection"],
                         name=name, summaries=row["selection"], new_fallback=True, partition="selection",
                         oracle=oracle, popularity_tiebreak=True, **common)
            counts["cosine_selections"] += 1
        for seed in SEEDS:
            assert len({row["initial_trainable_sha256"] for row in split_candidates if row["initialization_seed"] == seed}) == 1
        for row in split_candidates:
            name = row["name"]
            model_dir = final / f"models/{name}"
            assert read_json(model_dir / "candidate_result.json") == row
            for fingerprints in (row["initial_model_fingerprint"], row["selected_model_fingerprints"]):
                for relative, expected in fingerprints.items():
                    check_file(model_dir, relative, expected)
            history = row["history"]
            assert [epoch["epoch"] for epoch in history] == list(range(1, row["epochs_completed"]+1))
            best = max(history, key=lambda epoch: (epoch["selection_ndcg_at_10"], epoch["selection_recall_at_10"]))
            assert best["epoch"] == row["best_epoch"]
            assert bool(row["best_at_epoch_cap"]) == (row["best_epoch"] == 20)
            assert bool(row["stopped_at_epoch_cap"]) == (len(history) == 20)
            if len(history) < 20:
                assert len(history)-row["best_epoch"] == 4
            updates = (len(eligible_pairs)+1023)//1024
            assert row["training_examples_per_epoch"] == len(eligible_pairs)
            assert row["optimizer_iterations"] == len(history)*updates
            for epoch in history:
                assert epoch["training_examples"] == len(eligible_pairs)
                assert epoch["optimizer_iterations"] == epoch["epoch"]*updates
                close(epoch["train_total_loss"], epoch["train_data_loss"]+epoch["reg_loss"], atol=2e-6)
                assert epoch["gradient_norm"] > 0 and np.isfinite(epoch["gradient_norm"])
            trace = row["first_step_trace"]
            order = np.random.default_rng(row["initialization_seed"]).permutation(len(eligible_pairs))
            first_users = eligible_pairs[order[:8], 0]
            np.testing.assert_array_equal(trace["context_positive_counts_after_target_removal"], positive_counts[first_users]-1)
            assert np.min(trace["context_positive_counts_after_target_removal"]) >= 1
            assert trace["optimizer_iterations"] == 1 and trace["batch_size"] == 1024
            assert np.isfinite(trace["context_pooled_vectors_before"]).all()
            assert trace["loss_before_update"]["gradient_norm"] > 0
            assert any(value > 0 for key, value in trace["variable_update_norms"].items() if "context_embedding" in key)
            close(selection_key(row), [best["selection_ndcg_at_10"], best["selection_recall_at_10"]])
            audit_tables(read_csv(model_dir / f"selection/{name}_outcomes.csv"),
                         read_csv(model_dir / f"selection/{name}_top10.csv"), targets["selection"],
                         name=name, summaries=row["selection"], new_fallback=True, partition="selection", **common)
            counts["neural_selections"] += 1
        selected = next(row for row in results["primary_selection"] if row["split_seed"] == split)
        assert selected["selection_only"]
        best_baseline = max(split_baselines, key=selection_key)
        assert selected["baseline_candidate"] == best_baseline["name"]
        assert [row["name"] for row in split_baselines if row["selected"]] == [best_baseline["name"]]
        for arm in ARMS:
            rows = [row for row in split_candidates if row["arm"] == arm]
            assert [row["initialization_seed"] for row in rows] == list(SEEDS)
            chosen = next(row for row in selected["arms"] if row["arm"] == arm)
            primary = max(rows, key=selection_key)
            assert chosen["primary_candidate"] == primary["name"] and chosen["primary_selection"] == primary["selection"]
            assert chosen["candidates"] == [row["name"] for row in rows]
            scores = [selection_key(row)[0] for row in rows]
            close([chosen["selection_ndcg_at_10_mean"], chosen["selection_ndcg_at_10_std"]], [np.mean(scores), np.std(scores, ddof=1)])
        ranking = next(row for row in results["ranking_results"] if row["split_seed"] == split)
        assert ranking["status"] == "completed" and ranking["baseline_name"] == best_baseline["name"]
        assert ranking["primary_models"] == {row["arm"]: row["primary_candidate"] for row in selected["arms"]}
        previous_ranking = next(row for row in previous["ranking_results"] if row["split_seed"] == split)
        assert ranking["previous_primary_models"] == previous_ranking["primary_models"]
        outcomes = read_csv(final / f"ranking/split_{split}/ranking_user_outcomes.csv")
        top10 = read_csv(final / f"ranking/split_{split}/ranking_top10.csv")
        previous_outcomes = read_csv(final / f"inputs/previous_ranking/split_{split}/ranking_user_outcomes.csv")
        previous_top = read_csv(final / f"inputs/previous_ranking/split_{split}/ranking_top10.csv")
        assert len(set(previous_outcomes.model)) == 17
        new_names = {row["name"] for row in split_candidates+split_baselines}
        assert set(outcomes.model) == set(top10.model) == set(ranking["metrics"]) == new_names | set(previous_outcomes.model)
        assert set(ranking["new_models"]) == new_names
        assert set(ranking["reused_models"]) == set(previous_outcomes.model)-{"popularity"}
        expected_counts = targets["assessment"].user_id.map(positive_count_map).fillna(0).to_numpy(int)
        assert ranking["cohort_counts"] == {cohort: int(mask.sum()) for cohort, mask in cohort_masks(expected_counts).items()}
        table_map = {}
        for name, metrics in ranking["metrics"].items():
            oracle = None
            if name in baseline_objects:
                baseline = baseline_objects[name]
                def oracle(original_users, baseline=baseline):
                    return baseline.score_catalog(np.asarray([user_index[u] for u in original_users], dtype=np.int32))
            frame, top = audit_tables(outcomes[outcomes.model == name], top10[top10.model == name], targets["assessment"],
                name=name, summaries=metrics, new_fallback=name in new_names, partition="assessment",
                oracle=oracle, popularity_tiebreak=name in baseline_objects, **common)
            table_map[name] = frame
            if name in new_names:
                row = next(row for row in split_candidates+split_baselines if row["name"] == name)
                assert row["assessment"] == metrics
                direct = final / f"assessment/split_{split}/{name}"
                pd.testing.assert_frame_equal(frame, read_csv(direct / f"{name}_outcomes.csv").sort_values("user_id").reset_index(drop=True), check_exact=False, rtol=0, atol=1e-14)
                pd.testing.assert_frame_equal(top, read_csv(direct / f"{name}_top10.csv").sort_values("user_id").reset_index(drop=True), check_exact=True)
            else:
                old = previous_outcomes[previous_outcomes.model == name].sort_values("user_id").reset_index(drop=True)
                old_top = previous_top[previous_top.model == name].sort_values("user_id").reset_index(drop=True)
                pd.testing.assert_frame_equal(frame[old.columns], old, check_exact=False, rtol=0, atol=1e-14)
                pd.testing.assert_frame_equal(top, old_top, check_exact=True)
                counts["reused_model_tables"] += 1
        expected_comparisons = set()
        for row in split_candidates+split_baselines:
            references = {"popularity", ranking["previous_primary_models"]["bpr_id"]}
            if row in split_candidates:
                other = "history_softmax" if row["arm"] == "history_bpr" else "history_bpr"
                references.update([best_baseline["name"], f"split{split}_{other}_seed{row['initialization_seed']}"])
            expected_comparisons.update((row["name"], reference, cohort, metric)
                for reference in references-{row["name"]} for cohort in COMPARISON_COHORTS for metric in METRICS)
        audit_comparisons(ranking["paired_bootstrap"], table_map, expected_comparisons)
        shuffle = [row for row in results["history_shuffle_results"] if row["split_seed"] == split]
        assert {row["candidate"] for row in shuffle} == set(ranking["primary_models"].values())
        donor_details[str(split)] = {"positive_fit_users": int((positive_counts > 0).sum()),
            "changed_fit_users": int((donors != np.arange(len(users))).sum())}
        for row in shuffle:
            name, shuffled_name = row["candidate"], row["shuffled_model"]
            assert shuffled_name == name+"__shuffled" and row["diagnostic_only"] and not row["used_for_selection"]
            assert row["normal_metrics"] == ranking["metrics"][name]
            assert row["donor_mapping"] == f"histories/split_{split}/training_data.npz"
            folder = final / f"history_shuffle/split_{split}/{name}"
            frame, _ = audit_tables(read_csv(folder / f"{shuffled_name}_outcomes.csv"),
                read_csv(folder / f"{shuffled_name}_top10.csv"), targets["assessment"],
                name=shuffled_name, summaries=row["shuffled_metrics"], new_fallback=True, partition="shuffle", **common)
            expected = {(name, shuffled_name, cohort, metric) for cohort in COMPARISON_COHORTS for metric in METRICS}
            audit_comparisons(row["paired_normal_minus_shuffled"], {name: table_map[name], shuffled_name: frame}, expected, shuffled=True)
        print(f"Split {split}: history, selection, cosine, all cohorts, prior parity and shuffle verified", flush=True)

    summary = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(), **counts,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bootstrap_oracle_script_sha256": hashlib.sha256((root / "scripts/audit_ranking_assessment.py").read_bytes()).hexdigest(),
        "downloaded_final_files_rehashed": len(downloaded["files"]),
        "all_12_neural_and_8_cosine_completion_gate_verified": True,
        "frozen_inputs_source_and_locked_models_verified": True,
        "fit_only_positive_histories_reconstructed": True, "training_targets_excluded_from_first_step_context_counts": True,
        "selection_decisions_reproduced": True, "all_five_cohort_metrics_reproduced": True,
        "cosine_models_and_full_catalogue_predictions_recomputed": True,
        "independent_bootstrap_oracles": True, "recipient_exclusions_and_fixed_donor_mapping_verified": True,
        "donor_mapping_summary": donor_details, "total_neural_epochs": results["training"]["total_neural_epochs"],
        "limitations": [
            "Neural scores were not recomputed locally; locked models, remote roundtrip records, saved score ordering and all recommendation outcomes were checked.",
            "Cosine models were refitted from frozen histories with the implementation covered by independent small-formula tests; ranking order uses a separate heap oracle.",
            "Shuffle mapping and recipient exclusions are verified, but shuffled neural forward passes were not rerun locally.",
            "This repeatedly inspected development assessment and fixed perturbation do not establish independent confirmation or a complete causal explanation.",
        ]}
    (final.parent / "local_result_verification.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    arguments = parser.parse_args()
    verify(arguments.artifact_dir.resolve(), arguments.verification_dir.resolve())
