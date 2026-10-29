"""Independent TensorFlow-free audit of the downloaded ranking-objective study."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


METRICS = ("recall_at_10", "ndcg_at_10", "mrr_at_10")
COHORTS = ("all_relevant", "fit_positive_history", "no_fit_positive_history")
ARMS = ("mse_id", "bpr_id", "bpr_metadata")
SEEDS = (42, 2026, 7)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def close(actual, expected, *, atol=1e-11):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def verify_file_fingerprint(path, expected):
    """Reject a missing or modified file against an already trusted fingerprint."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Verified file is missing: {path}")
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Verified file size differs: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError(f"Verified file SHA-256 differs: {path}")


def bootstrap_mean(candidate, reference, draws, seed):
    """Independent scalar-draw oracle, not the production batched helper."""
    difference = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        chosen = generator.integers(0, len(difference), size=len(difference))
        samples.append(float(np.mean(difference[chosen])))
    return np.quantile(samples, [.025, .975])


def verify(final, verification_dir):
    root = Path(__file__).resolve().parents[1]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from book_recommender.content_based import book_work_key
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    # Trust starts with the separately verified download, not hashes read from
    # a potentially changed result file. The monitor's record is retained locally.
    backups = read_json(final.parent / "backup_verification.json")
    downloaded = backups["final_backup.tar.gz"]
    if (downloaded["kind"] != "completed_experiment"
            or Path(downloaded["extracted_directory"]).resolve() != final.resolve()):
        raise ValueError("Expected the monitor-verified final experiment extraction.")
    verify_file_fingerprint(final.parent / "final_backup.tar.gz", downloaded)
    for relative, expected_hash in downloaded["files"].items():
        path = final / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Downloaded experiment file changed: {relative}")
    results = read_json(final / "results.json")
    locked = read_json(final / "weights_locked.json")
    manifest = read_json(final / "inputs/input_manifest.json")
    previous = read_json(final / "inputs/previous_results.json")
    source = read_json(final / "source_metadata.json")
    protocol = read_json(verification_dir / "predeclared_protocol.json")
    counts = {"fingerprinted_files": 0, "selection_candidates": 0,
              "selection_user_outcomes": 0, "assessment_user_outcomes": 0,
              "cohort_summaries": 0, "ranking_bootstraps": 0,
              "exact_popularity_or_fallback_top10": 0, "reused_prior_models": 0}
    metadata_zero_feature_works = {}
    metadata_vocabulary_cutoffs = {}

    def check_file(base, name, expected):
        verify_file_fingerprint(base / name, expected)
        counts["fingerprinted_files"] += 1

    assert results["experiment_kind"] == "ranking_objective_metadata_reused_development"
    assert results["status"] == "completed"
    assert results["all_candidates_completed"] and results["assessment_performed"]
    assert not results["original_test_used"] and not results["original_validation_used"]
    assert not results["model_promotion_to_application"]
    assert not locked["assessment_scored"] and locked["all_initializations_paired"]
    assert datetime.fromisoformat(protocol["declared_utc"]) < datetime.fromisoformat(
        results["configuration"]["started_utc"])
    assert datetime.fromisoformat(locked["locked_utc"]) < datetime.fromisoformat(results["completed_utc"])
    assert protocol["input_manifest_sha256"] == hashlib.sha256(
        (final / "inputs/input_manifest.json").read_bytes()).hexdigest()
    for name, expected in locked["files"].items():
        check_file(final, name, expected)
    for name, expected in manifest["files"].items():
        check_file(final / "inputs", name, expected)
    assert manifest["files"]["previous_results.json"]["sha256"] == manifest["previous_results_sha256"]
    previous_local = root / "artifacts/colab/run-20261014-comparators/final/results.json"
    assert hashlib.sha256(previous_local.read_bytes()).hexdigest() == manifest["previous_results_sha256"]
    actual_inputs = {str(path.relative_to(final / "inputs"))
                     for path in (final / "inputs").rglob("*") if path.is_file()}
    assert actual_inputs == set(manifest["files"]) | {"input_manifest.json"}
    for name, expected in source["source_files"].items():
        check_file(verification_dir / "executed_source", name, expected)
    for name, expected in manifest["previous_ranking_fingerprints"].items():
        assert manifest["files"][name] == expected
        original = name.replace("previous_ranking/", "ranking/", 1)
        check_file(previous_local.parent, original, expected)

    candidates = results["candidate_validation_results"]
    declared = [f"split{split}_{arm}_seed{seed}"
                for split in (17, 29) for seed in SEEDS for arm in ARMS]
    assert [row["name"] for row in candidates] == declared
    assert len(candidates) == 18
    assert all(row["status"] == "completed" and row["model_roundtrip_verified"]
               and row["assessment_scored"] for row in candidates)
    assert sum(row["epochs_completed"] for row in candidates) == results["training"]["total_neural_epochs"]
    assert results["primary_selection"] == locked["primary_selection"]
    assert results["primary_selection"] == read_json(final / "primary_selection.json")
    assert results["ranking_results"] == read_json(final / "ranking_results.json")
    assert results["configuration"]["learning_rate"] == .001
    assert results["configuration"]["regularization_lambda"] == 1e-5
    assert results["configuration"]["batch_size"] == 1024
    assert results["configuration"]["max_neural_epochs_each"] == 12
    assert results["configuration"]["patience"] == 3
    books = read_csv(final / "inputs/train_books.csv", dtype={"isbn": str})
    book_work = {str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
                 for row in books.itertuples(index=False)}
    works = sorted(set(book_work.values()))
    assert len(books) == 5000 and len(works) == 4364

    def key(row):
        return row["selection_ndcg_at_10"], row["selection_recall_at_10"]

    def selection_key(row):
        selected = row["selection"]["all_relevant"]
        return selected["ndcg_at_10"], selected["recall_at_10"]

    def masks(outcomes):
        positive = outcomes.has_positive_fit_history.to_numpy(bool)
        return {"all_relevant": np.ones(len(outcomes), dtype=bool),
                "fit_positive_history": positive, "no_fit_positive_history": ~positive}

    def audit_outcomes(outcomes, top10, targets, seen, positive_users, popularity,
                       expected_metrics, *, name, is_new, partition):
        outcomes = outcomes.sort_values("user_id").reset_index(drop=True)
        top10 = top10.sort_values("user_id").reset_index(drop=True)
        assert outcomes.user_id.is_unique and top10.user_id.is_unique
        np.testing.assert_array_equal(outcomes.user_id, targets.user_id)
        np.testing.assert_array_equal(top10.user_id, targets.user_id)
        assert set(outcomes.model) == set(top10.model) == {name}
        positive = outcomes.user_id.isin(positive_users).to_numpy()
        np.testing.assert_array_equal(outcomes.has_positive_fit_history, positive)
        expected_fallback = ~positive if is_new else np.zeros(len(outcomes), dtype=bool)
        np.testing.assert_array_equal(outcomes.used_popularity_fallback, expected_fallback)
        global_order = sorted(works, key=lambda work: (-popularity.get(work, 0), work))
        frequencies = {cohort: Counter() for cohort in COHORTS}
        cohort_masks = masks(outcomes)
        for index, row in enumerate(outcomes.itertuples(index=False)):
            target = targets.iloc[index]
            saved = top10.iloc[index]
            recommendations = json.loads(saved.work_keys)
            scores = np.asarray(json.loads(saved.raw_scores), dtype=float)
            excluded = seen.get(row.user_id, set())
            assert len(recommendations) == len(set(recommendations))
            assert len(recommendations) == len(scores) == min(10, len(works) - len(excluded))
            assert set(recommendations).issubset(works)
            assert not set(recommendations).intersection(excluded)
            assert np.isfinite(scores).all() and np.all(scores[:-1] >= scores[1:])
            assert all(recommendations[j] <= recommendations[j+1]
                       for j in range(len(scores)-1) if scores[j] == scores[j+1])
            if name == "popularity" or expected_fallback[index]:
                expected = [work for work in global_order if work not in excluded][:10]
                assert recommendations == expected
                close(scores, [popularity.get(work, 0) for work in expected])
                counts["exact_popularity_or_fallback_top10"] += 1
            rank = recommendations.index(target.work_key)+1 if target.work_key in recommendations else 0
            assert row.target_work == target.work_key and row.rank_at_10 == rank
            assert bool(row.target_available) == (target.work_key in works)
            assert bool(row.known_user) == (row.user_id in seen)
            assert row.candidate_works == len(works)-len(excluded)
            close([row.recall_at_10, row.ndcg_at_10, row.mrr_at_10],
                  [bool(rank), 1/np.log2(rank+1) if rank else 0, 1/rank if rank else 0])
            for cohort, mask in cohort_masks.items():
                if mask[index]:
                    frequencies[cohort].update(recommendations)
            counts[f"{partition}_user_outcomes"] += 1
        for cohort, mask in cohort_masks.items():
            frame = outcomes.loc[mask]
            summary = expected_metrics[cohort]
            assert summary["users"] == len(frame)
            assert summary["hits_at_10"] == int(frame.recall_at_10.sum())
            assert summary["popularity_fallback_users"] == int(frame.used_popularity_fallback.sum())
            for metric in METRICS:
                if len(frame):
                    close(frame[metric].mean(), summary[metric])
                else:
                    assert summary[metric] is None
            frequency = np.asarray(list(frequencies[cohort].values()), dtype=float)
            probability = frequency/frequency.sum() if len(frequency) else frequency
            entropy = float(-np.sum(probability*np.log(probability)))
            assert summary["recommended_unique_works"] == len(frequency)
            close([summary["catalogue_coverage"], summary["frequency_entropy"],
                   summary["normalized_frequency_entropy"]],
                  [len(frequency)/len(works), entropy, entropy/np.log(len(works))])
            counts["cohort_summaries"] += 1
        return outcomes, top10

    for split in (17, 29):
        input_dir = final / f"inputs/splits/split_{split}"
        frames = {part: read_csv(input_dir / f"{part}_ratings.csv", dtype={"isbn": str})
                  for part in ("fit", "selection", "assessment")}
        fit = frames["fit"]
        split_previous = next(row for row in previous["dataset"]["splits"] if row["split_seed"] == split)
        for part, frame in frames.items():
            check_file(input_dir, f"{part}_ratings.csv",
                       split_previous["partition_file_fingerprints"][f"{part}_ratings.csv"])
            np.testing.assert_array_equal(frame.work_key, frame.isbn.map(book_work))
        source_rows = np.concatenate([frame.source_row.to_numpy() for frame in frames.values()])
        np.testing.assert_array_equal(np.sort(source_rows), np.arange(111875))
        assert set(frames["selection"].user_id).isdisjoint(frames["assessment"].user_id)
        fit_user_works = set(zip(fit.user_id, fit.work_key))
        for part in ("selection", "assessment"):
            assert not set(zip(frames[part].user_id, frames[part].work_key)).intersection(fit_user_works)
        seen = fit.groupby("user_id").work_key.agg(set).to_dict()
        liked = fit.loc[fit.book_rating >= 8, ["user_id", "work_key"]].drop_duplicates()
        positive_users = set(liked.user_id)
        popularity = liked.groupby("work_key").size().to_dict()
        targets = {}
        for part in ("selection", "assessment"):
            target = frames[part].groupby(["user_id", "work_key"]).book_rating.max().reset_index()
            targets[part] = target[target.book_rating >= 8].sort_values("user_id").reset_index(drop=True)
        vocabulary = read_json(final / f"vocabularies/split_{split}.json")
        assert vocabulary["user_ids"] == sorted(fit.user_id.unique().tolist())
        assert vocabulary["work_keys"] == works
        assert vocabulary["isbn_to_work"] == {isbn: book_work[isbn] for isbn in sorted(fit.isbn.unique())}
        feature_path = final / f"metadata_features/split_{split}"
        metadata = read_json(feature_path / "tfidf.json")
        features = np.load(feature_path / "features.npz")["features"]
        assert metadata["fit_only"] and features.shape == (len(works), 1024)
        assert np.isfinite(features).all()
        # A capped vocabulary can legitimately leave a rare work with no token.
        # Reconstruct both documents and TF-IDF to distinguish that from damage.
        catalog = books.loc[books.isbn.isin(fit.isbn)].copy()
        catalog["work_key"] = catalog.isbn.map(book_work)
        grouped_catalog = {work: frame for work, frame in catalog.groupby("work_key")}
        documents = [" ".join(
            text for field in ("book_title", "book_author", "publisher")
            for text in sorted(set(grouped_catalog[work][field].fillna("").astype(str)) - {""})
        ).strip() or "missingmetadata" for work in works]
        assert documents == metadata["documents"]
        counter = CountVectorizer(lowercase=True, ngram_range=(1, 2),
                                  token_pattern=r"(?u)\b\w+\b")
        term_counts = counter.fit_transform(documents)
        frequencies = dict(zip(counter.get_feature_names_out(),
                               np.asarray(term_counts.sum(axis=0)).ravel()))
        vocabulary = metadata["vocabulary"]
        assert len(vocabulary) == min(1024, len(frequencies))
        assert set(vocabulary).issubset(frequencies)
        assert sorted(vocabulary.values()) == list(range(len(vocabulary)))
        assert sorted(vocabulary, key=vocabulary.get) == sorted(vocabulary)
        cutoff = min(int(frequencies[word]) for word in vocabulary)
        excluded_max = max((int(count) for word, count in frequencies.items()
                            if word not in vocabulary), default=0)
        assert cutoff >= excluded_max
        # sklearn/NumPy versions may choose different terms tied at the frequency
        # cutoff. Preserve the locked runtime vocabulary, after validating that
        # every included term ranks at least as high as every excluded term.
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
            vocabulary=vocabulary, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        reconstructed = vectorizer.fit_transform(documents).toarray()
        close(features[:, :reconstructed.shape[1]], reconstructed, atol=2e-6)
        assert not features[:, reconstructed.shape[1]:].any()
        norm = np.linalg.norm(features, axis=1)
        zero_rows = np.count_nonzero(features, axis=1) == 0
        close(norm[~zero_rows], np.ones(int((~zero_rows).sum())), atol=2e-6)
        metadata_zero_feature_works[str(split)] = int(zero_rows.sum())
        metadata_vocabulary_cutoffs[str(split)] = {
            "selected_minimum_term_frequency": cutoff,
            "excluded_maximum_term_frequency": excluded_max,
            "terms_tied_at_cutoff": sum(int(value) == cutoff for value in frequencies.values()),
            "selected_terms_at_cutoff": sum(int(frequencies[word]) == cutoff for word in vocabulary),
        }
        work_summary = next(row for row in results["dataset"]["training_work_summaries"]
                            if row["split_seed"] == split)
        assert work_summary["fit_user_work_pairs"] == len(fit_user_works)
        assert work_summary["fit_positive_user_work_pairs"] == len(liked)
        assert work_summary["fit_positive_users"] == len(positive_users)
        assert work_summary["fit_no_positive_users"] == len(seen)-len(positive_users)
        assert work_summary["selection_relevant_users"] == len(targets["selection"])
        assert work_summary["selection_fit_positive_users"] == int(targets["selection"].user_id.isin(positive_users).sum())
        close(work_summary["fit_mean_rating"], fit.groupby(["user_id", "work_key"]).book_rating.mean().mean())
        split_candidates = [row for row in candidates if row["split_seed"] == split]
        for seed in SEEDS:
            assert len({row["initial_trainable_sha256"] for row in split_candidates
                        if row["initialization_seed"] == seed}) == 1
        for row in split_candidates:
            name = row["name"]
            model_dir = final / f"models/{name}"
            assert read_json(model_dir / "candidate_result.json") == row
            for name_, fingerprint in row["selected_model_fingerprints"].items():
                check_file(model_dir, name_, fingerprint)
            for name_, fingerprint in row["initial_model_fingerprint"].items():
                check_file(model_dir, name_, fingerprint)
            history = row["history"]
            assert [epoch["epoch"] for epoch in history] == list(range(1, row["epochs_completed"]+1))
            best = max(history, key=key)
            assert best["epoch"] == row["best_epoch"]
            assert bool(row["best_at_epoch_cap"]) == (row["best_epoch"] == 12)
            assert bool(row["stopped_at_epoch_cap"]) == (len(history) == 12)
            if len(history) < 12:
                assert len(history)-row["best_epoch"] == 3
            examples = len(fit_user_works) if row["objective"] == "mse" else len(liked)
            updates = (examples + 1023)//1024
            assert row["training_examples_per_epoch"] == examples
            assert row["optimizer_iterations"] == len(history)*updates
            for epoch in history:
                assert epoch["training_examples"] == examples
                assert epoch["optimizer_iterations"] == epoch["epoch"]*updates
                # Per-step totals are float32 additions before Python averaging.
                close(epoch["train_total_loss"], epoch["train_data_loss"]+epoch["reg_loss"], atol=1e-7)
                assert epoch["gradient_norm"] > 0 and np.isfinite(epoch["gradient_norm"])
            trace = row["first_step_trace"]
            assert trace["optimizer_iterations"] == 1 and trace["batch_size"] == 1024
            assert trace["loss_before_update"]["gradient_norm"] > 0
            assert any(value > 0 for key_, value in trace["variable_update_norms"].items()
                       if "embedding" in key_)
            assert np.isfinite(list(trace["variable_update_norms"].values())).all()
            close(selection_key(row), key(best))
            selection_dir = model_dir / "selection"
            audit_outcomes(read_csv(selection_dir / f"{name}_outcomes.csv"),
                           read_csv(selection_dir / f"{name}_top10.csv"), targets["selection"],
                           seen, positive_users, popularity, row["selection"],
                           name=name, is_new=True, partition="selection")
            counts["selection_candidates"] += 1
        selected = next(row for row in results["primary_selection"] if row["split_seed"] == split)
        assert selected["selection_only"]
        for arm in ARMS:
            rows = [row for row in split_candidates if row["arm"] == arm]
            assert [row["initialization_seed"] for row in rows] == list(SEEDS)
            chosen = next(row for row in selected["arms"] if row["arm"] == arm)
            assert chosen["primary_candidate"] == max(rows, key=selection_key)["name"]
            assert chosen["candidates"] == [row["name"] for row in rows]
            assert chosen["n_seeds"] == 3
            primary = next(row for row in rows if row["name"] == chosen["primary_candidate"])
            assert chosen["primary_selection"] == primary["selection"]
            scores = [selection_key(row)[0] for row in rows]
            close([chosen["selection_ndcg_at_10_mean"], chosen["selection_ndcg_at_10_std"]],
                  [np.mean(scores), np.std(scores, ddof=1)])
        ranking = next(row for row in results["ranking_results"] if row["split_seed"] == split)
        assert ranking["status"] == "completed" and ranking["catalogue_works"] == len(works)
        assert ranking["primary_models"] == {row["arm"]: row["primary_candidate"] for row in selected["arms"]}
        assert ranking["old_primary_mlp"] == protocol["prior_primary_mlp"][str(split)]
        assert ranking["new_models"] == [row["name"] for row in split_candidates]
        rank_dir = final / f"ranking/split_{split}"
        outcomes = read_csv(rank_dir / "ranking_user_outcomes.csv")
        top10 = read_csv(rank_dir / "ranking_top10.csv")
        prior_outcomes = read_csv(final / f"inputs/previous_ranking/split_{split}/ranking_user_outcomes.csv")
        prior_top10 = read_csv(final / f"inputs/previous_ranking/split_{split}/ranking_top10.csv")
        assert set(outcomes.model) == set(top10.model) == set(ranking["metrics"])
        assert set(outcomes.model) == {row["name"] for row in split_candidates} | set(prior_outcomes.model)
        assert set(ranking["reused_models"]) == set(prior_outcomes.model)-{"popularity"}
        positive_mask = targets["assessment"].user_id.isin(positive_users)
        assert ranking["cohort_counts"] == {
            "all_relevant": len(positive_mask), "fit_positive_history": int(positive_mask.sum()),
            "no_fit_positive_history": int((~positive_mask).sum()),
        }
        audited = {}
        for name, metrics in ranking["metrics"].items():
            frame, top = audit_outcomes(outcomes[outcomes.model == name], top10[top10.model == name],
                targets["assessment"], seen, positive_users, popularity, metrics,
                name=name, is_new=name in ranking["new_models"], partition="assessment")
            audited[name] = frame
            if name in ranking["new_models"]:
                row = next(row for row in split_candidates if row["name"] == name)
                assert row["assessment"] == metrics
                direct_dir = final / f"assessment/split_{split}/{name}"
                direct_outcomes = read_csv(direct_dir / f"{name}_outcomes.csv").sort_values("user_id").reset_index(drop=True)
                direct_top = read_csv(direct_dir / f"{name}_top10.csv").sort_values("user_id").reset_index(drop=True)
                pd.testing.assert_frame_equal(frame, direct_outcomes, check_exact=False, rtol=0, atol=1e-14)
                pd.testing.assert_frame_equal(top, direct_top, check_exact=True)
            else:
                old = prior_outcomes[prior_outcomes.model == name].sort_values("user_id").reset_index(drop=True)
                old_top = prior_top10[prior_top10.model == name].sort_values("user_id").reset_index(drop=True)
                pd.testing.assert_frame_equal(frame[old.columns], old, check_exact=False, rtol=0, atol=1e-14)
                pd.testing.assert_frame_equal(top, old_top, check_exact=True)
                counts["reused_prior_models"] += 1
        expected_comparisons = set()
        for row in split_candidates:
            name, seed = row["name"], row["initialization_seed"]
            references = {"popularity", ranking["old_primary_mlp"],
                          f"split{split}_mse_id_seed{seed}", f"split{split}_bpr_id_seed{seed}"}-{name}
            for reference in references:
                for cohort in COHORTS[:2]:
                    for metric in METRICS:
                        expected_comparisons.add((name, reference, cohort, metric))
        actual_comparisons = {(row["model"], row["reference"], row["cohort"], row["metric"])
                              for row in ranking["paired_bootstrap"]}
        assert actual_comparisons == expected_comparisons
        assert len(actual_comparisons) == len(ranking["paired_bootstrap"])
        for row in ranking["paired_bootstrap"]:
            mask = masks(audited[row["model"]])[row["cohort"]]
            candidate = audited[row["model"]].loc[mask, row["metric"]].to_numpy(float)
            reference = audited[row["reference"]].loc[mask, row["metric"]].to_numpy(float)
            assert row["draws"] == 1000 and row["seed"] == 123 and row["users"] == len(candidate)
            close((candidate-reference).mean(), row["difference"])
            close(bootstrap_mean(candidate, reference, 1000, 123), row["ci95"])
            counts["ranking_bootstraps"] += 1
        print(f"Split {split}: frozen inputs, selections, fallback, ranking, cohorts and intervals verified", flush=True)

    summary = {"verified_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", **counts,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "downloaded_final_files_rehashed": len(downloaded["files"]),
        "fit_only_metadata_features_reproduced": True,
        "metadata_zero_feature_works": metadata_zero_feature_works,
        "metadata_vocabulary_cutoffs": metadata_vocabulary_cutoffs,
        "locked_artifact_and_frozen_input_hashes_verified": True,
        "executed_source_matches_verification_archive": True,
        "all_18_completion_gate_verified": True, "paired_initialization_records_verified": True,
        "selection_decisions_reproduced": True, "ranking_and_cohort_metrics_reproduced": True,
        "prior_ranking_parity_verified": True, "independent_bootstrap_oracles": True,
        "total_neural_epochs": results["training"]["total_neural_epochs"],
        "limitations": [
            "Neural scores were not recomputed from Keras locally; locked model hashes, remote roundtrip records, saved score order, and target-outcome consistency were checked.",
            "All popularity and shared-fallback recommendations were independently recomputed over the complete fit catalogue.",
            "Initial trainable equality is verified from recorded GPU hashes; nontrainable metadata intentionally differs between ID and metadata arms.",
            "This is reused development assessment with adaptive experiment design, not an independent confirmation test.",
        ]}
    target = final.parent / "local_result_verification.json"
    target.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    arguments = parser.parse_args()
    verify(arguments.artifact_dir.resolve(), arguments.verification_dir.resolve())
