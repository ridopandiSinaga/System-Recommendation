"""Bounded history-encoder ranking study on reused development data.

Only selection ranking determines checkpoints and primary seeds. Assessment is
opened after every declared fit has completed and selected artifacts are hashed.
This adaptive experiment is not a fresh independent test of generalization.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

try:
    from .colab_neural_experiment import (
        ROOT, SECRET_PATTERN, atomic_json, emit, fingerprint_files, gpu_probe,
        save_network, source_metadata, utc_now, write_archive,
    )
    from .colab_neural_multiseed import FROZEN_INPUT_SHA256
except ImportError:
    from colab_neural_experiment import (
        ROOT, SECRET_PATTERN, atomic_json, emit, fingerprint_files, gpu_probe,
        save_network, source_metadata, utc_now, write_archive,
    )
    from colab_neural_multiseed import FROZEN_INPUT_SHA256

SPLIT_SEEDS = (17, 29)
TRAINING_SEEDS = (42, 2026, 7)
ARMS = ("history_bpr", "history_softmax")
BASELINE_GRID = tuple((neighbors, shrinkage) for neighbors in (50, 200) for shrinkage in (0.0, 10.0))
CANDIDATES = tuple(
    {"name": f"split{split}_{arm}_seed{seed}", "split_seed": split,
     "initialization_seed": seed, "arm": arm,
     "objective": "bpr" if arm == "history_bpr" else "softmax",
     "architecture": "positive_history_encoder", "kind": "neural"}
    for split in SPLIT_SEEDS for seed in TRAINING_SEEDS for arm in ARMS
)


def verify_inputs(input_dir):
    """Hash frozen partitions and prior ranking CSVs without parsing assessment."""
    input_dir = Path(input_dir)
    manifest = json.loads((input_dir / "input_manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    required = {"train_books.csv", "previous_results.json"}
    required.update(f"splits/split_{seed}/{part}_ratings.csv"
                    for seed in SPLIT_SEEDS for part in ("fit", "selection", "assessment"))
    required.update(f"previous_ranking/split_{seed}/{filename}"
                    for seed in SPLIT_SEEDS for filename in
                    ("ranking_user_outcomes.csv", "ranking_top10.csv"))
    if set(files) != required:
        raise RuntimeError("Only declared frozen metadata, reused partitions, and prior ranking evidence are allowed")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe input manifest path")
        if fingerprint_files(input_dir, [relative])[relative] != expected:
            raise RuntimeError(f"Input hash/size mismatch: {relative}")
    actual = {path.relative_to(input_dir).as_posix() for path in input_dir.rglob("*") if path.is_file()}
    if actual != required | {"input_manifest.json"}:
        raise RuntimeError("Input directory contains undeclared files")
    if files["train_books.csv"]["sha256"] != FROZEN_INPUT_SHA256["train_books.csv"]:
        raise RuntimeError("Metadata differs from the frozen catalog")
    if files["previous_results.json"]["sha256"] != manifest["previous_results_sha256"]:
        raise RuntimeError("Prior result fingerprint differs")
    previous = json.loads((input_dir / "previous_results.json").read_text(encoding="utf-8"))
    if (previous.get("experiment_kind") != "ranking_objective_metadata_reused_development"
            or not previous.get("all_candidates_completed") or not previous.get("assessment_performed")):
        raise RuntimeError("Expected the completed prior objective/metadata development experiment")
    for seed in SPLIT_SEEDS:
        split = next(row for row in previous["dataset"]["splits"] if row["split_seed"] == seed)
        for part in ("fit", "selection", "assessment"):
            filename = f"{part}_ratings.csv"
            if files[f"splits/split_{seed}/{filename}"] != split["partition_file_fingerprints"][filename]:
                raise RuntimeError("Reused partition differs from its prior fingerprint")
    prior_files = {name: expected for name, expected in files.items() if name.startswith("previous_ranking/")}
    if prior_files != manifest["previous_ranking_fingerprints"]:
        raise RuntimeError("Prior ranking evidence does not match independently verified fingerprints")
    return manifest, previous


def trainable_fingerprint(model):
    """Trainable initialization must be identical across objective arms for each seed."""
    digest = hashlib.sha256()
    for variable in model.trainable_variables:
        value = variable.numpy()
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def selection_key(metrics):
    values = metrics["all_relevant"]
    return float(values["ndcg_at_10"]), float(values["recall_at_10"])


def select_primaries(candidates, np):
    selected = []
    for split in SPLIT_SEEDS:
        arms = []
        for arm in ARMS:
            rows = [row for row in candidates if row["split_seed"] == split
                    and row["arm"] == arm and row["status"] == "completed"]
            if len(rows) != len(TRAINING_SEEDS):
                continue
            # Stable declaration order breaks exact seed ties, not assessment.
            primary = max(rows, key=lambda row: selection_key(row["selection"]))
            scores = [row["selection"]["all_relevant"]["ndcg_at_10"] for row in rows]
            arms.append({"arm": arm, "primary_candidate": primary["name"],
                         "candidates": [row["name"] for row in rows],
                         "selection_ndcg_at_10_mean": float(np.mean(scores)),
                         "selection_ndcg_at_10_std": float(np.std(scores, ddof=1)),
                         "primary_selection": primary["selection"], "n_seeds": len(rows)})
        selected.append({"split_seed": split, "arms": arms, "selection_only": True})
    return selected


def donor_history_indices(positive_counts, np):
    """Fixed cyclic shift of every fit-positive user, without heldout labels."""
    counts = np.asarray(positive_counts)
    donors = np.arange(len(counts), dtype=np.int32)
    eligible = np.flatnonzero(counts > 0)
    if len(eligible):
        donors[eligible] = np.roll(eligible, -1)
    return donors


def first_step(model, step, users, targets, negatives, candidate_mask, objective, np):
    before = [variable.numpy().copy() for variable in model.trainable_variables]
    pooled, counts = model.context_summary(users[:8], exclude_items=targets[:8])
    positive_before = model.score_targets(users[:8], targets[:8], exclude_items=targets[:8]).numpy()
    negative_before = model.score_targets(users[:8], negatives[:8], exclude_items=targets[:8]).numpy()
    values = {key: float(value.numpy()) for key, value in step(users, targets, negatives, candidate_mask).items()}
    return values, {"objective": objective, "loss_before_update": values,
        "context_positive_counts_after_target_removal": counts.numpy().tolist(),
        "context_pooled_vectors_before": pooled.numpy().tolist(),
        "raw_positive_scores_before": positive_before.tolist(),
        "raw_negative_scores_before": negative_before.tolist() if objective == "bpr" else None,
        "raw_positive_scores_after": model.score_targets(users[:8], targets[:8], exclude_items=targets[:8]).numpy().tolist(),
        "variable_update_norms": {variable.path: float(np.linalg.norm(variable.numpy() - old))
                                  for variable, old in zip(model.trainable_variables, before)},
        "batch_size": len(users), "gradient_source": "Actual compiled optimizer step before Adam update",
        "context_policy": "The positive target is removed before history pooling in both objectives"}


def run_experiment(args):
    import numpy as np
    import pandas as pd
    from scipy.sparse import load_npz, save_npz
    import tensorflow as tf

    from book_recommender.ranking_neural import prepare_work_data, sample_negative_works
    from book_recommender.history_neural import (
        ItemCosineBaseline, build_history_data, build_item_cosine,
        create_history_model, load_history_model, make_history_train_step,
    )
    try:
        from .history_objective_evaluation import (
            assemble_history_assessment_report, compare_history_outcomes,
            evaluate_history_scores, prepare_history_evaluation,
        )
    except ImportError:
        from history_objective_evaluation import (
            assemble_history_assessment_report, compare_history_outcomes,
            evaluate_history_scores, prepare_history_evaluation,
        )

    output = args.output.resolve()
    started = time.monotonic()
    manifest, previous = verify_inputs(args.input_dir)
    configuration = {
        "started_utc": utc_now(), "candidates": list(CANDIDATES),
        "split_seeds": list(SPLIT_SEEDS), "training_seeds": list(TRAINING_SEEDS), "arms": list(ARMS),
        "embedding_size": 32, "architecture": "Untied input/output item embeddings; sum positive history / sqrt(count) -> Dense64 tanh -> Dense32; dot output item / sqrt32 + item bias; no user ID embedding",
        "paired_initialization": "Same seed gives identical trainable values across BPR and full softmax",
        "random_streams": "Minibatch order uses seed; BPR negatives use seed+10000 independently, preserving identical epoch orders across objectives",
        "context": "Fit positive works (any edition rated>=8), deduplicated; remove target before encoding during training; all fit positives at inference",
        "training_examples": "One positive user/work target for users with at least two fit-positive works; per-example weighting",
        "bpr": "softplus(score_negative-score_target); one uniform work absent from ALL user's fit works",
        "softmax": "Full catalogue sparse cross-entropy; mask all other fit-observed works, retaining target and unobserved works",
        "negative_limitation": "Unobserved works have unknown preference, not confirmed dislike",
        "regularization_lambda": 1e-5, "regularization": "Mean squared value of ALL trainable parameters, multiplied by 1e-5",
        "optimizer": "Adam", "learning_rate": 0.001, "batch_size": args.batch_size,
        "max_neural_epochs_each": args.epochs, "patience": 4,
        "max_training_seconds": args.max_training_seconds, "max_ranking_seconds": args.max_ranking_seconds,
        "snapshot_interval_epochs": 5,
        "selection_metric": "Full catalogue selection NDCG@10; tie Recall@10 then earlier checkpoint; seed ties declared order",
        "baseline_grid": [{"k": k, "shrink": shrink} for k, shrink in BASELINE_GRID],
        "baseline_similarity": "Cij/sqrt(ci*cj) * Cij/(Cij+shrink), diagonal zero; binary positive user/work C=X.T@X; top-k per source row, lexical ties",
        "baseline_scoring": "Positive user/work row @ source-to-target similarity; score ties by descending fit-positive popularity then lexical work",
        "fallback": "Identical fit-only popularity for new arms and baselines when user has no fit-positive work",
        "assessment_gate": "All 12 neural fits and 8 baseline configurations complete; selected artifacts hash-locked before reading assessment CSVs",
        "history_shuffle": "For primary neural candidates only: fixed cyclic shift=1 over all sorted fit-positive encoded users; original recipient exclusions retained; no heldout labels in donor mapping",
        "scope": "Adaptive reused internal development; assessment was inspected previously",
        "original_test_used": False, "original_validation_used": False, "model_promotion_to_application": False,
    }
    atomic_json(output / "configuration.json", configuration)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    probe = gpu_probe(tf, args.require_gpu)
    probe["deterministic_operations_enabled"] = True
    atomic_json(output / "gpu_probe.json", probe)
    metadata = source_metadata()
    metadata["source_files"].update(fingerprint_files(ROOT, [
        "src/book_recommender/history_neural.py", "src/book_recommender/ranking_neural.py",
        "scripts/colab_history_experiment.py", "scripts/history_objective_evaluation.py",
        "scripts/ranking_objective_evaluation.py", "scripts/ranking_comparison.py",
    ]))
    metadata["experiment_label"] = "history_encoder_ranking_reused_development"
    atomic_json(output / "source_metadata.json", metadata)
    shutil.copytree(args.input_dir, output / "inputs")
    inputs = output / "inputs"
    verify_inputs(inputs)
    atomic_json(output / "input_fingerprints.json", manifest)
    books = pd.read_csv(inputs / "train_books.csv", dtype={"isbn": str})
    dataset = {"splits": previous["dataset"]["splits"], "training_work_summaries": [],
               "reused_prior_assessment": True, "previous_results_sha256": manifest["previous_results_sha256"]}
    split_context = {}
    for split_seed in SPLIT_SEEDS:
        split_dir = inputs / "splits" / f"split_{split_seed}"
        fit = pd.read_csv(split_dir / "fit_ratings.csv", dtype={"isbn": str})
        selection = pd.read_csv(split_dir / "selection_ratings.csv", dtype={"isbn": str})
        work_data = prepare_work_data(books, fit, max_features=1)
        history_data = build_history_data(work_data)
        context = prepare_history_evaluation(books, fit, selection, work_data, history_data)
        donors = donor_history_indices(history_data.positive_counts, np)
        vocabulary = {"user_ids": [int(value) for value in work_data.users], "work_keys": list(work_data.work_keys),
                      "isbn_to_work": work_data.isbn_to_work,
                      "fit_input_fingerprint": manifest["files"][f"splits/split_{split_seed}/fit_ratings.csv"]}
        atomic_json(output / "vocabularies" / f"split_{split_seed}.json", vocabulary)
        feature_dir = output / "histories" / f"split_{split_seed}"
        feature_dir.mkdir(parents=True)
        save_npz(feature_dir / "positive_history.npz", history_data.positive_csr)
        np.savez_compressed(feature_dir / "training_data.npz", eligible_pairs=history_data.eligible_pairs,
                            positive_counts=history_data.positive_counts, history_donor_indices=donors)
        counts = history_data.positive_counts
        eligible_users = np.flatnonzero(counts >= 2)
        eligible_works = np.unique(history_data.eligible_pairs[:, 1])
        summary = {"split_seed": split_seed, "fit_user_work_pairs": len(work_data.pairs),
            "fit_positive_user_work_pairs": len(work_data.positive_pairs), "training_examples": len(history_data.eligible_pairs),
            "fit_users": len(work_data.users), "fit_works": len(work_data.work_keys),
            "fit_positive_users": int((counts > 0).sum()), "fit_no_positive_users": int((counts == 0).sum()),
            "fit_single_positive_users": int((counts == 1).sum()), "fit_multiple_positive_users": len(eligible_users),
            "trained_positive_target_works": len(eligible_works),
            "positive_target_work_coverage": len(eligible_works) / len(work_data.work_keys),
            "max_positive_history": int(counts.max()),
            "selection_relevant_users": len(context["targets"]),
            "selection_fit_positive_users": int(context["positive_history"].sum()),
            "selection_no_fit_positive_users": int((~context["positive_history"]).sum()),
            "history_shuffle_changed_users": int((donors != np.arange(len(donors))).sum()),
            "metadata_used_by_network": False}
        dataset["training_work_summaries"].append(summary)
        split_context[split_seed] = {"fit": fit, "selection": selection, "work_data": work_data,
            "history_data": history_data, "evaluation": context, "vocabulary": vocabulary, "donors": donors}
    atomic_json(output / "dataset_summary.json", dataset)
    static_entries = ["inputs", "configuration.json", "gpu_probe.json", "source_metadata.json",
                      "input_fingerprints.json", "dataset_summary.json", "vocabularies", "histories"]
    archive = write_archive(output, "inputs_backup.tar.gz", static_entries, "inputs_manifest.json", kind="training_inputs")
    emit("INPUTS_BACKUP_READY", **archive)
    deadline = time.monotonic() + args.max_training_seconds
    candidate_results, baseline_results, initial_hashes = [], [], {}
    global_epoch, snapshot_number = 0, 0
    current_candidate = {"name": "baseline_grid"}

    def progress(status, **extra):
        atomic_json(output / "progress.json", {"status": status, "epoch": global_epoch,
                    "candidate": current_candidate, "updated_utc": utc_now(), **extra})
        atomic_json(output / "candidate_validation_results.json", candidate_results)
        atomic_json(output / "baseline_grid_results.json", baseline_results)

    def snapshot(reason, completed_model=None):
        nonlocal snapshot_number
        snapshot_number += 1
        entries = ["current_best_network.keras", "current_checkpoint", "current_checkpoint_state.json",
                   "candidate_validation_results.json", "baseline_grid_results.json", "progress.json",
                   "configuration.json", "weights_locked.json", "primary_selection.json", "worker.log"]
        if completed_model is not None:
            entries.append(str(completed_model.relative_to(output)))
        archive = write_archive(output, f"epoch_{snapshot_number:03d}_snapshot.tar.gz", entries,
                                f"epoch_{snapshot_number:03d}_manifest.json", kind="epoch_checkpoint",
                                epoch=global_epoch, candidate=current_candidate["name"], reason=reason)
        atomic_json(output / "snapshot_manifest.json", archive)
        emit("EPOCH_BACKUP_READY", **archive)

    for split_seed in SPLIT_SEEDS:
        context = split_context[split_seed]
        for k, shrink in BASELINE_GRID:
            current_candidate = {"name": f"split{split_seed}_itemcos_k{k}_shrink{int(shrink)}",
                                 "kind": "classical", "split_seed": split_seed, "k": k, "shrink": shrink}
            record = {**current_candidate, "status": "not_started", "selected": False}
            if time.monotonic() >= deadline:
                record["status"] = "skipped_budget"
                baseline_results.append(record)
                continue
            began = time.monotonic()
            baseline = build_item_cosine(context["history_data"], k=k, shrink=shrink)
            directory = output / "baselines" / record["name"]
            directory.mkdir(parents=True)
            evaluation = evaluate_history_scores(context["evaluation"], baseline.score_catalog,
                model_name=record["name"], split_seed=split_seed, output_dir=directory / "selection", popularity_tiebreak=True)
            save_npz(directory / "similarity.npz", baseline.similarity)
            restored = ItemCosineBaseline(context["history_data"].positive_csr, load_npz(directory / "similarity.npz"))
            check_users = np.arange(min(7, context["history_data"].num_users), dtype=np.int32)
            np.testing.assert_allclose(baseline.score_catalog(check_users), restored.score_catalog(check_users), rtol=1e-7, atol=1e-8)
            record.update(status="completed", selection=evaluation["metrics"], model_roundtrip_verified=True,
                          similarity_nonzero=int(baseline.similarity.nnz), seconds=time.monotonic() - began)
            atomic_json(directory / "metadata.json", {"candidate": current_candidate,
                "similarity_definition": configuration["baseline_similarity"], "selection": record["selection"],
                "positive_history": f"histories/split_{split_seed}/positive_history.npz"})
            record["selected_model_fingerprints"] = fingerprint_files(directory, ["similarity.npz", "metadata.json"])
            baseline_results.append(record)
            progress("baseline_grid", candidate_status=record["status"])
        successful = [row for row in baseline_results if row["split_seed"] == split_seed and row["status"] == "completed"]
        if len(successful) == len(BASELINE_GRID):
            max(successful, key=lambda row: selection_key(row["selection"]))["selected"] = True
    progress("baseline_grid_completed")
    snapshot("baseline_grid_completed", output / "baselines")

    for candidate in CANDIDATES:
        current_candidate = dict(candidate)
        record = {**candidate, "status": "not_started", "history": [], "epochs_completed": 0,
                  "best_epoch": None, "assessment_scored": False}
        if time.monotonic() >= deadline:
            record["status"] = "skipped_budget"
            candidate_results.append(record)
            progress("budget_exhausted")
            continue
        context = split_context[candidate["split_seed"]]
        data = context["history_data"]
        pairs = data.eligible_pairs
        objective = candidate["objective"]
        directory = output / "models" / candidate["name"]
        directory.mkdir(parents=True)
        began = time.monotonic()
        model, optimizer, step = None, None, None
        history, best_epoch, first_trace, rng_state = [], None, None, None
        try:
            tf.keras.backend.clear_session()
            model = create_history_model(data, random_state=candidate["initialization_seed"])
            model(pairs[:1], training=False)
            init_hash = trainable_fingerprint(model)
            group = (candidate["split_seed"], candidate["initialization_seed"])
            if initial_hashes.setdefault(group, init_hash) != init_hash:
                raise RuntimeError("History objective arms have different initial trainable weights")
            record["initial_trainable_sha256"] = init_hash
            save_network(model, directory / "initial_network.keras")
            record["initial_model_fingerprint"] = fingerprint_files(directory, ["initial_network.keras"])
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            step = make_history_train_step(model, optimizer, objective=objective, regularization=1e-5)
            order_rng = np.random.default_rng(candidate["initialization_seed"])
            negative_rng = np.random.default_rng(candidate["initialization_seed"] + 10000)
            best_key, best_weights, wait = (-1.0, -1.0), None, 0

            @tf.function(reduce_retracing=True)
            def compiled_catalogue(users):
                return model.score_catalog(users)

            def score_catalogue(users):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Shared training budget exhausted during selection scoring")
                return compiled_catalogue(tf.convert_to_tensor(users, dtype=tf.int32)).numpy()

            def checkpoint(epoch, partial=False):
                checkpoint_dir = output / "current_checkpoint"
                checkpoint_dir.mkdir(exist_ok=True)
                tf.train.Checkpoint(model=model, optimizer=optimizer).write(str(checkpoint_dir / "training"))
                atomic_json(output / "current_checkpoint_state.json", {
                    "candidate": candidate, "candidate_epoch": epoch, "epoch": global_epoch,
                    "history": history, "best_epoch": best_epoch, "partial_epoch": partial,
                    "optimizer_iterations": int(optimizer.iterations.numpy()), "training_rng_state": rng_state,
                    "checkpoint_prefix": "current_checkpoint/training", "updated_utc": utc_now(),
                    "resume_scope": "Matched network + Adam + RNG at epoch boundary; partial minibatch order not resumable"})

            for epoch in range(1, args.epochs + 1):
                order = order_rng.permutation(len(pairs))
                totals = {"loss": 0.0, "data_loss": 0.0, "regularization_loss": 0.0, "gradient_norm": 0.0}
                observed_count = 0
                for start in range(0, len(order), args.batch_size):
                    if time.monotonic() >= deadline:
                        rng_state = {"order_rng": order_rng.bit_generator.state, "negative_rng": negative_rng.bit_generator.state}
                        checkpoint(epoch, partial=True)
                        raise TimeoutError("Shared training budget exhausted at minibatch boundary")
                    batch = pairs[order[start:start + args.batch_size]]
                    users, targets = batch[:, 0], batch[:, 1]
                    if objective == "bpr":
                        negatives = sample_negative_works(users, data.seen_work_indices, data.num_works, negative_rng)
                        candidate_mask = np.ones((len(batch), 1), dtype=bool)
                    else:
                        negatives = targets.copy()
                        candidate_mask = data.softmax_candidate_mask(users, targets)
                    if first_trace is None:
                        values, first_trace = first_step(model, step, users, targets, negatives, candidate_mask, objective, np)
                        first_trace["optimizer_iterations"] = int(optimizer.iterations.numpy())
                    else:
                        values = {key: float(value.numpy()) for key, value in step(users, targets, negatives, candidate_mask).items()}
                    if not np.isfinite(list(values.values())).all():
                        raise RuntimeError("Non-finite training objective or gradient")
                    for key in totals:
                        totals[key] += values[key] * len(batch)
                    observed_count += len(batch)
                rng_state = {"order_rng": order_rng.bit_generator.state, "negative_rng": negative_rng.bit_generator.state}
                evaluation = evaluate_history_scores(context["evaluation"], score_catalogue,
                    model_name=candidate["name"], split_seed=candidate["split_seed"])
                key = selection_key(evaluation["metrics"])
                epoch_record = {"epoch": epoch, "train_data_loss": totals["data_loss"] / observed_count,
                    "train_total_loss": totals["loss"] / observed_count, "reg_loss": totals["regularization_loss"] / observed_count,
                    "gradient_norm": totals["gradient_norm"] / observed_count, "training_examples": observed_count,
                    "optimizer_iterations": int(optimizer.iterations.numpy()),
                    "selection_ndcg_at_10": key[0], "selection_recall_at_10": key[1],
                    "selection_fit_positive_ndcg_at_10": evaluation["metrics"]["fit_positive_history"]["ndcg_at_10"],
                    "selection_multiple_positive_ndcg_at_10": evaluation["metrics"]["multiple_fit_positive"]["ndcg_at_10"]}
                if key > best_key:
                    best_key, best_weights, wait, best_epoch = key, model.get_weights(), 0, epoch
                    save_network(model, output / "current_best_network.keras")
                else:
                    wait += 1
                history.append(epoch_record)
                global_epoch += 1
                checkpoint(epoch)
                progress("training_history", candidate_epoch=epoch, record=epoch_record)
                emit("EPOCH_FINISHED", candidate=candidate["name"], **epoch_record)
                if epoch % 5 == 0:
                    snapshot("periodic_five_epochs")
                if wait >= 4:
                    break
            model.set_weights(best_weights)
            evaluation = evaluate_history_scores(context["evaluation"], score_catalogue,
                model_name=candidate["name"], split_seed=candidate["split_seed"], output_dir=directory / "selection")
            record.update(status="completed", history=history, epochs_completed=len(history), best_epoch=best_epoch,
                best_at_epoch_cap=(best_epoch == args.epochs), stopped_at_epoch_cap=(len(history) == args.epochs),
                selection=evaluation["metrics"], first_step_trace=first_trace,
                optimizer_iterations=int(optimizer.iterations.numpy()), training_examples_per_epoch=len(pairs))
            save_network(model, directory / "network.keras")
            atomic_json(directory / "metadata.json", {"format_version": 1, "architecture": configuration["architecture"],
                "candidate": candidate, "vocabulary": f"vocabularies/split_{candidate['split_seed']}.json",
                "positive_history": f"histories/split_{candidate['split_seed']}/positive_history.npz",
                "history": history, "best_epoch": best_epoch, "selection": record["selection"],
                "initial_trainable_sha256": init_hash, "regularization": configuration["regularization"]})
            restored = load_history_model(directory / "network.keras")
            check_users = np.arange(min(7, data.num_users), dtype=np.int32)
            np.testing.assert_allclose(model.score_catalog(check_users).numpy(), restored.score_catalog(check_users).numpy(), rtol=1e-6, atol=1e-6)
            record["model_roundtrip_verified"] = True
            record["selected_model_fingerprints"] = fingerprint_files(directory, ["network.keras", "metadata.json"])
            del restored
        except Exception as error:  # noqa: BLE001 -- every bounded attempt remains in evidence
            record.update(status="budget_exhausted" if isinstance(error, TimeoutError) else "failed",
                message=SECRET_PATTERN.sub("<redacted>", str(error)), exception_type=type(error).__name__,
                history=history, epochs_completed=len(history), best_epoch=best_epoch)
            print(SECRET_PATTERN.sub("<redacted>", traceback.format_exc()), file=sys.stderr, flush=True)
        record["seconds"] = time.monotonic() - began
        candidate_results.append(record)
        atomic_json(directory / "candidate_result.json", record)
        progress("candidate_completed", candidate_status=record["status"])
        snapshot("candidate_completion", directory)
        model, optimizer, step = None, None, None
        gc.collect()

    complete = [row for row in candidate_results if row["status"] == "completed"]
    baseline_complete = [row for row in baseline_results if row["status"] == "completed"]
    assessment_performed = len(complete) == len(CANDIDATES) and len(baseline_complete) == len(BASELINE_GRID) * len(SPLIT_SEEDS)
    primary_selection = select_primaries(candidate_results, np)
    for entry in primary_selection:
        chosen = [row for row in baseline_complete if row["split_seed"] == entry["split_seed"] and row["selected"]]
        entry["baseline_candidate"] = chosen[0]["name"] if chosen else None
    atomic_json(output / "primary_selection.json", primary_selection)
    ranking_results, shuffle_results = [], []
    if assessment_performed:
        locked_files = {}
        for folder in ("models", "baselines", "vocabularies", "histories"):
            for path in sorted((output / folder).rglob("*")):
                if path.is_file() and path.name != "candidate_result.json":
                    relative = str(path.relative_to(output))
                    locked_files[relative] = fingerprint_files(output, [relative])[relative]
        atomic_json(output / "weights_locked.json", {"locked_utc": utc_now(), "files": locked_files,
            "primary_selection": primary_selection, "assessment_scored": False,
            "selection_metric": configuration["selection_metric"], "all_initializations_paired": True})
        progress("all_weights_locked")
        snapshot("all_weights_locked_before_reassessment")
        ranking_deadline = time.monotonic() + args.max_ranking_seconds
        for split_seed in SPLIT_SEEDS:
            context = split_context[split_seed]
            assessment = pd.read_csv(inputs / "splits" / f"split_{split_seed}" / "assessment_ratings.csv", dtype={"isbn": str})
            evaluation_context = prepare_history_evaluation(books, context["fit"], assessment, context["work_data"], context["history_data"])
            primary = next(row for row in primary_selection if row["split_seed"] == split_seed)
            primary_models = {row["arm"]: row["primary_candidate"] for row in primary["arms"]}
            evaluations = {}
            for candidate in (row for row in complete if row["split_seed"] == split_seed):
                directory = output / "models" / candidate["name"]
                if fingerprint_files(directory, ["network.keras", "metadata.json"]) != candidate["selected_model_fingerprints"]:
                    raise RuntimeError("Selected history model changed after hash lock")
                model = load_history_model(directory / "network.keras")

                @tf.function(reduce_retracing=True)
                def assessment_scores(users):
                    return model.score_catalog(users)

                def budgeted_scores(users):
                    if time.monotonic() >= ranking_deadline:
                        raise TimeoutError("Shared assessment ranking budget exhausted")
                    return assessment_scores(tf.convert_to_tensor(users, dtype=tf.int32)).numpy()

                evaluation = evaluate_history_scores(evaluation_context, budgeted_scores,
                    model_name=candidate["name"], split_seed=split_seed,
                    output_dir=output / "assessment" / f"split_{split_seed}" / candidate["name"])
                evaluations[candidate["name"]] = evaluation
                candidate["assessment"] = evaluation["metrics"]
                candidate["assessment_scored"] = True
                atomic_json(directory / "candidate_result.json", candidate)
                if candidate["name"] in primary_models.values():
                    shuffled_name = candidate["name"] + "__shuffled"
                    shuffled = evaluate_history_scores(evaluation_context,
                        lambda users: budgeted_scores(context["donors"][users]),
                        model_name=shuffled_name, split_seed=split_seed,
                        output_dir=output / "history_shuffle" / f"split_{split_seed}" / candidate["name"])
                    comparisons = compare_history_outcomes(
                        evaluation["outcomes"], shuffled["outcomes"],
                        model=candidate["name"], reference=shuffled_name)
                    shuffle_results.append({"split_seed": split_seed, "candidate": candidate["name"],
                        "shuffled_model": shuffled_name, "normal_metrics": evaluation["metrics"],
                        "shuffled_metrics": shuffled["metrics"], "paired_normal_minus_shuffled": comparisons,
                        "diagnostic_only": True, "used_for_selection": False,
                        "donor_mapping": f"histories/split_{split_seed}/training_data.npz",
                        "scope": "Fixed cyclic perturbation of fit history, retaining recipient exclusions; not a fresh test or complete causal explanation"})
                model = None
                gc.collect()
            for baseline_record in (row for row in baseline_complete if row["split_seed"] == split_seed):
                if time.monotonic() >= ranking_deadline:
                    raise TimeoutError("Shared assessment budget exhausted before classical scoring")
                directory = output / "baselines" / baseline_record["name"]
                if fingerprint_files(directory, ["similarity.npz", "metadata.json"]) != baseline_record["selected_model_fingerprints"]:
                    raise RuntimeError("Classical baseline changed after hash lock")
                baseline = ItemCosineBaseline(context["history_data"].positive_csr, load_npz(directory / "similarity.npz"))
                evaluation = evaluate_history_scores(evaluation_context, baseline.score_catalog,
                    model_name=baseline_record["name"], split_seed=split_seed,
                    output_dir=output / "assessment" / f"split_{split_seed}" / baseline_record["name"], popularity_tiebreak=True)
                evaluations[baseline_record["name"]] = evaluation
                baseline_record["assessment"] = evaluation["metrics"]
            previous_dir = inputs / "previous_ranking" / f"split_{split_seed}"
            old_outcomes = pd.read_csv(previous_dir / "ranking_user_outcomes.csv", float_precision="round_trip")
            old_top10 = pd.read_csv(previous_dir / "ranking_top10.csv", float_precision="round_trip")
            previous_primary = next(row for row in previous["ranking_results"] if row["split_seed"] == split_seed)["primary_models"]
            ranking = assemble_history_assessment_report(evaluation_context, evaluations, old_outcomes, old_top10,
                primary_models=primary_models, baseline_name=primary["baseline_candidate"],
                previous_primary_models=previous_primary, split_seed=split_seed,
                output_dir=output / "ranking" / f"split_{split_seed}")
            ranking["status"] = "completed"
            ranking_results.append(ranking)
            atomic_json(output / "ranking_results.json", ranking_results)
            atomic_json(output / "history_shuffle_results.json", shuffle_results)
            progress("assessment_split_completed", split_seed=split_seed)
        for relative, expected in locked_files.items():
            if fingerprint_files(output, [relative])[relative] != expected:
                raise RuntimeError(f"Locked artifact changed during reassessment: {relative}")
    results = {"schema_version": 1, "experiment_kind": "history_encoder_ranking_reused_development",
        "status": "completed" if assessment_performed else "incomplete", "completed_utc": utc_now(),
        "configuration": configuration, "dataset": dataset, "gpu": probe,
        "candidate_validation_results": candidate_results, "baseline_grid_results": baseline_results,
        "primary_selection": primary_selection, "ranking_results": ranking_results,
        "history_shuffle_results": shuffle_results, "assessment_performed": assessment_performed,
        "all_candidates_completed": assessment_performed, "training": {"total_neural_epochs": global_epoch},
        "original_test_used": False, "original_validation_used": False,
        "model_promotion_to_application": False, "total_seconds": time.monotonic() - started,
        "limitations": [
            "Repeatedly inspected internal development assessment is not an independent confirmation set.",
            "BPR and full softmax share context, architecture, eligible target examples and initialization, but differ in sampled versus full-catalogue contrast and loss scale.",
            "Only users with two or more fit-positive works supply training examples; inference can use one positive work, with separate cohort results.",
            "History is an unordered set with no timestamps or sequence claim; high-activity users contribute more target examples.",
            "All unobserved works are unknown preferences; masking observed works does not create true negative labels.",
            "The new cosine control uses binary positive work co-occurrence, not the application's adjusted-cosine ISBN model.",
            "Cosine ties use fit popularity; neural ties use lexical work order. All models share no-positive-user popularity fallback.",
            "Fixed history shuffling is a diagnostic perturbation; recipients retain their own exclusions and it is not an independent test.",
            "Three seeds per split, overlapping split users, unadjusted conditional bootstrap; do not pool split observations.",
            "A best checkpoint at the epoch cap does not establish convergence; no model is automatically promoted to the application.",
        ]}
    atomic_json(output / "results.json", results)
    progress("completed", assessment_performed=assessment_performed)
    sys.stdout.flush()
    archive = write_archive(output, "final_backup.tar.gz", [*static_entries, "models", "baselines", "assessment", "ranking", "history_shuffle",
        "current_best_network.keras", "current_checkpoint", "current_checkpoint_state.json",
        "candidate_validation_results.json", "baseline_grid_results.json", "primary_selection.json", "weights_locked.json",
        "ranking_results.json", "history_shuffle_results.json", "results.json", "progress.json", "worker.log"],
        "final_manifest.json", kind="completed_experiment", epochs=global_epoch)
    (output / "completed.flag").write_text(json.dumps({"completed_utc": utc_now(), "sha256": archive["sha256"]}) + "\n")
    emit("EXPERIMENT_COMPLETED", **archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-training-seconds", type=int, default=1200)
    parser.add_argument("--max-ranking-seconds", type=int, default=300)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 20 or args.batch_size < 1 or not 1 <= args.max_training_seconds <= 1200:
        parser.error("Epochs must be 1–20, batch size positive, and training budget 1–1200 seconds")
    if not 1 <= args.max_ranking_seconds <= 300:
        parser.error("Ranking budget must be 1–300 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("configuration.json", "completed.flag", "failed.json")):
        parser.error("Output already contains an experiment")
    try:
        run_experiment(args)
    except Exception as error:  # noqa: BLE001 -- preserve terminal remote failure
        message = SECRET_PATTERN.sub("<redacted>", str(error))
        atomic_json(args.output / "failed.json", {"status": "failed", "failed_utc": utc_now(),
                    "exception_type": type(error).__name__, "message": message})
        emit("EXPERIMENT_FAILED", exception_type=type(error).__name__, message=message)
        print(SECRET_PATTERN.sub("<redacted>", traceback.format_exc()), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
