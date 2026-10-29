"""Bounded internal-development comparison across split and initialization seeds.

Inputs are exactly the previously frozen training books and ratings. Old
validation/test files are prohibited. All twelve candidate weights are locked
before either new assessment partition is scored. These partitions come from
previously used training data and are not fresh independent external tests.
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
    from .colab_residual_experiment import regression_metrics
except ImportError:
    from colab_neural_experiment import (
        ROOT, SECRET_PATTERN, atomic_json, emit, fingerprint_files, gpu_probe,
        save_network, source_metadata, utc_now, write_archive,
    )
    from colab_residual_experiment import regression_metrics

FROZEN_INPUT_SHA256 = {
    "train_books.csv": "c3082bc48dc429e8bd90df18ed0938437d8b37deb968399efc8fa8a752035b6d",
    "train_ratings.csv": "ce2c45880b2e09ef899131ae953493502619c300b9ce7362a9788ad50a45d07a",
}
SPLIT_SEEDS = (17, 29)
INITIALIZATION_SEEDS = (42, 2026, 7)
CANDIDATES = tuple(
    {"name": f"split{split_seed}_{architecture}_seed{initialization_seed}",
     "kind": "neural", "split_seed": split_seed, "initialization_seed": initialization_seed,
     "architecture": architecture, "l2": 1e-5 if architecture == "mlp" else 0.0}
    for split_seed in SPLIT_SEEDS
    for initialization_seed in INITIALIZATION_SEEDS
    for architecture in ("mlp", "residual")
)
FALLBACK_POLICY = (
    "All compared models: both IDs known uses their own predictor; otherwise known user uses "
    "fit user mean, otherwise known ISBN uses fit item mean, otherwise fit global mean."
)


def records_digest(records):
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_development_split(books, ratings, seed, np, pd, *, users_per_partition=3000):
    """Sample users/works without consulting rating values; retain every edition."""
    from book_recommender.content_based import book_work_key

    if users_per_partition < 1:
        raise ValueError("Each holdout partition must contain at least one user.")
    work_map = {str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
                for row in books.itertuples(index=False)}
    rows = ratings.reset_index(drop=True).copy()
    rows["source_row"] = np.arange(len(rows))
    rows["work_key"] = rows.isbn.map(work_map)
    if rows.work_key.isna().any():
        raise ValueError("Every rating ISBN must have frozen book metadata.")
    user_works = rows.groupby("user_id", sort=True).work_key.agg(lambda values: sorted(set(values)))
    eligible = user_works[user_works.map(len) >= 2]
    if len(eligible) < 2 * users_per_partition:
        raise ValueError("Insufficient users with two canonical works for disjoint holdout users.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible.index.to_numpy(), size=2 * users_per_partition, replace=False)
    membership = []
    heldout = {}
    for position, user_id in enumerate(chosen):
        works = eligible.loc[user_id]
        work = works[int(rng.integers(len(works)))]
        partition = "selection" if position < users_per_partition else "assessment"
        heldout[(int(user_id), str(work))] = partition
        membership.append({"user_id": int(user_id), "work_key": str(work), "partition": partition})
    membership.sort(key=lambda row: (row["partition"], row["user_id"], row["work_key"]))
    rows["partition"] = [heldout.get((int(user), str(work)), "fit")
                         for user, work in zip(rows.user_id, rows.work_key)]
    partitions = {name: rows.loc[rows.partition == name].drop(columns="partition").copy()
                  for name in ("fit", "selection", "assessment")}
    keys = {name: set(zip(frame.user_id, frame.work_key)) for name, frame in partitions.items()}
    if any(keys[left] & keys[right] for left, right in
           (("fit", "selection"), ("fit", "assessment"), ("selection", "assessment"))):
        raise RuntimeError("A user/canonical-work group crossed partition boundaries.")
    if set(partitions["selection"].user_id) & set(partitions["assessment"].user_id):
        raise RuntimeError("Selection and assessment users overlap within one split.")
    if set(rows.user_id) != set(partitions["fit"].user_id):
        raise RuntimeError("A holdout removed all fit history for a source user.")
    if sum(len(frame) for frame in partitions.values()) != len(rows):
        raise RuntimeError("Partition row accounting is incomplete.")
    summary = {
        "split_seed": seed, "eligible_users": len(eligible), "source_pairs": len(rows),
        "fit_pairs": len(partitions["fit"]), "fit_users": int(partitions["fit"].user_id.nunique()),
        "fit_isbns": int(partitions["fit"].isbn.nunique()),
        "selection_pairs": len(partitions["selection"]),
        "assessment_pairs": len(partitions["assessment"]),
        "selection_users": int(partitions["selection"].user_id.nunique()),
        "assessment_users": int(partitions["assessment"].user_id.nunique()),
        "selection_works": len(keys["selection"]), "assessment_works": len(keys["assessment"]),
        "membership_sha256": records_digest(membership),
        "partition_row_sha256": {
            name: records_digest(frame.source_row.astype(int).tolist())
            for name, frame in partitions.items()
        },
        "rating_values_used_for_split": False, "all_heldout_editions_retained": True,
        "selection_assessment_users_disjoint": True, "all_users_keep_fit_history": True,
        "user_work_overlap_count": 0,
        "metric_weighting": "Rating-pair weighted; repeated editions remain in the same user cluster",
    }
    if summary["selection_users"] != users_per_partition or summary["assessment_users"] != users_per_partition:
        raise RuntimeError("Holdout user count differs from the declared protocol.")
    return partitions, membership, summary


def user_cluster_bootstrap(user_ids, targets, predictions, reference, np, *, draws=1000, seed=123):
    """Paired RMSE differences, resampling whole users and retaining their rows."""
    users = np.asarray(user_ids)
    targets, predictions, reference = (np.asarray(values, dtype=float).reshape(-1)
                                      for values in (targets, predictions, reference))
    if not (len(users) == len(targets) == len(predictions) == len(reference)) or not len(users):
        raise ValueError("Bootstrap arrays must be aligned and nonempty.")
    if not all(np.isfinite(values).all() for values in (targets, predictions, reference)):
        raise ValueError("Bootstrap inputs must be finite.")
    if draws < 1:
        raise ValueError("Bootstrap draw count must be positive.")
    unique_users, index = np.unique(users, return_inverse=True)
    count = np.bincount(index)
    error = np.bincount(index, weights=(predictions - targets) ** 2)
    reference_error = np.bincount(index, weights=(reference - targets) ** 2)
    rng = np.random.default_rng(seed)
    differences = np.empty(draws)
    # Small chunks avoid a draws × all-rating-pairs allocation.
    for start in range(0, draws, 100):
        chosen = rng.integers(len(unique_users), size=(min(100, draws - start), len(unique_users)))
        denominators = count[chosen].sum(axis=1)
        differences[start:start + len(chosen)] = (
            np.sqrt(error[chosen].sum(axis=1) / denominators)
            - np.sqrt(reference_error[chosen].sum(axis=1) / denominators)
        )
    interval = np.quantile(differences, [0.025, 0.975])
    return {
        "draws": draws, "seed": seed, "clusters": len(unique_users), "pairs": len(users),
        "point_estimate": float(np.sqrt(error.sum() / count.sum())
                                - np.sqrt(reference_error.sum() / count.sum())),
        "interval_95_percent": [float(value) for value in interval],
        "method": "Paired percentile bootstrap of users; all rating rows retained within each user",
        "difference": "candidate_rmse_minus_reference_rmse",
        "limitation": "Conditional development-partition interval; not adjusted for multiple comparisons",
    }


def fit_statistics(fit):
    return {
        "user_mean": fit.groupby("user_id").book_rating.mean(),
        "item_mean": fit.groupby("isbn").book_rating.mean(),
        "global_mean": float(fit.book_rating.mean()),
    }


def evaluate_with_fallback(model, frame, statistics, np, *, predictor="neural"):
    users_known = frame.user_id.isin(statistics["user_mean"].index).to_numpy()
    items_known = frame.isbn.isin(statistics["item_mean"].index).to_numpy()
    known = users_known & items_known
    user_mean = frame.user_id.map(statistics["user_mean"]).fillna(statistics["global_mean"]).to_numpy(float)
    item_mean = frame.isbn.map(statistics["item_mean"]).fillna(statistics["global_mean"]).to_numpy(float)
    predictions = np.where(users_known, user_mean,
                           np.where(items_known, item_mean, statistics["global_mean"]))
    if predictor == "global_mean":
        predictions[known] = statistics["global_mean"]
    elif predictor == "user_mean":
        predictions[known] = user_mean[known]
    elif predictor == "neural":
        known_frame = frame.loc[known]
        pairs = np.column_stack([
            known_frame.user_id.map(model.user_to_index).to_numpy(np.int32),
            known_frame.isbn.map(model.isbn_to_index).to_numpy(np.int32),
        ])
        if len(pairs):
            predictions[known] = 1.0 + 9.0 * model._predict_encoded(pairs).reshape(-1)
    else:
        raise ValueError("Unknown predictor kind.")
    targets = frame.book_rating.to_numpy(float)
    groups = {
        "both_known": known,
        "known_user_unknown_isbn": users_known & ~items_known,
        "unknown_user_known_isbn": ~users_known & items_known,
        "both_unknown": ~users_known & ~items_known,
    }
    metrics = {
        "known_pairs": regression_metrics(targets[known], predictions[known], np),
        "all_pairs": regression_metrics(targets, predictions, np),
        "availability": {name: int(mask.sum()) for name, mask in groups.items()},
        "known_pair_retention_rate": float(known.mean()), "fallback_policy": FALLBACK_POLICY,
    }
    return metrics, predictions, known


def run_experiment(args):
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from book_recommender.neural import NeuralRecommender
    from book_recommender.residual_neural import ResidualNeuralRecommender

    output = args.output.resolve()
    started = time.monotonic()
    configuration = {
        "started_utc": utc_now(), "candidates": list(CANDIDATES), "candidate_count": len(CANDIDATES),
        "split_seeds": list(SPLIT_SEEDS), "initialization_seeds": list(INITIALIZATION_SEEDS),
        "architectures": ["mlp", "residual"], "users_per_partition": 3000,
        "training_seed_scope": "Weight initialization and minibatch shuffle; identical seed paired across architectures",
        "deterministic_operations_requested": True,
        "max_neural_epochs_each": args.epochs, "patience": 3, "batch_size": args.batch_size,
        "embedding_size": 16, "hidden_units": [32, 16], "learning_rate": 0.001,
        "residual_scale": 0.25, "user_shrinkage": 0.0, "item_shrinkage": 20.0,
        "max_training_seconds": args.max_training_seconds, "bootstrap_draws": 1000,
        "selection_metric": "known_selection_rmse_rating_scale",
        "selection_tie_breaker": "earlier predeclared candidate",
        "split_policy": "Sample 6000 users with >=2 canonical works uniformly; disjoint 3000 "
                        "selection/3000 assessment users; one uniform work each; retain all editions",
        "metric_weighting": "Rating-pair weighted with user-cluster bootstrap",
        "fallback_policy": FALLBACK_POLICY,
        "assessment_gate": "All 12 successful candidate networks frozen and hashed before assessment",
        "original_test_used": False, "original_validation_used": False,
        "scope": "Internal development repartition of previously used training data, not a fresh test",
        "model_promotion_to_application": False,
    }
    atomic_json(output / "configuration.json", configuration)
    csv_inputs = {path.name for path in args.input_dir.glob("*.csv")}
    if csv_inputs != set(FROZEN_INPUT_SHA256):
        raise RuntimeError("Input directory must contain exactly the two frozen training CSVs.")
    fingerprints = fingerprint_files(args.input_dir, FROZEN_INPUT_SHA256)
    if any(fingerprints[name]["sha256"] != digest for name, digest in FROZEN_INPUT_SHA256.items()):
        raise RuntimeError("Frozen training inputs differ from their predeclared hashes.")
    tf.keras.utils.set_random_seed(42)
    determinism = {"requested": True, "enabled": False}
    try:
        tf.config.experimental.enable_op_determinism()
        determinism["enabled"] = True
    except AttributeError:
        determinism["reason"] = "TensorFlow build does not expose enable_op_determinism"
    probe = gpu_probe(tf, args.require_gpu)
    probe["determinism"] = determinism
    atomic_json(output / "gpu_probe.json", probe)
    emit("GPU_PROBE_PASSED", gpu=probe["nvidia_smi"], determinism=determinism)
    metadata = source_metadata()
    metadata["source_files"].update(fingerprint_files(ROOT, [
        "scripts/colab_neural_multiseed.py", "scripts/colab_residual_experiment.py",
        "src/book_recommender/residual_neural.py",
    ]))
    metadata["experiment_label"] = "multiseed_internal_development"
    atomic_json(output / "source_metadata.json", metadata)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in FROZEN_INPUT_SHA256:
        shutil.copyfile(args.input_dir / name, inputs / name)
    if fingerprint_files(inputs, FROZEN_INPUT_SHA256) != fingerprints:
        raise RuntimeError("Copied input bytes changed.")
    atomic_json(output / "input_fingerprints.json", fingerprints)
    books = pd.read_csv(inputs / "train_books.csv", dtype={"isbn": str})
    source = pd.read_csv(inputs / "train_ratings.csv", dtype={"isbn": str})
    split_summaries = []
    for split_seed in SPLIT_SEEDS:
        partitions, membership, summary = build_development_split(books, source, split_seed, np, pd)
        directory = output / "splits" / f"split_{split_seed}"
        directory.mkdir(parents=True)
        for name, frame in partitions.items():
            frame.to_csv(directory / f"{name}_ratings.csv", index=False)
        atomic_json(directory / "membership.json", membership)
        summary["partition_file_fingerprints"] = fingerprint_files(directory, [
            "fit_ratings.csv", "selection_ratings.csv", "assessment_ratings.csv", "membership.json",
        ])
        atomic_json(directory / "split_summary.json", summary)
        split_summaries.append(summary)
        del partitions
    dataset = {"source_pairs": len(source), "source_users": int(source.user_id.nunique()),
               "source_isbns": int(source.isbn.nunique()), "splits": split_summaries,
               "original_test_used": False, "original_validation_used": False}
    atomic_json(output / "dataset_summary.json", dataset)
    static_entries = ["inputs", "splits", "input_fingerprints.json", "configuration.json",
                      "gpu_probe.json", "source_metadata.json", "dataset_summary.json"]
    manifest = write_archive(output, "inputs_backup.tar.gz", static_entries,
                             "inputs_manifest.json", kind="training_inputs")
    emit("INPUTS_BACKUP_READY", **manifest)
    candidate_results, controls = [], []
    global_epoch, snapshot_number = 0, 0
    deadline = time.monotonic() + args.max_training_seconds
    current_candidate = None
    model_classes = {"mlp": NeuralRecommender, "residual": ResidualNeuralRecommender}

    def persist_progress(status, **extra):
        atomic_json(output / "progress.json", {
            "status": status, "epoch": global_epoch, "candidate": current_candidate,
            "updated_utc": utc_now(), **extra,
        })
        atomic_json(output / "candidate_validation_results.json", candidate_results)
        atomic_json(output / "controls.json", controls)

    def snapshot(reason):
        nonlocal snapshot_number
        snapshot_number += 1
        manifest = write_archive(output, f"epoch_{snapshot_number:03d}_snapshot.tar.gz", [
            "current_network.keras", "current_best_network.keras", "current_checkpoint",
            "current_checkpoint_state.json", "candidate_validation_results.json", "controls.json",
            "weights_locked.json", "progress.json", "configuration.json", "worker.log",
        ], f"epoch_{snapshot_number:03d}_manifest.json", kind="epoch_checkpoint",
            epoch=global_epoch, candidate=current_candidate["name"], reason=reason)
        atomic_json(output / "snapshot_manifest.json", manifest)
        emit("EPOCH_BACKUP_READY", **manifest)

    def save_current(model, epoch, *, partial_epoch=False):
        save_network(model.model, output / "current_network.keras")
        checkpoint = output / "current_checkpoint"
        checkpoint.mkdir(exist_ok=True)
        tf.train.Checkpoint(model=model.model, optimizer=model.optimizer).write(str(checkpoint / "training"))
        atomic_json(output / "current_checkpoint_state.json", {
            "candidate": current_candidate, "candidate_epoch": epoch, "epoch": global_epoch,
            "partial_epoch": partial_epoch, "history": model.history, "best_epoch": model.best_epoch,
            "optimizer_iterations": int(model.optimizer.iterations.numpy()),
            "training_rng_state": getattr(model, "training_rng_state", None),
            "checkpoint_prefix": "current_checkpoint/training", "updated_utc": utc_now(),
            "resume_scope": "Matched current network+Adam+RNG; partial epoch order not resumable",
            "vocabulary_source": "Reinitialize from the matching split fit partition",
        })

    for split_seed in SPLIT_SEEDS:
        split_dir = output / "splits" / f"split_{split_seed}"
        fit = pd.read_csv(split_dir / "fit_ratings.csv", dtype={"isbn": str})
        selection_frame = pd.read_csv(split_dir / "selection_ratings.csv", dtype={"isbn": str})
        statistics = fit_statistics(fit)
        tf.keras.backend.clear_session()
        baseline = ResidualNeuralRecommender(l2=0.0, residual_scale=0.0,
                                            user_shrinkage=0.0, item_shrinkage=20.0).initialize(books, fit)
        baseline.best_epoch = None
        baseline.save(output / "controls" / f"split_{split_seed}" / "bias_model")
        for control_kind in ("global_mean", "user_mean", "bias"):
            metrics, _, _ = evaluate_with_fallback(
                baseline, selection_frame, statistics, np,
                predictor="neural" if control_kind == "bias" else control_kind,
            )
            controls.append({"name": f"split{split_seed}_{control_kind}", "kind": "control",
                             "control_kind": control_kind, "split_seed": split_seed,
                             "label": f"{control_kind} on known pairs; common train-only fallback on cold pairs",
                             "selection": metrics})
        del baseline
        for candidate in (row for row in CANDIDATES if row["split_seed"] == split_seed):
            current_candidate = dict(candidate)
            record = {**candidate, "status": "not_started", "history": [], "epochs_completed": 0,
                      "best_epoch": None, "original_test_used": False, "assessment_scored": False}
            if time.monotonic() >= deadline:
                record["status"] = "skipped_budget"
                candidate_results.append(record)
                persist_progress("budget_exhausted")
                continue
            candidate_started = time.monotonic()
            model = None
            candidate_dir = output / "candidates" / candidate["name"]
            candidate_dir.mkdir(parents=True)
            emit("CANDIDATE_STARTED", **candidate)
            try:
                tf.keras.backend.clear_session()
                kwargs = {"embedding_size": 16, "hidden_units": (32, 16), "learning_rate": 0.001,
                          "l2": candidate["l2"], "random_state": candidate["initialization_seed"]}
                if candidate["architecture"] == "residual":
                    kwargs.update(user_shrinkage=0.0, item_shrinkage=20.0, residual_scale=0.25)
                model = model_classes[candidate["architecture"]](**kwargs).initialize(books, fit)
                model.best_epoch = None
                model._check_validation_overlap(selection_frame)
                pairs, targets, _ = model.encode_ratings(model.ratings)
                val_pairs, val_targets, ignored = model.encode_ratings(selection_frame)
                if not len(val_pairs):
                    raise RuntimeError("No known selection pairs.")
                record["selection_known_pairs"] = len(val_pairs)
                record["selection_unknown_pairs"] = ignored
                save_network(model.model, candidate_dir / "initial_network.keras")
                record["initial_network_fingerprint"] = fingerprint_files(candidate_dir, ["initial_network.keras"])
                rng = np.random.default_rng(candidate["initialization_seed"])
                best_mse, best_weights, wait = float("inf"), None, 0
                for epoch in range(1, args.epochs + 1):
                    order = rng.permutation(len(pairs))
                    model.training_rng_state = rng.bit_generator.state
                    for start in range(0, len(order), args.batch_size):
                        if time.monotonic() >= deadline:
                            save_current(model, epoch, partial_epoch=True)
                            raise TimeoutError("Training time budget exhausted at minibatch boundary.")
                        batch = order[start:start + args.batch_size]
                        if not model.first_step_trace:
                            model.first_step_trace = model.trace_training_step(pairs[batch], targets[batch])
                        else:
                            model._compiled_step(pairs[batch], targets[batch])
                    train_mse, val_mse = float(model._mse(pairs, targets)), float(model._mse(val_pairs, val_targets))
                    epoch_record = {"epoch": epoch, "train_mse": train_mse,
                                    "train_rmse": float(9 * np.sqrt(train_mse)),
                                    "val_mse": val_mse, "val_rmse": float(9 * np.sqrt(val_mse))}
                    if not np.isfinite([train_mse, val_mse]).all():
                        raise RuntimeError("Non-finite training or selection loss.")
                    if val_mse < best_mse:
                        best_mse, best_weights, wait = val_mse, model.model.get_weights(), 0
                        model.best_epoch = epoch
                        save_network(model.model, output / "current_best_network.keras")
                    else:
                        wait += 1
                    model.history.append(epoch_record)
                    global_epoch += 1
                    save_current(model, epoch)
                    persist_progress("training", candidate_epoch=epoch, record=epoch_record)
                    emit("EPOCH_FINISHED", candidate=candidate["name"], **epoch_record)
                    if epoch % 4 == 0:
                        snapshot("periodic_four_epochs")
                    if wait >= 3:
                        break
                if best_weights is not None:
                    model.model.set_weights(best_weights)
                selection_metrics, prediction, _ = evaluate_with_fallback(model, selection_frame, statistics, np)
                record.update({"status": "completed", "validation": selection_metrics["known_pairs"],
                               "selection": selection_metrics, "history": model.history,
                               "epochs_completed": len(model.history), "best_epoch": model.best_epoch,
                               "first_step_trace": model.first_step_trace})
                model.save(candidate_dir / "model")
                restored = model_classes[candidate["architecture"]].load(candidate_dir / "model")
                np.testing.assert_allclose(model._predict_encoded(val_pairs), restored._predict_encoded(val_pairs),
                                           rtol=1e-6, atol=1e-7)
                record["model_roundtrip_verified"] = True
                record["selected_model_fingerprints"] = fingerprint_files(candidate_dir / "model", ["network.keras", "metadata.json"])
                del restored
                prediction_rows = selection_frame.copy()
                prediction_rows["prediction"] = prediction
                prediction_rows.to_csv(candidate_dir / "selection_predictions.csv", index=False)
                emit("CANDIDATE_VALIDATED", name=candidate["name"], **record["validation"])
            except Exception as error:  # noqa: BLE001 -- preserve incomplete attempts
                record.update({"status": "budget_exhausted" if isinstance(error, TimeoutError) else "failed",
                               "exception_type": type(error).__name__,
                               "message": SECRET_PATTERN.sub("<redacted>", str(error)),
                               "history": model.history if model is not None else [],
                               "epochs_completed": len(model.history) if model is not None else 0,
                               "best_epoch": getattr(model, "best_epoch", None)})
                emit("CANDIDATE_FAILED", name=candidate["name"], message=record["message"])
                print(SECRET_PATTERN.sub("<redacted>", traceback.format_exc()), file=sys.stderr, flush=True)
            record["seconds"] = time.monotonic() - candidate_started
            candidate_results.append(record)
            atomic_json(candidate_dir / "candidate_result.json", record)
            persist_progress("candidate_completed", candidate_status=record["status"])
            snapshot("candidate_completion")
            model = None
            gc.collect()

    completed = [row for row in candidate_results if row["status"] == "completed"]
    assessment_performed = len(completed) == len(CANDIDATES)
    selection = {"metric": configuration["selection_metric"], "by_split": [],
                 "all_weights_locked": assessment_performed, "assessment_used_for_selection": False}
    for split_seed in SPLIT_SEEDS:
        successful = [row for row in completed if row["split_seed"] == split_seed]
        split_controls = [row for row in controls if row["split_seed"] == split_seed]
        winner = min(successful, key=lambda row: row["validation"]["rmse_rating_scale"], default=None)
        options = [*successful, *split_controls]
        overall = min(options, key=lambda row: row["selection"]["known_pairs"]["rmse_rating_scale"])
        selection["by_split"].append({"split_seed": split_seed,
                                      "best_neural_candidate": winner["name"] if winner else None,
                                      "best_candidate": overall["name"]})
    assessment_results, seed_summary = [], []
    if assessment_performed:
        selection["weights_locked_utc"] = utc_now()
        locked = {row["name"]: row["selected_model_fingerprints"] for row in completed}
        locked_controls = {
            str(seed): fingerprint_files(output / "controls" / f"split_{seed}" / "bias_model",
                                         ["network.keras", "metadata.json"])
            for seed in SPLIT_SEEDS
        }
        recovery_files = ["current_network.keras", "current_best_network.keras", "current_checkpoint_state.json"]
        recovery_files.extend(str(path.relative_to(output)) for path in sorted((output / "current_checkpoint").glob("*"))
                              if path.is_file())
        atomic_json(output / "weights_locked.json", {
            "locked_utc": selection["weights_locked_utc"], "models": locked,
            "initial_models": {row["name"]: row["initial_network_fingerprint"] for row in completed},
            "controls": locked_controls, "recovery_checkpoint": fingerprint_files(output, recovery_files),
            "assessment_scored": False, "selection": selection,
        })
        atomic_json(output / "selection.json", selection)
        persist_progress("weights_locked", selection=selection)
        snapshot("all_candidates_locked_before_assessment")
        for split_seed in SPLIT_SEEDS:
            persist_progress("assessing", assessment_split_seed=split_seed)
            split_dir = output / "splits" / f"split_{split_seed}"
            fit = pd.read_csv(split_dir / "fit_ratings.csv", dtype={"isbn": str})
            # First assessment prediction/metric access occurs after all 12 locks.
            assessment = pd.read_csv(split_dir / "assessment_ratings.csv", dtype={"isbn": str})
            statistics = fit_statistics(fit)
            baseline_dir = output / "controls" / f"split_{split_seed}" / "bias_model"
            if fingerprint_files(baseline_dir, ["network.keras", "metadata.json"]) != locked_controls[str(split_seed)]:
                raise RuntimeError("A locked bias control changed before assessment.")
            baseline = ResidualNeuralRecommender.load(baseline_dir)
            model_metrics, predictions = {}, {}
            for control in (row for row in controls if row["split_seed"] == split_seed):
                metrics, prediction, known = evaluate_with_fallback(
                    baseline, assessment, statistics, np,
                    predictor="neural" if control["control_kind"] == "bias" else control["control_kind"],
                )
                control["assessment"] = metrics
                model_metrics[control["name"]], predictions[control["name"]] = metrics, prediction
            del baseline
            for row in (candidate for candidate in completed if candidate["split_seed"] == split_seed):
                model_dir = output / "candidates" / row["name"] / "model"
                if fingerprint_files(model_dir, ["network.keras", "metadata.json"]) != locked[row["name"]]:
                    raise RuntimeError("A locked model changed before assessment.")
                restored = model_classes[row["architecture"]].load(model_dir)
                metrics, prediction, candidate_known = evaluate_with_fallback(restored, assessment, statistics, np)
                np.testing.assert_array_equal(candidate_known, known)
                row["assessment"], row["assessment_scored"] = metrics, True
                model_metrics[row["name"]], predictions[row["name"]] = metrics, prediction
                atomic_json(output / "candidates" / row["name"] / "candidate_result.json", row)
                del restored
                gc.collect()
            prediction_rows = assessment.copy()
            prediction_rows["both_ids_known"] = known
            for name, prediction in predictions.items():
                prediction_rows[name] = prediction
            prediction_rows.to_csv(split_dir / "assessment_predictions.csv", index=False)
            comparisons = []
            for row in (candidate for candidate in completed if candidate["split_seed"] == split_seed):
                references = [f"split{split_seed}_bias"]
                if row["architecture"] == "residual":
                    references.append(f"split{split_seed}_mlp_seed{row['initialization_seed']}")
                for reference in references:
                    for scope, mask in (("known_pairs", known), ("all_pairs", np.ones(len(known), dtype=bool))):
                        comparison = user_cluster_bootstrap(
                            assessment.user_id.to_numpy()[mask], assessment.book_rating.to_numpy(float)[mask],
                            predictions[row["name"]][mask], predictions[reference][mask], np,
                        )
                        comparisons.append({"candidate": row["name"], "reference": reference,
                                            "scope": scope, "bootstrap": comparison})
            assessment_results.append({"split_seed": split_seed, "models": model_metrics,
                                       "paired_comparisons": comparisons,
                                       "predictions_file": f"splits/split_{split_seed}/assessment_predictions.csv"})
            for architecture in ("mlp", "residual"):
                rows = [row for row in completed if row["split_seed"] == split_seed
                        and row["architecture"] == architecture]
                known_scores = [row["assessment"]["known_pairs"]["rmse_rating_scale"] for row in rows]
                all_scores = [row["assessment"]["all_pairs"]["rmse_rating_scale"] for row in rows]
                seed_summary.append({"split_seed": split_seed, "architecture": architecture,
                                     "initialization_seeds": [row["initialization_seed"] for row in rows],
                                     "models": [row["name"] for row in rows],
                                     "known_rmse_mean": float(np.mean(known_scores)),
                                     "known_rmse_std": float(np.std(known_scores, ddof=1)),
                                     "all_rmse_mean": float(np.mean(all_scores)),
                                     "all_rmse_std": float(np.std(all_scores, ddof=1)),
                                     "best_epochs": [row["best_epoch"] for row in rows],
                                     "variation_scope": "Training seeds change initialization and minibatch order; split held fixed",
                                     "interpretation": "Descriptive sample SD across training seeds, not a confidence interval"})
            atomic_json(output / "assessment_results.json", assessment_results)
            atomic_json(output / "seed_summary.json", seed_summary)
    else:
        selection["assessment_skipped_reason"] = "At least one declared candidate was incomplete; assessment gate kept closed"
    atomic_json(output / "selection.json", selection)
    results = {
        "schema_version": 1, "experiment_kind": "multiseed_internal_development", "status": "completed",
        "completed_utc": utc_now(), "configuration": configuration, "dataset": dataset, "gpu": probe,
        "candidate_validation_results": candidate_results, "controls": controls, "selection": selection,
        "assessment_performed": assessment_performed, "assessment_results": assessment_results,
        "seed_summary": seed_summary, "all_candidates_completed": len(completed) == len(CANDIDATES),
        "training": {"total_neural_epochs": global_epoch}, "original_test_used": False,
        "original_validation_used": False, "model_promotion_to_application": False,
        "total_seconds": time.monotonic() - started,
        "limitations": [
            "Internal development assessment derives from previously used training ratings; it is not a fresh independent test.",
            "All assessment users retain fit history; this is not a cold-user evaluation.",
            "The fixed book catalog was originally selected from the full original training data before this repartition.",
            "Split populations overlap across seeds; no observations or bootstrap draws are pooled across splits.",
            "RMSE/MAE weight rating rows; multiple editions are clustered by user during bootstrap.",
            "Seed means and sample SD are descriptive; confidence intervals are conditional and unadjusted for multiple comparisons.",
            "Only rating prediction is evaluated; ranking and production promotion remain out of scope.",
        ],
    }
    atomic_json(output / "results.json", results)
    report = ["Multiseed internal-development neural comparison", configuration["scope"],
              "Original validation/test files were not uploaded or read.",
              f"Completed candidates: {len(completed)}/{len(CANDIDATES)}", f"Assessment performed: {assessment_performed}"]
    for row in seed_summary:
        report.append(f"Split {row['split_seed']} {row['architecture']}: known RMSE "
                      f"{row['known_rmse_mean']:.6f} ± {row['known_rmse_std']:.6f} (seed SD)")
    (output / "report.txt").write_text("\n".join([*report, *results["limitations"]]) + "\n", encoding="utf-8")
    persist_progress("completed", assessment_performed=assessment_performed)
    sys.stdout.flush()
    manifest = write_archive(output, "final_backup.tar.gz", [
        *static_entries, "candidates", "controls", "current_network.keras", "current_best_network.keras",
        "current_checkpoint", "current_checkpoint_state.json", "candidate_validation_results.json",
        "controls.json", "weights_locked.json", "selection.json", "assessment_results.json",
        "seed_summary.json", "results.json", "progress.json", "report.txt", "worker.log",
    ], "final_manifest.json", kind="completed_experiment", epochs=global_epoch)
    (output / "completed.flag").write_text(json.dumps({"completed_utc": utc_now(), "sha256": manifest["sha256"]}) + "\n")
    emit("EXPERIMENT_COMPLETED", **manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-training-seconds", type=int, default=900)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 12 or args.batch_size < 1 or not 1 <= args.max_training_seconds <= 900:
        parser.error("epochs must be 1–12, batch size positive, and training budget 1–900 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("configuration.json", "completed.flag", "failed.json")):
        parser.error("Output already contains an experiment; choose a new path")
    try:
        run_experiment(args)
    except Exception as error:  # noqa: BLE001 -- persist terminal infrastructure failure
        message = SECRET_PATTERN.sub("<redacted>", str(error))
        atomic_json(args.output / "failed.json", {"status": "failed", "failed_utc": utc_now(),
                                                "exception_type": type(error).__name__, "message": message})
        emit("EXPERIMENT_FAILED", exception_type=type(error).__name__, message=message)
        print(SECRET_PATTERN.sub("<redacted>", traceback.format_exc()), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
