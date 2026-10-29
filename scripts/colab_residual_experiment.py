"""Bounded, validation-selected residual experiment on the frozen Colab split.

The earlier test results have already been inspected. This is a development
replication, not an untouched-test benchmark. The four candidates below are
declared before training; no candidate is chosen using test outcomes. Runtime
ownership, authentication, downloads and shutdown belong to the caller.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

try:
    from .colab_neural_experiment import (
        ROOT, SECRET_PATTERN, atomic_json, emit, evaluate_ranking, fingerprint_files,
        gpu_probe, source_metadata, utc_now, write_archive,
    )
except ImportError:
    from colab_neural_experiment import (
        ROOT, SECRET_PATTERN, atomic_json, emit, evaluate_ranking, fingerprint_files,
        gpu_probe, source_metadata, utc_now, write_archive,
    )

# These exact bytes were archived before the initial experiment was evaluated.
# Refusing a different split prevents silently comparing unrelated RMSE values.
FROZEN_INPUT_SHA256 = {
    "train_books.csv": "c3082bc48dc429e8bd90df18ed0938437d8b37deb968399efc8fa8a752035b6d",
    "train_ratings.csv": "ce2c45880b2e09ef899131ae953493502619c300b9ce7362a9788ad50a45d07a",
    "validation_ratings.csv": "7fd5277f139a9b390bf7ed3900c7e639a7f8ab4bd594a66bc0fe372f449e83e2",
    "test_ratings.csv": "e87e5387156aa08628fbc482df8718e0c25c3a2fdb64af206a73bfac92ab161d",
}
CANDIDATES = (
    {"name": "bias_only_u0_i20", "kind": "bias_only", "user_shrinkage": 0.0,
     "item_shrinkage": 20.0, "residual_scale": 0.0, "l2": 1e-5},
    {"name": "residual_u0_i20_s025", "kind": "neural", "user_shrinkage": 0.0,
     "item_shrinkage": 20.0, "residual_scale": 0.25, "l2": 1e-5},
    {"name": "residual_u0_i20_s100", "kind": "neural", "user_shrinkage": 0.0,
     "item_shrinkage": 20.0, "residual_scale": 1.0, "l2": 1e-5},
    {"name": "residual_u0_i20_s025_l2e4", "kind": "neural", "user_shrinkage": 0.0,
     "item_shrinkage": 20.0, "residual_scale": 0.25, "l2": 1e-4},
)
PREDICTION_CONVENTION = (
    "Train loss uses raw linear normalized predictions plus L2. Validation, test and "
    "recommendations clip normalized predictions to [0,1], then convert to rating 1+9*p."
)
FALLBACK_POLICY = (
    "Both IDs known: selected predictor. Otherwise known user: train user mean; "
    "otherwise known ISBN: train item mean; otherwise: train global mean."
)


def regression_metrics(targets, predictions, np):
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if targets.shape != predictions.shape or not np.isfinite(predictions).all():
        raise ValueError("Rating predictions must be finite and aligned with targets.")
    error = predictions - targets
    return {
        "pairs": len(targets),
        "rmse_rating_scale": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
        "mae_rating_scale": float(np.mean(np.abs(error))) if len(error) else None,
    }


def encoded_predictions(model, frame, np):
    """Return aligned known-pair predictions without reordering frame rows."""
    pairs = np.column_stack([
        frame.user_id.map(model.user_to_index).to_numpy(dtype=np.int32),
        frame.isbn.map(model.isbn_to_index).to_numpy(dtype=np.int32),
    ])
    return 1.0 + 9.0 * model._predict_encoded(pairs).reshape(-1)


def bootstrap_rmse_difference(targets, predictions, reference, np, draws=1000):
    """Paired percentile interval; negative delta favors the neural predictor."""
    if not len(targets):
        return {"draws": draws, "interval_95_percent": None}
    rng = np.random.default_rng(123)
    indices = rng.integers(0, len(targets), size=(draws, len(targets)))
    errors = (np.asarray(predictions) - np.asarray(targets)) ** 2
    reference_errors = (np.asarray(reference) - np.asarray(targets)) ** 2
    differences = np.sqrt(errors[indices].mean(axis=1)) - np.sqrt(
        reference_errors[indices].mean(axis=1)
    )
    interval = np.quantile(differences, [0.025, 0.975])
    return {
        "draws": draws, "seed": 123, "pairs": len(targets),
        "difference": "best_neural_rmse_minus_train_user_mean_rmse",
        "point_estimate": float(np.sqrt(errors.mean()) - np.sqrt(reference_errors.mean())),
        "interval_95_percent": [float(value) for value in interval],
        "method": "paired bootstrap percentile; one held-out rating per test user",
        "limitation": "Conditional on this reused split; no correction for development reuse.",
    }


def assert_holdout_separation(train_ratings, validation, work_map, test=None):
    train_pairs = {(int(row.user_id), work_map[str(row.isbn)])
                   for row in train_ratings.itertuples(index=False)}
    holdouts = [validation] if test is None else [validation, test]
    if test is not None and not set(validation.user_id).isdisjoint(test.user_id):
        raise RuntimeError("Frozen validation and test users overlap.")
    for holdout in holdouts:
        if any((int(row.user_id), str(row.work_key)) in train_pairs
               for row in holdout.itertuples(index=False)):
            raise RuntimeError("Frozen holdout overlaps a training user/canonical work.")


def full_rating_evaluation(models, train, test, output, np, pd):
    """One evaluation pass for all preselected models, preserving every test row."""
    train_user_means = train.groupby("user_id").book_rating.mean()
    train_item_means = train.groupby("isbn").book_rating.mean()
    global_mean = float(train.book_rating.mean())
    users_known = test.user_id.isin(train_user_means.index).to_numpy()
    items_known = test.isbn.isin(train_item_means.index).to_numpy()
    known_mask = users_known & items_known
    known = test.loc[known_mask]
    targets = test.book_rating.to_numpy(dtype=float)
    user_mean = test.user_id.map(train_user_means).fillna(global_mean).to_numpy(dtype=float)
    item_mean = test.isbn.map(train_item_means).fillna(global_mean).to_numpy(dtype=float)
    fallback = np.where(users_known, user_mean, np.where(items_known, item_mean, global_mean))
    predictions = {
        "train_global_mean": np.full(len(test), global_mean),
        "train_user_mean_global_fallback": user_mean,
        "train_item_mean_global_fallback": item_mean,
        "train_user_item_global_fallback": fallback,
    }
    out_of_range = {}
    for name, model in models.items():
        prediction = fallback.copy()
        prediction[known_mask] = encoded_predictions(model, known, np)
        predictions[name] = prediction
        pairs = np.column_stack([
            known.user_id.map(model.user_to_index).to_numpy(dtype=np.int32),
            known.isbn.map(model.isbn_to_index).to_numpy(dtype=np.int32),
        ])
        raw = model.model(pairs, training=False).numpy().reshape(-1)
        out_of_range[name] = int(((raw < 0) | (raw > 1)).sum())
    groups = {
        "both_known": known_mask,
        "known_user_unknown_isbn": users_known & ~items_known,
        "unknown_user_known_isbn": ~users_known & items_known,
        "both_unknown": ~users_known & ~items_known,
    }
    rows = test.copy()
    rows["known_user"] = users_known
    rows["known_isbn"] = items_known
    for name, prediction in predictions.items():
        rows[name] = prediction
    rows.to_csv(output / "rating_predictions.csv", index=False)
    metrics = {
        name: {
            "all_pairs": regression_metrics(targets, prediction, np),
            "known_pairs": regression_metrics(targets[known_mask], prediction[known_mask], np),
            "by_availability": {
                group: regression_metrics(targets[mask], prediction[mask], np)
                for group, mask in groups.items()
            },
        }
        for name, prediction in predictions.items()
    }
    return {
        "test_explicit_pairs": len(test), "known_pairs": int(known_mask.sum()),
        "known_pair_retention_rate": float(known_mask.mean()),
        "availability": {name: int(mask.sum()) for name, mask in groups.items()},
        "scale": "explicit ratings 1 through 10",
        "prediction_convention": PREDICTION_CONVENTION, "fallback_policy": FALLBACK_POLICY,
        "known_predictions_clipped": out_of_range, "models": metrics,
        "best_neural_vs_user_mean_paired_bootstrap": bootstrap_rmse_difference(
            targets[known_mask], predictions["best_neural"][known_mask], user_mean[known_mask], np,
        ),
    }


def plot_candidates(candidate_results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for candidate in candidate_results:
        history = candidate["history"]
        if history:
            axis.plot([row["epoch"] for row in history],
                      [row["val_rmse"] for row in history], marker="o", label=candidate["name"])
        else:
            axis.axhline(candidate["validation"]["rmse_rating_scale"], linestyle="--",
                         label=candidate["name"])
    axis.set(xlabel="Epoch", ylabel="Validation RMSE (ratings 1–10)",
             title="Four predeclared candidates on the frozen validation split")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "validation_comparison.png", dpi=160)
    plt.close(figure)


def write_report(results, output):
    selection = results["selection"]
    rating = results["rating_test"]
    lines = [
        "Eksperimen lanjutan residual neural — GPU Colab gratis", "",
        "Split train/validation/test identik byte-for-byte dengan eksperimen sebelumnya.",
        "Test sebelumnya sudah dilihat: hasil ini replikasi pengembangan, bukan blind test baru.",
        f"Pilihan serving dari validation: {selection['serving_candidate']}.",
        f"Neural terbaik dari validation: {selection['best_neural_candidate']}.",
        f"Validation dikenal: {results['dataset']['validation_known_pairs']} dari 500 pasangan.",
        "Bias menggunakan statistik train; MLP mempelajari koreksi residual melalui Adam.",
        "Prediksi evaluasi dibatasi pada rating 1–10; loss training menggunakan output linear.",
        "", "Test: semua 500 pasangan dipertahankan; metrik warm memakai pasangan yang sama.",
    ]
    for name, metrics in rating["models"].items():
        lines.append(f"{name}: RMSE all={metrics['all_pairs']['rmse_rating_scale']:.6f}; "
                     f"RMSE warm={metrics['known_pairs']['rmse_rating_scale']:.6f}")
    interval = rating["best_neural_vs_user_mean_paired_bootstrap"]["interval_95_percent"]
    lines.extend([
        "", f"95% paired bootstrap delta RMSE neural − user mean (warm): {interval}.",
        "Fallback cold-start tetap: user mean, lalu item mean, lalu global mean dari train.",
        "Pemilihan kandidat dan epoch memakai validation saja; test dinilai setelah pemilihan.",
        "Hanya 140 pasangan validation dan 135 warm test; variasi sampel dapat membalik urutan model.",
        "Belum ada klaim superioritas umum atau peningkatan pada test baru yang independen.",
        "final_model adalah pilihan serving; best_neural_model selalu model neural terbaik.",
        "Checkpoint epoch menyimpan current_model, current_best_model, optimizer dan RNG.",
    ])
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args):
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from book_recommender.content_based import book_work_key
    from book_recommender.residual_neural import ResidualNeuralRecommender

    output = args.output.resolve()
    started = time.monotonic()
    configuration = {
        "started_utc": utc_now(), "input_dir": str(args.input_dir.resolve()),
        "output": str(output), "candidates": list(CANDIDATES), "candidate_count": 4,
        "max_neural_epochs_each": args.epochs, "batch_size": args.batch_size, "patience": 3,
        "embedding_size": 16, "hidden_units": [32, 16], "learning_rate": 0.001,
        "regularization": "L2 is predeclared separately for each candidate",
        "shared_bias_backbone": "All candidates use user_shrinkage=0 and item_shrinkage=20",
        "seed": 42, "require_gpu": args.require_gpu,
        "max_training_seconds": args.max_training_seconds,
        "selection_metric": "known_validation_rmse_rating_scale",
        "selection_tie_breaker": "earlier candidate in the predeclared list",
        "test_has_been_seen_previously": True,
        "test_scope": "development replication on reused test, not a new blind test",
        "prediction_convention": PREDICTION_CONVENTION, "fallback_policy": FALLBACK_POLICY,
        "fixed_train_only_fallback_not_selected_using_test": True,
    }
    atomic_json(output / "configuration.json", configuration)
    fingerprints = fingerprint_files(args.input_dir, FROZEN_INPUT_SHA256)
    if any(fingerprints[name]["sha256"] != digest
           for name, digest in FROZEN_INPUT_SHA256.items()):
        raise RuntimeError("Inputs differ from the frozen original experiment; refusing comparison.")
    probe = gpu_probe(tf, args.require_gpu)
    atomic_json(output / "gpu_probe.json", probe)
    emit("GPU_PROBE_PASSED", gpu=probe["nvidia_smi"], device=probe["probe_device"])
    metadata = source_metadata()
    metadata["source_files"].update(fingerprint_files(ROOT, [
        "scripts/colab_residual_experiment.py", "src/book_recommender/residual_neural.py",
    ]))
    metadata["experiment_label"] = "frozen_split_validation_selected_residual_development_replication"
    atomic_json(output / "source_metadata.json", metadata)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in FROZEN_INPUT_SHA256:
        shutil.copyfile(args.input_dir / name, inputs / name)
    if fingerprint_files(inputs, FROZEN_INPUT_SHA256) != fingerprints:
        raise RuntimeError("Copied frozen input bytes changed.")
    atomic_json(output / "input_fingerprints.json", fingerprints)
    train_books = pd.read_csv(inputs / "train_books.csv", dtype={"isbn": str})
    train = pd.read_csv(inputs / "train_ratings.csv", dtype={"isbn": str})
    validation = pd.read_csv(inputs / "validation_ratings.csv", dtype={"isbn": str})
    # Do not load test ratings into a dataframe until candidate selection is saved.
    work_map = {
        str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
        for row in train_books.itertuples(index=False)
    }
    assert_holdout_separation(train, validation, work_map)
    known_validation = validation[
        validation.user_id.isin(train.user_id) & validation.isbn.isin(train.isbn)
    ]
    if known_validation.empty:
        raise RuntimeError("No known validation pairs are available for candidate selection.")
    dataset = {
        "train_pairs": len(train), "train_users": int(train.user_id.nunique()),
        "train_isbns": int(train.isbn.nunique()), "validation_pairs": len(validation),
        "validation_known_pairs": len(known_validation), "test_pairs_frozen_reference": 500,
        "input_bytes_match_original_experiment": True,
        "split_policy": "Reuse prior work-level holdout and train-only catalog selection exactly",
    }
    atomic_json(output / "dataset_summary.json", dataset)
    manifest = write_archive(
        output, "inputs_backup.tar.gz", ["inputs", "input_fingerprints.json", "configuration.json",
        "gpu_probe.json", "source_metadata.json", "dataset_summary.json"],
        "inputs_manifest.json", kind="training_inputs",
    )
    emit("INPUTS_BACKUP_READY", **manifest)
    candidate_results = []
    global_epoch = 0
    training_started = time.monotonic()
    current_candidate = None

    def checkpoint_epoch(model, record):
        nonlocal global_epoch
        if record["epoch"]:
            global_epoch += 1
        # Called before fit restores best parameters: optimizer and current model
        # belong to the same step. The separate best model is for inference.
        model.save(output / "current_model")
        if not record["epoch"] or record["epoch"] == model.best_epoch:
            model.save(output / "current_best_model")
        checkpoint = output / "current_checkpoint"
        checkpoint.mkdir(exist_ok=True)
        tf.train.Checkpoint(model=model.model, optimizer=model.optimizer).write(
            str(checkpoint / "training")
        )
        state = {
            "candidate": current_candidate, "candidate_epoch": record["epoch"],
            "epoch": global_epoch, "record": record, "history": model.history,
            "best_epoch": getattr(model, "best_epoch", None),
            "optimizer_iterations": int(model.optimizer.iterations.numpy()),
            "training_rng_state": getattr(model, "training_rng_state", None),
            "baseline_summary": model.baseline_summary,
            "elapsed_seconds": time.monotonic() - started, "updated_utc": utc_now(),
            "status": "training", "checkpoint_prefix": "current_checkpoint/training",
            "resume_scope": "Restore current model + optimizer + RNG manually; fit starts fresh.",
        }
        atomic_json(output / "current_checkpoint_state.json", state)
        atomic_json(output / "progress.json", state)
        atomic_json(output / "candidate_validation_results.json", candidate_results)
        manifest = write_archive(
            output, f"epoch_{global_epoch:03d}_snapshot.tar.gz",
            ["current_model", "current_best_model", "current_checkpoint",
             "current_checkpoint_state.json", "progress.json",
             "configuration.json", "gpu_probe.json", "source_metadata.json",
             "input_fingerprints.json", "dataset_summary.json", "candidate_validation_results.json"],
            f"epoch_{global_epoch:03d}_manifest.json", kind="epoch_checkpoint",
            epoch=global_epoch, candidate=current_candidate["name"], candidate_epoch=record["epoch"],
        )
        atomic_json(output / "snapshot_manifest.json", manifest)
        emit("EPOCH_BACKUP_READY", **manifest)
        if time.monotonic() - training_started > args.max_training_seconds:
            raise TimeoutError("Bounded training time reached; current checkpoint has been archived.")

    for candidate in CANDIDATES:
        current_candidate = dict(candidate)
        tf.keras.backend.clear_session()
        model = ResidualNeuralRecommender(
            embedding_size=16, hidden_units=(32, 16), learning_rate=0.001, l2=candidate["l2"],
            random_state=42, user_shrinkage=candidate["user_shrinkage"],
            item_shrinkage=candidate["item_shrinkage"], residual_scale=candidate["residual_scale"],
        )
        candidate_started = time.monotonic()
        emit("CANDIDATE_STARTED", **candidate)
        if candidate["kind"] == "bias_only":
            model.initialize(train_books, train)
            checkpoint_epoch(model, {"epoch": 0})
        else:
            model.fit(train_books, train, validation_ratings=validation, epochs=args.epochs,
                      batch_size=args.batch_size, patience=3, epoch_callback=checkpoint_epoch)
        validation_predictions = encoded_predictions(model, known_validation, np)
        metrics = regression_metrics(known_validation.book_rating, validation_predictions, np)
        candidate_dir = output / "candidates" / candidate["name"]
        model.save(candidate_dir / "model")
        record = {
            **candidate, "validation": metrics, "best_epoch": getattr(model, "best_epoch", None),
            "epochs_completed": len(model.history), "history": model.history,
            "baseline_summary": model.baseline_summary,
            "first_step_trace": model.first_step_trace,
            "seconds": time.monotonic() - candidate_started,
        }
        candidate_results.append(record)
        atomic_json(candidate_dir / "validation_result.json", record)
        atomic_json(output / "candidate_validation_results.json", candidate_results)
        emit("CANDIDATE_VALIDATED", name=candidate["name"], **metrics)
        del model
        gc.collect()

    selected = min(candidate_results, key=lambda row: row["validation"]["rmse_rating_scale"])
    best_neural = min((row for row in candidate_results if row["kind"] == "neural"),
                      key=lambda row: row["validation"]["rmse_rating_scale"])
    user_means = train.groupby("user_id").book_rating.mean()
    validation_user_mean = regression_metrics(
        known_validation.book_rating,
        known_validation.user_id.map(user_means).to_numpy(dtype=float), np,
    )
    selection = {
        "selected_utc": utc_now(), "metric": configuration["selection_metric"],
        "serving_candidate": selected["name"], "serving_kind": selected["kind"],
        "serving_validation": selected["validation"],
        "best_neural_candidate": best_neural["name"],
        "best_neural_validation": best_neural["validation"],
        "validation_user_mean": validation_user_mean,
        "candidate_count": len(candidate_results), "test_evaluated_at_selection": False,
        "limitation": "140 known validation pairs; sample uncertainty can reverse model ordering.",
    }
    atomic_json(output / "selection.json", selection)
    emit("CANDIDATE_SELECTION_FROZEN", **selection)
    atomic_json(output / "progress.json", {
        "status": "evaluating", "epoch": global_epoch, "selection": selection,
        "updated_utc": utc_now(),
    })
    # Only now inspect the reused test. No subsequent training, tuning or retries
    # based on these numbers are performed by this worker.
    test = pd.read_csv(inputs / "test_ratings.csv", dtype={"isbn": str})
    assert_holdout_separation(train, validation, work_map, test)
    dataset["test_pairs"] = len(test)
    dataset["canonical_work_leakage_checks_passed"] = True
    atomic_json(output / "dataset_summary.json", dataset)
    chosen = {}
    for name in dict.fromkeys([CANDIDATES[0]["name"], selected["name"], best_neural["name"]]):
        chosen[name] = ResidualNeuralRecommender.load(output / "candidates" / name / "model")
    neural = chosen[best_neural["name"]]
    serving = chosen[selected["name"]]
    models = {"bias_only": chosen[CANDIDATES[0]["name"]],
              "best_neural": neural, "selected_serving": serving}
    rating_test = full_rating_evaluation(models, train, test, output, np, pd)
    ranking_test = evaluate_ranking(neural, train_books, train, test, work_map, output)
    ranking_test["evaluated_neural_candidate"] = best_neural["name"]
    ranking_test["selection_objective"] = "known validation rating RMSE, not ranking metrics"
    serving.save(output / "final_model")
    neural.save(output / "best_neural_model")
    roundtrips = {}
    for name, original in (("final_model", serving), ("best_neural_model", neural)):
        restored = ResidualNeuralRecommender.load(output / name)
        np.testing.assert_allclose(encoded_predictions(original, train.head(8), np),
                                   encoded_predictions(restored, train.head(8), np),
                                   rtol=1e-6, atol=1e-7)
        roundtrips[name] = True
    previous = None
    if args.reference_results.is_file():
        previous_result = json.loads(args.reference_results.read_text(encoding="utf-8"))
        previous_rating = previous_result["rating_test"]
        if previous_rating["known_pairs"] != rating_test["known_pairs"]:
            raise RuntimeError("Original experiment's known test count does not match frozen inputs.")
        previous = {
            "source": str(args.reference_results), "same_input_sha256_verified": True,
            "previous_completed_utc": previous_result["completed_utc"],
            "scope": "Published prior run on these identical known test pairs; not retrained here",
            "known_pairs": previous_rating["known_pairs"],
            "neural": previous_rating["models"]["neural"],
            "user_mean": previous_rating["models"]["train_user_mean"],
            "previous_validation_rmse": min(
                row["val_rmse"] for row in previous_result["training"]["history"]
            ),
        }
    results = {
        "status": "completed", "completed_utc": utc_now(), "configuration": configuration,
        "gpu": probe, "dataset": dataset, "selection": selection,
        "candidate_validation_results": candidate_results,
        "training": {"total_neural_epochs": global_epoch,
                     "best_neural_epochs_completed": best_neural["epochs_completed"],
                     "best_neural_best_epoch": best_neural["best_epoch"],
                     "restored_best_validation_weights": True},
        "rating_test": rating_test, "ranking_test": ranking_test,
        "previous_published_original_nn": previous,
        "model_roundtrip_verified": roundtrips,
        "interpretation": "Development replication on a previously inspected test split. "
                          "Four candidates predeclared, selected only by validation. Includes "
                          "all 500 test pairs with deterministic train-only cold fallback. "
                          "A bias-only winner is not evidence of neural improvement. "
                          "No claim of superiority on independent unseen data.",
        "total_seconds": time.monotonic() - started,
    }
    atomic_json(output / "results.json", results)
    plot_candidates(candidate_results, output)
    write_report(results, output)
    atomic_json(output / "progress.json", {
        "status": "completed", "epoch": global_epoch, "selection": selection,
        "updated_utc": utc_now(),
    })
    sys.stdout.flush()
    manifest = write_archive(
        output, "final_backup.tar.gz",
        ["final_model", "best_neural_model", "candidates", "current_checkpoint", "current_model",
         "current_checkpoint_state.json",
         "current_best_model", "inputs", "input_fingerprints.json", "configuration.json",
         "gpu_probe.json", "source_metadata.json", "dataset_summary.json", "selection.json",
         "candidate_validation_results.json", "results.json", "rating_predictions.csv",
         "ranking_rows.csv", "evaluation_progress.json", "validation_comparison.png",
         "progress.json", "report.txt", "worker.log"],
        "final_manifest.json", kind="completed_experiment", epochs=global_epoch,
    )
    (output / "completed.flag").write_text(
        json.dumps({"completed_utc": utc_now(), "sha256": manifest["sha256"]}) + "\n",
        encoding="utf-8",
    )
    emit("EXPERIMENT_COMPLETED", **manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path,
                        default=ROOT / "docs/evidence/neural_colab_t4_results.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-training-seconds", type=int, default=600)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 10 or args.batch_size < 1 or args.max_training_seconds < 1:
        parser.error("epochs must be 1–10; batch size and time limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in
           ("completed.flag", "configuration.json", "failed.json")):
        parser.error("output already contains an experiment; choose a fresh output path")
    try:
        run_experiment(args)
    except Exception as error:  # noqa: BLE001 -- preserve a terminal worker failure
        message = SECRET_PATTERN.sub("<redacted>", str(error))
        atomic_json(args.output / "failed.json", {
            "status": "failed", "failed_utc": utc_now(),
            "exception_type": type(error).__name__, "message": message,
        })
        emit("EXPERIMENT_FAILED", exception_type=type(error).__name__, message=message)
        print(SECRET_PATTERN.sub("<redacted>", traceback.format_exc()), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
