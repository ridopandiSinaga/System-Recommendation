"""Validation-only, bounded diagnostics for residual neural learning on Colab.

Only the three frozen train/books/validation CSVs are accepted. No test file is
needed or evaluated. The caller owns the runtime and verifies local backups
before shutting it down. This worker uses the existing monitor archive protocol.
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
        ROOT, SECRET_PATTERN, atomic_json, emit, fingerprint_files, gpu_probe,
        save_network, source_metadata, utc_now, write_archive,
    )
    from .colab_residual_experiment import assert_holdout_separation, regression_metrics
except ImportError:
    from colab_neural_experiment import (
        ROOT, SECRET_PATTERN, atomic_json, emit, fingerprint_files, gpu_probe,
        save_network, source_metadata, utc_now, write_archive,
    )
    from colab_residual_experiment import assert_holdout_separation, regression_metrics

FROZEN_INPUT_SHA256 = {
    "train_books.csv": "c3082bc48dc429e8bd90df18ed0938437d8b37deb968399efc8fa8a752035b6d",
    "train_ratings.csv": "ce2c45880b2e09ef899131ae953493502619c300b9ce7362a9788ad50a45d07a",
    "validation_ratings.csv": "7fd5277f139a9b390bf7ed3900c7e639a7f8ab4bd594a66bc0fe372f449e83e2",
}
CANDIDATES = (
    {"name": "bias_only", "kind": "bias_only", "l2": 0.0, "head_initialization": "zeros"},
    {"name": "bias_plus_train_fitted_scalar", "kind": "bias_plus_scalar", "l2": 0.0,
     "head_initialization": "zeros"},
    *({"name": f"residual_l2_{label}_zero_head", "kind": "neural", "l2": value,
       "head_initialization": "zeros"}
      for label, value in (("1e4", 1e-4), ("1e5", 1e-5), ("1e6", 1e-6),
                           ("1e7", 1e-7), ("0", 0.0))),
    {"name": "residual_l2_1e6_tiny_random_head", "kind": "neural", "l2": 1e-6,
     "head_initialization": "normal_std_0.001_seed_2026"},
)
WARMUP_CANDIDATES = (
    {"name": "residual_l2_0_zero_head_fixed12", "kind": "neural", "l2": 0.0,
     "head_initialization": "zeros"},
    {"name": "residual_l2_1e7_zero_head_fixed12", "kind": "neural", "l2": 1e-7,
     "head_initialization": "zeros"},
)


def experiment_protocol(preset, epochs):
    """Keep the original ablation fixed while declaring the adaptive followup."""
    if preset == "ablation":
        if not 1 <= epochs <= 10:
            raise ValueError("ablation epochs must be 1–10")
        return {
            "preset": preset, "candidates": list(CANDIDATES), "candidate_count": len(CANDIDATES),
            "max_neural_epochs_each": epochs, "min_neural_epochs_each": 1, "patience": 3,
            "early_stopping_enabled": True, "snapshot_interval_epochs": 2,
            "rationale": "Predeclared regularization and output-head initialization ablation",
        }
    if preset != "warmup" or epochs != 12:
        raise ValueError("warmup preset requires exactly --epochs 12")
    return {
        "preset": preset, "candidates": list(WARMUP_CANDIDATES),
        "candidate_count": len(WARMUP_CANDIDATES),
        "max_neural_epochs_each": 12, "min_neural_epochs_each": 12, "patience": 12,
        "early_stopping_enabled": False, "snapshot_interval_epochs": 4,
        "rationale": "Adaptive followup after observing delayed signal at epoch 3–4 in ablation; "
                     "decided 12 epochs before followup training",
        "previous_controls_reference": "docs/evidence/neural_colab_ablation_results.json",
        "previous_controls_rerun": False,
        "selection_note": "Best validation epoch selected after all 12 epochs; no early stopping",
    }


def correction_summary(values, np):
    """Describe raw neural corrections in original rating points."""
    values = np.asarray(values, dtype=float).reshape(-1) * 9.0
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Corrections must be nonempty and finite.")
    return {
        "pairs": len(values), "mean_rating": float(values.mean()),
        "std_rating": float(values.std()), "min_rating": float(values.min()),
        "max_rating": float(values.max()),
        "absolute_p95_rating": float(np.quantile(np.abs(values), 0.95)),
    }


def fit_scalar_intercept(targets, baseline, np):
    """Raw-MSE optimal intercept from aligned training arrays, before clipping."""
    targets = np.asarray(targets, dtype=float).reshape(-1)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    if targets.shape != baseline.shape or not len(targets):
        raise ValueError("Scalar fitting requires nonempty aligned training arrays.")
    if not np.isfinite(targets).all() or not np.isfinite(baseline).all():
        raise ValueError("Scalar fitting requires finite training values.")
    return float(np.mean(targets - baseline))


def observational_diagnostics(model, train_pairs, val_pairs, probe_pairs, probe_targets,
                              np, tf, *, stage, epoch):
    """Inspect gradients without updating weights or advancing Adam iterations.

    IndexedSlices are densified so repeated user/book indices are summed before
    computing gradient norms. This measures the actual parameter gradient, not
    the norm of unaggregated per-occurrence embedding updates.
    """
    before = int(model.optimizer.iterations.numpy())
    with tf.GradientTape(persistent=True) as tape:
        prediction = model.model(probe_pairs, training=False)
        data_loss = tf.reduce_mean(tf.square(probe_targets - prediction))
        regularization = tf.add_n(model.model.losses)
    variables = model.model.trainable_variables
    data_gradients = tape.gradient(data_loss, variables)
    l2_gradients = tape.gradient(regularization, variables)
    del tape

    def norm(gradient):
        if gradient is None:
            return 0.0
        return float(tf.linalg.norm(tf.convert_to_tensor(gradient)).numpy())

    gradient_norms = {
        variable.path: {"data": norm(data), "l2": norm(l2)}
        for variable, data, l2 in zip(variables, data_gradients, l2_gradients)
    }
    corrections = {}
    for label, pairs in (("train_sample", train_pairs), ("validation", val_pairs)):
        raw = model.model(pairs, training=False).numpy()
        baseline = model.baseline_predictions(pairs, clip=False)
        corrections[label] = correction_summary(raw - baseline, np)
    result = {
        "stage": stage, "epoch": epoch, "optimizer_iterations": before,
        "gradient_probe_pairs": len(probe_pairs), "data_mse": float(data_loss.numpy()),
        "regularization_loss": float(regularization.numpy()),
        "gradient_norms": gradient_norms,
        "gradient_norm_totals": {
            kind: float(np.sqrt(sum(row[kind] ** 2 for row in gradient_norms.values())))
            for kind in ("data", "l2")
        },
        "embedding_norms": {
            "user_l2": norm(model.model.user_embedding.embeddings),
            "book_l2": norm(model.model.book_embedding.embeddings),
        },
        "output_head_kernel_norm": norm(model.model.output_layer.kernel),
        "corrections": corrections,
        "observational_only": True,
        "gradient_definition": "L2 norm after summing duplicate embedding indices",
    }
    if int(model.optimizer.iterations.numpy()) != before:
        raise RuntimeError("Observational diagnostics unexpectedly updated the optimizer.")
    return result


def small_batch_overfit(books, train, model_class, np, tf, deadline, output, updates):
    """Check real-data memorization on a fixed train-only batch; never score validation."""
    result = {
        "status": "not_started", "pairs": min(128, len(train)), "updates_requested": updates,
        "updates_completed": 0, "l2": 0.0, "seed": 42, "subset_seed": 2026,
        "purpose": "train-only capacity diagnostic, not a generalization benchmark",
        "test_used": False, "validation_used": False, "history": [],
    }
    if time.monotonic() >= deadline:
        result["status"] = "skipped_budget"
        return result
    tf.keras.backend.clear_session()
    model = model_class(embedding_size=16, hidden_units=(32, 16), learning_rate=0.001,
                        l2=0.0, random_state=42, user_shrinkage=0.0,
                        item_shrinkage=20.0, residual_scale=0.25).initialize(books, train)
    pairs, targets, _ = model.encode_ratings(model.ratings)
    indices = np.random.default_rng(2026).choice(len(pairs), result["pairs"], replace=False)
    pairs, targets = pairs[indices], targets[indices]

    def raw_mse():
        return float(np.mean((model.model(pairs, training=False).numpy() - targets) ** 2))

    result["initial_mse"] = raw_mse()
    result["history"].append({"update": 0, "mse": result["initial_mse"]})
    for update in range(1, updates + 1):
        if time.monotonic() >= deadline:
            result["status"] = "budget_exhausted"
            break
        model._compiled_step(pairs, targets)
        result["updates_completed"] = update
        if update == 1 or update % 25 == 0 or update == updates:
            result["history"].append({"update": update, "mse": raw_mse()})
            atomic_json(output / "small_batch_overfit.json", result)
    else:
        result["status"] = "completed"
    result["final_mse"] = raw_mse()
    initial = result["initial_mse"]
    result["relative_mse_reduction"] = 1.0 - result["final_mse"] / initial if initial else None
    result["sample_encoded_pairs"] = pairs.tolist()
    result["sample_targets_normalized"] = targets[:, 0].tolist()
    result["training_only_model_saved"] = "small_batch_network.keras"
    save_network(model.model, output / "small_batch_network.keras")
    model = None
    gc.collect()
    return result


def plot_candidates(candidates, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for candidate in candidates:
        if candidate["status"] != "completed":
            continue
        history = candidate["history"]
        if history:
            axes[0].plot([row["epoch"] for row in history], [row["val_rmse"] for row in history],
                         marker=".", label=candidate["name"])
            diagnostics = [row for row in candidate["diagnostics"] if row["stage"] == "epoch_end"]
            axes[1].plot([row["epoch"] for row in diagnostics],
                         [row["corrections"]["validation"]["std_rating"] for row in diagnostics],
                         marker=".", label=candidate["name"])
        else:
            axes[0].axhline(candidate["validation"]["rmse_rating_scale"], linestyle="--",
                            label=candidate["name"])
    axes[0].set(xlabel="Epoch", ylabel="Validation RMSE (rating points)")
    axes[1].set(xlabel="Epoch", ylabel="Neural correction std (rating points)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=6)
    figure.suptitle("Validation-only regularization and head-initialization ablation")
    figure.tight_layout()
    figure.savefig(output / "ablation_validation.png", dpi=160)
    plt.close(figure)


def run_experiment(args):
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from book_recommender.content_based import book_work_key
    from book_recommender.residual_neural import ResidualNeuralRecommender

    output = args.output.resolve()
    started = time.monotonic()
    protocol = experiment_protocol(args.preset, args.epochs)
    candidates = protocol["candidates"]
    configuration = {
        "started_utc": utc_now(), **protocol, "batch_size": args.batch_size,
        "embedding_size": 16, "hidden_units": [32, 16], "learning_rate": 0.001,
        "user_shrinkage": 0.0, "item_shrinkage": 20.0, "neural_residual_scale": 0.25,
        "seed": 42, "diagnostic_sample_seed": 2026, "max_training_seconds": args.max_training_seconds,
        "require_gpu": args.require_gpu, "test_used": False,
        "selection_metric": "known_validation_rmse_rating_scale",
        "selection_tie_breaker": "earlier candidate in predeclared order",
        "initialization_ablation": "Only output kernel changes: N(0,0.001), numpy seed 2026; bias zero",
        "paired_training": "Every neural candidate has identical seed, row order and batch order",
        "scalar_control": "Mean normalized train target minus raw frozen baseline; no validation fit",
        "diagnostic_protocol": "Fixed train sample, gradient batch and known validation IDs across candidates",
        "prediction_convention": "Train raw normalized MSE plus L2; validation clipped to ratings 1–10",
        "checkpoint_protocol": "Network+Adam+RNG each epoch; immutable backup every "
                               f"{protocol['snapshot_interval_epochs']} epochs and completion",
        "test_policy": "Test CSV prohibited in input directory; no test metrics computed",
        "model_promotion_to_application": False,
    }
    atomic_json(output / "configuration.json", configuration)
    if (args.input_dir / "test_ratings.csv").exists():
        raise RuntimeError("Validation-only run must not receive a test_ratings.csv file.")
    fingerprints = fingerprint_files(args.input_dir, FROZEN_INPUT_SHA256)
    if any(fingerprints[name]["sha256"] != digest
           for name, digest in FROZEN_INPUT_SHA256.items()):
        raise RuntimeError("Inputs differ from the three frozen train/validation CSVs.")
    probe = gpu_probe(tf, args.require_gpu)
    atomic_json(output / "gpu_probe.json", probe)
    emit("GPU_PROBE_PASSED", gpu=probe["nvidia_smi"], device=probe["probe_device"])
    metadata = source_metadata()
    metadata["source_files"].update(fingerprint_files(ROOT, [
        "scripts/colab_neural_ablation.py", "scripts/colab_residual_experiment.py",
        "src/book_recommender/residual_neural.py",
    ]))
    metadata["experiment_label"] = f"validation_only_neural_{args.preset}"
    atomic_json(output / "source_metadata.json", metadata)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in FROZEN_INPUT_SHA256:
        shutil.copyfile(args.input_dir / name, inputs / name)
    if fingerprint_files(inputs, FROZEN_INPUT_SHA256) != fingerprints:
        raise RuntimeError("Copied frozen inputs changed.")
    atomic_json(output / "input_fingerprints.json", fingerprints)
    books = pd.read_csv(inputs / "train_books.csv", dtype={"isbn": str})
    train = pd.read_csv(inputs / "train_ratings.csv", dtype={"isbn": str})
    validation = pd.read_csv(inputs / "validation_ratings.csv", dtype={"isbn": str})
    work_map = {str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
                for row in books.itertuples(index=False)}
    assert_holdout_separation(train, validation, work_map)
    known = validation[validation.user_id.isin(train.user_id) & validation.isbn.isin(train.isbn)]
    if known.empty:
        raise RuntimeError("Frozen validation has no known pairs.")
    dataset = {
        "train_pairs": len(train), "train_users": int(train.user_id.nunique()),
        "train_isbns": int(train.isbn.nunique()), "validation_pairs": len(validation),
        "validation_known_pairs": len(known), "input_bytes_match_original_experiment": True,
        "canonical_work_leakage_checks_passed": True, "test_used": False,
        "split_policy": "Unchanged previous user/work holdout and train-only catalog",
    }
    atomic_json(output / "dataset_summary.json", dataset)
    static_entries = ["inputs", "configuration.json", "gpu_probe.json", "source_metadata.json",
                      "input_fingerprints.json", "dataset_summary.json"]
    manifest = write_archive(output, "inputs_backup.tar.gz", static_entries,
                             "inputs_manifest.json", kind="training_inputs")
    emit("INPUTS_BACKUP_READY", **manifest)
    candidate_results = []
    global_epoch = 0
    snapshot_number = 0
    deadline = time.monotonic() + args.max_training_seconds
    current_candidate = None
    diagnostics = []
    candidate_dir = None

    def persist_progress(status, **extra):
        atomic_json(output / "progress.json", {
            "status": status, "epoch": global_epoch, "candidate": current_candidate,
            "updated_utc": utc_now(), **extra,
        })
        atomic_json(output / "candidate_validation_results.json", candidate_results)

    def snapshot(reason):
        nonlocal snapshot_number
        snapshot_number += 1
        entries = ["current_network.keras", "current_best_network.keras", "current_checkpoint",
                   "current_checkpoint_state.json", "candidate_validation_results.json",
                   "progress.json", "current_diagnostics.json", "configuration.json", "worker.log"]
        if candidate_dir is not None:
            entries.append(str((candidate_dir / "validation_result.json").relative_to(output)))
        manifest = write_archive(
            output, f"epoch_{snapshot_number:03d}_snapshot.tar.gz", entries,
            f"epoch_{snapshot_number:03d}_manifest.json", kind="epoch_checkpoint",
            epoch=global_epoch, candidate=current_candidate["name"], reason=reason,
        )
        atomic_json(output / "snapshot_manifest.json", manifest)
        emit("EPOCH_BACKUP_READY", **manifest)

    def save_current(model, epoch, *, partial_epoch=False):
        save_network(model.model, output / "current_network.keras")
        checkpoint = output / "current_checkpoint"
        checkpoint.mkdir(exist_ok=True)
        tf.train.Checkpoint(model=model.model, optimizer=model.optimizer).write(
            str(checkpoint / "training")
        )
        atomic_json(output / "current_checkpoint_state.json", {
            "candidate": current_candidate, "candidate_epoch": epoch, "epoch": global_epoch,
            "partial_epoch": partial_epoch, "history": model.history,
            "best_epoch": getattr(model, "best_epoch", None),
            "optimizer_iterations": int(model.optimizer.iterations.numpy()),
            "training_rng_state": getattr(model, "training_rng_state", None),
            "elapsed_seconds": time.monotonic() - started, "updated_utc": utc_now(),
            "checkpoint_prefix": "current_checkpoint/training",
            "resume_scope": "Manual restore of network+Adam+RNG; partial epoch ordering not resumable",
            "vocabulary_source": "Reinitialize from immutable input backup before restoring network",
        })
        atomic_json(output / "current_diagnostics.json", diagnostics)

    for candidate in candidates:
        current_candidate = dict(candidate)
        record = {**candidate, "status": "not_started", "history": [], "diagnostics": [],
                  "epochs_completed": 0, "best_epoch": None, "test_used": False}
        if time.monotonic() >= deadline:
            record["status"] = "skipped_budget"
            candidate_results.append(record)
            persist_progress("budget_exhausted")
            continue
        candidate_started = time.monotonic()
        model = None
        diagnostics = []
        candidate_dir = output / "candidates" / candidate["name"]
        candidate_dir.mkdir(parents=True)
        emit("CANDIDATE_STARTED", **candidate)
        try:
            tf.keras.backend.clear_session()
            model = ResidualNeuralRecommender(
                embedding_size=16, hidden_units=(32, 16), learning_rate=0.001,
                l2=candidate["l2"], random_state=42, user_shrinkage=0.0, item_shrinkage=20.0,
                residual_scale=0.0 if candidate["kind"] == "bias_only" else 0.25,
            ).initialize(books, train)
            model.best_epoch = None
            model._check_validation_overlap(validation)
            pairs, targets, _ = model.encode_ratings(model.ratings)
            val_pairs, val_targets, _ = model.encode_ratings(known)
            expected_val_pairs = np.column_stack([
                known.user_id.map(model.user_to_index).to_numpy(dtype=np.int32),
                known.isbn.map(model.isbn_to_index).to_numpy(dtype=np.int32),
            ])
            np.testing.assert_array_equal(val_pairs, expected_val_pairs)
            indices = np.random.default_rng(2026).permutation(len(pairs))
            sample_pairs = pairs[indices[:min(2048, len(pairs))]]
            gradient_indices = indices[:min(args.batch_size, len(pairs))]
            probe_pairs, probe_targets = pairs[gradient_indices], targets[gradient_indices]

            def inspect(stage, epoch):
                row = observational_diagnostics(model, sample_pairs, val_pairs, probe_pairs,
                                                probe_targets, np, tf, stage=stage, epoch=epoch)
                diagnostics.append(row)
                return row

            if candidate["head_initialization"] != "zeros":
                kernel = np.random.default_rng(2026).normal(
                    0.0, 0.001, tuple(model.model.output_layer.kernel.shape)
                ).astype(np.float32)
                model.model.output_layer.kernel.assign(kernel)
            if candidate["kind"] != "neural":
                baseline = model.baseline_predictions(pairs, clip=False).reshape(-1)
                scalar = fit_scalar_intercept(targets, baseline, np)
                if candidate["kind"] == "bias_only":
                    scalar = 0.0
                # Store the analytic scalar in the output bias so the saved
                # network predicts exactly the control evaluated here. There is
                # no optimizer update and no learned interaction in this control.
                if candidate["kind"] == "bias_plus_scalar":
                    model.model.output_layer.bias.assign([scalar / model.residual_scale])
                prediction = model._predict_encoded(val_pairs).reshape(-1)
                record["scalar_intercept_normalized"] = scalar
                record["scalar_intercept_rating"] = 9.0 * scalar
                record["scalar_fit_source"] = "training rows only; raw normalized baseline residual mean"
                record["scalar_objective"] = "Analytic raw-MSE optimum, not clipped-validation calibration"
                atomic_json(candidate_dir / "control.json", {
                    "kind": candidate["kind"], "scalar_intercept_normalized": scalar,
                    "inference": "Saved model output includes the analytic scalar in its output bias",
                    "optimizer_updates": 0, "learned_interactions": False,
                })
                inspect("initial", 0)
                save_current(model, 0)
                save_network(model.model, output / "current_best_network.keras")
            else:
                inspect("initial", 0)
                rng = np.random.default_rng(42)
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
                            model.first_step_trace["head_initialization"] = candidate["head_initialization"]
                            if candidate["head_initialization"] != "zeros":
                                model.first_step_trace["embedding_first_step_note"] = (
                                    "Tiny random output kernel permits a data gradient from the first update."
                                )
                            inspect("after_first_update", 0)
                        else:
                            model._compiled_step(pairs[batch], targets[batch])
                    train_mse = float(model._mse(pairs, targets))
                    val_mse = float(model._mse(val_pairs, val_targets))
                    epoch_record = {"epoch": epoch, "train_mse": train_mse,
                                    "train_rmse": float(9 * np.sqrt(train_mse)),
                                    "val_mse": val_mse, "val_rmse": float(9 * np.sqrt(val_mse))}
                    if val_mse < best_mse:
                        best_mse, best_weights, wait = val_mse, model.model.get_weights(), 0
                        model.best_epoch = epoch
                        save_network(model.model, output / "current_best_network.keras")
                    else:
                        wait += 1
                    model.history.append(epoch_record)
                    global_epoch += 1
                    inspect("epoch_end", epoch)
                    save_current(model, epoch)
                    persist_progress("training", candidate_epoch=epoch, record=epoch_record)
                    emit("EPOCH_FINISHED", candidate=candidate["name"], **epoch_record)
                    if epoch % protocol["snapshot_interval_epochs"] == 0:
                        snapshot(f"periodic_{protocol['snapshot_interval_epochs']}_epochs")
                    if (protocol["early_stopping_enabled"]
                            and epoch >= protocol["min_neural_epochs_each"]
                            and wait >= protocol["patience"]):
                        break
                if best_weights is not None:
                    model.model.set_weights(best_weights)
                inspect("restored_best", model.best_epoch)
                diagnostics[-1]["weights_epoch"] = model.best_epoch
                diagnostics[-1]["optimizer_state_epoch"] = len(model.history)
                diagnostics[-1]["state_note"] = (
                    "Inference-only best weights; optimizer remains at last epoch. "
                    "Recovery checkpoint separately retains matched last-epoch network+Adam."
                )
                prediction = model._predict_encoded(val_pairs).reshape(-1)
            record.update({
                "status": "completed", "validation": regression_metrics(
                    known.book_rating.to_numpy(dtype=float), 1.0 + 9.0 * prediction, np),
                "history": model.history, "epochs_completed": len(model.history),
                "best_epoch": model.best_epoch, "diagnostics": diagnostics,
                "first_step_trace": model.first_step_trace, "baseline_summary": model.baseline_summary,
            })
            model.save(candidate_dir / "model")
            metadata_path = candidate_dir / "model" / "metadata.json"
            model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_metadata["experiment_candidate_kind"] = candidate["kind"]
            model_metadata["experiment_candidate_name"] = candidate["name"]
            model_metadata["experiment_optimizer_updates"] = int(model.optimizer.iterations.numpy())
            atomic_json(metadata_path, model_metadata)
            restored = ResidualNeuralRecommender.load(candidate_dir / "model")
            np.testing.assert_allclose(model._predict_encoded(val_pairs),
                                       restored._predict_encoded(val_pairs), rtol=1e-6, atol=1e-7)
            record["model_roundtrip_verified"] = True
            del restored
            validation_rows = known.copy()
            validation_rows["prediction"] = 1.0 + 9.0 * prediction
            validation_rows.to_csv(candidate_dir / "validation_predictions.csv", index=False)
            emit("CANDIDATE_VALIDATED", name=candidate["name"], **record["validation"])
        except Exception as error:  # noqa: BLE001 -- preserve every attempted candidate
            record.update({
                "status": "budget_exhausted" if isinstance(error, TimeoutError) else "failed",
                "exception_type": type(error).__name__,
                "message": SECRET_PATTERN.sub("<redacted>", str(error)),
                "diagnostics": diagnostics,
                "history": model.history if model is not None else [],
                "epochs_completed": len(model.history) if model is not None else 0,
                "best_epoch": getattr(model, "best_epoch", None),
            })
            emit("CANDIDATE_FAILED", name=candidate["name"], status=record["status"],
                 message=record["message"])
            print(SECRET_PATTERN.sub("<redacted>", traceback.format_exc()), file=sys.stderr, flush=True)
        record["seconds"] = time.monotonic() - candidate_started
        candidate_results.append(record)
        atomic_json(candidate_dir / "validation_result.json", record)
        atomic_json(output / "current_diagnostics.json", diagnostics)
        persist_progress("candidate_completed", candidate_status=record["status"])
        snapshot("candidate_completion")
        model = None
        gc.collect()

    current_candidate = {"name": "small_batch_overfit", "kind": "train_only_diagnostic"}
    persist_progress("small_batch_overfit")
    try:
        if args.preset == "warmup":
            overfit = {
                "status": "not_run_scope", "test_used": False, "validation_used": False,
                "reason": "Train-only capacity diagnostic already completed in the ablation run",
                "previous_results_reference": "docs/evidence/neural_colab_ablation_results.json",
            }
        else:
            overfit = small_batch_overfit(books, train, ResidualNeuralRecommender, np, tf,
                                         deadline, output, args.overfit_updates)
    except Exception as error:  # noqa: BLE001 -- preserve diagnostics even when capacity check fails
        overfit = {"status": "failed", "exception_type": type(error).__name__,
                   "message": SECRET_PATTERN.sub("<redacted>", str(error)),
                   "test_used": False, "validation_used": False}
    atomic_json(output / "small_batch_overfit.json", overfit)
    completed = [row for row in candidate_results if row["status"] == "completed"]
    neural = [row for row in completed if row["kind"] == "neural"]
    selected = min(completed, key=lambda row: row["validation"]["rmse_rating_scale"], default=None)
    best_neural = min(neural, key=lambda row: row["validation"]["rmse_rating_scale"], default=None)
    selection = {
        "selected_utc": utc_now(), "metric": configuration["selection_metric"],
        "best_candidate": selected["name"] if selected else None,
        "best_neural_candidate": best_neural["name"] if best_neural else None,
        "validation": selected["validation"] if selected else None,
        "best_neural_validation": best_neural["validation"] if best_neural else None,
        "completed_candidates": len(completed), "declared_candidates": len(candidates),
        "test_used": False, "model_promotion_to_application": False,
        "scope": "Diagnostic candidate ranking on the existing development validation only",
    }
    atomic_json(output / "selection.json", selection)
    results = {
        "schema_version": 1, "status": "completed", "completed_utc": utc_now(),
        "test_used": False, "configuration": configuration, "gpu": probe, "dataset": dataset,
        "candidate_validation_results": candidate_results, "selection": selection,
        "all_candidates_completed": len(completed) == len(candidates),
        "training": {"total_neural_epochs": global_epoch}, "small_batch_overfit": overfit,
        "limitations": [
            "One seed and only 140 known validation pairs on a previously used development split.",
            "No fresh holdout, test, ranking evaluation or claim of independent superiority.",
            "Correction variability demonstrates nonconstant output, not useful personalization by itself.",
            "Controls use fixed train-derived biases; scalar is fitted on train only.",
        ],
        "model_promotion_to_application": False, "total_seconds": time.monotonic() - started,
    }
    atomic_json(output / "results.json", results)
    plot_candidates(candidate_results, output)
    lines = ["Validation-only neural ablation", "No test data uploaded, read, or evaluated.",
             "Candidates were declared before training; selection uses validation RMSE only."]
    for row in candidate_results:
        value = row.get("validation", {}).get("rmse_rating_scale")
        lines.append(f"{row['name']}: {row['status']}; validation RMSE={value}")
    lines.extend([f"Train-only overfit diagnostic: {overfit['status']}", *results["limitations"]])
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    persist_progress("completed", selection=selection)
    sys.stdout.flush()
    manifest = write_archive(
        output, "final_backup.tar.gz", [
            *static_entries, "candidates", "current_network.keras", "current_best_network.keras",
            "current_checkpoint", "current_checkpoint_state.json", "current_diagnostics.json",
            "candidate_validation_results.json", "results.json", "selection.json", "progress.json",
            "small_batch_overfit.json", "small_batch_network.keras", "ablation_validation.png",
            "report.txt", "worker.log",
        ], "final_manifest.json", kind="completed_experiment", epochs=global_epoch,
    )
    (output / "completed.flag").write_text(json.dumps({
        "completed_utc": utc_now(), "sha256": manifest["sha256"],
    }) + "\n", encoding="utf-8")
    emit("EXPERIMENT_COMPLETED", **manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("ablation", "warmup"), default="ablation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-training-seconds", type=int, default=900)
    parser.add_argument("--overfit-updates", type=int, default=150)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        experiment_protocol(args.preset, args.epochs)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if not 1 <= args.max_training_seconds <= 900 or not 1 <= args.overfit_updates <= 150:
        parser.error("training budget must be 1–900 seconds and overfit updates 1–150")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in
           ("configuration.json", "completed.flag", "failed.json")):
        parser.error("output already contains an experiment; choose a fresh output path")
    try:
        run_experiment(args)
    except Exception as error:  # noqa: BLE001 -- persist terminal infrastructure failures
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
