"""Tune joint biases and biased MF, then reassess the reused development splits.

The two development assessment partitions have already been inspected. This run
is an adaptive comparator study, not an independent test. Epochs, bias settings,
MF regularization, and primary seeds are selected using selection partitions.
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
    from .colab_neural_multiseed import FROZEN_INPUT_SHA256, fit_statistics, user_cluster_bootstrap
    from .colab_residual_experiment import regression_metrics
except ImportError:
    from colab_neural_experiment import (
        ROOT, SECRET_PATTERN, atomic_json, emit, fingerprint_files, gpu_probe,
        save_network, source_metadata, utc_now, write_archive,
    )
    from colab_neural_multiseed import FROZEN_INPUT_SHA256, fit_statistics, user_cluster_bootstrap
    from colab_residual_experiment import regression_metrics

SPLIT_SEEDS = (17, 29)
TRAINING_SEEDS = (42, 2026, 7)
SHRINKAGES = (0.0, 5.0, 20.0, 100.0)
MF_LAMBDAS = (("1e4", 1e-4), ("1e3", 1e-3), ("1e2", 1e-2))
CANDIDATES = tuple(
    {"name": f"split{split}_mf_lam{label}_seed{seed}", "split_seed": split,
     "initialization_seed": seed, "regularization_lambda": penalty,
     "architecture": "biased_mf", "kind": "neural"}
    for split in SPLIT_SEEDS for label, penalty in MF_LAMBDAS for seed in TRAINING_SEEDS
)


def verify_inputs(input_dir):
    """Accept only the exact previous split/model artifacts named by the manifest."""
    manifest = json.loads((input_dir / "input_manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe input manifest path.")
        actual = fingerprint_files(input_dir, [relative])[relative]
        if actual != expected:
            raise RuntimeError(f"Input hash/size mismatch: {relative}")
    actual_files = {str(path.relative_to(input_dir)) for path in input_dir.rglob("*") if path.is_file()}
    if actual_files != set(files) | {"input_manifest.json"}:
        raise RuntimeError("Input directory contains undeclared files.")
    allowed_csvs = {"train_books.csv", *{
        f"splits/split_{seed}/{partition}_ratings.csv"
        for seed in SPLIT_SEEDS for partition in ("fit", "selection", "assessment")
    }}
    if {name for name in files if name.lower().endswith(".csv")} != allowed_csvs:
        raise RuntimeError("Only frozen book metadata and the reused internal split CSVs are accepted.")
    if files["train_books.csv"]["sha256"] != FROZEN_INPUT_SHA256["train_books.csv"]:
        raise RuntimeError("Book metadata differs from the frozen source catalog.")
    previous = json.loads((input_dir / "previous_results.json").read_text(encoding="utf-8"))
    if files["previous_results.json"]["sha256"] != manifest["previous_results_sha256"]:
        raise RuntimeError("Previous result fingerprint differs from the declared reference.")
    if (previous.get("experiment_kind") != "multiseed_internal_development"
            or not previous.get("all_candidates_completed") or not previous.get("assessment_performed")):
        raise RuntimeError("Expected the completed prior multiseed development experiment.")
    for split_seed in SPLIT_SEEDS:
        split = next(row for row in previous["dataset"]["splits"] if row["split_seed"] == split_seed)
        for partition in ("fit", "selection", "assessment"):
            name = f"{partition}_ratings.csv"
            if files[f"splits/split_{split_seed}/{name}"] != split["partition_file_fingerprints"][name]:
                raise RuntimeError("Reused split differs from the previous run.")
        for seed in TRAINING_SEEDS:
            name = f"split{split_seed}_mlp_seed{seed}"
            candidate = next(row for row in previous["candidate_validation_results"] if row["name"] == name)
            for filename, expected in candidate["selected_model_fingerprints"].items():
                if files[f"mlp_models/{name}/{filename}"] != expected:
                    raise RuntimeError("A reused MLP differs from its previous selected artifact.")
    return manifest, previous


def encode(frame, user_to_index, book_to_index, np):
    return np.column_stack([frame.user_id.map(user_to_index).to_numpy(np.int32),
                            frame.isbn.map(book_to_index).to_numpy(np.int32)])


def rating_report(frame, predictor, statistics, np):
    known = (frame.user_id.isin(statistics["user_mean"].index)
             & frame.isbn.isin(statistics["item_mean"].index)).to_numpy()
    prediction = predictor(frame.user_id.to_numpy(), frame.isbn.to_numpy())
    prediction = np.clip(np.asarray(prediction, dtype=float).reshape(-1), 1.0, 10.0)
    targets = frame.book_rating.to_numpy(float)
    return {"known_pairs": regression_metrics(targets[known], prediction[known], np),
            "all_pairs": regression_metrics(targets, prediction, np),
            "known_pair_retention_rate": float(known.mean()),
            "availability": {"both_known": int(known.sum()), "cold_pairs": int((~known).sum())}}, prediction, known


def make_predictor(user_to_index, book_to_index, statistics, raw_encoded, np, pd):
    """Raw rating-scale scorer shared by rating evaluation and unclipped ranking."""
    def predict(user_ids, isbns):
        users, books = pd.Series(user_ids), pd.Series(isbns, dtype=str)
        user_indices = users.map(user_to_index)
        book_indices = books.map(book_to_index)
        known = user_indices.notna().to_numpy() & book_indices.notna().to_numpy()
        user_mean = users.map(statistics["user_mean"])
        item_mean = books.map(statistics["item_mean"])
        values = user_mean.fillna(item_mean).fillna(statistics["global_mean"]).to_numpy(float)
        if known.any():
            pairs = np.column_stack([user_indices[known].to_numpy(np.int32), book_indices[known].to_numpy(np.int32)])
            values[known] = 1.0 + 9.0 * np.asarray(raw_encoded(pairs)).reshape(-1)
        return values
    return predict


def raw_network(model, pairs, np, batch_size=8192):
    return np.concatenate([model(pairs[start:start + batch_size], training=False).numpy().reshape(-1)
                           for start in range(0, len(pairs), batch_size)]) if len(pairs) else np.empty(0)


def compile_scorer(network, tf):
    @tf.function(reduce_retracing=True)
    def score(batch):
        return network(batch, training=False)
    return score


def mf_trace(model, optimizer, pairs, targets, penalty, np, tf):
    """One actual update with variable-identity-safe gradient evidence."""
    before = model.user_embedding.embeddings.numpy().copy()
    with tf.GradientTape() as tape:
        prediction = model(pairs, training=True)
        mse = tf.reduce_mean(tf.square(targets - prediction))
        regularization = penalty * model.observed_pair_penalty(pairs)
        loss = mse + regularization
    gradients = tape.gradient(loss, model.trainable_variables)
    norms = {variable.path: float(tf.linalg.norm(tf.convert_to_tensor(gradient)).numpy())
             for variable, gradient in zip(model.trainable_variables, gradients)}
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {"mse_before": float(mse.numpy()), "regularization_loss": float(regularization.numpy()),
            "total_loss_before": float(loss.numpy()), "gradient_norms": norms,
            "embedding_delta_norm": float(np.linalg.norm(model.user_embedding.embeddings.numpy() - before)),
            "optimizer_iterations": int(optimizer.iterations.numpy()),
            "predictions_before": prediction.numpy()[:8, 0].tolist(),
            "predictions_after": model(pairs[:8], training=False).numpy()[:, 0].tolist(),
            "regularization_definition": "lambda * batch mean(sum(user factors²)+sum(item factors²)+user bias²+item bias²)"}


def run_experiment(args):
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from book_recommender.factorization import (
        create_factorization_model, fit_joint_bias, load_factorization_model,
    )
    from book_recommender.neural import NeuralRecommender

    try:
        from .ranking_comparison import run_ranking_comparison
    except ImportError:
        from ranking_comparison import run_ranking_comparison

    output = args.output.resolve()
    started = time.monotonic()
    manifest, previous = verify_inputs(args.input_dir)
    configuration = {
        "started_utc": utc_now(), "candidates": list(CANDIDATES), "split_seeds": list(SPLIT_SEEDS),
        "training_seeds": list(TRAINING_SEEDS), "bias_user_shrinkages": list(SHRINKAGES),
        "bias_item_shrinkages": list(SHRINKAGES), "bias_max_iterations": 1000, "bias_tolerance": 1e-10,
        "bias_objective": "Sum squared normalized fit residuals + user_alpha*sum(user_bias²) + item_alpha*sum(item_bias²); fixed fit global mean",
        "mf_lambdas": [value for _, value in MF_LAMBDAS], "embedding_size": 16,
        "embedding_initialization": "Normal stddev .05; user seed=training seed, item seed=training seed+1",
        "mf_bias_initialization": "Selected joint additive ridge biases, subsequently trainable",
        "mf_regularization": "lambda * observed-pair batch mean(||user factors||²+||item factors||²+user bias²+item bias²)",
        "mf_regularization_scope": "Frequency-weighted sampled-pair penalty; no global Keras L2 or double counting",
        "mf_learning_rate": 0.005, "max_neural_epochs_each": args.epochs, "patience": 5,
        "batch_size": args.batch_size, "max_training_seconds": args.max_training_seconds,
        "max_ranking_seconds": args.max_ranking_seconds, "snapshot_interval_epochs": 10,
        "selection_metric": "Known selection RMSE; MF lambda chosen by mean across all three training seeds",
        "training_seed_scope": "Weight initialization and minibatch order, paired across MF lambda settings",
        "assessment_gate": "All 32 bias fits and 18 MF models complete and hashes locked before reassessment",
        "ranking_selection": "No ranking-based tuning; rank the three MLPs, three MF seeds of selected lambda, selected bias and popularity",
        "rating_reporting": "Clip rating predictions to [1,10]; training and ranking use raw scores",
        "scope": "Adaptive reassessment of previously inspected internal development partitions",
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
        "src/book_recommender/factorization.py", "scripts/colab_comparator_experiment.py",
        "scripts/colab_neural_multiseed.py", "scripts/colab_residual_experiment.py", "scripts/ranking_comparison.py",
    ]))
    metadata["experiment_label"] = "tuned_bias_mf_ranking_reused_development"
    atomic_json(output / "source_metadata.json", metadata)
    shutil.copytree(args.input_dir, output / "inputs")
    inputs = output / "inputs"
    verify_inputs(inputs)
    atomic_json(output / "input_fingerprints.json", manifest)
    books = pd.read_csv(inputs / "train_books.csv", dtype={"isbn": str})
    dataset = {"splits": previous["dataset"]["splits"], "reused_prior_assessment": True,
               "previous_results_sha256": manifest["previous_results_sha256"]}
    atomic_json(output / "dataset_summary.json", dataset)
    static_entries = ["inputs", "configuration.json", "gpu_probe.json", "source_metadata.json",
                      "input_fingerprints.json", "dataset_summary.json"]
    archive = write_archive(output, "inputs_backup.tar.gz", static_entries,
                            "inputs_manifest.json", kind="training_inputs")
    emit("INPUTS_BACKUP_READY", **archive)
    deadline = time.monotonic() + args.max_training_seconds
    bias_results, candidate_results, chosen_bias, split_context = [], [], {}, {}
    global_epoch, snapshot_number = 0, 0
    current_candidate = {"name": "bias_grid"}

    def progress(status, **extra):
        atomic_json(output / "progress.json", {"status": status, "epoch": global_epoch,
                    "candidate": current_candidate, "updated_utc": utc_now(), **extra})
        atomic_json(output / "bias_grid_results.json", bias_results)
        atomic_json(output / "candidate_validation_results.json", candidate_results)

    def snapshot(reason, completed_model=None):
        nonlocal snapshot_number
        snapshot_number += 1
        entries = ["current_best_network.keras", "current_checkpoint", "current_checkpoint_state.json",
                   "candidate_validation_results.json", "bias_grid_results.json", "progress.json",
                   "configuration.json", "weights_locked.json", "lambda_selection.json", "worker.log"]
        if completed_model is not None:
            entries.append(str(completed_model.relative_to(output)))
        archive = write_archive(output, f"epoch_{snapshot_number:03d}_snapshot.tar.gz", entries,
                                 f"epoch_{snapshot_number:03d}_manifest.json", kind="epoch_checkpoint",
                                 epoch=global_epoch, candidate=current_candidate["name"], reason=reason)
        atomic_json(output / "snapshot_manifest.json", archive)
        emit("EPOCH_BACKUP_READY", **archive)

    for split_seed in SPLIT_SEEDS:
        split_dir = inputs / "splits" / f"split_{split_seed}"
        fit = pd.read_csv(split_dir / "fit_ratings.csv", dtype={"isbn": str})
        selection_frame = pd.read_csv(split_dir / "selection_ratings.csv", dtype={"isbn": str})
        users, isbns = sorted(fit.user_id.unique()), sorted(fit.isbn.unique())
        user_to_index = {int(value): index for index, value in enumerate(users)}
        book_to_index = {str(value): index for index, value in enumerate(isbns)}
        pairs = encode(fit, user_to_index, book_to_index, np)
        targets64 = (fit.book_rating.to_numpy(float) - 1.0) / 9.0
        statistics = fit_statistics(fit)
        vocabulary = {"user_ids": [int(value) for value in users], "isbns": [str(value) for value in isbns],
                      "fit_input_fingerprint": manifest["files"][f"splits/split_{split_seed}/fit_ratings.csv"]}
        atomic_json(output / "vocabularies" / f"split_{split_seed}.json", vocabulary)
        split_context[split_seed] = {"fit": fit, "selection": selection_frame, "pairs": pairs,
                                    "targets": targets64, "statistics": statistics,
                                    "user_to_index": user_to_index, "book_to_index": book_to_index,
                                    "vocabulary": vocabulary}
        for user_alpha in SHRINKAGES:
            for item_alpha in SHRINKAGES:
                name = f"split{split_seed}_bias_u{int(user_alpha)}_i{int(item_alpha)}"
                current_candidate = {"name": name, "kind": "bias", "split_seed": split_seed}
                record = {**current_candidate, "user_shrinkage": user_alpha, "item_shrinkage": item_alpha,
                          "selected": False, "status": "not_started"}
                if time.monotonic() >= deadline:
                    record["status"] = "skipped_budget"
                    bias_results.append(record)
                    continue
                bias = fit_joint_bias(pairs, targets64, len(users), len(isbns),
                                      user_shrinkage=user_alpha, item_shrinkage=item_alpha)
                scorer = make_predictor(user_to_index, book_to_index, statistics,
                    lambda batch, bias=bias: bias["global_mean"] + bias["user_bias"][batch[:, 0]] + bias["book_bias"][batch[:, 1]], np, pd)
                metrics, _, _ = rating_report(selection_frame, scorer, statistics, np)
                record.update({key: value for key, value in bias.items() if key not in ("user_bias", "book_bias")})
                record.update(status="completed", selection=metrics)
                directory = output / "bias_models" / name
                directory.mkdir(parents=True)
                np.savez_compressed(directory / "biases.npz", global_mean=bias["global_mean"],
                                    user_bias=bias["user_bias"], book_bias=bias["book_bias"])
                atomic_json(directory / "metadata.json", {**record, "vocabulary": f"vocabularies/split_{split_seed}.json"})
                record["artifact_fingerprints"] = fingerprint_files(directory, ["biases.npz", "metadata.json"])
                bias_results.append(record)
                progress("tuning_bias", record={"name": name, "selection_rmse": metrics["known_pairs"]["rmse_rating_scale"], "converged": bias["converged"]})
        successful = [row for row in bias_results if row["split_seed"] == split_seed and row["status"] == "completed"]
        if successful:
            winner = min(successful, key=lambda row: row["selection"]["known_pairs"]["rmse_rating_scale"])
            winner["selected"] = True
            directory = output / "bias_models" / winner["name"]
            stored_metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            stored_metadata["selected"] = True
            atomic_json(directory / "metadata.json", stored_metadata)
            winner["artifact_fingerprints"] = fingerprint_files(directory, ["biases.npz", "metadata.json"])
            chosen_bias[split_seed] = winner
    progress("bias_grid_completed")
    snapshot("all_bias_grid_results", output / "bias_models")

    for candidate in CANDIDATES:
        current_candidate = dict(candidate)
        record = {**candidate, "status": "not_started", "history": [], "epochs_completed": 0,
                  "best_epoch": None, "assessment_scored": False}
        if time.monotonic() >= deadline or candidate["split_seed"] not in chosen_bias:
            record["status"] = "skipped_budget"
            candidate_results.append(record)
            progress("budget_exhausted")
            continue
        context = split_context[candidate["split_seed"]]
        pairs, targets = context["pairs"], context["targets"].astype(np.float32).reshape(-1, 1)
        val_frame = context["selection"]
        known_val = val_frame[val_frame.user_id.isin(context["user_to_index"]) & val_frame.isbn.isin(context["book_to_index"])]
        val_pairs = encode(known_val, context["user_to_index"], context["book_to_index"], np)
        val_targets = (known_val.book_rating.to_numpy(float) - 1.0) / 9.0
        directory = output / "mf_models" / candidate["name"]
        directory.mkdir(parents=True)
        began = time.monotonic()
        model, optimizer = None, None
        history, best_epoch, rng_state = [], None, None
        try:
            tf.keras.backend.clear_session()
            bias_record = chosen_bias[candidate["split_seed"]]
            with np.load(output / "bias_models" / bias_record["name"] / "biases.npz") as stored:
                model = create_factorization_model(len(context["user_to_index"]), len(context["book_to_index"]),
                    random_state=candidate["initialization_seed"], global_mean=float(stored["global_mean"]),
                    user_bias=stored["user_bias"], book_bias=stored["book_bias"])
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
            penalty = candidate["regularization_lambda"]

            @tf.function(reduce_retracing=True)
            def step(batch, values):
                with tf.GradientTape() as tape:
                    raw = model(batch, training=True)
                    mse = tf.reduce_mean(tf.square(values - raw))
                    reg = penalty * model.observed_pair_penalty(batch)
                    loss = mse + reg
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return reg

            def checkpoint(epoch, partial=False):
                checkpoint_dir = output / "current_checkpoint"
                checkpoint_dir.mkdir(exist_ok=True)
                tf.train.Checkpoint(model=model, optimizer=optimizer).write(str(checkpoint_dir / "training"))
                atomic_json(output / "current_checkpoint_state.json", {
                    "candidate": candidate, "candidate_epoch": epoch, "epoch": global_epoch,
                    "history": history, "best_epoch": best_epoch, "partial_epoch": partial,
                    "optimizer_iterations": int(optimizer.iterations.numpy()), "training_rng_state": rng_state,
                    "checkpoint_prefix": "current_checkpoint/training", "updated_utc": utc_now(),
                    "resume_scope": "Matched network+Adam+RNG; partial epoch order is not resumable",
                })

            save_network(model, directory / "initial_network.keras")
            record["initial_model_fingerprint"] = fingerprint_files(directory, ["initial_network.keras"])
            record["bias_initialization_source"] = bias_record["name"]
            record["bias_initialization_fingerprints"] = bias_record["artifact_fingerprints"]
            rng = np.random.default_rng(candidate["initialization_seed"])
            best_mse, best_weights, wait = float("inf"), None, 0
            first_trace = None
            for epoch in range(1, args.epochs + 1):
                order = rng.permutation(len(pairs))
                rng_state = rng.bit_generator.state
                regularization_sum, observed_count = 0.0, 0
                for start in range(0, len(order), args.batch_size):
                    if time.monotonic() >= deadline:
                        checkpoint(epoch, partial=True)
                        raise TimeoutError("Shared training budget exhausted at minibatch boundary")
                    batch = order[start:start + args.batch_size]
                    if first_trace is None:
                        first_trace = mf_trace(model, optimizer, pairs[batch], targets[batch], penalty, np, tf)
                        reg = first_trace["regularization_loss"]
                    else:
                        reg = float(step(pairs[batch], targets[batch]).numpy())
                    regularization_sum += reg * len(batch)
                    observed_count += len(batch)
                train_mse = float(np.mean((raw_network(model, pairs, np) - targets[:, 0]) ** 2))
                val_mse = float(np.mean((np.clip(raw_network(model, val_pairs, np), 0, 1) - val_targets) ** 2))
                if not np.isfinite([train_mse, val_mse]).all():
                    raise RuntimeError("Non-finite MF training/selection loss")
                epoch_record = {"epoch": epoch, "train_mse": train_mse, "train_rmse": float(9 * np.sqrt(train_mse)),
                                "val_mse": val_mse, "val_rmse": float(9 * np.sqrt(val_mse)),
                                "reg_loss": regularization_sum / observed_count}
                if val_mse < best_mse:
                    best_mse, best_weights, wait, best_epoch = val_mse, model.get_weights(), 0, epoch
                    save_network(model, output / "current_best_network.keras")
                else:
                    wait += 1
                history.append(epoch_record)
                global_epoch += 1
                checkpoint(epoch)
                progress("training_mf", candidate_epoch=epoch, record=epoch_record)
                emit("EPOCH_FINISHED", candidate=candidate["name"], **epoch_record)
                if epoch % 10 == 0:
                    snapshot("periodic_ten_epochs")
                if wait >= 5:
                    break
            model.set_weights(best_weights)
            scorer = make_predictor(context["user_to_index"], context["book_to_index"], context["statistics"],
                                     lambda values: raw_network(model, values, np), np, pd)
            metrics, prediction, _ = rating_report(val_frame, scorer, context["statistics"], np)
            record.update(status="completed", history=history, epochs_completed=len(history), best_epoch=best_epoch,
                          best_at_epoch_cap=(best_epoch == args.epochs), stopped_at_epoch_cap=(len(history) == args.epochs),
                          validation=metrics["known_pairs"], selection=metrics, first_step_trace=first_trace)
            save_network(model, directory / "network.keras")
            atomic_json(directory / "metadata.json", {
                "format_version": 1, "architecture": "biased_mf", "candidate": candidate,
                "vocabulary": context["vocabulary"], "history": history, "best_epoch": best_epoch,
                "global_mean_fixed": True, "user_item_biases_trainable": True,
                "bias_initialization_source": bias_record["name"], "regularization": configuration["mf_regularization"],
            })
            restored = load_factorization_model(directory / "network.keras")
            np.testing.assert_allclose(raw_network(model, val_pairs, np), raw_network(restored, val_pairs, np), rtol=1e-6, atol=1e-7)
            record["model_roundtrip_verified"] = True
            record["selected_model_fingerprints"] = fingerprint_files(directory, ["network.keras", "metadata.json"])
            del restored
            rows = val_frame.copy()
            rows["prediction"] = prediction
            rows.to_csv(directory / "selection_predictions.csv", index=False)
        except Exception as error:  # noqa: BLE001 -- retain bounded failed attempts
            record.update(status="budget_exhausted" if isinstance(error, TimeoutError) else "failed",
                          message=SECRET_PATTERN.sub("<redacted>", str(error)), exception_type=type(error).__name__,
                          history=history, epochs_completed=len(history), best_epoch=best_epoch)
            print(SECRET_PATTERN.sub("<redacted>", traceback.format_exc()), file=sys.stderr, flush=True)
        record["seconds"] = time.monotonic() - began
        candidate_results.append(record)
        atomic_json(directory / "candidate_result.json", record)
        progress("candidate_completed", candidate_status=record["status"])
        snapshot("candidate_completion", directory)
        model, optimizer = None, None
        gc.collect()

    complete = [row for row in candidate_results if row["status"] == "completed"]
    complete_bias = [row for row in bias_results if row["status"] == "completed"]
    assessment_performed = len(complete) == 18 and len(complete_bias) == 32
    lambda_selection, reused_mlp_results, assessment_results, ranking_results = [], [], [], []
    for split_seed in SPLIT_SEEDS:
        grid = []
        for _, penalty in MF_LAMBDAS:
            rows = [row for row in complete if row["split_seed"] == split_seed and row["regularization_lambda"] == penalty]
            if len(rows) == 3:
                scores = [row["validation"]["rmse_rating_scale"] for row in rows]
                grid.append({"regularization_lambda": penalty, "selection_rmse_mean": float(np.mean(scores)),
                             "selection_rmse_std": float(np.std(scores, ddof=1)), "n_seeds": len(rows)})
        previous_mlps = [row for row in previous["candidate_validation_results"]
                         if row["split_seed"] == split_seed and row["architecture"] == "mlp"]
        primary_mlp = min(previous_mlps, key=lambda row: row["validation"]["rmse_rating_scale"])["name"]
        if len(grid) == 3:
            penalty = min(grid, key=lambda row: row["selection_rmse_mean"])["regularization_lambda"]
            rows = [row for row in complete if row["split_seed"] == split_seed and row["regularization_lambda"] == penalty]
            primary = min(rows, key=lambda row: row["validation"]["rmse_rating_scale"])
            lambda_selection.append({"split_seed": split_seed, "grid": grid, "selected_lambda": penalty,
                                     "selected_mf_candidates": [row["name"] for row in rows],
                                     "primary_mf_candidate": primary["name"], "primary_mlp_candidate": primary_mlp,
                                     "bias_candidate": chosen_bias[split_seed]["name"], "selection_only": True})
    atomic_json(output / "lambda_selection.json", lambda_selection)
    if assessment_performed:
        ranking_seconds_used = 0.0
        locked_files = {}
        for directory in (output / "mf_models", output / "bias_models", output / "vocabularies"):
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in (".keras", ".npz", ".json") and path.name != "candidate_result.json":
                    relative = str(path.relative_to(output))
                    locked_files[relative] = fingerprint_files(output, [relative])[relative]
        for name, expected in manifest["files"].items():
            if name.startswith("mlp_models/"):
                locked_files["inputs/" + name] = expected
        for path in sorted((output / "current_checkpoint").glob("*")):
            if path.is_file():
                relative = str(path.relative_to(output))
                locked_files[relative] = fingerprint_files(output, [relative])[relative]
        atomic_json(output / "weights_locked.json", {"locked_utc": utc_now(), "files": locked_files,
                    "lambda_selection": lambda_selection, "assessment_scored": False,
                    "ranking_used_for_selection": False})
        progress("all_weights_locked")
        snapshot("all_weights_locked_before_reassessment")
        for split_seed in SPLIT_SEEDS:
            context = split_context[split_seed]
            assessment = pd.read_csv(inputs / "splits" / f"split_{split_seed}" / "assessment_ratings.csv", dtype={"isbn": str})
            statistics = context["statistics"]
            selected = next(row for row in lambda_selection if row["split_seed"] == split_seed)
            predictors, model_metrics, predictions, loaded_mlps = {}, {}, {}, []
            bias_record = chosen_bias[split_seed]
            bias_path = output / "bias_models" / bias_record["name"] / "biases.npz"
            with np.load(bias_path) as stored:
                bias_global, bias_user, bias_book = float(stored["global_mean"]), stored["user_bias"].copy(), stored["book_bias"].copy()
            bias_predictor = make_predictor(context["user_to_index"], context["book_to_index"], statistics,
                lambda batch, mu=bias_global, ub=bias_user, ib=bias_book: mu + ub[batch[:, 0]] + ib[batch[:, 1]], np, pd)
            predictors[bias_record["name"]] = bias_predictor
            metrics, prediction, known = rating_report(assessment, bias_predictor, statistics, np)
            bias_record["assessment"] = metrics
            model_metrics[bias_record["name"]], predictions[bias_record["name"]] = metrics, prediction
            for old in (row for row in previous["candidate_validation_results"]
                        if row["split_seed"] == split_seed and row["architecture"] == "mlp"):
                neural = NeuralRecommender.load(inputs / "mlp_models" / old["name"])
                loaded_mlps.append(neural)
                compiled = compile_scorer(neural.model, tf)
                scorer = make_predictor(neural.user_to_index, neural.isbn_to_index, statistics,
                    lambda batch, compiled=compiled: compiled(tf.convert_to_tensor(batch, dtype=tf.int32)).numpy().reshape(-1), np, pd)
                predictors[old["name"]] = scorer
                selection_metrics, _, _ = rating_report(context["selection"], scorer, statistics, np)
                np.testing.assert_allclose(selection_metrics["known_pairs"]["rmse_rating_scale"],
                                           old["validation"]["rmse_rating_scale"], rtol=1e-6, atol=1e-7)
                metrics, prediction, _ = rating_report(assessment, scorer, statistics, np)
                model_metrics[old["name"]], predictions[old["name"]] = metrics, prediction
                reused_mlp_results.append({"name": old["name"], "split_seed": split_seed,
                    "initialization_seed": old["initialization_seed"], "selection": selection_metrics,
                    "assessment": metrics, "source_model_fingerprints": old["selected_model_fingerprints"],
                    "trained_in_this_run": False})
            for row in (candidate for candidate in complete if candidate["split_seed"] == split_seed):
                directory = output / "mf_models" / row["name"]
                if fingerprint_files(directory, ["network.keras", "metadata.json"]) != row["selected_model_fingerprints"]:
                    raise RuntimeError("MF artifact changed after its selection was locked")
                model = load_factorization_model(directory / "network.keras")
                mu = float(model.global_mean.numpy())
                ub, ib = model.user_bias.numpy(), model.book_bias.numpy()
                uf, bf = model.user_embedding.embeddings.numpy(), model.book_embedding.embeddings.numpy()
                check_frame = context["selection"]
                check_frame = check_frame[check_frame.user_id.isin(context["user_to_index"])
                                          & check_frame.isbn.isin(context["book_to_index"])]
                check_pairs = encode(check_frame, context["user_to_index"], context["book_to_index"], np)
                numpy_raw = mu + ub[check_pairs[:, 0]] + ib[check_pairs[:, 1]] + np.sum(uf[check_pairs[:, 0]] * bf[check_pairs[:, 1]], axis=1)
                np.testing.assert_allclose(numpy_raw, raw_network(model, check_pairs, np), rtol=1e-6, atol=1e-7)
                row["numpy_tensorflow_prediction_parity_verified"] = True
                scorer = make_predictor(context["user_to_index"], context["book_to_index"], statistics,
                    lambda batch, mu=mu, ub=ub, ib=ib, uf=uf, bf=bf: mu + ub[batch[:, 0]] + ib[batch[:, 1]]
                    + np.sum(uf[batch[:, 0]] * bf[batch[:, 1]], axis=1), np, pd)
                metrics, prediction, candidate_known = rating_report(assessment, scorer, statistics, np)
                np.testing.assert_array_equal(candidate_known, known)
                row["assessment"], row["assessment_scored"] = metrics, True
                model_metrics[row["name"]], predictions[row["name"]] = metrics, prediction
                if row["name"] in selected["selected_mf_candidates"]:
                    predictors[row["name"]] = scorer
                atomic_json(directory / "candidate_result.json", row)
                model = None
            prediction_rows = assessment.copy()
            prediction_rows["both_ids_known"] = known
            for name, prediction in predictions.items():
                prediction_rows[name] = prediction
            assessment_dir = output / "assessment" / f"split_{split_seed}"
            assessment_dir.mkdir(parents=True)
            prediction_rows.to_csv(assessment_dir / "rating_predictions.csv", index=False)
            comparisons = []
            for name in predictions:
                if name == bias_record["name"]:
                    continue
                for reference in (bias_record["name"], selected["primary_mlp_candidate"]):
                    if name == reference:
                        continue
                    for scope, mask in (("known_pairs", known), ("all_pairs", np.ones(len(known), dtype=bool))):
                        comparison = user_cluster_bootstrap(assessment.user_id.to_numpy()[mask],
                            assessment.book_rating.to_numpy(float)[mask], predictions[name][mask], predictions[reference][mask], np)
                        comparisons.append({"candidate": name, "reference": reference, "scope": scope, "bootstrap": comparison})
            assessment_results.append({"split_seed": split_seed, "models": model_metrics,
                                       "paired_comparisons": comparisons})
            atomic_json(output / "assessment_results.json", assessment_results)
            progress("ranking", split_seed=split_seed)
            ranking_started = time.monotonic()
            ranking_deadline = ranking_started + max(0.0, args.max_ranking_seconds - ranking_seconds_used)

            def budgeted(predictor):
                def checked(users, isbns):
                    if time.monotonic() >= ranking_deadline:
                        raise TimeoutError("Shared ranking time budget exhausted at scoring-block boundary")
                    return predictor(users, isbns)
                return checked

            try:
                ranking = run_ranking_comparison(split_seed=split_seed, books=books, fit=context["fit"],
                    assessment=assessment, predictors={name: budgeted(predictor) for name, predictor in predictors.items()},
                    primary_mlp=selected["primary_mlp_candidate"], primary_mf=selected["primary_mf_candidate"],
                    bias_name=bias_record["name"], output_dir=output / "ranking" / f"split_{split_seed}", np=np, pd=pd)
                ranking["status"] = "completed"
            except TimeoutError as error:
                ranking = {"split_seed": split_seed, "status": "budget_exhausted", "message": str(error)}
            ranking["seconds"] = time.monotonic() - ranking_started
            ranking_seconds_used += ranking["seconds"]
            ranking["budget_seconds"] = args.max_ranking_seconds
            ranking["shared_budget_seconds_used"] = ranking_seconds_used
            ranking["budget_exceeded"] = ranking_seconds_used > args.max_ranking_seconds
            ranking_results.append(ranking)
            atomic_json(output / "ranking_results.json", ranking_results)
            del predictors, loaded_mlps
            gc.collect()
        for relative, expected in locked_files.items():
            if fingerprint_files(output, [relative])[relative] != expected:
                raise RuntimeError(f"Locked artifact changed during reassessment: {relative}")
    results = {
        "schema_version": 1, "experiment_kind": "tuned_bias_mf_ranking_reused_development",
        "status": "completed", "completed_utc": utc_now(), "configuration": configuration,
        "dataset": dataset, "gpu": probe, "bias_grid_results": bias_results,
        "candidate_validation_results": candidate_results, "lambda_selection": lambda_selection,
        "reused_mlp_results": reused_mlp_results, "assessment_results": assessment_results,
        "ranking_results": ranking_results, "assessment_performed": assessment_performed,
        "all_candidates_completed": assessment_performed, "training": {"total_neural_epochs": global_epoch},
        "original_test_used": False, "original_validation_used": False,
        "model_promotion_to_application": False, "total_seconds": time.monotonic() - started,
        "limitations": [
            "These internal assessment partitions were already inspected in the previous multiseed run; this is adaptive reused-data development.",
            "Bias settings, MF lambda and primary seeds were chosen using selection ratings only; ranking was not tuned.",
            "All users retain fit history and the original fixed catalog was selected before repartition; cold-user generalization is not evaluated.",
            "Bootstrap intervals are conditional user-cluster percentiles without multiplicity correction; overlapping splits are never pooled.",
            "MF sampled-pair regularization is frequency weighted and is not numerically comparable with previous global embedding L2.",
            "MF also updates the initialized bias terms; a null MF result does not by itself disprove useful latent factors.",
            "A best checkpoint at the epoch cap does not establish convergence; bias convergence flags are reported separately.",
        ],
    }
    atomic_json(output / "results.json", results)
    progress("completed", assessment_performed=assessment_performed)
    sys.stdout.flush()
    archive = write_archive(output, "final_backup.tar.gz", [
        *static_entries, "vocabularies", "bias_models", "mf_models", "assessment", "ranking",
        "current_best_network.keras", "current_checkpoint", "current_checkpoint_state.json",
        "bias_grid_results.json", "candidate_validation_results.json", "lambda_selection.json",
        "weights_locked.json", "assessment_results.json", "ranking_results.json", "results.json",
        "progress.json", "worker.log",
    ], "final_manifest.json", kind="completed_experiment", epochs=global_epoch)
    (output / "completed.flag").write_text(json.dumps({"completed_utc": utc_now(), "sha256": archive["sha256"]}) + "\n")
    emit("EXPERIMENT_COMPLETED", **archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-training-seconds", type=int, default=900)
    parser.add_argument("--max-ranking-seconds", type=int, default=300)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 30 or args.batch_size < 1 or not 1 <= args.max_training_seconds <= 900:
        parser.error("Epochs must be 1–30, batch size positive and training budget 1–900 seconds")
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
