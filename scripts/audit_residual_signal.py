#!/usr/bin/env python3
"""Audit saved residual checkpoints with NumPy, without TensorFlow or training.

Expects a completed ``colab_residual_experiment.py`` run's extracted ``final``
directory, including inputs and candidate models. Requires NumPy and h5py; h5py
is already included with the optional TensorFlow environment. This is a manual
forward pass for the documented float32, two-hidden-layer ResidualRatingModel,
not a general Keras loader. No test data is read or newly evaluated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "bias_only_u0_i20"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    try:
        label = str(path.resolve().relative_to(ROOT))
    except ValueError:
        label = str(path.resolve())
    return {
        "path": label, "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_ratings(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [(int(row["user_id"]), str(row["isbn"]), float(row["book_rating"])) for row in rows]


def load_weights(path, metadata, user_count, book_count):
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("This read-only audit needs h5py; use the optional neural environment.") from error

    with zipfile.ZipFile(path) as archive:
        architecture = json.loads(archive.read("config.json"))
        require(architecture["class_name"] == "ResidualRatingModel", "Unsupported Keras model class.")
        require(architecture["registered_name"] == "book_recommender>ResidualRatingModel",
                "Unsupported registered model.")
        config = architecture["config"]
        require(config["dtype"]["config"]["name"] == "float32", "Only float32 weights supported.")
        require(config["num_users"] == user_count and config["num_books"] == book_count,
                "Saved vocabulary sizes do not match frozen training inputs.")
        require(len(config["hidden_units"]) == 2, "Only the documented two-hidden-layer model is supported.")
        require(metadata["architecture"] == "residual"
                and metadata["model_kind"] == "bias_residual_neural", "Unsupported metadata architecture.")
        require(metadata["baseline_summary"]["hidden_activation"] == "leaky_relu (negative_slope=0.2)",
                "Only LeakyReLU with negative_slope=0.2 is supported.")
        for key in ("embedding_size", "hidden_units", "l2", "residual_scale"):
            require(config[key] == metadata["config"][key], f"Configuration mismatch: {key}")
        weights = {}
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as saved:
            def visit(name, value):
                if isinstance(value, h5py.Dataset):
                    weights[name] = value[()]
            saved.visititems(visit)
    embedding = config["embedding_size"]
    first, second = config["hidden_units"]
    shapes = {
        "layers/embedding/vars/0": (user_count, embedding),
        "book_embedding/vars/0": (book_count, embedding),
        "hidden/dense/vars/0": (3 * embedding, first),
        "hidden/dense/vars/1": (first,),
        "hidden/dense_1/vars/0": (first, second),
        "hidden/dense_1/vars/1": (second,),
        "layers/dense_2/vars/0": (second, 1),
        "layers/dense_2/vars/1": (1,),
        "vars/0": (), "vars/1": (user_count,), "vars/2": (book_count,),
    }
    require(set(weights) == set(shapes), "Unexpected checkpoint variables; update the manual architecture audit.")
    for key, shape in shapes.items():
        require(weights[key].shape == shape, f"Unsupported shape for {key}")
        require(weights[key].dtype == np.float32 and np.isfinite(weights[key]).all(),
                f"Expected finite float32 weights for {key}")
    return weights, config


def correction(weights, config, pairs):
    user = weights["layers/embedding/vars/0"][pairs[:, 0]]
    book = weights["book_embedding/vars/0"][pairs[:, 1]]
    features = np.concatenate([user, book, user * book], axis=1)
    for layer in ("hidden/dense", "hidden/dense_1"):
        features = features @ weights[f"{layer}/vars/0"] + weights[f"{layer}/vars/1"]
        features = np.maximum(features, 0) + np.float32(0.2) * np.minimum(features, 0)
    output = features @ weights["layers/dense_2/vars/0"] + weights["layers/dense_2/vars/1"]
    return (np.float32(config["residual_scale"]) * output).reshape(-1)


def summary(values):
    values = values.astype(np.float64)
    return {
        "pairs": int(values.size), "mean": float(values.mean()),
        "std": float(values.std()), "min": float(values.min()), "max": float(values.max()),
    }


def audit(run_dir, batch_size=1024):
    results_path = run_dir / "results.json"
    results = read_json(results_path)
    require(results["status"] == "completed", "Only completed experiment runs can be audited.")
    train_path = run_dir / "inputs/train_ratings.csv"
    validation_path = run_dir / "inputs/validation_ratings.csv"
    train = read_ratings(train_path)
    validation = read_ratings(validation_path)
    users = {user: index for index, user in enumerate(sorted({row[0] for row in train}))}
    books = {book: index for index, book in enumerate(sorted({row[1] for row in train}))}
    known_validation = [row for row in validation if row[0] in users and row[1] in books]
    require(train and known_validation, "Training and known validation inputs must be nonempty.")
    require(len({(user, book) for user, book, _ in train}) == len(train), "Duplicate training pairs.")
    require(all(1 <= rating <= 10 for _, _, rating in train + validation), "Expected explicit 1-10 ratings.")
    train_pairs = np.array([(users[user], books[book]) for user, book, _ in train], dtype=np.int32)
    validation_pairs = np.array([(users[user], books[book]) for user, book, _ in known_validation], dtype=np.int32)
    validation_targets = np.array([rating for _, _, rating in known_validation])
    candidates = results["configuration"]["candidates"]
    require(BASELINE in {candidate["name"] for candidate in candidates}, "Untrained bias-only control is required.")
    records = []
    reference = None
    reference_weights = None
    for candidate in sorted(candidates, key=lambda item: (item["name"] != BASELINE, item["name"])):
        name = candidate["name"]
        require(Path(name).name == name and name not in (".", ".."), "Invalid candidate directory name.")
        directory = run_dir / "candidates" / name
        metadata_path = directory / "model/metadata.json"
        checkpoint_path = directory / "model/network.keras"
        validation_result_path = directory / "validation_result.json"
        metadata = read_json(metadata_path)
        recorded = read_json(validation_result_path)
        metadata_train = [(int(row["user_id"]), str(row["isbn"]), float(row["book_rating"]))
                          for row in metadata["ratings"]]
        require(metadata_train == train, f"Frozen input order/content differs from {name}'s saved metadata.")
        require(metadata["config"]["random_state"] == results["configuration"]["seed"], "Seed mismatch.")
        weights, config = load_weights(checkpoint_path, metadata, len(users), len(books))
        train_delta = np.concatenate([
            correction(weights, config, train_pairs[start:start + batch_size])
            for start in range(0, len(train_pairs), batch_size)
        ])
        validation_delta = correction(weights, config, validation_pairs)
        baseline = (weights["vars/0"] + weights["vars/1"][validation_pairs[:, 0]]
                    + weights["vars/2"][validation_pairs[:, 1]])
        predictions = 1 + 9 * np.clip(baseline + validation_delta, 0, 1)
        rmse = float(np.sqrt(np.mean(np.square(predictions - validation_targets))))
        expected_rmse = recorded["validation"]["rmse_rating_scale"]
        tolerance = 1e-5
        require(abs(rmse - expected_rmse) <= tolerance, f"Manual validation prediction check failed for {name}.")
        require(recorded["validation"]["pairs"] == len(known_validation), "Validation pair count mismatch.")
        norms = {
            "user_embedding": float(np.linalg.norm(weights["layers/embedding/vars/0"].astype(np.float64))),
            "book_embedding": float(np.linalg.norm(weights["book_embedding/vars/0"].astype(np.float64))),
        }
        if name == BASELINE:
            require(config["residual_scale"] == 0 and recorded["epochs_completed"] == 0,
                    "Bias reference must be an untrained zero-residual control.")
            require(not metadata["first_step_trace"], "Bias reference must have no optimizer trace step.")
            reference = norms
            reference_weights = weights
        require(all(np.array_equal(weights[key], reference_weights[key])
                    for key in ("vars/0", "vars/1", "vars/2")),
                f"Frozen bias weights differ from the control for {name}.")
        initial_row_matches = None
        if name != BASELINE:
            trace = metadata["first_step_trace"]
            traced_user = trace["encoded_pairs"][0][0]
            initial_row_matches = np.array_equal(
                np.asarray(trace["user_embedding_before"], dtype=np.float32),
                reference_weights["layers/embedding/vars/0"][traced_user],
            )
            require(initial_row_matches, f"Initial traced embedding differs from the control for {name}.")
        records.append({
            "name": name, "kind": candidate["kind"], "l2": config["l2"],
            "residual_scale": config["residual_scale"],
            "best_epoch": recorded["best_epoch"], "epochs_completed": recorded["epochs_completed"],
            "train_residual_rating": summary(9 * train_delta),
            "validation_residual_rating": summary(9 * validation_delta),
            "embedding_norms": norms,
            "embedding_norm_ratio_to_untrained_control": {key: value / reference[key] for key, value in norms.items()},
            "untrained_control_checks": {
                "frozen_bias_arrays_identical": True,
                "traced_initial_user_embedding_matches_control": initial_row_matches,
                "initial_embedding_check_scope": "First traced user embedding row, plus matching seed and architecture; no optimizer step exists for the control.",
            },
            "validation_check": {
                "saved_rmse": expected_rmse, "numpy_rmse": rmse,
                "absolute_difference": abs(rmse - expected_rmse), "absolute_tolerance": tolerance, "passed": True,
            },
            "artifacts": [fingerprint(path) for path in (checkpoint_path, metadata_path, validation_result_path)],
        })
    return {
        "schema_version": 1, "status": "completed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Read saved float32 weights using h5py; manual NumPy forward pass; no TensorFlow import or training.",
        "architecture_contract": "ResidualRatingModel; concat(user,book,user*book); two LeakyReLU(0.2) layers; linear scaled residual; frozen additive biases.",
        "scope": "Training and validation only. No test file read or new test evaluation.",
        "residual_statistics_convention": "Additive neural correction in original rating points, before clipping final predictions to 1-10; population standard deviation.",
        "baseline_reference": BASELINE,
        "selected_candidate": results["selection"]["best_neural_candidate"],
        "train_pairs": len(train), "validation_pairs": len(validation), "validation_known_pairs": len(known_validation),
        "provenance": [fingerprint(path) for path in (Path(__file__), results_path, train_path, validation_path)],
        "candidates": records,
        "interpretation": "The saved residual corrections are nearly constant across training and known validation pairs. The learned embeddings are much smaller than the same-seed untrained control. This supports auditing regularization and initialization before expanding the network.",
        "limitations": [
            "This manual forward pass supports only the explicitly checked architecture/dtype and must be updated when model semantics change.",
            "Agreement of aggregate validation RMSE within 1e-5 checks the reconstruction; it is not a fresh TensorFlow per-prediction parity test.",
            "Nearly constant is not exactly constant. This diagnoses saved best checkpoints, not every optimizer step.",
            "Large embedding shrinkage is observed; causation by L2 or zero-head initialization requires controlled ablations.",
            "Embedding norm comparisons use the same-seed untrained bias-only control; they do not measure parameter displacement.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "artifacts/colab/run-20261014-residual/final")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/evidence/neural_residual_signal_audit.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    require(args.batch_size > 0, "batch-size must be positive.")
    report = audit(args.run_dir.resolve(), args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(report['candidates'])} saved models; audit written to {args.output}")


if __name__ == "__main__":
    main()
