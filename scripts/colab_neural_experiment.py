"""Run one bounded neural Book-Crossing experiment inside this project's Colab.

The caller owns Colab authentication, uploads, downloads, and runtime shutdown.
This worker accepts only data/experiment paths and emits independently verifiable
input, per-epoch, and final archives; it never reads authentication configuration.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA_FILES = ("Books.csv", "Ratings.csv", "Users.csv")
DATA_SOURCE = "https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset"
DATA_ORIGIN = "https://www2.informatik.uni-freiburg.de/~cziegler/BX/"
SECRET_PATTERN = re.compile(
    r"(?:ya29\.[A-Za-z0-9._~-]+|1//[A-Za-z0-9._~-]+|"
    r"(?i:Bearer)\s+[^\s]+|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(base, names):
    return {
        name: {"sha256": sha256_file(Path(base) / name),
               "bytes": (Path(base) / name).stat().st_size}
        for name in names
    }


def emit(event, **values):
    print(json.dumps({"event": event, "time_utc": utc_now(), **values}), flush=True)


def write_archive(output, archive_name, entries, manifest_name, **metadata):
    """Replace one archive atomically; its external manifest never hashes itself."""
    output = Path(output)
    archive = output / archive_name
    temporary = archive.with_name(archive.name + ".partial")
    with tarfile.open(temporary, "w:gz", compresslevel=3) as tar:
        for entry in entries:
            path = output / entry
            if path.exists():
                tar.add(path, arcname=entry, recursive=True)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(archive)
    manifest = {
        "archive": archive.name, "remote_path": str(archive),
        "sha256": sha256_file(archive), "bytes": archive.stat().st_size,
        "created_utc": utc_now(), **metadata,
    }
    # Validate archive structure and checksum before publishing its manifest.
    with tarfile.open(archive, "r:gz") as tar:
        member_count = len(tar.getmembers())
        if not member_count:
            raise RuntimeError("A backup archive contains no files.")
    if sha256_file(archive) != manifest["sha256"]:
        raise RuntimeError("Backup archive changed during verification.")
    manifest["members"] = member_count
    atomic_json(output / manifest_name, manifest)
    return manifest


def save_network(model, path):
    path = Path(path)
    temporary = path.with_name(path.stem + ".partial.keras")
    model.save(temporary)
    temporary.replace(path)


def gpu_probe(tf, require_gpu):
    """Fail before dataset processing when the required GPU is unavailable."""
    physical = tf.config.list_physical_devices("GPU")
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
    if require_gpu and not physical:
        raise RuntimeError("GPU required, but TensorFlow detects no GPU in this Colab runtime.")
    device_name = "/GPU:0" if physical else "/CPU:0"
    started = time.monotonic()
    with tf.device(device_name):
        value = tf.linalg.matmul(tf.ones((128, 128)), tf.ones((128, 128)))
        scalar = float(value[0, 0].numpy())
    if require_gpu and "GPU" not in value.device:
        raise RuntimeError("TensorFlow did not place the GPU probe on a GPU.")
    if scalar != 128.0:
        raise RuntimeError("GPU matrix multiplication probe returned the wrong value.")
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        nvidia_output = process.stdout.strip()
        nvidia_code = process.returncode
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        nvidia_output = type(error).__name__
        nvidia_code = None
    return {
        "checked_utc": utc_now(), "tensorflow_version": tf.__version__,
        "cuda_build": bool(tf.test.is_built_with_cuda()),
        "physical_gpu_names": [device.name for device in physical],
        "gpu_details": [tf.config.experimental.get_device_details(device) for device in physical],
        "logical_gpu_names": [device.name for device in tf.config.list_logical_devices("GPU")],
        "probe_device": value.device, "probe_value": scalar,
        "probe_seconds": time.monotonic() - started,
        "nvidia_smi_exit_code": nvidia_code, "nvidia_smi": nvidia_output,
    }


def source_metadata():
    selected = [
        "scripts/colab_neural_experiment.py", "src/book_recommender/neural.py",
        "src/book_recommender/data.py", "src/book_recommender/evaluation.py",
        "src/book_recommender/collaborative.py", "src/book_recommender/content_based.py",
        "pyproject.toml",
    ]
    packages = {}
    for name in ("tensorflow", "keras", "numpy", "pandas", "scikit-learn", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "source_files": fingerprint_files(ROOT, selected),
        "python": sys.version, "packages": packages,
        "source_label": "uploaded_project_source_with_neural_epoch_callback",
        "experiment_label": "single_seed_single_split_free_colab_gpu",
    }


def persist_inputs(output, train_books, train_ratings, validation, test):
    directory = output / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    frames = {
        "train_books.csv": train_books, "train_ratings.csv": train_ratings,
        "validation_ratings.csv": validation, "test_ratings.csv": test,
    }
    for name, frame in frames.items():
        frame.to_csv(directory / name, index=False)
    hashes = fingerprint_files(directory, list(frames))
    atomic_json(output / "input_fingerprints.json", hashes)
    manifest = write_archive(
        output, "inputs_backup.tar.gz",
        ["inputs", "input_fingerprints.json", "configuration.json", "gpu_probe.json",
         "source_metadata.json", "dataset_fingerprints.json", "dataset_summary.json"],
        "inputs_manifest.json", kind="training_inputs",
    )
    emit("INPUTS_BACKUP_READY", **manifest)


def rating_baselines(neural, train_ratings, test_ratings, np):
    """Compare every rating predictor on the exact same known test pairs."""
    known_user = test_ratings.user_id.isin(neural.user_to_index)
    known_book = test_ratings.isbn.isin(neural.isbn_to_index)
    known = test_ratings.loc[known_user & known_book].copy()
    mean_rating = float(train_ratings.book_rating.mean())
    user_means = train_ratings.groupby("user_id").book_rating.mean()
    metrics = {}
    for label, predictions in {
        "train_global_mean": np.full(len(known), mean_rating),
        "train_user_mean": known.user_id.map(user_means).to_numpy(dtype=float),
    }.items():
        errors = predictions - known.book_rating.to_numpy(dtype=float)
        metrics[label] = {
            "known_pairs": len(known), "scope": "same_known_user_and_isbn_pairs_as_neural",
            "rmse_rating_scale": float(np.sqrt(np.mean(errors ** 2))) if len(known) else None,
            "mae_rating_scale": float(np.mean(np.abs(errors))) if len(known) else None,
        }
    neural_metrics = neural.evaluate_ratings(test_ratings)
    if len(known):
        pairs, _, _ = neural.encode_ratings(known)
        predictions = 1 + 9 * neural._predict_encoded(pairs).ravel()
        neural_metrics["mae_rating_scale"] = float(np.mean(np.abs(
            predictions - known.book_rating.to_numpy(dtype=float)
        )))
    else:
        neural_metrics["mae_rating_scale"] = None
    metrics["neural"] = neural_metrics
    return {
        "test_explicit_pairs": len(test_ratings), "known_pairs": len(known),
        "unknown_user_pairs": int((~known_user).sum()),
        "unknown_isbn_pairs": int((~known_book).sum()),
        "unknown_user_and_isbn_pairs": int((~known_user & ~known_book).sum()),
        "known_pair_retention_rate": len(known) / len(test_ratings) if len(test_ratings) else None,
        "scale": "original explicit ratings 1 through 10", "models": metrics,
    }


def evaluate_ranking(neural, train_books, train_ratings, test_ratings, work_map, output):
    import pandas as pd

    from book_recommender.collaborative import ItemBasedCollaborativeRecommender
    from book_recommender.evaluation import mean_ranking_metrics_at_k, ranking_metrics_at_k

    top_k = 10
    relevant = test_ratings[test_ratings.book_rating >= 8].copy()
    candidate_works = {work_map[str(isbn)] for isbn in train_books.isbn}
    candidate_available = relevant.work_key.isin(candidate_works)
    history_available = relevant.user_id.isin(neural.user_to_index)
    scope = {
        "test_holdouts": len(test_ratings), "relevant_test_holdouts_rating_ge_8": len(relevant),
        "candidate_work_available": int(candidate_available.sum()),
        "train_user_history_available": int(history_available.sum()),
        "candidate_and_history_available": int((candidate_available & history_available).sum()),
        "candidate_isbns": len(train_books), "candidate_canonical_works": len(candidate_works),
        "scope": "all relevant test holdouts; unavailable targets remain explicit misses",
        "paired": True, "top_k": top_k,
    }
    labels = ("neural_with_popularity_fallback", "item_based_with_popularity_fallback", "popularity")
    if relevant.empty:
        return {"availability": scope, "models": {label: {"count": 0, "metrics": None}
                                                   for label in labels}}
    item = ItemBasedCollaborativeRecommender().fit(train_books, train_ratings)
    histories = train_ratings.groupby("user_id").isbn.agg(set).to_dict()
    records = []
    for offset, row in enumerate(relevant.itertuples(index=False), start=1):
        user_id = int(row.user_id)
        recommendations = {
            labels[0]: neural.recommend_for_user(user_id, top_n=top_k),
            labels[1]: item.recommend_for_user(user_id, top_n=top_k),
            labels[2]: item.recommend_popular(
                top_n=top_k, exclude_isbns=histories.get(user_id, set()),
            ),
        }
        for label, frame in recommendations.items():
            recommended_works = list(dict.fromkeys(work_map[str(isbn)] for isbn in frame.isbn))
            records.append({
                "model": label, "user_id": user_id, "heldout_work_key": str(row.work_key),
                "candidate_work_available": str(row.work_key) in candidate_works,
                "user_history_available": user_id in neural.user_to_index,
                **ranking_metrics_at_k(recommended_works, [str(row.work_key)], top_k),
            })
        if offset == 1 or offset % 25 == 0 or offset == len(relevant):
            atomic_json(output / "evaluation_progress.json", {
                "stage": "ranking", "completed_holdouts": offset,
                "total_holdouts": len(relevant), "updated_utc": utc_now(),
            })
            emit("RANKING_PROGRESS", completed_holdouts=offset, total_holdouts=len(relevant))
    ranking = pd.DataFrame(records)
    ranking.to_csv(output / "ranking_rows.csv", index=False)
    summary = {}
    metric_columns = [column for column in ranking if "_at_" in column]
    for label in labels:
        model_rows = ranking[ranking.model == label]
        if len(model_rows) != len(relevant):
            raise RuntimeError("Ranking baseline models were evaluated on different holdouts.")
        retained = model_rows.candidate_work_available & model_rows.user_history_available
        summary[label] = {
            "count": len(model_rows),
            "metrics": mean_ranking_metrics_at_k(model_rows[metric_columns], k=top_k),
            "conditional_count": int(retained.sum()),
            "conditional_metrics": (
                mean_ranking_metrics_at_k(model_rows.loc[retained, metric_columns], k=top_k)
                if retained.any() else None
            ),
        }
    return {"availability": scope, "models": summary}


def plot_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["epoch"] for row in history], [row["train_mse"] for row in history],
              marker="o", label="Train")
    axis.plot([row["epoch"] for row in history], [row["val_mse"] for row in history],
              marker="o", label="Validation (known user/ISBN)")
    axis.set(xlabel="Epoch", ylabel="MSE on normalized 0–1 rating", title="Neural training history")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "training_history.png", dpi=160)
    plt.close(figure)


def write_report(results, output):
    data = results["dataset"]
    rating = results["rating_test"]
    lines = [
        "Eksperimen rekomendasi buku neural — GPU Google Colab gratis",
        "", (f"Training: {data['train_pairs']} pasangan, {data['train_users']} pengguna, "
             f"{data['train_isbns']} ISBN."),
        (f"Epoch selesai: {results['training']['epochs_completed']}; "
         f"epoch terbaik menurut validation: {results['training']['best_epoch']}."),
        ("Arsitektur: embedding pengguna/buku 16 dimensi, interaksi elementwise, "
         "Dense ReLU 32 dan 16 unit, output sigmoid."),
        "Target adalah (rating−1)/9; loss training = MSE + regularisasi L2; optimizer Adam.",
        ("first_step_trace.json menyimpan forward propagation, loss, gradien backpropagation, "
         "dan embedding sebelum/sesudah pembaruan Adam."),
        "", "Rating test hanya memakai pasangan pengguna dan ISBN yang dikenal semua model:",
        f"{rating['known_pairs']} dari {rating['test_explicit_pairs']} pasangan test.",
    ]
    for name, metrics in rating["models"].items():
        lines.append(f"{name}: RMSE={metrics['rmse_rating_scale']}; "
                     f"MAE={metrics['mae_rating_scale']}")
    lines.extend([
        "", "Ranking memakai holdout test rating >=8 dan kandidat karya canonical yang sama.",
        "Target yang hilang setelah filtering tetap dihitung sebagai miss dalam metrik keseluruhan.",
        "Kondisi ketersediaan kandidat/histori dan metrik kondisional tercatat dalam results.json.",
        "", "Eksperimen ini satu seed/split, bukan bukti bahwa neural mengalahkan baseline.",
        "Rating 0 dikeluarkan karena merupakan interaksi implisit, bukan nilai ketidaksukaan.",
        ("Pemilihan katalog dan k-core dilakukan setelah penghapusan karya holdout. Test tidak "
         "dipakai untuk early stopping atau pengaturan hyperparameter."),
        ("latest_checkpoint menyimpan model dan optimizer pada epoch terakhir untuk recovery; "
         "final_model menyimpan bobot terbaik menurut validation untuk inference."),
    ])
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args):
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from book_recommender.content_based import book_work_key
    from book_recommender.data import BookDataset, clean_dataset, load_raw_data
    from book_recommender.evaluation import prepare_validation_test_evaluation
    from book_recommender.neural import NeuralRecommender

    output = args.output.resolve()
    started_clock = time.monotonic()
    configuration = {
        "data_dir": str(args.data_dir.resolve()), "output": str(output),
        "requested_epochs": args.epochs, "sample_users": args.sample_users,
        "max_books": args.max_books, "max_content_books": args.max_books,
        "batch_size": args.batch_size, "seed": 42, "embedding_size": 16,
        "hidden_units": [32, 16], "learning_rate": 0.001, "l2": 1e-5,
        "patience": 3, "validation_fraction": 0.5, "min_interactions": 3,
        "min_relevant_rating_for_split": 1, "ranking_relevant_rating": 8,
        "min_user_ratings": 2, "min_book_ratings": 2, "require_gpu": args.require_gpu,
        "started_utc": utc_now(), "dataset_source": DATA_SOURCE,
        "dataset_origin": DATA_ORIGIN,
        "dataset_format": "Kaggle distribution of Book-Crossing in three CSV schemas",
    }
    atomic_json(output / "configuration.json", configuration)
    probe = gpu_probe(tf, args.require_gpu)
    atomic_json(output / "gpu_probe.json", probe)
    emit("GPU_PROBE_PASSED", gpu=probe["nvidia_smi"], device=probe["probe_device"])
    metadata = source_metadata()
    atomic_json(output / "source_metadata.json", metadata)
    fingerprints = fingerprint_files(args.data_dir, DATA_FILES)
    atomic_json(output / "dataset_fingerprints.json", {
        "source": DATA_SOURCE, "origin": DATA_ORIGIN, "files": fingerprints,
        "combined_sha256": hashlib.sha256(
            json.dumps(fingerprints, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    })
    raw = load_raw_data(args.data_dir)
    rating_column = "Book-Rating" if "Book-Rating" in raw.ratings else "book_rating"
    raw_rating_values = pd.to_numeric(raw.ratings[rating_column], errors="coerce")
    raw_summary = {
        "raw_books_rows": len(raw.books), "raw_ratings_rows": len(raw.ratings),
        "raw_users_rows": len(raw.users),
        "raw_explicit_rating_rows": int(raw_rating_values.between(1, 10).sum()),
        "implicit_interactions_excluded": int(raw_rating_values.eq(0).sum()),
    }
    clean = clean_dataset(raw)
    del raw, raw_rating_values
    gc.collect()
    explicit = clean.ratings[clean.ratings.book_rating >= 1].reset_index(drop=True)
    evaluation = prepare_validation_test_evaluation(
        BookDataset(clean.books, explicit, clean.users), sample_users=args.sample_users,
        validation_fraction=0.5, min_interactions=3, min_relevant_rating=1,
        min_user_ratings=2, min_book_ratings=2, max_books=args.max_books,
        max_content_books=args.max_books, random_state=42,
    )
    train_ratings = evaluation.train_dataset.ratings
    train_books = evaluation.train_dataset.books[
        evaluation.train_dataset.books.isbn.isin(train_ratings.isbn)
    ].reset_index(drop=True)
    validation = evaluation.validation_holdout_ratings
    test = evaluation.test_holdout_ratings
    if train_ratings.empty:
        raise RuntimeError("No training interactions remain after train-only filtering.")
    work_map = {
        str(row.isbn): book_work_key(row.book_title, row.book_author, row.isbn)
        for row in clean.books.itertuples(index=False)
    }
    train_pairs = {(int(row.user_id), work_map[str(row.isbn)])
                   for row in train_ratings.itertuples(index=False)}
    if not set(validation.user_id).isdisjoint(test.user_id):
        raise RuntimeError("Validation and test users overlap.")
    for holdout in (validation, test):
        if any((int(row.user_id), str(row.work_key)) in train_pairs
               for row in holdout.itertuples(index=False)):
            raise RuntimeError("Canonical holdout leakage found in training interactions.")
    validation_known = validation[
        validation.user_id.isin(train_ratings.user_id) & validation.isbn.isin(train_ratings.isbn)
    ]
    if validation_known.empty:
        raise RuntimeError("Validation has no known user/ISBN pairs; cannot choose weights safely.")
    dataset_summary = {
        **raw_summary,
        "clean_catalog_books": len(clean.books),
        "clean_explicit_interactions_with_metadata": len(clean.ratings),
        "explicit_interactions": len(explicit),
        "train_pairs": len(train_ratings), "train_users": int(train_ratings.user_id.nunique()),
        "train_isbns": int(train_ratings.isbn.nunique()),
        "validation_pairs": len(validation), "validation_known_pairs": len(validation_known),
        "test_pairs": len(test), "canonical_work_leakage_checks_passed": True,
        "selection_scope": "split canonical works first, then select train-only catalog/k-core",
    }
    atomic_json(output / "dataset_summary.json", dataset_summary)
    emit("DATA_PREPARED", **dataset_summary)
    persist_inputs(output, train_books, train_ratings, validation, test)
    checkpoint_dir = output / "latest_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def epoch_finished(recommender, record):
        # fit() calls this before restoring best weights, so checkpoint optimizer
        # moments and network parameters represent the same most recent update.
        save_network(recommender.model, output / "latest_network.keras")
        checkpoint = tf.train.Checkpoint(model=recommender.model, optimizer=recommender.optimizer)
        checkpoint.write(str(checkpoint_dir / "training"))
        if record["epoch"] == recommender.best_epoch:
            save_network(recommender.model, output / "best_network.keras")
        atomic_json(output / "vocabulary.json", {
            "user_to_index": recommender.user_to_index,
            "isbn_to_index": recommender.isbn_to_index,
        })
        atomic_json(output / "first_step_trace.json", recommender.first_step_trace)
        atomic_json(output / "history.json", recommender.history)
        atomic_json(output / "progress.json", {
            "status": "training", "epoch": record["epoch"], "record": record,
            "history": recommender.history, "best_epoch": recommender.best_epoch,
            "optimizer_iterations": int(recommender.optimizer.iterations.numpy()),
            "training_rng_state": recommender.training_rng_state,
            "elapsed_seconds": time.monotonic() - started_clock, "updated_utc": utc_now(),
            "checkpoint_prefix": "latest_checkpoint/training",
            "resume_scope": "checkpoint model/optimizer + vocabulary, inputs, history, RNG state; "
                            "fit itself initializes fresh parameters, so restore via a resume loop",
        })
        manifest = write_archive(
            output, f"epoch_{record['epoch']:03d}_snapshot.tar.gz",
            ["latest_checkpoint", "latest_network.keras", "best_network.keras", "vocabulary.json",
             "first_step_trace.json", "history.json", "progress.json", "configuration.json",
             "gpu_probe.json", "source_metadata.json", "dataset_fingerprints.json",
             "dataset_summary.json", "input_fingerprints.json", "inputs_manifest.json"],
            f"epoch_{record['epoch']:03d}_manifest.json",
            kind="epoch_checkpoint", epoch=record["epoch"],
        )
        # Publish a small latest index only after the immutable epoch backup is
        # ready. A download can safely finish while the next epoch is training.
        atomic_json(output / "snapshot_manifest.json", manifest)
        emit("EPOCH_BACKUP_READY", **manifest)

    neural = NeuralRecommender(
        embedding_size=16, hidden_units=(32, 16), learning_rate=0.001, l2=1e-5, random_state=42,
    ).fit(
        train_books, train_ratings, validation_ratings=validation,
        epochs=args.epochs, batch_size=args.batch_size, patience=3,
        epoch_callback=epoch_finished,
    )
    training_seconds = time.monotonic() - started_clock
    atomic_json(output / "progress.json", {
        "status": "evaluating", "epoch": len(neural.history), "history": neural.history,
        "best_epoch": neural.best_epoch, "updated_utc": utc_now(),
    })
    emit("TRAINING_FINISHED", epochs=len(neural.history), best_epoch=neural.best_epoch)
    # Test is first scored here, after all training/validation decisions finish.
    rating_test = rating_baselines(neural, train_ratings, test, np)
    ranking_test = evaluate_ranking(neural, train_books, train_ratings, test, work_map, output)
    neural.save(output / "final_model")
    # Verify deployable serialization on deterministic train pairs before backup.
    restored = NeuralRecommender.load(output / "final_model")
    probe_pairs, _, _ = neural.encode_ratings(train_ratings.head(8))
    np.testing.assert_allclose(neural._predict_encoded(probe_pairs),
                               restored._predict_encoded(probe_pairs), rtol=1e-6, atol=1e-7)
    results = {
        "status": "completed", "completed_utc": utc_now(), "configuration": configuration,
        "gpu": probe, "dataset": dataset_summary,
        "training": {
            "epochs_completed": len(neural.history), "best_epoch": neural.best_epoch,
            "history": neural.history, "training_and_preparation_seconds": training_seconds,
            "optimizer_iterations": int(neural.optimizer.iterations.numpy()),
            "normalized_loss": "MSE plus L2 regularization; history MSE excludes L2",
            "restored_best_validation_weights": True,
        },
        "rating_test": rating_test, "ranking_test": ranking_test,
        "final_model_roundtrip_verified": True,
        "interpretation": "One seed/split exploratory run; no claim that neural is superior. "
                          "Rating metrics are conditional on known user/ISBN pairs. Ranking "
                          "compares all relevant test holdouts using a shared candidate catalog.",
        "total_seconds": time.monotonic() - started_clock,
    }
    atomic_json(output / "results.json", results)
    plot_history(neural.history, output)
    write_report(results, output)
    atomic_json(output / "progress.json", {
        "status": "completed", "epoch": len(neural.history), "best_epoch": neural.best_epoch,
        "history": neural.history, "updated_utc": utc_now(),
    })
    sys.stdout.flush()
    manifest = write_archive(
        output, "final_backup.tar.gz",
        ["final_model", "latest_checkpoint", "latest_network.keras", "best_network.keras",
         "inputs", "vocabulary.json", "first_step_trace.json", "history.json", "progress.json",
         "configuration.json", "gpu_probe.json", "source_metadata.json",
         "dataset_fingerprints.json", "dataset_summary.json", "input_fingerprints.json",
         "results.json", "ranking_rows.csv", "evaluation_progress.json", "training_history.png",
         "report.txt", "worker.log"],
        "final_manifest.json", kind="completed_experiment", epochs=len(neural.history),
    )
    (output / "completed.flag").write_text(
        json.dumps({"completed_utc": utc_now(), "sha256": manifest["sha256"]}) + "\n",
        encoding="utf-8",
    )
    emit("EXPERIMENT_COMPLETED", **manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sample-users", type=int, default=1000)
    parser.add_argument("--max-books", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.max_books, args.batch_size) < 1 or args.sample_users < 2:
        parser.error("epochs/max-books/batch-size must be positive and sample-users must be >=2")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in
           ("completed.flag", "configuration.json", "failed.json")):
        parser.error("output directory already contains an experiment; choose a fresh output path")
    try:
        run_experiment(args)
    except Exception as error:  # noqa: BLE001 -- worker boundary must persist terminal failure
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
