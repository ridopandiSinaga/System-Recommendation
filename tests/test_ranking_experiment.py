"""Pure input/gating invariants for the bounded ranking worker."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import colab_ranking_experiment as worker
from scripts.colab_neural_experiment import fingerprint_files


def make_inputs(tmp_path, monkeypatch):
    (tmp_path / "train_books.csv").write_text("isbn,book_title,book_author\na,A,Author\n")
    splits = []
    for seed in worker.SPLIT_SEEDS:
        directory = tmp_path / "splits" / f"split_{seed}"
        directory.mkdir(parents=True)
        names = []
        for part in ("fit", "selection", "assessment"):
            name = f"{part}_ratings.csv"
            (directory / name).write_text("user_id,isbn,book_rating\n1,a,8\n")
            names.append(name)
        splits.append({"split_seed": seed, "partition_file_fingerprints": fingerprint_files(directory, names)})
        prior = tmp_path / "previous_ranking" / f"split_{seed}"
        prior.mkdir(parents=True)
        for name in ("ranking_user_outcomes.csv", "ranking_top10.csv"):
            (prior / name).write_text("model,user_id\nold,1\n")
    previous = {"experiment_kind": "tuned_bias_mf_ranking_reused_development",
                "all_candidates_completed": True, "assessment_performed": True,
                "dataset": {"splits": splits}}
    (tmp_path / "previous_results.json").write_text(json.dumps(previous))
    files = fingerprint_files(tmp_path, [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()])
    manifest = {"files": files, "previous_results_sha256": files["previous_results.json"]["sha256"],
                "previous_ranking_fingerprints": {name: value for name, value in files.items()
                                                  if name.startswith("previous_ranking/")}}
    (tmp_path / "input_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(worker, "FROZEN_INPUT_SHA256", {"train_books.csv": files["train_books.csv"]["sha256"]})
    return manifest, previous


def test_accepts_only_exact_hash_verified_reused_partitions_and_prior_rankings(tmp_path, monkeypatch):
    expected = make_inputs(tmp_path, monkeypatch)
    assert worker.verify_inputs(tmp_path) == expected
    (tmp_path / "splits/split_17/assessment_ratings.csv").write_text("changed\n")
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        worker.verify_inputs(tmp_path)


def test_refuses_undeclared_files_even_when_declared_inputs_are_unchanged(tmp_path, monkeypatch):
    make_inputs(tmp_path, monkeypatch)
    (tmp_path / "test_ratings.csv").write_text("unapproved test partition\n")
    with pytest.raises(RuntimeError, match="undeclared"):
        worker.verify_inputs(tmp_path)


def test_cannot_replace_prior_split_and_merely_rehash_local_manifest(tmp_path, monkeypatch):
    manifest, _ = make_inputs(tmp_path, monkeypatch)
    name = "splits/split_29/selection_ratings.csv"
    (tmp_path / name).write_text("user_id,isbn,book_rating\n1,a,9\n")
    manifest["files"][name] = fingerprint_files(tmp_path, [name])[name]
    (tmp_path / "input_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="prior fingerprint"):
        worker.verify_inputs(tmp_path)


def test_primary_choice_uses_selection_and_stable_ties_only():
    rows = []
    for candidate in worker.CANDIDATES:
        seed = candidate["initialization_seed"]
        rows.append({**candidate, "status": "completed",
                     "selection": {"all_relevant": {"ndcg_at_10": 0.2 if seed != 7 else 0.1,
                                                     "recall_at_10": 0.3}},
                     "assessment": {"ndcg_at_10": 1 if seed == 7 else 0}})
    result = worker.select_primaries(rows, np)
    assert len(result) == 2
    assert len(rows) == 18 and len({row["name"] for row in rows}) == 18
    for split in result:
        assert len(split["arms"]) == 3
        for arm in split["arms"]:
            assert arm["primary_candidate"].endswith("seed42")
            assert arm["n_seeds"] == 3
    rows[0]["status"] = "failed"
    incomplete = worker.select_primaries(rows, np)
    assert {arm["arm"] for arm in incomplete[0]["arms"]} == {"bpr_id", "bpr_metadata"}


def test_initialization_fingerprint_checks_values_shapes_and_dtypes_not_layer_names():
    def fake(value):
        return SimpleNamespace(trainable_variables=[SimpleNamespace(numpy=lambda: value)])
    first = np.arange(4, dtype=np.float32).reshape(2, 2)
    assert worker.trainable_fingerprint(fake(first)) == worker.trainable_fingerprint(fake(first.copy()))
    assert worker.trainable_fingerprint(fake(first)) != worker.trainable_fingerprint(fake(first.reshape(4)))
    assert worker.trainable_fingerprint(fake(first)) != worker.trainable_fingerprint(fake(first.astype(np.float64)))
    assert worker.trainable_fingerprint(fake(first)) != worker.trainable_fingerprint(fake(first + 1))
