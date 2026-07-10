import json
from dataclasses import replace

import pandas as pd
import pytest

from book_recommender.artifacts import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    fingerprint_dataset,
    load_model_bundle,
    manifest_path,
    save_model_bundle,
)


def sample_fingerprint():
    books = pd.DataFrame({"isbn": ["B", "A"], "book_title": ["Beta", "Alpha"]})
    ratings = pd.DataFrame(
        {"user_id": [2, 1], "isbn": ["B", "A"], "book_rating": [8.0, 9.0]}
    )
    return fingerprint_dataset(books, ratings)


def test_dataset_fingerprint_is_order_independent():
    books = pd.DataFrame({"isbn": ["A", "B"], "book_title": ["Alpha", "Beta"]})
    ratings = pd.DataFrame(
        {"user_id": [1, 2], "isbn": ["A", "B"], "book_rating": [9.0, 8.0]}
    )

    first = fingerprint_dataset(books, ratings)
    second = fingerprint_dataset(books.iloc[::-1], ratings.iloc[::-1])

    assert first.fingerprint == second.fingerprint
    assert first.books_rows == 2
    assert first.ratings_rows == 2


def test_model_bundle_roundtrip_and_manifest(tmp_path):
    artifact = tmp_path / "model.joblib"
    model = {"weights": [0.4, 0.45, 0.15]}
    fingerprint = sample_fingerprint()

    saved = save_model_bundle(
        model,
        artifact,
        model_type="hybrid",
        dataset_fingerprint=fingerprint,
        training_config={"max_books": 5000},
        metrics={"hit_rate_at_10": 0.2},
    )
    loaded, manifest = load_model_bundle(
        artifact,
        expected_type=dict,
        expected_model_type="hybrid",
        expected_dataset_fingerprint=fingerprint.fingerprint,
        expected_training_config={"max_books": 5000},
    )

    assert loaded == model
    assert manifest.artifact_id == saved.artifact_id
    assert manifest.metrics["hit_rate_at_10"] == 0.2
    assert json.loads(manifest_path(artifact).read_text())["artifact_sha256"]


def test_model_bundle_rejects_tampered_artifact(tmp_path):
    artifact = tmp_path / "model.joblib"
    save_model_bundle(
        {"ok": True},
        artifact,
        model_type="content",
        dataset_fingerprint=sample_fingerprint(),
        training_config={},
    )
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        load_model_bundle(artifact)


def test_model_bundle_rejects_incompatible_config(tmp_path):
    artifact = tmp_path / "model.joblib"
    save_model_bundle(
        {"ok": True},
        artifact,
        model_type="content",
        dataset_fingerprint=sample_fingerprint(),
        training_config={"max_books": 100},
    )

    with pytest.raises(ArtifactCompatibilityError, match="max_books"):
        load_model_bundle(artifact, expected_training_config={"max_books": 200})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"dataset": []}),
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload.update({"artifact_sha256": "not-a-hash"}),
    ],
)
def test_model_bundle_rejects_malformed_manifest(tmp_path, mutation):
    artifact = tmp_path / "model.joblib"
    save_model_bundle(
        {"ok": True},
        artifact,
        model_type="content",
        dataset_fingerprint=sample_fingerprint(),
        training_config={},
    )
    sidecar = manifest_path(artifact)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    mutation(payload)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="Invalid artifact manifest"):
        load_model_bundle(artifact)


def test_model_bundle_rejects_dependency_minor_mismatch(tmp_path):
    artifact = tmp_path / "model.joblib"
    save_model_bundle(
        {"ok": True},
        artifact,
        model_type="content",
        dataset_fingerprint=sample_fingerprint(),
        training_config={},
    )
    sidecar = manifest_path(artifact)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["dependencies"]["scikit-learn"] = "0.0.1"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="scikit-learn"):
        load_model_bundle(artifact)


def test_model_bundle_rejects_non_finite_json_metadata(tmp_path):
    with pytest.raises(ValueError):
        save_model_bundle(
            {"ok": True},
            tmp_path / "model.joblib",
            model_type="content",
            dataset_fingerprint=sample_fingerprint(),
            training_config={"weight": float("nan")},
        )


def test_model_bundle_rejects_non_finite_manifest_on_load(tmp_path):
    artifact = tmp_path / "model.joblib"
    save_model_bundle(
        {"ok": True},
        artifact,
        model_type="content",
        dataset_fingerprint=sample_fingerprint(),
        training_config={},
    )
    sidecar = manifest_path(artifact)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["metrics"] = {"score": float("inf")}
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="Invalid artifact manifest"):
        load_model_bundle(artifact)


def test_model_bundle_refuses_invalid_fingerprint_before_publish(tmp_path):
    artifact = tmp_path / "model.joblib"
    malformed = replace(sample_fingerprint(), fingerprint="not-a-sha256")

    with pytest.raises(ValueError, match="SHA-256"):
        save_model_bundle(
            {"ok": True},
            artifact,
            model_type="content",
            dataset_fingerprint=malformed,
            training_config={},
        )

    assert not artifact.exists()
    assert not manifest_path(artifact).exists()
