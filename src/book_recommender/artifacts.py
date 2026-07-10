from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd


ARTIFACT_FORMAT_VERSION = 1
MANIFEST_SUFFIX = ".manifest.json"
TRACKED_DEPENDENCIES = (
    "joblib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
)


class ArtifactError(RuntimeError):
    """Base error for model artifact contract failures."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when an artifact checksum or manifest is invalid."""


class ArtifactCompatibilityError(ArtifactError):
    """Raised when an artifact does not match the requested runtime contract."""


@dataclass(frozen=True)
class DatasetFingerprint:
    fingerprint: str
    books_rows: int
    ratings_rows: int
    users_rows: int
    books_schema: dict[str, str]
    ratings_schema: dict[str, str]
    users_schema: dict[str, str]


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_format_version: int
    artifact_id: str
    artifact_sha256: str
    created_at: str
    model_type: str
    model_class: str
    project_version: str
    python_version: str
    dependencies: dict[str, str]
    code_sha: str | None
    code_dirty: bool | None
    dataset: dict[str, Any]
    training_config: dict[str, Any]
    metrics: dict[str, Any]


def fingerprint_dataset(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    users: pd.DataFrame | None = None,
) -> DatasetFingerprint:
    """Return an order-independent fingerprint for the prepared training dataset."""

    users = users if users is not None else pd.DataFrame()
    books_hash = _fingerprint_dataframe(books)
    ratings_hash = _fingerprint_dataframe(ratings)
    users_hash = _fingerprint_dataframe(users)

    digest = hashlib.sha256()
    for name, value in (
        ("books", books_hash),
        ("ratings", ratings_hash),
        ("users", users_hash),
    ):
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))

    return DatasetFingerprint(
        fingerprint=digest.hexdigest(),
        books_rows=int(len(books)),
        ratings_rows=int(len(ratings)),
        users_rows=int(len(users)),
        books_schema=_schema(books),
        ratings_schema=_schema(ratings),
        users_schema=_schema(users),
    )


def save_model_bundle(
    model: Any,
    path: str | Path,
    *,
    model_type: str,
    dataset_fingerprint: DatasetFingerprint,
    training_config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Atomically save a trusted joblib artifact and its verifiable sidecar manifest."""

    output = Path(path)
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type must not be empty.")
    if not isinstance(dataset_fingerprint, DatasetFingerprint):
        raise TypeError("dataset_fingerprint must be a DatasetFingerprint.")
    if not isinstance(training_config, dict):
        raise TypeError("training_config must be a dictionary.")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output = manifest_path(output)
    token = uuid4().hex
    model_tmp = output.with_name(f".{output.name}.{token}.tmp")
    manifest_tmp = manifest_output.with_name(f".{manifest_output.name}.{token}.tmp")

    try:
        try:
            setattr(model, "artifact_dataset_fingerprint", dataset_fingerprint.fingerprint)
        except (AttributeError, TypeError):
            # Immutable/builtin payloads still use the sidecar contract; fitted model
            # classes expose this marker for stronger serving-time consistency checks.
            pass
        joblib.dump(model, model_tmp)
        checksum = sha256_file(model_tmp)
        code_sha, code_dirty = _git_state(output.parent)
        manifest = ArtifactManifest(
            artifact_format_version=ARTIFACT_FORMAT_VERSION,
            artifact_id=f"{model_type}-{checksum[:12]}",
            artifact_sha256=checksum,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_type=model_type,
            model_class=f"{type(model).__module__}.{type(model).__qualname__}",
            project_version=_package_version("book-recommender"),
            python_version=platform.python_version(),
            dependencies={name: _package_version(name) for name in TRACKED_DEPENDENCIES},
            code_sha=code_sha,
            code_dirty=code_dirty,
            dataset=asdict(dataset_fingerprint),
            training_config=_json_safe(training_config),
            metrics=_json_safe(metrics or {}),
        )
        # Validate the exact sidecar contract before publishing either file.
        _parse_manifest(asdict(manifest))
        manifest_tmp.write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

        os.replace(model_tmp, output)
        os.replace(manifest_tmp, manifest_output)
        return manifest
    finally:
        model_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def load_model_bundle(
    path: str | Path,
    *,
    expected_type: type | None = None,
    expected_model_type: str | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_training_config: dict[str, Any] | None = None,
    enforce_runtime_compatibility: bool = True,
) -> tuple[Any, ArtifactManifest]:
    """Verify checksum and compatibility before deserializing a trusted local artifact."""

    artifact_path = Path(path)
    sidecar = manifest_path(artifact_path)
    if not artifact_path.is_file():
        raise ArtifactError(f"Model artifact not found: {artifact_path}")
    if not sidecar.is_file():
        raise ArtifactIntegrityError(f"Artifact manifest not found: {sidecar}")

    try:
        payload = json.loads(
            sidecar.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
        manifest = _parse_manifest(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"Invalid artifact manifest: {sidecar}") from exc

    if manifest.artifact_format_version != ARTIFACT_FORMAT_VERSION:
        raise ArtifactCompatibilityError(
            "Unsupported artifact format version: "
            f"{manifest.artifact_format_version}; expected {ARTIFACT_FORMAT_VERSION}."
        )

    if enforce_runtime_compatibility:
        _validate_runtime_compatibility(manifest)
    if expected_model_type and manifest.model_type != expected_model_type:
        raise ArtifactCompatibilityError(
            f"Artifact model_type is {manifest.model_type!r}, expected {expected_model_type!r}."
        )
    if expected_type is not None:
        expected_class = f"{expected_type.__module__}.{expected_type.__qualname__}"
        if manifest.model_class != expected_class:
            raise ArtifactCompatibilityError(
                f"Artifact model_class is {manifest.model_class!r}, expected {expected_class!r}."
            )
    if expected_dataset_fingerprint:
        actual_fingerprint = str(manifest.dataset.get("fingerprint", ""))
        if actual_fingerprint != expected_dataset_fingerprint:
            raise ArtifactCompatibilityError(
                "Artifact dataset fingerprint does not match the active dataset."
            )
    if expected_training_config:
        for key, expected in expected_training_config.items():
            actual = manifest.training_config.get(key)
            if actual != expected:
                raise ArtifactCompatibilityError(
                    f"Artifact training config {key!r} is {actual!r}, expected {expected!r}."
                )

    # joblib/pickle can execute code. Only immutable, repository-controlled artifacts
    # should ever reach this point. Read a single byte snapshot so neither path replacement
    # nor an in-place write can change bytes between verification and deserialization.
    with artifact_path.open("rb") as artifact_handle:
        artifact_bytes = artifact_handle.read()
    if hashlib.sha256(artifact_bytes).hexdigest() != manifest.artifact_sha256:
        raise ArtifactIntegrityError(
            f"Artifact checksum mismatch for {artifact_path}; refuse to deserialize it."
        )
    model = joblib.load(io.BytesIO(artifact_bytes))
    actual_model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if actual_model_class != manifest.model_class:
        raise ArtifactCompatibilityError(
            "Artifact payload class does not match the class declared by its manifest."
        )
    if expected_type is not None and not isinstance(model, expected_type):
        raise ArtifactCompatibilityError(
            f"Artifact contains {type(model).__name__}, expected {expected_type.__name__}."
        )
    return model, manifest


def manifest_path(path: str | Path) -> Path:
    artifact_path = Path(path)
    return artifact_path.with_suffix(artifact_path.suffix + MANIFEST_SUFFIX)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_dataframe(dataframe: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    column_order = sorted(dataframe.columns, key=str)
    columns = [str(column) for column in column_order]
    digest.update(json.dumps(columns, separators=(",", ":")).encode("utf-8"))
    digest.update(
        json.dumps(_schema(dataframe), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(str(len(dataframe)).encode("ascii"))
    if dataframe.empty or not columns:
        return digest.hexdigest()

    canonical = dataframe.loc[:, column_order]
    row_hashes = pd.util.hash_pandas_object(canonical, index=False, categorize=True).to_numpy(
        dtype=np.uint64,
        copy=True,
    )
    row_hashes.sort()
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _schema(dataframe: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _git_state(start: Path) -> tuple[str | None, bool | None]:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return sha or None, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _parse_manifest(payload: object) -> ArtifactManifest:
    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be an object.")
    expected_fields = set(ArtifactManifest.__dataclass_fields__)
    if set(payload) != expected_fields:
        missing = sorted(expected_fields.difference(payload))
        extra = sorted(set(payload).difference(expected_fields))
        raise ValueError(f"Manifest fields mismatch; missing={missing}, extra={extra}.")
    manifest = ArtifactManifest(**payload)
    _require_int(manifest.artifact_format_version, "artifact_format_version", minimum=1)
    for name in (
        "artifact_id",
        "created_at",
        "model_type",
        "model_class",
        "project_version",
        "python_version",
    ):
        _require_non_empty_string(getattr(manifest, name), name)
    _require_sha256(manifest.artifact_sha256, "artifact_sha256")
    parsed_time = datetime.fromisoformat(manifest.created_at.replace("Z", "+00:00"))
    if parsed_time.tzinfo is None:
        raise ValueError("created_at must include a timezone.")
    _require_string_mapping(manifest.dependencies, "dependencies")
    if manifest.code_sha is not None:
        _require_non_empty_string(manifest.code_sha, "code_sha")
        if len(manifest.code_sha) not in {40, 64} or not _is_hex(manifest.code_sha):
            raise ValueError("code_sha must be a 40- or 64-character hexadecimal digest.")
    if manifest.code_dirty is not None and not isinstance(manifest.code_dirty, bool):
        raise ValueError("code_dirty must be a boolean or null.")
    if not isinstance(manifest.training_config, dict) or not isinstance(manifest.metrics, dict):
        raise ValueError("training_config and metrics must be objects.")
    _validate_json_tree(manifest.training_config, "training_config")
    _validate_json_tree(manifest.metrics, "metrics")
    _validate_dataset_manifest(manifest.dataset)
    return manifest


def _validate_dataset_manifest(dataset: object) -> None:
    expected = {
        "fingerprint",
        "books_rows",
        "ratings_rows",
        "users_rows",
        "books_schema",
        "ratings_schema",
        "users_schema",
    }
    if not isinstance(dataset, dict) or set(dataset) != expected:
        raise ValueError("dataset manifest has an invalid schema.")
    _require_sha256(dataset["fingerprint"], "dataset.fingerprint")
    for name in ("books_rows", "ratings_rows", "users_rows"):
        _require_int(dataset[name], f"dataset.{name}", minimum=0)
    for name in ("books_schema", "ratings_schema", "users_schema"):
        _require_string_mapping(dataset[name], f"dataset.{name}")
    _validate_json_tree(dataset, "dataset")


def _validate_runtime_compatibility(manifest: ArtifactManifest) -> None:
    running_python = platform.python_version()
    if _major_minor(manifest.python_version) != _major_minor(running_python):
        raise ArtifactCompatibilityError(
            f"Artifact Python {manifest.python_version} is incompatible with runtime {running_python}."
        )
    running_project = _package_version("book-recommender")
    if (
        manifest.project_version != "unknown"
        and running_project != "unknown"
        and manifest.project_version != running_project
    ):
        raise ArtifactCompatibilityError(
            f"Artifact project version {manifest.project_version} is incompatible with "
            f"runtime {running_project}."
        )
    for distribution in TRACKED_DEPENDENCIES:
        trained = manifest.dependencies.get(distribution)
        running = _package_version(distribution)
        if not trained or trained == "unknown" or running == "unknown":
            raise ArtifactCompatibilityError(
                f"Cannot verify runtime compatibility for dependency {distribution}."
            )
        if _major_minor(trained) != _major_minor(running):
            raise ArtifactCompatibilityError(
                f"Artifact {distribution} {trained} is incompatible with runtime {running}."
            )


def _major_minor(version: str) -> tuple[int, int]:
    pieces = version.split(".")
    try:
        return int(pieces[0]), int(pieces[1])
    except (IndexError, ValueError) as exc:
        raise ArtifactCompatibilityError(f"Invalid version string in manifest: {version!r}.") from exc


def _require_non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or not _is_hex(value):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest.")


def _require_int(value: object, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _require_string_mapping(value: object, name: str) -> None:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{name} must be an object containing only string values.")


def _is_hex(value: str) -> bool:
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_json_tree(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} must contain only string object keys.")
            _validate_json_tree(item, f"{name}.{key}")
        return
    raise ValueError(f"{name} contains a non-JSON value of type {type(value).__name__}.")


def _reject_json_constant(value: str):
    raise ValueError(f"Invalid non-finite JSON constant: {value}")
