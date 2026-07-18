"""Deterministic resource measurements for recommender benchmarks.

Process RSS is deliberately not measured here: allocator reuse and benchmark order can
make it look as though a later configuration uses more or less memory than it actually
does.  Serialized size and sparse-matrix payload are stable, configuration-comparable
proxies that can also be written to JSON benchmark artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from io import BytesIO
import math
import numbers
from time import perf_counter
from typing import Any, TypeVar

import joblib
import numpy as np
from scipy import sparse


T = TypeVar("T")
R = TypeVar("R")


def timed_call(call: Callable[..., R], /, *args: Any, **kwargs: Any) -> tuple[R, float]:
    """Call ``call`` and return its result plus non-negative wall time in milliseconds."""

    if not callable(call):
        raise TypeError("call must be callable.")

    started_at = perf_counter()
    result = call(*args, **kwargs)
    elapsed_ms = max(0.0, (perf_counter() - started_at) * 1_000.0)
    if not math.isfinite(elapsed_ms):  # Defensive guard for benchmark artifact validity.
        raise RuntimeError("Measured elapsed time is not finite.")
    return result, float(elapsed_ms)


def summarize_latency_ms(samples_ms: Iterable[float]) -> dict[str, float | int]:
    """Return a JSON-safe latency summary after validating every sample.

    Empty, negative, and non-finite samples are rejected rather than silently producing
    misleading resource evidence.
    """

    samples = np.asarray(list(samples_ms), dtype="float64")
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("latency samples must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("latency samples must be finite.")
    if np.any(samples < 0.0):
        raise ValueError("latency samples must be non-negative.")

    return {
        "sample_count": int(samples.size),
        "min_ms": float(np.min(samples)),
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "max_ms": float(np.max(samples)),
    }


def measure_callable_latencies_ms(
    call: Callable[[T], Any],
    inputs: Iterable[T],
    *,
    warmup_calls: int = 1,
) -> list[float]:
    """Warm a callable, then measure one invocation for every supplied input.

    The helper intentionally returns raw samples.  Callers can aggregate samples across
    validation folds before passing them to :func:`summarize_latency_ms`.
    """

    if not callable(call):
        raise TypeError("call must be callable.")
    if (
        isinstance(warmup_calls, bool)
        or not isinstance(warmup_calls, numbers.Integral)
        or int(warmup_calls) < 0
    ):
        raise ValueError("warmup_calls must be a non-negative integer.")

    materialized_inputs = list(inputs)
    if not materialized_inputs:
        raise ValueError("inputs must contain at least one value.")

    for index in range(int(warmup_calls)):
        call(materialized_inputs[index % len(materialized_inputs)])

    samples: list[float] = []
    for value in materialized_inputs:
        _, elapsed_ms = timed_call(call, value)
        samples.append(elapsed_ms)
    return samples


def model_serialized_size_bytes(model: object) -> int:
    """Measure the uncompressed in-memory joblib artifact size without a temp file."""

    buffer = BytesIO()
    joblib.dump(model, buffer, compress=0)
    size_bytes = int(buffer.getbuffer().nbytes)
    if size_bytes <= 0:
        raise RuntimeError("Serialized model size must be greater than zero.")
    return size_bytes


def sparse_matrix_payload(matrix: object) -> dict[str, object]:
    """Describe the structural byte payload of a SciPy sparse matrix.

    Measurements use CSR's three storage arrays.  The hybrid recommender already stores
    both measured matrices as CSR, while converting another sparse format makes this
    function predictable for tests and future implementations.
    """

    if not sparse.issparse(matrix):
        raise TypeError("matrix must be a SciPy sparse matrix.")

    source_format = str(matrix.format)
    measured = matrix.tocsr(copy=False)
    rows, columns = (int(value) for value in measured.shape)
    nnz = int(measured.nnz)
    data_bytes = int(measured.data.nbytes)
    indices_bytes = int(measured.indices.nbytes)
    indptr_bytes = int(measured.indptr.nbytes)
    payload_bytes = data_bytes + indices_bytes + indptr_bytes
    capacity = rows * columns

    return {
        "shape": [rows, columns],
        "nnz": nnz,
        "density": float(nnz / capacity) if capacity else 0.0,
        "source_format": source_format,
        "measured_format": "csr",
        "data_dtype": str(measured.data.dtype),
        "index_dtype": str(measured.indices.dtype),
        "data_bytes": data_bytes,
        "indices_bytes": indices_bytes,
        "indptr_bytes": indptr_bytes,
        "payload_bytes": payload_bytes,
    }


def hybrid_sparse_payload(model: object) -> dict[str, object]:
    """Measure fitted content and collaborative matrices of a hybrid recommender."""

    content_model = getattr(model, "content_model", None)
    collaborative_model = getattr(model, "collaborative_model", None)
    content_matrix = getattr(content_model, "matrix", None)
    collaborative_matrix = getattr(collaborative_model, "item_user_matrix", None)
    if content_matrix is None or collaborative_matrix is None:
        raise RuntimeError("Hybrid recommender must be fitted before measuring matrix payload.")

    content = sparse_matrix_payload(content_matrix)
    collaborative = sparse_matrix_payload(collaborative_matrix)
    return {
        "content": content,
        "collaborative": collaborative,
        "total_payload_bytes": int(content["payload_bytes"])
        + int(collaborative["payload_bytes"]),
    }
