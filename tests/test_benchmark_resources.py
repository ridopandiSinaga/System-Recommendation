from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from book_recommender.benchmark_resources import hybrid_sparse_payload
from book_recommender.benchmark_resources import measure_callable_latencies_ms
from book_recommender.benchmark_resources import model_serialized_size_bytes
from book_recommender.benchmark_resources import sparse_matrix_payload
from book_recommender.benchmark_resources import summarize_latency_ms
from book_recommender.benchmark_resources import timed_call


def test_timed_call_returns_result_and_finite_non_negative_latency():
    result, elapsed_ms = timed_call(lambda left, right: left + right, 2, right=3)

    assert result == 5
    assert np.isfinite(elapsed_ms)
    assert elapsed_ms >= 0.0


def test_latency_summary_is_json_safe_and_rejects_invalid_samples():
    summary = summarize_latency_ms([1.0, 2.0, 3.0, 4.0])

    assert summary == {
        "sample_count": 4,
        "min_ms": 1.0,
        "mean_ms": 2.5,
        "p50_ms": 2.5,
        "p95_ms": pytest.approx(3.85),
        "max_ms": 4.0,
    }
    for invalid in ([], [-0.1], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            summarize_latency_ms(invalid)


def test_callable_latency_helper_warms_then_returns_each_measured_sample():
    calls: list[int] = []

    def record(value: int) -> int:
        calls.append(value)
        return value * 2

    samples = measure_callable_latencies_ms(record, [10, 20], warmup_calls=3)

    assert calls == [10, 20, 10, 10, 20]
    assert len(samples) == 2
    assert all(np.isfinite(value) and value >= 0.0 for value in samples)


@pytest.mark.parametrize("warmup_calls", [-1, 1.5, True])
def test_callable_latency_helper_rejects_invalid_warmup_count(warmup_calls):
    with pytest.raises(ValueError, match="warmup_calls"):
        measure_callable_latencies_ms(lambda value: value, [1], warmup_calls=warmup_calls)


def test_model_serialized_size_is_positive_without_creating_a_file():
    size_bytes = model_serialized_size_bytes({"weights": np.arange(20, dtype="float32")})

    assert isinstance(size_bytes, int)
    assert size_bytes > 0


def test_sparse_payload_reports_csr_storage_arrays_and_shape():
    matrix = csr_matrix([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]], dtype="float32")
    payload = sparse_matrix_payload(matrix)

    assert payload["shape"] == [2, 3]
    assert payload["nnz"] == 3
    assert payload["density"] == 0.5
    assert payload["source_format"] == "csr"
    assert payload["payload_bytes"] == (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    )


def test_sparse_payload_normalizes_other_sparse_formats_to_csr():
    matrix = coo_matrix(([1.0, 2.0], ([0, 1], [1, 0])), shape=(2, 2))

    payload = sparse_matrix_payload(matrix)

    assert payload["source_format"] == "coo"
    assert payload["measured_format"] == "csr"
    assert payload["shape"] == [2, 2]
    assert payload["nnz"] == 2


def test_hybrid_sparse_payload_reports_both_components_and_total():
    content = csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    collaborative = csr_matrix([[0.0, 3.0], [1.0, 0.0]])
    model = SimpleNamespace(
        content_model=SimpleNamespace(matrix=content),
        collaborative_model=SimpleNamespace(item_user_matrix=collaborative),
    )

    payload = hybrid_sparse_payload(model)

    assert payload["content"]["shape"] == [2, 2]
    assert payload["collaborative"]["shape"] == [2, 2]
    assert payload["total_payload_bytes"] == (
        payload["content"]["payload_bytes"]
        + payload["collaborative"]["payload_bytes"]
    )


def test_hybrid_sparse_payload_requires_a_fitted_model():
    with pytest.raises(RuntimeError, match="must be fitted"):
        hybrid_sparse_payload(SimpleNamespace(content_model=None, collaborative_model=None))
