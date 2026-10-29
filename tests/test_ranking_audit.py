"""Independent resampling arithmetic and trust-boundary checks for local audit."""
import hashlib

import numpy as np
import pytest

from scripts.audit_ranking_assessment import bootstrap_mean, verify_file_fingerprint


def test_independent_bootstrap_matches_fixed_draw_arithmetic():
    # Six draw means for seed11: .0625, .125, .375, -.25, 1.625, .4375.
    # The interpolated 2.5th/97.5th percentiles are computed from this fixed list.
    candidate = np.array([-1., .25, .5, 2.])
    actual = bootstrap_mean(candidate, np.zeros(4), draws=6, seed=11)
    np.testing.assert_array_equal(actual, [-.2109375, 1.4765625])
    # Pairing must cancel a shared user offset, rather than bootstrap arms separately.
    offset = np.array([100., -8., 30., 4.])
    np.testing.assert_array_equal(
        bootstrap_mean(candidate+offset, offset, draws=6, seed=11), actual,
    )


def test_fingerprint_rejects_same_size_corruption_and_truncation(tmp_path):
    path = tmp_path / "result.json"
    original = b'{"metric":0.1}\n'
    path.write_bytes(original)
    trusted = {"bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()}
    verify_file_fingerprint(path, trusted)
    path.write_bytes(original.replace(b"0.1", b"0.9"))
    with pytest.raises(ValueError, match="SHA-256 differs"):
        verify_file_fingerprint(path, trusted)
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="size differs"):
        verify_file_fingerprint(path, trusted)
