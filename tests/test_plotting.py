import numpy as np

from analysis.plotting import SUPERMAG_COLS, supermag_col, supermag_dbH


def _fake_supermag(n=3):
    """(N, 6) array with distinct, known column values."""
    arr = np.zeros((n, 6))
    arr[:, SUPERMAG_COLS["dbe"]] = [3.0, 0.0, -3.0]
    arr[:, SUPERMAG_COLS["dbn"]] = [4.0, 5.0, 4.0]
    return arr


def test_supermag_col_extracts_named_column():
    arr = _fake_supermag()
    np.testing.assert_array_equal(supermag_col(arr, "dbe"), [3.0, 0.0, -3.0])


def test_supermag_col_reduces_3d_to_latest_timestep():
    arr = np.stack([np.zeros((3, 6)), _fake_supermag()])  # (T=2, N=3, 6)
    np.testing.assert_array_equal(supermag_col(arr, "dbn"), [4.0, 5.0, 4.0])


def test_supermag_dbH_is_euclidean_magnitude():
    arr = _fake_supermag()
    np.testing.assert_allclose(supermag_dbH(arr), [5.0, 5.0, 5.0])
