import numpy as np
import torch

from dagger_harmonics.models.recreation import DAGGER
from dagger_harmonics.train import (
    _check_early_stop,
    _eval_epoch,
    _prepare_coords_and_target,
    _prepare_omni,
)


# ---------------------------------------------------------------------------
# Shared fake dataset helpers
# ---------------------------------------------------------------------------


class _FakeDS:
    def __init__(self, records, features):
        self._data = records
        self.features = features

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def _make_fake_datasets(n_records=3, T=4, n_omni=5, n_stations=6):
    omni_records = [np.ones((T, n_omni), dtype=np.float32) for _ in range(n_records)]
    sm_records = [_sm_record(n=n_stations) for _ in range(n_records)]
    omni_ds = _FakeDS(omni_records, [f"f{i}" for i in range(n_omni)])
    supermag_ds = _FakeDS(sm_records, SM_FEATURES)
    omni_mean = np.zeros(n_omni, dtype=np.float32)
    omni_std = np.ones(n_omni, dtype=np.float32)
    return omni_ds, supermag_ds, omni_mean, omni_std

SM_FEATURES = ["MAGLAT", "MLT", "dbe_nez", "dbn_nez", "ddbe_dt", "ddbn_dt"]


def _sm_record(maglat=70.0, mlt=12.0, dbe=3.0, dbn=4.0, n=4):
    """(N, 6) SuperMAG record with uniform values across stations."""
    arr = np.zeros((n, len(SM_FEATURES)), dtype=np.float32)
    arr[:, SM_FEATURES.index("MAGLAT")] = maglat
    arr[:, SM_FEATURES.index("MLT")] = mlt
    arr[:, SM_FEATURES.index("dbe_nez")] = dbe
    arr[:, SM_FEATURES.index("dbn_nez")] = dbn
    return arr


# ---------------------------------------------------------------------------
# _prepare_coords_and_target
# ---------------------------------------------------------------------------


def test_mcolat_rad_equals_radians_of_90_minus_maglat():
    mcolat, _, _, _ = _prepare_coords_and_target(_sm_record(maglat=70.0), SM_FEATURES)
    np.testing.assert_allclose(mcolat.numpy(), np.radians(20.0), rtol=1e-5)


def test_mlt_rad_equals_mlt_hours_times_pi_over_12():
    _, mlt_rad, _, _ = _prepare_coords_and_target(_sm_record(mlt=6.0), SM_FEATURES)
    np.testing.assert_allclose(mlt_rad.numpy(), np.pi / 2, rtol=1e-5)


def test_dbh_is_euclidean_magnitude_of_dbe_dbn():
    _, _, dbh, _ = _prepare_coords_and_target(_sm_record(dbe=3.0, dbn=4.0), SM_FEATURES)
    np.testing.assert_allclose(dbh.numpy(), 5.0, rtol=1e-5)


def test_prepare_coords_returns_tensors_of_shape_n_stations():
    n = 7
    mcolat, mlt_rad, dbh, valid = _prepare_coords_and_target(_sm_record(n=n), SM_FEATURES)
    for t in (mcolat, mlt_rad, dbh, valid):
        assert isinstance(t, torch.Tensor)
        assert t.shape == (n,)


def test_valid_mask_excludes_nan_maglat():
    arr = _sm_record(n=4)
    arr[1, SM_FEATURES.index("MAGLAT")] = np.nan
    _, _, _, valid = _prepare_coords_and_target(arr, SM_FEATURES)
    assert valid.sum() == 3 and not valid[1]


def test_valid_mask_excludes_nan_dbh():
    arr = _sm_record(n=4)
    arr[2, SM_FEATURES.index("dbe_nez")] = np.nan
    _, _, _, valid = _prepare_coords_and_target(arr, SM_FEATURES)
    assert valid.sum() == 3 and not valid[2]


# ---------------------------------------------------------------------------
# _prepare_omni
# ---------------------------------------------------------------------------


def test_omni_is_z_scored_by_mean_and_std():
    record = np.full((5, 3), 3.0, dtype=np.float32)
    mean = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    std = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    out = _prepare_omni(record, mean, std)
    np.testing.assert_allclose(out.numpy(), 1.0, rtol=1e-5)


def test_omni_returns_float32_tensor_with_shape_T_by_n_features():
    T, n = 10, 5
    record = np.ones((T, n), dtype=np.float32)
    out = _prepare_omni(record, np.zeros(n), np.ones(n))
    assert out.dtype == torch.float32
    assert out.shape == (T, n)


def test_omni_std_clamp_prevents_division_by_zero():
    record = np.ones((3, 2), dtype=np.float32)
    out = _prepare_omni(record, np.zeros(2), np.zeros(2))  # std=0 must not raise
    assert torch.isfinite(out).all()


def test_omni_nan_inputs_replaced_with_zero():
    record = np.full((4, 3), np.nan, dtype=np.float32)
    out = _prepare_omni(record, np.zeros(3), np.ones(3))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# _eval_epoch
# ---------------------------------------------------------------------------


def test_eval_epoch_returns_finite_float():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    result = _eval_epoch(model, [0, 1, 2], omni_ds, supermag_ds, omni_mean, omni_std)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_eval_epoch_restores_model_to_train_mode():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    _eval_epoch(model, [0, 1], omni_ds, supermag_ds, omni_mean, omni_std)
    assert model.training


def test_eval_epoch_empty_indices_returns_zero():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    result = _eval_epoch(model, [], omni_ds, supermag_ds, omni_mean, omni_std)
    assert result == 0.0


# ---------------------------------------------------------------------------
# _check_early_stop
# ---------------------------------------------------------------------------


def test_check_early_stop_improvement_resets_counter():
    best, count, stop = _check_early_stop(val_loss=1.0, best=2.0, no_improve=3, patience=5)
    assert best == 1.0
    assert count == 0
    assert stop is False


def test_check_early_stop_no_improvement_increments_counter():
    best, count, stop = _check_early_stop(val_loss=2.0, best=1.0, no_improve=2, patience=5)
    assert best == 1.0
    assert count == 3
    assert stop is False


def test_check_early_stop_triggers_when_patience_exceeded():
    _, _, stop = _check_early_stop(val_loss=2.0, best=1.0, no_improve=4, patience=5)
    assert stop is True
