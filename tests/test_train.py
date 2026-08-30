import numpy as np
import pytest
import torch

from dagger_harmonics.models.recreation import DAGGER
from dagger_harmonics.train import (
    _check_early_stop,
    _predict_and_target,
    _prepare_coords_and_target,
    _prepare_omni,
    _run_epoch,
    get_device,
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
    mcolat, _, _, _, _ = _prepare_coords_and_target(
        _sm_record(maglat=70.0), SM_FEATURES
    )
    np.testing.assert_allclose(mcolat.numpy(), np.radians(20.0), rtol=1e-5)


def test_mlt_rad_equals_mlt_hours_times_pi_over_12():
    _, mlt_rad, _, _, _ = _prepare_coords_and_target(_sm_record(mlt=6.0), SM_FEATURES)
    np.testing.assert_allclose(mlt_rad.numpy(), np.pi / 2, rtol=1e-5)


def test_prepare_coords_returns_dbe_separately():
    _, _, dbe, _, _ = _prepare_coords_and_target(_sm_record(dbe=3.0), SM_FEATURES)
    np.testing.assert_allclose(dbe.numpy(), 3.0, rtol=1e-5)


def test_prepare_coords_returns_dbn_separately():
    _, _, _, dbn, _ = _prepare_coords_and_target(_sm_record(dbn=4.0), SM_FEATURES)
    np.testing.assert_allclose(dbn.numpy(), 4.0, rtol=1e-5)


def test_prepare_coords_returns_tensors_of_shape_n_stations():
    n = 7
    mcolat, mlt_rad, dbe, dbn, valid = _prepare_coords_and_target(
        _sm_record(n=n), SM_FEATURES
    )
    for t in (mcolat, mlt_rad, dbe, dbn, valid):
        assert isinstance(t, torch.Tensor)
        assert t.shape == (n,)


def test_valid_mask_excludes_nan_maglat():
    arr = _sm_record(n=4)
    arr[1, SM_FEATURES.index("MAGLAT")] = np.nan
    _, _, _, _, valid = _prepare_coords_and_target(arr, SM_FEATURES)
    assert valid.sum() == 3 and not valid[1]


def test_valid_mask_excludes_station_with_both_components_nan():
    # Station is invalid only when BOTH dbe and dbn are NaN
    arr = _sm_record(n=4)
    arr[2, SM_FEATURES.index("dbe_nez")] = np.nan
    arr[2, SM_FEATURES.index("dbn_nez")] = np.nan
    _, _, _, _, valid = _prepare_coords_and_target(arr, SM_FEATURES)
    assert valid.sum() == 3 and not valid[2]


def test_valid_mask_includes_station_with_only_dbn_nan():
    # Station with dbn=NaN but finite dbe is still valid (paper §2.3: zero that component)
    arr = _sm_record(n=4)
    arr[2, SM_FEATURES.index("dbn_nez")] = np.nan
    _, _, _, _, valid = _prepare_coords_and_target(arr, SM_FEATURES)
    assert valid.sum() == 4  # all stations valid


def test_nan_dbe_replaced_with_zero_in_output():
    arr = _sm_record(n=3, dbe=5.0)
    arr[1, SM_FEATURES.index("dbe_nez")] = np.nan
    _, _, dbe, _, _ = _prepare_coords_and_target(arr, SM_FEATURES)
    assert dbe[1].item() == 0.0


def test_nan_dbn_replaced_with_zero_in_output():
    arr = _sm_record(n=3, dbn=7.0)
    arr[0, SM_FEATURES.index("dbn_nez")] = np.nan
    _, _, _, dbn, _ = _prepare_coords_and_target(arr, SM_FEATURES)
    assert dbn[0].item() == 0.0


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
# _predict_and_target
# ---------------------------------------------------------------------------


def test_predict_and_target_returns_pred_and_target_same_shape():
    model = DAGGER(input_size=5)
    omni = torch.ones(1, 4, 5)
    mcolat, mlt, dbe, dbn, valid = _prepare_coords_and_target(
        _sm_record(n=6), SM_FEATURES
    )
    pred, target = _predict_and_target(model, omni, mcolat, mlt, dbe, dbn, valid)
    assert pred.shape == target.shape == (6, 2)


def test_predict_and_target_builds_target_from_dbe_dbn():
    model = DAGGER(input_size=5)
    omni = torch.ones(1, 4, 5)
    mcolat, mlt, dbe, dbn, valid = _prepare_coords_and_target(
        _sm_record(n=3, dbe=3.0, dbn=4.0), SM_FEATURES
    )
    _, target = _predict_and_target(model, omni, mcolat, mlt, dbe, dbn, valid)
    np.testing.assert_allclose(target[:, 0].numpy(), 3.0)
    np.testing.assert_allclose(target[:, 1].numpy(), 4.0)


def test_predict_and_target_restricts_to_valid_stations():
    model = DAGGER(input_size=5)
    omni = torch.ones(1, 4, 5)
    arr = _sm_record(n=4)
    arr[1, SM_FEATURES.index("MAGLAT")] = np.nan
    mcolat, mlt, dbe, dbn, valid = _prepare_coords_and_target(arr, SM_FEATURES)
    pred, target = _predict_and_target(model, omni, mcolat, mlt, dbe, dbn, valid)
    assert pred.shape == target.shape == (3, 2)


# ---------------------------------------------------------------------------
# _run_epoch (shared by training and validation passes)
# ---------------------------------------------------------------------------


def test_run_epoch_eval_mode_returns_dict_of_finite_metrics():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    result = _run_epoch(model, [0, 1, 2], omni_ds, supermag_ds, omni_mean, omni_std)
    assert set(result) == {"mae", "mse", "rmse"}
    assert all(isinstance(v, float) and np.isfinite(v) for v in result.values())


def test_run_epoch_rmse_is_sqrt_of_mse():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    result = _run_epoch(model, [0, 1, 2], omni_ds, supermag_ds, omni_mean, omni_std)
    assert result["rmse"] == pytest.approx(result["mse"] ** 0.5, rel=1e-4)


def test_run_epoch_without_optimizer_leaves_model_in_eval_mode():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    _run_epoch(model, [0, 1], omni_ds, supermag_ds, omni_mean, omni_std)
    assert not model.training


def test_run_epoch_empty_indices_returns_all_zero():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    result = _run_epoch(model, [], omni_ds, supermag_ds, omni_mean, omni_std)
    assert result == {"mae": 0.0, "mse": 0.0, "rmse": 0.0}


def test_run_epoch_without_optimizer_does_not_change_weights():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    before = [p.clone() for p in model.parameters()]
    _run_epoch(model, [0, 1, 2], omni_ds, supermag_ds, omni_mean, omni_std)
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_run_epoch_with_optimizer_leaves_model_in_train_mode():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    optimizer = torch.optim.Adam(model.parameters())
    _run_epoch(
        model, [0, 1], omni_ds, supermag_ds, omni_mean, omni_std, optimizer=optimizer
    )
    assert model.training


def test_run_epoch_with_optimizer_updates_weights():
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    optimizer = torch.optim.Adam(model.parameters())
    before = [p.clone() for p in model.parameters()]
    _run_epoch(
        model,
        [0, 1, 2],
        omni_ds,
        supermag_ds,
        omni_mean,
        omni_std,
        optimizer=optimizer,
    )
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


# ---------------------------------------------------------------------------
# get_device / hardware-agnostic dispatch
# ---------------------------------------------------------------------------


def test_get_device_returns_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device().type == "cuda"


def test_get_device_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device().type == "cpu"


def test_run_epoch_moves_model_inputs_to_requested_device():
    # No GPU needed to exercise the dispatch path: CPU is itself a valid
    # explicit `device`, so this pins the contract that `_run_epoch` honors
    # whatever device it's given rather than hardcoding CPU tensors.
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5)
    device = torch.device("cpu")
    result = _run_epoch(
        model, [0, 1, 2], omni_ds, supermag_ds, omni_mean, omni_std, device=device
    )
    assert all(np.isfinite(v) for v in result.values())
    assert next(model.parameters()).device == device


def test_run_epoch_defaults_device_to_model_device():
    # When `device` is omitted, tensors should follow the model's own
    # device rather than assuming CPU, so a pre-moved model still works.
    omni_ds, supermag_ds, omni_mean, omni_std = _make_fake_datasets(n_omni=5)
    model = DAGGER(input_size=5).to(torch.device("cpu"))
    result = _run_epoch(model, [0, 1, 2], omni_ds, supermag_ds, omni_mean, omni_std)
    assert all(np.isfinite(v) for v in result.values())


# ---------------------------------------------------------------------------
# _check_early_stop
# ---------------------------------------------------------------------------


def test_check_early_stop_improvement_resets_counter():
    best, count, stop = _check_early_stop(
        val_loss=1.0, best=2.0, no_improve=3, patience=5
    )
    assert best == 1.0
    assert count == 0
    assert stop is False


def test_check_early_stop_no_improvement_increments_counter():
    best, count, stop = _check_early_stop(
        val_loss=2.0, best=1.0, no_improve=2, patience=5
    )
    assert best == 1.0
    assert count == 3
    assert stop is False


def test_check_early_stop_triggers_when_patience_exceeded():
    _, _, stop = _check_early_stop(val_loss=2.0, best=1.0, no_improve=4, patience=5)
    assert stop is True
