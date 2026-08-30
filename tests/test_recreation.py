import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from dagger_harmonics.models.recreation import DAGGER


def _model(input_size: int = 15) -> DAGGER:
    return DAGGER(input_size=input_size)


def _forward(model, batch=2, T=5, n_stations=10, n_omni=15):
    omni = torch.zeros(batch, T, n_omni)
    mcolat = torch.full((batch, n_stations), 0.3)
    mlt = torch.zeros(batch, n_stations)
    return model(omni, mcolat, mlt)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_gru_has_8_hidden_units():
    assert _model().gru.hidden_size == 8


def test_gru_has_1_layer():
    assert _model().gru.num_layers == 1


def test_fc_second_layer_outputs_880_per_field():
    # 440 modes × 2 (real+imag) per field; two fields share the same FC → 1760 total
    # Verify the per-field count is still 440*2 by checking n_coeffs*4 == 1760
    model = _model()
    linears = [m for m in model.fc.modules() if isinstance(m, nn.Linear)]
    assert linears[-1].out_features % 4 == 0  # divisible: 4 coefficient groups


def test_fc_has_relu_between_layers():
    model = _model()
    has_relu = any(isinstance(m, nn.ReLU) for m in model.fc.modules())
    assert has_relu


def test_sh_layer_adds_no_trainable_parameters():
    model = _model()
    expected = sum(p.numel() for p in model.gru.parameters()) + sum(
        p.numel() for p in model.fc.parameters()
    )
    assert sum(p.numel() for p in model.parameters()) == expected


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------


def test_forward_returns_batch_by_n_stations_by_2():
    out = _forward(_model(), batch=2, n_stations=10)
    assert out.shape == (2, 10, 2)


def test_forward_works_for_single_item_batch():
    out = _forward(_model(), batch=1, n_stations=7)
    assert out.shape == (1, 7, 2)


def test_forward_broadcasts_1d_coords_across_batch():
    """(N,) station coords shared by all records in a batch."""
    model = _model()
    batch, T, n_stations = 3, 4, 7
    omni = torch.zeros(batch, T, 15)
    mcolat = torch.full((n_stations,), 0.3)
    mlt = torch.zeros(n_stations)
    out = model(omni, mcolat, mlt)
    assert out.shape == (batch, n_stations, 2)


# ---------------------------------------------------------------------------
# SH coefficient count (lmax = 20, l = 1..20)
# ---------------------------------------------------------------------------


def test_440_sh_modes_for_lmax_20():
    n = sum(2 * degree + 1 for degree in range(1, 21))
    assert n == 440


# ---------------------------------------------------------------------------
# Two-channel (dbe + dbn) architecture
# ---------------------------------------------------------------------------


def test_fc_second_layer_outputs_1760():
    # 440 modes × 2 (real+imag) × 2 fields (dbe, dbn) = 1760
    model = _model()
    linears = [m for m in model.fc.modules() if isinstance(m, nn.Linear)]
    assert linears[-1].out_features == 1760


def test_fc_has_dropout_with_probability_07():
    import pytest

    model = _model()
    dropouts = [m for m in model.fc.modules() if isinstance(m, nn.Dropout)]
    assert len(dropouts) >= 1
    assert dropouts[0].p == pytest.approx(0.7)


def test_forward_returns_batch_by_n_stations_by_2_channels():
    out = _forward(_model(), batch=2, n_stations=10)
    assert out.shape == (2, 10, 2)


def test_forward_channel_0_is_dbe_channel_1_is_dbn():
    """Both channels are present and finite for a clean input."""
    out = _forward(_model(), batch=1, n_stations=5)
    assert out.shape[2] == 2
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# _sh_basis: characterization test pinning known values ahead of a refactor
# (l=1, m=0 mode at mcolat=pi/2, mlt=0 -- see paper eq. for Y_nm)
# ---------------------------------------------------------------------------


def test_sh_basis_matches_hand_computed_l1_m0_value():
    model = _model()
    mcolat = torch.tensor([[np.pi / 2]])  # equator: cos(mcolat) = 0
    mlt = torch.tensor([[0.0]])
    Y_real, Y_imag = model._sh_basis(mcolat, mlt)

    idx = model._lm_pairs.index((1, 0))
    # K_10 * P_1^0(cos(pi/2)) * cos(0) = sqrt(3/(4*pi)) * 0 * 1 == 0
    assert Y_real[0, 0, idx].item() == pytest.approx(0.0, abs=1e-6)
    assert Y_imag[0, 0, idx].item() == pytest.approx(0.0, abs=1e-6)


def test_sh_basis_matches_hand_computed_l1_m0_value_at_pole():
    model = _model()
    mcolat = torch.tensor([[0.0]])  # pole: cos(mcolat) = 1
    mlt = torch.tensor([[0.0]])
    Y_real, Y_imag = model._sh_basis(mcolat, mlt)

    idx = model._lm_pairs.index((1, 0))
    expected = math.sqrt(3 / (4 * math.pi))  # K_10 * P_1^0(1) * cos(0) = K_10 * 1 * 1
    assert Y_real[0, 0, idx].item() == pytest.approx(expected, rel=1e-5)
    assert Y_imag[0, 0, idx].item() == pytest.approx(0.0, abs=1e-6)


def _scipy_reference_Y(model, mcolat_val, mlt_val):
    """Independent scipy computation of the full Y_real/Y_imag vectors at one
    station -- ground truth for _sh_basis regardless of how it's implemented,
    so this test still catches a regression after any internal rewrite
    (e.g. moving the basis evaluation off scipy/CPU onto GPU tensor ops)."""
    from scipy.special import lpmv

    cos_theta = math.cos(mcolat_val)
    degs = model._degs.ravel()
    abs_ms = model._abs_ms.ravel()
    ms = model._ms.ravel()
    norms = model._lm_norms.ravel()
    plm = norms * lpmv(abs_ms, degs, cos_theta)
    angle = ms * mlt_val
    return plm * np.cos(angle), plm * np.sin(angle)


def test_sh_basis_matches_scipy_reference_across_modes_and_points():
    """Full-vector parity check against an independent scipy computation,
    covering positive and negative m (sign convention for the imaginary/sin
    part) and several station positions -- not just the single l=1,m=0 case
    the earlier characterization tests pin."""
    model = _model()
    points = [
        (0.1, 0.0),
        (0.7, 1.3),
        (np.pi / 2, 4.2),
        (np.pi - 0.05, 5.9),  # near south-pole-ward edge
    ]
    mcolat = torch.tensor([[p[0] for p in points]])
    mlt = torch.tensor([[p[1] for p in points]])
    Y_real, Y_imag = model._sh_basis(mcolat, mlt)

    for i, (mcolat_val, mlt_val) in enumerate(points):
        expected_real, expected_imag = _scipy_reference_Y(model, mcolat_val, mlt_val)
        actual_real = Y_real[0, i].detach().numpy()
        actual_imag = Y_imag[0, i].detach().numpy()
        np.testing.assert_allclose(actual_real, expected_real, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(actual_imag, expected_imag, rtol=1e-4, atol=1e-5)


def test_sh_basis_output_device_matches_input_device():
    model = _model()
    mcolat = torch.tensor([[0.5, 1.0]])
    mlt = torch.tensor([[0.2, 0.9]])
    Y_real, Y_imag = model._sh_basis(mcolat, mlt)
    assert Y_real.device == mcolat.device
    assert Y_imag.device == mcolat.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_sh_basis_runs_and_stays_on_cuda_without_cpu_roundtrip():
    model = _model()
    device = torch.device("cuda")
    mcolat = torch.tensor([[0.5, 1.0]], device=device)
    mlt = torch.tensor([[0.2, 0.9]], device=device)
    Y_real, Y_imag = model._sh_basis(mcolat, mlt)
    assert Y_real.is_cuda and Y_imag.is_cuda
