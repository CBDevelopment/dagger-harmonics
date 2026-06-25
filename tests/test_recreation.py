import numpy as np
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


def test_fc_second_layer_outputs_880():
    # 440 complex SH coefficients × 2 (real + imag) = 880
    model = _model()
    linears = [m for m in model.fc.modules() if isinstance(m, nn.Linear)]
    assert linears[-1].out_features == 880


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


def test_forward_returns_batch_by_n_stations():
    out = _forward(_model(), batch=2, n_stations=10)
    assert out.shape == (2, 10)


def test_forward_works_for_single_item_batch():
    out = _forward(_model(), batch=1, n_stations=7)
    assert out.shape == (1, 7)


def test_forward_broadcasts_1d_coords_across_batch():
    """(N,) station coords shared by all records in a batch."""
    model = _model()
    batch, T, n_stations = 3, 4, 7
    omni = torch.zeros(batch, T, 15)
    mcolat = torch.full((n_stations,), 0.3)
    mlt = torch.zeros(n_stations)
    out = model(omni, mcolat, mlt)
    assert out.shape == (batch, n_stations)


# ---------------------------------------------------------------------------
# SH coefficient count (lmax = 20, l = 1..20)
# ---------------------------------------------------------------------------


def test_440_sh_modes_for_lmax_20():
    n = sum(2 * l + 1 for l in range(1, 21))
    assert n == 440
