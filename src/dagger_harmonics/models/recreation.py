import math

import numpy as np
import torch
import torch.nn as nn
from scipy.special import gammaln, lpmv


def _sh_norm(degree: int, abs_m: int) -> float:
    """Normalization for Y_l^m; log-space to avoid factorial overflow at high degree."""
    log_k = 0.5 * (
        math.log((2 * degree + 1) / (4 * math.pi))
        + gammaln(degree - abs_m + 1)
        - gammaln(degree + abs_m + 1)
    )
    return math.exp(log_k)


class DAGGER(nn.Module):
    """
    DAGGER geomagnetic perturbation model (Upendran et al., Table 1).

    Architecture:
      GRU      : 8 hidden units — solar wind time series encoder
      FC Layer 1: 8 → 16, ReLU
      FC Layer 2: 16 → 880  (440 complex SH coefficients × 2: real + imag)
      SH layer  : non-trainable basis contraction at station (mcolat, mlt) positions

    lmax=20 spans l=1..20 (l=0 DC term excluded):
      Σ_{l=1}^{20} (2l+1) = 20·(20+2) = 440 modes.
    """

    def __init__(self, input_size: int, lmax: int = 20):
        super().__init__()
        self.lmax = lmax
        self._lm_pairs = [
            (deg, m) for deg in range(1, lmax + 1) for m in range(-deg, deg + 1)
        ]
        # Normalization constants K_lm — fixed, computed once
        self._lm_norms: np.ndarray = np.array(
            [_sh_norm(deg, abs(m)) for deg, m in self._lm_pairs], dtype=np.float32
        )
        n_coeffs = len(self._lm_pairs)  # 440 for lmax=20

        self.gru = nn.GRU(input_size=input_size, hidden_size=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, n_coeffs * 2),  # real and imaginary parts
        )

    def _sh_basis(
        self, mcolat_rad: torch.Tensor, mlt_rad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate complex SH basis at station positions. Non-trainable.

        mcolat_rad: (batch, N) — magnetic co-latitude in radians
        mlt_rad:    (batch, N) — MLT azimuth in radians (MLT_hours * π/12)

        Returns Y_real, Y_imag: (batch, N, n_coeffs)
        """
        B, N = mcolat_rad.shape
        cos_theta = np.cos(mcolat_rad.detach().cpu().numpy().ravel())  # (B*N,)
        phi = mlt_rad.detach().cpu().numpy().ravel()  # (B*N,)

        n_coeffs = len(self._lm_pairs)
        Y_real = np.empty((B * N, n_coeffs), dtype=np.float32)
        Y_imag = np.empty((B * N, n_coeffs), dtype=np.float32)

        for idx, (deg, m) in enumerate(self._lm_pairs):
            k_plm = self._lm_norms[idx] * lpmv(abs(m), deg, cos_theta)
            Y_real[:, idx] = k_plm * np.cos(m * phi)
            Y_imag[:, idx] = k_plm * np.sin(m * phi)

        dev = mcolat_rad.device
        Y_real_t = torch.from_numpy(Y_real.reshape(B, N, n_coeffs)).to(dev)
        Y_imag_t = torch.from_numpy(Y_imag.reshape(B, N, n_coeffs)).to(dev)
        return Y_real_t, Y_imag_t

    def forward(
        self,
        omni: torch.Tensor,
        mcolat_rad: torch.Tensor,
        mlt_rad: torch.Tensor,
    ) -> torch.Tensor:
        """
        omni:       (batch, T, n_omni_features) — solar wind time series
        mcolat_rad: (batch, N) or (N,) — magnetic co-latitude in radians
        mlt_rad:    (batch, N) or (N,) — MLT azimuth in radians

        Returns (batch, N) — predicted scalar perturbation at each station.
        """
        _, h_n = self.gru(omni)
        hidden = h_n.squeeze(0)  # (batch, 8)

        coeffs = self.fc(hidden)  # (batch, n_coeffs * 2)
        n_coeffs = len(self._lm_pairs)
        c_real = coeffs[:, :n_coeffs]  # (batch, n_coeffs)
        c_imag = coeffs[:, n_coeffs:]  # (batch, n_coeffs)

        if mcolat_rad.dim() == 1:
            mcolat_rad = mcolat_rad.unsqueeze(0).expand(omni.shape[0], -1)
            mlt_rad = mlt_rad.unsqueeze(0).expand(omni.shape[0], -1)

        Y_real, Y_imag = self._sh_basis(mcolat_rad, mlt_rad)  # (batch, N, n_coeffs)

        # Re[Σ_k c_k · Y_k] = Σ_k (c_real_k · Y_real_k − c_imag_k · Y_imag_k)
        return torch.einsum("bk,bnk->bn", c_real, Y_real) - torch.einsum(
            "bk,bnk->bn", c_imag, Y_imag
        )
