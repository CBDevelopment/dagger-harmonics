import math

import numpy as np
import torch
import torch.nn as nn
from scipy.special import gammaln


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
    DAGGER geomagnetic perturbation model (Upendran et al. 2022, Table 1).

    Architecture:
      GRU      : 8 hidden units — solar wind time series encoder
      FC Layer 1: 8 → 16, ReLU, dropout p=0.7
      FC Layer 2: 16 → 1760  (440 complex SH coefficients × 2 real/imag × 2 fields: dbe, dbn)
      SH layer  : non-trainable basis contraction at station (mcolat, mlt) positions

    lmax=20 spans l=1..20 (l=0 DC term excluded):
      Σ_{l=1}^{20} (2l+1) = 20·(20+2) = 440 modes.

    The paper's Table 1 lists the FC output as "440*2 (real and imaginary parts)"
    describing the per-field coefficient count; here both fields (dbe, dbn) share
    one GRU + hidden FC layer, with the final layer sized to emit both fields'
    coefficients at once (4 × 440 = 1760).
    """

    def __init__(self, input_size: int, lmax: int = 20):
        super().__init__()
        self.lmax = lmax
        lm_pairs = [
            (deg, m) for deg in range(1, lmax + 1) for m in range(-deg, deg + 1)
        ]
        self._lm_pairs = lm_pairs
        n_coeffs = len(lm_pairs)  # 440 for lmax=20
        self.n_coeffs = n_coeffs

        # Per-mode (degree, order, normalization) arrays for the vectorized SH
        # basis evaluation below -- computed once, non-trainable.
        degs = np.array([deg for deg, _ in lm_pairs], dtype=np.float64)
        ms = np.array([m for _, m in lm_pairs], dtype=np.float64)
        self._degs = degs[:, None]  # (n_coeffs, 1)
        self._abs_ms = np.abs(ms)[:, None]  # (n_coeffs, 1)
        self._ms = ms[:, None]  # (n_coeffs, 1)
        self._lm_norms = np.array(
            [_sh_norm(deg, abs(m)) for deg, m in lm_pairs], dtype=np.float64
        )[:, None]  # (n_coeffs, 1)

        # Group mode indices by |m|: computing the associated Legendre
        # function P_l^m(cos_theta) is a sequential recurrence over l for a
        # fixed m, so modes sharing the same |m| (e.g. (l=5,m=-2) and
        # (l=7,m=2)) are evaluated together in one recurrence in
        # _sh_basis. Built once here (pure Python/numpy, independent of any
        # station position) as (index, degree-offset) pairs so the per-call
        # gather is one vectorized indexed assignment per group rather than
        # a per-mode Python loop -- the latter turned into ~230 individual
        # tiny GPU kernel launches per forward pass and was *slower* on GPU
        # than the original scipy call it replaced.
        abs_ms_int = np.abs(ms).astype(int)
        self._m_groups: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for m0 in range(0, lmax + 1):
            idx = np.where(abs_ms_int == m0)[0]
            if idx.size == 0:
                continue
            offsets = degs[idx].astype(int) - m0  # position within that group's cache
            self._m_groups.append(
                (m0, torch.from_numpy(idx), torch.from_numpy(offsets))
            )

        self.gru = nn.GRU(input_size=input_size, hidden_size=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(16, n_coeffs * 4),  # 2 fields × 2 (real + imag)
        )

    def _sh_basis(
        self, mcolat_rad: torch.Tensor, mlt_rad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate complex SH basis at station positions. Non-trainable.

        Pure-PyTorch tensor implementation of the associated-Legendre
        recurrence (same algorithm and Condon-Shortley sign convention as
        `scipy.special.lpmv`, verified against it in
        tests/test_recreation.py), so the whole computation runs on
        mcolat_rad's own device -- no CPU/numpy round trip. Modes are
        evaluated in groups sharing the same |m| (see `_m_groups`), since the
        recurrence over degree l is sequential for a fixed m; each group's
        results are gathered into `plm` with one vectorized indexed
        assignment (not a per-mode Python loop -- see `_m_groups`'s comment
        for why that mattered).

        mcolat_rad: (batch, N) — magnetic co-latitude in radians
        mlt_rad:    (batch, N) — MLT azimuth in radians (MLT_hours * π/12)

        Returns Y_real, Y_imag: (batch, N, n_coeffs)
        """
        B, N = mcolat_rad.shape
        dev = mcolat_rad.device
        dtype = mcolat_rad.dtype if mcolat_rad.is_floating_point() else torch.float32

        cos_theta = torch.cos(mcolat_rad.reshape(-1).to(dtype))  # (B*N,)
        phi = mlt_rad.reshape(-1).to(dtype)  # (B*N,)
        somx2 = torch.sqrt((1 - cos_theta) * (1 + cos_theta))

        plm = torch.empty(self.n_coeffs, cos_theta.shape[0], device=dev, dtype=dtype)
        for m0, idx, offsets in self._m_groups:
            idx = idx.to(dev)
            offsets = offsets.to(dev)

            # P_m0^m0(x), including the Condon-Shortley phase (-1)^m.
            pmm = torch.ones_like(cos_theta)
            fact = 1.0
            for _ in range(m0):
                pmm = pmm * (-fact) * somx2
                fact += 2.0

            cache = [pmm]
            if m0 + 1 <= self.lmax:
                cache.append(cos_theta * (2 * m0 + 1) * pmm)
            for degree in range(m0 + 2, self.lmax + 1):
                cache.append(
                    (
                        cos_theta * (2 * degree - 1) * cache[-1]
                        - (degree + m0 - 1) * cache[-2]
                    )
                    / (degree - m0)
                )

            # One vectorized gather for the whole group instead of a
            # per-mode assignment loop.
            plm[idx] = torch.stack(cache, dim=0)[offsets]

        norms = torch.as_tensor(
            self._lm_norms.ravel(), device=dev, dtype=dtype
        ).unsqueeze(1)
        ms = torch.as_tensor(self._ms.ravel(), device=dev, dtype=dtype).unsqueeze(1)
        plm = plm * norms  # (n_coeffs, B*N)
        angle = ms * phi.unsqueeze(0)  # (n_coeffs, B*N)

        Y_real = (plm * torch.cos(angle)).T.reshape(B, N, self.n_coeffs)
        Y_imag = (plm * torch.sin(angle)).T.reshape(B, N, self.n_coeffs)
        return Y_real.to(torch.float32), Y_imag.to(torch.float32)

    @staticmethod
    def _complex_contract(
        c_real: torch.Tensor,
        c_imag: torch.Tensor,
        Y_real: torch.Tensor,
        Y_imag: torch.Tensor,
    ) -> torch.Tensor:
        """Real part of the (c_real + i·c_imag) coefficients contracted with the
        (Y_real + i·Y_imag) SH basis, summed over modes: Re(c · Y) = c_r·Y_r − c_i·Y_i.

        c_real, c_imag: (batch, n_coeffs)
        Y_real, Y_imag: (batch, N, n_coeffs)
        Returns (batch, N).
        """
        return torch.einsum("bk,bnk->bn", c_real, Y_real) - torch.einsum(
            "bk,bnk->bn", c_imag, Y_imag
        )

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

        Returns (batch, N, 2) — predicted [dbe, dbn] perturbation at each station.
        """
        _, h_n = self.gru(omni)
        hidden = h_n.squeeze(0)  # (batch, 8)

        coeffs = self.fc(hidden)  # (batch, n_coeffs * 4)
        c_real_dbe, c_imag_dbe, c_real_dbn, c_imag_dbn = coeffs.split(
            self.n_coeffs, dim=1
        )

        if mcolat_rad.dim() == 1:
            mcolat_rad = mcolat_rad.unsqueeze(0).expand(omni.shape[0], -1)
            mlt_rad = mlt_rad.unsqueeze(0).expand(omni.shape[0], -1)

        Y_real, Y_imag = self._sh_basis(mcolat_rad, mlt_rad)  # (batch, N, n_coeffs)

        dbe_pred = self._complex_contract(c_real_dbe, c_imag_dbe, Y_real, Y_imag)
        dbn_pred = self._complex_contract(c_real_dbn, c_imag_dbn, Y_real, Y_imag)
        return torch.stack([dbe_pred, dbn_pred], dim=-1)  # (batch, N, 2)
