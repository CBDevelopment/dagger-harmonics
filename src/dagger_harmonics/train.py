import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm

from dagger_harmonics.config import settings
from dagger_harmonics.data.datasets import (
    OMNIDataset,
    SuperMAGDataset,
    load_dagger_data,
)
from dagger_harmonics.models.recreation import DAGGER

_PROJECT_ROOT = Path(__file__).parents[2]
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "outputs" / "trained_models" / "dagger_model.pt"


def get_device() -> torch.device:
    """Pick CUDA when available, otherwise fall back to CPU.

    Single source of truth for hardware selection so training and inference
    run unmodified on either a GPU box or a CPU-only machine.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_omni(
    record: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """Z-score one OMNI record. NaN inputs are replaced with 0 after normalisation.
    Returns (T, n_omni) float32 tensor."""
    arr = np.nan_to_num(np.asarray(record, dtype=np.float32), nan=0.0)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return torch.from_numpy((arr - mean) / np.maximum(std, 1e-8))


def _prepare_coords_and_target(
    record: np.ndarray,
    feature_names,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract station coordinates, per-component targets, and a validity mask.

    record        : (N, n_features) array
    feature_names : ordered sequence of column names

    Returns mcolat_rad (N,), mlt_rad (N,), dbe (N,), dbn (N,), valid (N,).
    valid is True where coords are finite AND at least one of dbe/dbn is finite.
    NaN component values are replaced with 0.0 so they contribute zero loss.
    """
    feats = list(feature_names)
    arr = np.asarray(record, dtype=np.float32)

    maglat = arr[:, feats.index("MAGLAT")]
    mlt = arr[:, feats.index("MLT")]
    dbe_raw = arr[:, feats.index("dbe_nez")]
    dbn_raw = arr[:, feats.index("dbn_nez")]

    mcolat_rad = np.radians(90.0 - maglat)
    mlt_rad = mlt * (np.pi / 12.0)

    coords_ok = np.isfinite(mcolat_rad) & np.isfinite(mlt_rad)
    valid = coords_ok & (np.isfinite(dbe_raw) | np.isfinite(dbn_raw))

    dbe = np.where(np.isfinite(dbe_raw), dbe_raw, 0.0).astype(np.float32)
    dbn = np.where(np.isfinite(dbn_raw), dbn_raw, 0.0).astype(np.float32)

    return (
        torch.from_numpy(mcolat_rad),
        torch.from_numpy(mlt_rad),
        torch.from_numpy(dbe),
        torch.from_numpy(dbn),
        torch.from_numpy(valid),
    )


def _predict_and_target(
    model: DAGGER,
    omni: torch.Tensor,
    mcolat_rad: torch.Tensor,
    mlt_rad: torch.Tensor,
    dbe: torch.Tensor,
    dbn: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model forward restricted to valid stations, and build the matching
    [dbe, dbn] target tensor. Shared by `_run_epoch`'s train and eval passes so the
    prediction/target pairing is defined in exactly one place.

    Returns pred, target: both (n_valid, 2).
    """
    pred = model(omni, mcolat_rad[valid].unsqueeze(0), mlt_rad[valid].unsqueeze(0))[0]
    target = torch.stack([dbe[valid], dbn[valid]], dim=-1)
    return pred, target


def _run_epoch(
    model: DAGGER,
    indices,
    omni_ds,
    supermag_ds,
    omni_mean: np.ndarray,
    omni_std: np.ndarray,
    optimizer: torch.optim.Optimizer | None = None,
    progress_desc: str | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run one pass over `indices`, reporting MAE/MSE/RMSE.

    Pass `optimizer` to run a training pass: the model is put in train mode,
    and each record's MAE loss is backpropagated and stepped. Omit it to run
    a read-only validation pass under `torch.no_grad()`. Either way the
    returned metrics are computed the same way, so training and validation
    numbers are directly comparable.

    `device` moves each record's tensors (and the metrics) to that device
    before the forward pass; the model itself is expected to already live
    there (see `train`). Defaults to the model's own device, so callers
    working purely on CPU need not pass anything.
    """
    is_training = optimizer is not None
    model.train(is_training)
    if device is None:
        device = next(model.parameters()).device

    mae, mse, rmse = (
        MeanAbsoluteError().to(device),
        MeanSquaredError().to(device),
        MeanSquaredError(squared=False).to(device),
    )
    n_steps = 0

    indices = (
        tqdm(indices, desc=progress_desc, unit="rec") if progress_desc else indices
    )
    with torch.set_grad_enabled(is_training):
        for idx in indices:
            omni = _prepare_omni(omni_ds[idx], omni_mean, omni_std).unsqueeze(0)
            mcolat_rad, mlt_rad, dbe, dbn, valid = _prepare_coords_and_target(
                supermag_ds[idx], supermag_ds.features
            )
            if not valid.any():
                continue
            omni, mcolat_rad, mlt_rad, dbe, dbn, valid = (
                omni.to(device),
                mcolat_rad.to(device),
                mlt_rad.to(device),
                dbe.to(device),
                dbn.to(device),
                valid.to(device),
            )
            pred, target = _predict_and_target(
                model, omni, mcolat_rad, mlt_rad, dbe, dbn, valid
            )

            if is_training:
                loss = F.l1_loss(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred, target = pred.detach(), target.detach()

            mae.update(pred, target)
            mse.update(pred, target)
            rmse.update(pred, target)
            n_steps += 1

    if not n_steps:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0}
    return {
        "mae": mae.compute().item(),
        "mse": mse.compute().item(),
        "rmse": rmse.compute().item(),
    }


def _check_early_stop(
    val_loss: float,
    best: float,
    no_improve: int,
    patience: int,
) -> tuple[float, int, bool]:
    """
    Update early-stopping state.

    Returns (new_best, new_no_improve_count, should_stop).
    """
    if val_loss < best:
        return val_loss, 0, False
    new_count = no_improve + 1
    return best, new_count, new_count >= patience


def train(
    n_epochs: int = 20,
    lr: float = 5e-3,
    weight_decay: float = 5e-5,
    max_records: int | None = None,
    patience: int = 5,
    val_fraction: float = 0.1,
    save_path: Path | str | None = None,
    device: torch.device | str | None = None,
) -> DAGGER:
    """
    Train DAGGER on val_data_2010.p.

    Loss is Mean Absolute Error (L1), and validation is tracked/early-stopped on
    MAE too, following Upendran et al. 2022 §3.4; MSE/RMSE are also reported
    each epoch for comparison against the paper's Table 3.

    lr, weight_decay : Adam hyperparameters; defaults are the paper's Table 2 values.
    max_records      : cap dataset size (e.g. 2000 for a quick run).
    patience         : early-stopping epochs without val-MAE improvement (0 = disabled).
    val_fraction     : fraction of records held out for validation / early stopping.
    save_path        : where to write the trained model weights; defaults to
                       outputs/trained_models/dagger_model.pt at the project root.
    device           : where to train; defaults to `get_device()` (CUDA if
                       available, else CPU), so this runs unmodified on either.
    """
    device = torch.device(device) if device is not None else get_device()

    dagger_data = load_dagger_data(settings.DATA_PATH / "val_data_2010.p")
    omni_ds = OMNIDataset(dagger_data, split="past")
    supermag_ds = SuperMAGDataset(dagger_data, split="future")

    omni_mean = np.asarray(omni_ds.scalers["omni_mean"], dtype=np.float32)
    omni_std = np.asarray(omni_ds.scalers["omni_std"], dtype=np.float32)

    model = DAGGER(input_size=len(omni_ds.features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = min(len(omni_ds), max_records) if max_records else len(omni_ds)
    all_indices = np.arange(n)
    split = max(1, int(n * (1.0 - val_fraction)))
    train_idx = all_indices[:split]
    val_idx = all_indices[split:]

    train_history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []

    best_val = float("inf")
    no_improve = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, n_epochs + 1):
        np.random.shuffle(train_idx)
        train_metrics = _run_epoch(
            model,
            train_idx,
            omni_ds,
            supermag_ds,
            omni_mean,
            omni_std,
            optimizer=optimizer,
            progress_desc=f"Epoch {epoch}/{n_epochs}",
            device=device,
        )
        val_metrics = _run_epoch(
            model, val_idx, omni_ds, supermag_ds, omni_mean, omni_std, device=device
        )

        train_history.append(train_metrics)
        val_history.append(val_metrics)
        print(
            f"  train_mae={train_metrics['mae']:.4f}  "
            f"val_mae={val_metrics['mae']:.4f}  "
            f"val_mse={val_metrics['mse']:.4f}  "
            f"val_rmse={val_metrics['rmse']:.4f}"
        )

        if patience:
            best_val, no_improve, stop = _check_early_stop(
                val_metrics["mae"], best_val, no_improve, patience
            )
            if val_metrics["mae"] <= best_val:
                best_state = copy.deepcopy(model.state_dict())
            if stop:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)."
                )
                break

    model.load_state_dict(best_state)

    # Save weights
    out_path = Path(save_path) if save_path else _DEFAULT_MODEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved → {out_path}")

    # Plot loss curves
    _plot_losses(train_history, val_history)

    return model


def _plot_losses(
    train_history: list[dict[str, float]], val_history: list[dict[str, float]]
) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_history) + 1)

    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [m["mae"] for m in train_history], label="train MAE")
    ax.plot(epochs, [m["mae"] for m in val_history], label="val MAE")
    ax.plot(epochs, [m["rmse"] for m in val_history], label="val RMSE", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (nT)")
    ax.set_title("DAGGER training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
