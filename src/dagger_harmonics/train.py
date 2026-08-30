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

    # Invalid-station coords are replaced with a safe finite placeholder too
    # (not just dbe/dbn) -- `valid` already excludes these stations from the
    # loss, but keeping every returned tensor finite means multiple records
    # can be stacked into a batch without a stray NaN ever reaching cos()/
    # sin() in the SH basis, regardless of which records land together.
    mcolat_rad = np.where(np.isfinite(mcolat_rad), mcolat_rad, 0.0)
    mlt_rad = np.where(np.isfinite(mlt_rad), mlt_rad, 0.0)

    dbe = np.where(np.isfinite(dbe_raw), dbe_raw, 0.0).astype(np.float32)
    dbn = np.where(np.isfinite(dbn_raw), dbn_raw, 0.0).astype(np.float32)

    return (
        torch.from_numpy(mcolat_rad),
        torch.from_numpy(mlt_rad),
        torch.from_numpy(dbe),
        torch.from_numpy(dbn),
        torch.from_numpy(valid),
    )


def _select_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the compute device.

    Pass an explicit device (e.g. "cuda", "cpu", a `torch.device`) to use it
    as-is. Pass nothing (the default) to auto-detect: CUDA if available,
    else CPU.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict_and_target(
    model: DAGGER,
    omni: torch.Tensor,
    mcolat_rad: torch.Tensor,
    mlt_rad: torch.Tensor,
    dbe: torch.Tensor,
    dbn: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model forward over every station, then mask to valid ones, and
    build the matching [dbe, dbn] target tensor. Shared by `_run_epoch`'s
    train and eval passes so the prediction/target pairing is defined in
    exactly one place.

    Accepts either a single record's 1D (N,) coordinate/target tensors
    (auto-promoted to a batch of 1, matching `omni`'s existing
    (1, T, n_omni) shape) or genuinely batched (B, N) tensors for B records
    run through the model in one forward pass -- same masking contract
    either way. Running every station (not pre-selecting valid ones, as
    earlier versions did) is what makes batching possible: different records
    have different numbers of valid stations, so only the batch's station
    *count* can be fixed, not which ones are valid.

    Returns pred, target: both (n_valid_total, 2), flattened across whatever
    batch dimension was passed in.
    """
    if mcolat_rad.dim() == 1:
        mcolat_rad = mcolat_rad.unsqueeze(0)
        mlt_rad = mlt_rad.unsqueeze(0)
        dbe = dbe.unsqueeze(0)
        dbn = dbn.unsqueeze(0)
        valid = valid.unsqueeze(0)

    pred = model(omni, mcolat_rad, mlt_rad)  # (B, N, 2)
    target = torch.stack([dbe, dbn], dim=-1)  # (B, N, 2)
    return pred[valid], target[valid]


def _prepare_batch(
    indices,
    omni_ds,
    supermag_ds,
    omni_mean: np.ndarray,
    omni_std: np.ndarray,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Stack several records into one batch's worth of model input.

    indices: record indices to include in this batch (any iterable).

    Returns omni (B, T, n_omni), and mcolat_rad/mlt_rad/dbe/dbn/valid, each
    (B, N) -- N fixed by the dataset (every record has the same station
    slots; which ones are `valid` varies per record, not N itself).
    """
    omni_list, mcolat_list, mlt_list, dbe_list, dbn_list, valid_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for idx in indices:
        omni_list.append(_prepare_omni(omni_ds[idx], omni_mean, omni_std))
        mcolat, mlt, dbe, dbn, valid = _prepare_coords_and_target(
            supermag_ds[idx], supermag_ds.features
        )
        mcolat_list.append(mcolat)
        mlt_list.append(mlt)
        dbe_list.append(dbe)
        dbn_list.append(dbn)
        valid_list.append(valid)

    return (
        torch.stack(omni_list),
        torch.stack(mcolat_list),
        torch.stack(mlt_list),
        torch.stack(dbe_list),
        torch.stack(dbn_list),
        torch.stack(valid_list),
    )


def _run_epoch(
    model: DAGGER,
    indices,
    omni_ds,
    supermag_ds,
    omni_mean: np.ndarray,
    omni_std: np.ndarray,
    optimizer: torch.optim.Optimizer | None = None,
    progress_desc: str | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 1,
) -> dict[str, float]:
    """Run one pass over `indices`, reporting MAE/MSE/RMSE.

    Pass `optimizer` to run a training pass: the model is put in train mode,
    and each batch's MAE loss is backpropagated and stepped once per batch.
    Omit it to run a read-only validation pass under `torch.no_grad()`.
    Either way the returned metrics are computed the same way, so training
    and validation numbers are directly comparable.

    `batch_size` groups `indices` into chunks of that many records, each
    processed in a single forward/backward pass (matching the paper's
    Table 2 `batch_size=8500`, though any value fits -- size it to whatever
    GPU memory allows). `batch_size=1` (the default) reproduces the
    original per-record behavior exactly.

    `device` moves the model and every tensor there before the forward pass
    (default: CPU, matching the model's own default placement). The SH
    basis in `DAGGER._sh_basis` is built fresh from the coordinate tensors'
    device each call, so moving them is enough to keep the whole forward
    pass on one device.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    model.to(device)
    is_training = optimizer is not None
    model.train(is_training)

    mae, mse, rmse = (
        MeanAbsoluteError().to(device),
        MeanSquaredError().to(device),
        MeanSquaredError(squared=False).to(device),
    )
    n_steps = 0

    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    batches = (
        tqdm(batches, desc=progress_desc, unit="batch") if progress_desc else batches
    )

    with torch.set_grad_enabled(is_training):
        for batch_indices in batches:
            if len(batch_indices) == 0:
                continue
            omni, mcolat_rad, mlt_rad, dbe, dbn, valid = _prepare_batch(
                batch_indices, omni_ds, supermag_ds, omni_mean, omni_std
            )
            omni, mcolat_rad, mlt_rad, dbe, dbn, valid = (
                t.to(device) for t in (omni, mcolat_rad, mlt_rad, dbe, dbn, valid)
            )
            if not valid.any():
                continue
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
    device: str | torch.device | None = None,
    batch_size: int = 256,
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
    device           : compute device ("cuda", "cpu", a `torch.device`, ...).
                       Defaults to CUDA if available, else CPU.
    batch_size       : records per optimizer step. The paper uses 8500
                       (Table 2); the default here (256) is a size that fits
                       comfortably on a modest GPU or CPU -- raise it toward
                       8500 if your hardware's memory allows, for a closer
                       match to the paper's exact training dynamics.
    """
    device = _select_device(device)
    print(f"Training on device: {device}, batch_size: {batch_size}")

    out_path = Path(save_path) if save_path else _DEFAULT_MODEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = out_path.with_suffix(".history.json")
    plot_path = out_path.with_suffix(".png")

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
            batch_size=batch_size,
        )
        val_metrics = _run_epoch(
            model,
            val_idx,
            omni_ds,
            supermag_ds,
            omni_mean,
            omni_std,
            device=device,
            batch_size=batch_size,
        )

        train_history.append(train_metrics)
        val_history.append(val_metrics)
        print(
            f"  train_mae={train_metrics['mae']:.4f}  "
            f"val_mae={val_metrics['mae']:.4f}  "
            f"val_mse={val_metrics['mse']:.4f}  "
            f"val_rmse={val_metrics['rmse']:.4f}"
        )

        # Persisted every epoch -- not just at the end -- so an interrupted
        # run (killed pod, OOM, etc.) still leaves a usable checkpoint and a
        # full record of what happened so far, not nothing.
        _save_history(train_history, val_history, history_path)

        stop = False
        if patience:
            best_val, no_improve, stop = _check_early_stop(
                val_metrics["mae"], best_val, no_improve, patience
            )
            if val_metrics["mae"] <= best_val:
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, out_path)
            if stop:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)."
                )
        else:
            # No early stopping configured -- every epoch's weights are the
            # "best" so far by definition; still checkpoint each one.
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_path)

        if stop:
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_path)  # final write, in case of ties above
    print(f"Model saved -> {out_path}")
    print(f"History saved -> {history_path}")

    # Plot loss curves -- saved to disk (works headless) and shown if a
    # display is attached.
    _plot_losses(train_history, val_history, plot_path)
    print(f"Loss plot saved -> {plot_path}")

    return model


def _save_history(
    train_history: list[dict[str, float]],
    val_history: list[dict[str, float]],
    path: Path,
) -> None:
    """Write per-epoch train/val metrics to disk as JSON, overwritten each
    call so the file always reflects every epoch completed so far."""
    import json

    with open(path, "w") as f:
        json.dump({"train": train_history, "val": val_history}, f, indent=2)


def _plot_losses(
    train_history: list[dict[str, float]],
    val_history: list[dict[str, float]],
    save_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_history) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [m["mae"] for m in train_history], label="train MAE")
    ax.plot(epochs, [m["mae"] for m in val_history], label="val MAE")
    ax.plot(epochs, [m["rmse"] for m in val_history], label="val RMSE", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (nT)")
    ax.set_title("DAGGER training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
