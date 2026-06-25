import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dagger_harmonics.config import settings
from dagger_harmonics.data.datasets import OMNIDataset, SuperMAGDataset
from dagger_harmonics.models.recreation import DAGGER
from dagger_harmonics.utils import load_data_df

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract station coordinates, dbH target, and a validity mask.

    record        : (N, n_features) array
    feature_names : ordered sequence of column names

    Returns mcolat_rad (N,), mlt_rad (N,), dbh (N,), valid (N,).
    valid is True only where coords AND dbh are all finite.
    """
    feats = list(feature_names)
    arr = np.asarray(record, dtype=np.float32)

    maglat = arr[:, feats.index("MAGLAT")]
    mlt = arr[:, feats.index("MLT")]
    dbe = arr[:, feats.index("dbe_nez")]
    dbn = arr[:, feats.index("dbn_nez")]

    mcolat_rad = np.radians(90.0 - maglat)
    mlt_rad = mlt * (np.pi / 12.0)
    dbh = np.sqrt(dbe**2 + dbn**2)

    valid = np.isfinite(mcolat_rad) & np.isfinite(mlt_rad) & np.isfinite(dbh)

    return (
        torch.from_numpy(mcolat_rad),
        torch.from_numpy(mlt_rad),
        torch.from_numpy(dbh),
        torch.from_numpy(valid),
    )


def _eval_epoch(
    model: DAGGER,
    indices,
    omni_ds,
    supermag_ds,
    omni_mean: np.ndarray,
    omni_std: np.ndarray,
) -> float:
    """Compute average MSE over a set of record indices without updating weights."""
    model.eval()
    total_loss = 0.0
    n_steps = 0
    with torch.no_grad():
        for idx in indices:
            omni = _prepare_omni(omni_ds[idx], omni_mean, omni_std).unsqueeze(0)
            mcolat_rad, mlt_rad, dbh, valid = _prepare_coords_and_target(
                supermag_ds[idx], supermag_ds.features
            )
            if not valid.any():
                continue
            pred = model(
                omni, mcolat_rad[valid].unsqueeze(0), mlt_rad[valid].unsqueeze(0)
            )
            total_loss += F.mse_loss(pred[0], dbh[valid]).item()
            n_steps += 1
    model.train()
    return total_loss / n_steps if n_steps else 0.0


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
    lr: float = 1e-3,
    max_records: int | None = None,
    patience: int = 5,
    val_fraction: float = 0.1,
    save_path: Path | str | None = None,
) -> DAGGER:
    """
    Train DAGGER on val_data_2010.p.

    max_records  : cap dataset size (e.g. 2000 for a quick run).
    patience     : early-stopping epochs without val-loss improvement (0 = disabled).
    val_fraction : fraction of records held out for validation / early stopping.
    save_path    : where to write the trained model weights; defaults to
                   outputs/trained_models/dagger_model.pt at the project root.
    """
    df = load_data_df(settings.DATA_PATH / "val_data_2010.p")
    omni_ds = OMNIDataset(df["past_omni"], df["past_dates"])
    supermag_ds = SuperMAGDataset(df["future_supermag"], df["future_dates"])

    omni_mean = np.asarray(omni_ds.scalers["omni_mean"], dtype=np.float32)
    omni_std = np.asarray(omni_ds.scalers["omni_std"], dtype=np.float32)

    model = DAGGER(input_size=len(omni_ds.features))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = min(len(omni_ds), max_records) if max_records else len(omni_ds)
    all_indices = np.arange(n)
    split = max(1, int(n * (1.0 - val_fraction)))
    train_idx = all_indices[:split]
    val_idx = all_indices[split:]

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val = float("inf")
    no_improve = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, n_epochs + 1):
        np.random.shuffle(train_idx)
        total_loss = 0.0
        n_steps = 0

        bar = tqdm(train_idx, desc=f"Epoch {epoch}/{n_epochs}", unit="rec", leave=True)
        for idx in bar:
            omni = _prepare_omni(omni_ds[idx], omni_mean, omni_std).unsqueeze(0)
            mcolat_rad, mlt_rad, dbh, valid = _prepare_coords_and_target(
                supermag_ds[idx], supermag_ds.features
            )
            if not valid.any():
                continue

            pred = model(
                omni, mcolat_rad[valid].unsqueeze(0), mlt_rad[valid].unsqueeze(0)
            )
            loss = F.mse_loss(pred[0], dbh[valid])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            bar.set_postfix(
                loss=f"{loss.item():.4f}", avg=f"{total_loss / n_steps:.4f}"
            )

        train_loss = total_loss / max(n_steps, 1)
        val_loss = _eval_epoch(
            model, val_idx, omni_ds, supermag_ds, omni_mean, omni_std
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"  val_loss={val_loss:.4f}")

        if patience:
            best_val, no_improve, stop = _check_early_stop(
                val_loss, best_val, no_improve, patience
            )
            if val_loss <= best_val:
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
    _plot_losses(train_losses, val_losses)

    return model


def _plot_losses(train_losses: list[float], val_losses: list[float]) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("DAGGER training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
