from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from dagger_harmonics.utils import load_data, load_data_df


@dataclass
class DaggerData:
    """Everything needed to build OMNI/SuperMAG datasets from one `.p` file.

    Bundles the record dataframe with its sidecar features/scalers so callers
    only ever need to point `load_dagger_data` at the main pickle.
    """

    df: pd.DataFrame
    omni_features: np.ndarray
    omni_scalers: dict
    supermag_features: np.ndarray
    supermag_scalers: dict


def _omni_scalers(scalers: dict) -> dict:
    mean, std = scalers["omni"]
    return {"omni_mean": np.asarray(mean), "omni_std": np.asarray(std)}


def _supermag_scalers(scalers: dict) -> dict:
    mean, std = scalers["supermag"]
    dbe_mean, dbn_mean = mean
    dbe_std, dbn_std = std
    return {
        "dbe_mean": dbe_mean,
        "dbe_std": dbe_std,
        "dbn_mean": dbn_mean,
        "dbn_std": dbn_std,
    }


def load_dagger_data(path: Path) -> DaggerData:
    """Load a val_data_*.p file plus its sidecar features/scalers.

    The sidecar files (`omni_features.p`, `supermag_features.p`, `scalers.p`)
    are read from `path.parent` — just point this at the main pickle and
    everything else is found alongside it.
    """
    data_dir = Path(path).parent
    scalers = load_data(data_dir / "scalers.p")

    return DaggerData(
        df=load_data_df(path),
        omni_features=np.asarray(load_data(data_dir / "omni_features.p")),
        omni_scalers=_omni_scalers(scalers),
        supermag_features=np.asarray(load_data(data_dir / "supermag_features.p")),
        supermag_scalers=_supermag_scalers(scalers),
    )


class OMNIDataset(Dataset):
    """OMNI solar-wind driver series, one entry per record.

    `split` selects which pickle column to read (`"past"` -> `past_omni`,
    `"future"` -> `future_omni` if present).
    """

    def __init__(self, dagger_data: DaggerData, split: str = "past"):
        self.data = dagger_data.df[f"{split}_omni"].reset_index(drop=True)
        self.dates = dagger_data.df[f"{split}_dates"].reset_index(drop=True)
        self.features = dagger_data.omni_features
        self.scalers = dagger_data.omni_scalers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_dates(self, idx) -> np.ndarray:
        """Return shape (T,) timestamps for record `idx`."""
        return self.dates[idx]

    def get_df(self, idx) -> pd.DataFrame:
        """Return a DataFrame for a single record at index `idx`."""
        df = pd.DataFrame(self[idx], columns=self.features)
        df["date"] = pd.to_datetime(self.get_dates(idx), unit="s", utc=True)
        return df

    def get_column(self, feature: str) -> np.ndarray:
        """Return shape (N_records, T) for a single named OMNI feature."""
        idx = list(self.features).index(feature)
        return np.stack([self[i] for i in range(len(self))])[:, :, idx]


class SuperMAGDataset(Dataset):
    """SuperMAG station arrays, one entry per record.

    `split` selects which pickle column to read (`"future"` -> `future_supermag`,
    `"past"` -> `past_supermag`). The `future_*` columns carry a single
    timestep per record, so that leading axis is squeezed away; `past_*`
    columns keep their full (T, N, 6) shape.
    """

    def __init__(self, dagger_data: DaggerData, split: str = "future"):
        self.split = split
        self.data = dagger_data.df[f"{split}_supermag"].reset_index(drop=True)
        self.dates = dagger_data.df[f"{split}_dates"].reset_index(drop=True)
        self.features = dagger_data.supermag_features
        self.scalers = dagger_data.supermag_scalers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        return record[0] if self.split == "future" else record

    def get_date(self, idx) -> np.ndarray:
        """Return the single timestamp for a `future`-split record `idx`."""
        return self.dates[idx][0][0]

    def get_df(self, idx) -> pd.DataFrame:
        """Return a DataFrame for a single record at index `idx`."""
        df = pd.DataFrame(self[idx], columns=self.features)
        df["date"] = pd.to_datetime(self.get_date(idx), unit="s", utc=True)
        return df

    def get_column(self, feature: str) -> np.ndarray:
        """Return shape (N_records, N_stations) for a single named SuperMAG feature."""
        idx = list(self.features).index(feature)
        return np.stack([self[i] for i in range(len(self))])[:, :, idx]
