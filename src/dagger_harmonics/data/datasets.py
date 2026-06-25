from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from dagger_harmonics.config import settings
from dagger_harmonics.utils import load_data


def get_all_scalers() -> dict:
    """Load scalers from the specified path and return as a dictionary."""
    scalers = load_data(settings.DATA_PATH / "scalers.p")
    return dict(scalers) if not isinstance(scalers, dict) else scalers


def get_omni_scalers() -> dict:
    scalers = get_all_scalers()
    _mean, _std = scalers.get("omni", {})

    return {
        "omni_mean": _mean,  # means for all OMNI features
        "omni_std": _std,  # stds for all OMNI features
    }


def get_omni_features() -> np.ndarray:
    """Load OMNI features and ensure an ndarray is returned."""
    features = load_data(settings.DATA_PATH / "omni_features.p")
    arr = np.asarray(features)
    return arr


class OMNIDataset(Dataset):
    def __init__(self, data, dates):
        self.data = data
        self.dates = dates
        self.features = get_omni_features()
        self.scalers = get_omni_scalers()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_dates(self, idx) -> np.ndarray:
        """Return shape (N_records, 1) for a single named OMNI feature."""
        return self.dates[idx]

    def get_df(self, idx) -> pd.DataFrame:
        """Return a DataFrame for a single record at index `idx`."""
        record = self[idx]
        df = pd.DataFrame(record, columns=self.features)
        df["date"] = pd.to_datetime(self.get_dates(idx), unit="s", utc=True)
        return df

    def get_column(self, feature: str) -> np.ndarray:
        """Return shape (N_records, T) for a single named OMNI feature."""
        idx = list(self.features).index(feature)
        return np.stack([self[i] for i in range(len(self))])[:, :, idx]


def get_supermag_scalers() -> dict:
    scalers = get_all_scalers()
    _mean, _std = scalers.get("supermag", {})

    dbe_mean, dbn_mean = _mean
    dbe_std, dbn_std = _std
    return {
        "dbe_mean": dbe_mean,
        "dbe_std": dbe_std,
        "dbn_mean": dbn_mean,
        "dbn_std": dbn_std,
    }


def get_supermag_features() -> np.ndarray:
    """Load SuperMAG features and ensure an ndarray is returned."""
    features = load_data(settings.DATA_PATH / "supermag_features.p")
    arr = np.asarray(features)
    return arr


class SuperMAGDataset(Dataset):
    def __init__(self, data, dates):
        self.data = data
        self.dates = dates
        self.features = get_supermag_features()
        self.scalers = get_supermag_scalers()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0]

    def get_date(self, idx) -> np.ndarray:
        """Return shape (N_records, 1) for a single named SuperMAG feature."""
        return self.dates[idx][0][0]

    def get_df(self, idx) -> pd.DataFrame:
        """Return a DataFrame for a single record at index `idx`."""
        record = self[idx]
        df = pd.DataFrame(record, columns=self.features)
        df["date"] = pd.to_datetime(self.get_date(idx), unit="s", utc=True)
        return df

    def get_column(self, feature: str) -> np.ndarray:
        """Return shape (N_records, N_stations) for a single named SuperMAG feature."""
        idx = list(self.features).index(feature)
        return np.stack([self[i] for i in range(len(self))])[:, :, idx]
