import pandas as pd

from dagger_harmonics.config import settings
from dagger_harmonics.data.datasets import (
    OMNIDataset,
    SuperMAGDataset,
    load_dagger_data,
)
from dagger_harmonics.train import train
from dagger_harmonics.utils import plot_omni, plot_supermag


def main():
    """Inspect one record of val_data_2010.p: the OMNI driver window and the
    SuperMAG station target it pairs with, using the same loader and dataset
    classes the training pipeline itself uses
    (dagger_harmonics.data.datasets.load_dagger_data).
    """
    dagger_data = load_dagger_data(settings.DATA_PATH / "val_data_2010.p")

    past_omni = OMNIDataset(dagger_data, split="past")
    # Bx, By, Bz are the GSM (Geocentric Solar Magnetospheric) components:
    # X points from Earth to Sun, Z is perpendicular to Earth's magnetic
    # dipole axis, Y completes the right-handed system.
    future_supermag = SuperMAGDataset(dagger_data, split="future")
    # MAGLAT between 40.18 and 84.72, 175 stations in the northern hemisphere

    index = 0

    omni_df = past_omni.get_df(index)
    print(omni_df.head())
    plot_omni(omni_df)

    maglat_data = future_supermag.get_column("MAGLAT").flatten()
    maglat = pd.DataFrame({"maglat": maglat_data})
    print(maglat.describe())

    supermag_df = future_supermag.get_df(index)
    print(supermag_df.head())
    plot_supermag(supermag_df, feature="dbh")
    # plot_supermag_geo(supermag_df, feature="dbh") also available (geographic
    # projection instead of MLT/MCOLAT), but needs cartopy/aacgmv2 --
    # `uv sync --group analysis` first.


def train_model():
    """Entry point for the `dagger-train` console script -- trains on the
    full dataset with train()'s own defaults (see dagger_harmonics.train.train
    for the paper-matched hyperparameters and batch_size guidance)."""
    return train()


if __name__ == "__main__":
    main()
