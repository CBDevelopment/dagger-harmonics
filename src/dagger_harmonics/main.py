import pandas as pd

from dagger_harmonics.utils import (
    load_data_df,
    plot_supermag,
    plot_supermag_geo,
    plot_omni,
)
from dagger_harmonics.config import settings
from dagger_harmonics.data.datasets import (
    OMNIDataset,
    SuperMAGDataset,
)
from dagger_harmonics.train import train


def main():
    df = load_data_df(settings.DATA_PATH / "val_data_2010.p")

    past_omni = OMNIDataset(df["past_omni"], df["past_dates"])
    # Bx, By, Bz are the GSM, or Geocentric Solar Magnetospheric coordinates
    # Origin for GSM is center of the Earth, X-axis points from Earth to Sun, Z-axis is perpendicular to the Earth's magnetic dipole axis, and Y-axis completes the right-handed system.
    future_supermag = SuperMAGDataset(df["future_supermag"], df["future_dates"])
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
    # plot_supermag(supermag_df, feature="dbh")
    # plot_supermag_geo(supermag_df, feature="dbh")


def train_model():
    model = train(max_records=2000)
    return model


if __name__ == "__main__":
    # train_model()
    main()
