from dagger_harmonics.utils import load_data_df, plot_supermag
from dagger_harmonics.config import settings
from dagger_harmonics.data.datasets import (
    OMNIDataset,
    SuperMAGDataset,
)


def main():
    df = load_data_df(settings.DATA_PATH / "val_data_2010.p")
    past_omni = OMNIDataset(df["past_omni"])
    future_supermag = SuperMAGDataset(df["future_supermag"])

    plot_supermag(future_supermag[300], feature="dbh")


if __name__ == "__main__":
    main()
