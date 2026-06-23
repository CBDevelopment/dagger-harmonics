from pathlib import Path
import pandas as pd
import numpy as np


def load_data(file_path: Path):
    import pickle

    with file_path.open("rb") as file:
        data = pickle.load(file)

    return data


def load_data_df(file_path: Path) -> pd.DataFrame:
    df = pd.read_pickle(file_path)
    return pd.DataFrame(df)


def unix_to_datetime(unix_time):
    """Convert Unix epoch seconds to a human-readable datetime string."""
    from datetime import datetime

    return datetime.fromtimestamp(unix_time)


def _setup_polar_ax(ax):
    # MLT frame: noon (mlt_rad=π) at top, clockwise progression
    ax.set_theta_offset(-np.pi / 2)
    ax.set_theta_direction(-1)

    # Set radial limits matching your co-latitude radians
    ax.set_ylim(0, np.radians(55))
    ax.set_yticks(np.radians([10, 20, 30, 40, 50]))

    # FIXED: Map co-latitude steps (10, 20...) to true Magnetic Latitudes (80°, 70°...)
    ax.set_yticklabels(["80°", "70°", "60°", "50°", "40°"], fontsize=8)
    ax.set_rlabel_position(
        45
    )  # Positions latitude ticks cleanly along a 45-degree diagonal line

    # Ticks at mlt_rad = MLT_hours * π/12
    mlts = [0, 3, 6, 9, 12, 15, 18, 21]
    ax.set_xticks([h * np.pi / 12 for h in mlts])

    # Extra padding (\n) keeps the text clear of the outer boundary circle
    ax.set_xticklabels(
        ["00\n☾", "\n03", "06\n", "09\n", "12\n☀", "\n15", "18\n", "21\n"], fontsize=9
    )


def plot_supermag(supermag_data: list[list[float]], feature: str = "dbh"):
    import matplotlib.pyplot as plt
    from dagger_harmonics.data.datasets import get_supermag_features

    columns = get_supermag_features()
    df = pd.DataFrame(supermag_data, columns=columns)

    if feature == "dbh":
        df["dbh"] = (df["dbe_nez"] ** 2 + df["dbn_nez"] ** 2) ** 0.5

    # Convert MLT to radians (theta)
    df["mlt_rad"] = np.deg2rad(df["MLT"] * 15)

    df["colat_rad"] = np.radians(90 - df["MAGLAT"])  # co-latitude in radians

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    _setup_polar_ax(ax)

    # Use transformed radial coordinate 'r_lat'
    sc = ax.scatter(
        df["mlt_rad"],
        df["colat_rad"],
        c=df[feature],
        cmap="plasma",
        s=20,
        edgecolors="none",
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(f"{feature} Units")
    ax.set_title("MLT vs. Magnetic Latitude Map", pad=30, fontsize=12)
    plt.show()
