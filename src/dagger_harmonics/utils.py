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
    """Convert Unix epoch seconds to a UTC datetime."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(unix_time, tz=timezone.utc)


def plot_supermag_geo(df: pd.DataFrame, feature: str = "dbh"):
    """Plot SuperMAG data on a geographic North Polar Stereo map.

    MLT + MAGLAT are converted to geographic lat/lon via aacgmv2. The
    timestamp is read from the ``date`` column (first row), so no separate
    ``dt`` argument is needed.

    Parameters
    ----------
    df : DataFrame with columns MAGLAT, MLT, date, and either the named
         feature column or dbe_nez/dbn_nez for the derived "dbh".
    feature : column name or "dbh" for derived sqrt(dbe_nez^2 + dbn_nez^2)
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import aacgmv2

    df = df.copy()
    if feature == "dbh":
        df["dbh"] = (df["dbe_nez"] ** 2 + df["dbn_nez"] ** 2) ** 0.5

    dt = pd.Timestamp(df["date"].iloc[0]).to_pydatetime()

    mlat = df["MAGLAT"].values
    mlt_hours = df["MLT"].values
    # MLT hours → AACGM magnetic longitude → geographic lat/lon
    mlon = aacgmv2.convert_mlt(mlt_hours, dt, m2a=True)
    geo_lat, geo_lon, _ = aacgmv2.convert_latlon_arr(
        mlat, mlon, 0.0, dt, method_code="A2G"
    )

    vals = df[feature].values
    valid = np.isfinite(geo_lat) & np.isfinite(geo_lon) & np.isfinite(vals)

    proj = ccrs.NorthPolarStereo()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj})
    ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="#b8d4e8", zorder=1)
    ax.add_feature(cfeature.LAND, facecolor="#e8e0d0", zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor="#b8d4e8", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#555555", zorder=4)
    ax.add_feature(
        cfeature.BORDERS, linewidth=0.4, edgecolor="#888888", linestyle=":", zorder=4
    )
    ax.gridlines(
        draw_labels=False,
        linewidth=0.5,
        color="#777777",
        alpha=0.5,
        linestyle="--",
        zorder=4,
    )

    cmap = "plasma" if feature == "dbh" else "PuOr_r"
    sc = ax.scatter(
        geo_lon[valid],
        geo_lat[valid],
        c=vals[valid],
        cmap=cmap,
        s=30,
        edgecolors="#333333",
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.05, shrink=0.7)
    cbar.set_label(f"{feature} (nT)", fontsize=10)
    ax.set_title(
        f"SuperMAG {feature.upper()} — {dt.strftime('%Y-%m-%d %H:%M UTC')}",
        pad=12,
        fontsize=12,
    )
    plt.show()


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


def plot_supermag(df: pd.DataFrame, feature: str = "dbh"):
    import matplotlib.pyplot as plt

    df = df.copy()
    if feature == "dbh":
        df["dbh"] = (df["dbe_nez"] ** 2 + df["dbn_nez"] ** 2) ** 0.5

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


def plot_omni(omni_data: pd.DataFrame):
    import matplotlib.pyplot as plt

    cols = [c for c in omni_data.columns if c != "date"]

    # Grid 3 wide x 5 tall
    ncols = 3
    nrows = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), sharex=True)
    axes = axes.flatten()

    for i, col in enumerate(cols[: nrows * ncols]):
        ax = axes[i]
        ax.plot(omni_data["date"], omni_data[col], label=col)
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid()

    # Hide any remaining empty axes
    for j in range(len(cols), nrows * ncols):
        axes[j].set_visible(False)

    fig.tight_layout()
    plt.show()
