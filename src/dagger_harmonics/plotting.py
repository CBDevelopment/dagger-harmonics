import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prep_coords(coords, values=None):
    """Unpack, ravel, and NaN-mask coords and optional values."""
    mlt_rad, mcolat_rad = coords
    mlt_rad = np.asarray(mlt_rad).ravel()
    mcolat_rad = np.asarray(mcolat_rad).ravel()
    mask = np.isfinite(mlt_rad) & np.isfinite(mcolat_rad)
    mlt_rad = mlt_rad[mask]
    mcolat_rad = mcolat_rad[mask]
    if values is not None:
        values = np.asarray(values).ravel()[mask]
    return mlt_rad, mcolat_rad, values


def supermag_dbH(past_supermag):
    """
    Compute horizontal perturbation magnitude δbH = √(δbe² + δbn²)
    from a past_supermag array of shape (T, N, 6) or (N, 6).

    Returns array of shape (N,) using the most recent timestep.
    """
    arr = np.asarray(past_supermag)
    if arr.ndim == 3:
        arr = arr[-1]  # most recent timestep -> (N, 6)
    dbe = arr[:, 2]
    dbn = arr[:, 3]
    return np.sqrt(dbe**2 + dbn**2)


# ---------------------------------------------------------------------------
# MLT / MCOLAT polar plot  —  matches DAGGER paper Figures 7 & 8
# ---------------------------------------------------------------------------


def plot_mlt_mcolat(
    coords,
    values=None,
    cmap="plasma",
    point_size=20,
    colorbar_label="δbH (nT)",
    title="SuperMAG — MLT / MCOLAT",
    save_path=None,
):
    """
    Polar plot in Magnetic Local Time / Magnetic Co-Latitude space.
    Reproduces the style of Figures 7 & 8 in Upendran et al. (2022).

    Parameters
    ----------
    coords : tuple of (mlt_rad, mcolat_rad)
        mlt_rad    : MLT in radians — 0/2π = midnight, π = noon
        mcolat_rad : magnetic co-latitude in radians — 0 = pole, π/2 = equator
    values : array-like, optional
        Scalar per station for colouring (e.g. δbH).  If None, points
        are plotted in a single colour.
    save_path : str or None
        Save PNG to this path; if None, opens an interactive window.
    """
    mlt_rad, mcolat_rad, values = _prep_coords(coords, values)

    fig = plt.figure(figsize=(7, 7), facecolor="#0d1117")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#0d1117")

    # Midnight at top, increasing clockwise — matches paper convention
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Radial axis: co-latitude in radians, labelled in degrees
    colat_ticks = np.radians([10, 20, 30, 40, 50])
    ax.set_yticks(colat_ticks)
    ax.set_yticklabels(
        [f"{int(np.degrees(c))}°" for c in colat_ticks],
        color="white",
        fontsize=8,
        alpha=0.6,
    )
    ax.set_ylim(0, np.radians(55))

    # Angular axis: MLT hours
    mlt_hours = np.arange(0, 24, 3)
    ax.set_xticks(np.radians(mlt_hours * 15))
    ax.set_xticklabels(
        [f"{h:02d}h" for h in mlt_hours],
        color="white",
        fontsize=9,
    )

    ax.grid(color="white", alpha=0.15, linewidth=0.5)
    ax.spines["polar"].set_color("#333355")

    # Data
    if values is not None:
        sc = ax.scatter(
            mlt_rad,
            mcolat_rad,
            c=values,
            cmap=cmap,
            s=point_size,
            zorder=5,
            edgecolors="k",
            linewidths=0.2,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(colorbar_label, color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    else:
        ax.scatter(
            mlt_rad,
            mcolat_rad,
            color="tomato",
            s=point_size,
            zorder=5,
            edgecolors="k",
            linewidths=0.2,
        )

    ax.set_title(title, color="white", fontsize=13, pad=22)
    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dagger_harmonics.data_utils import load_data, DEFAULT_DATA_PATH

    data = load_data(DEFAULT_DATA_PATH)

    for i in range(1):
        record = data[i * 50]

        coords = record["coords_radians"]
        dbH = supermag_dbH(record["past_supermag"])

        plot_mlt_mcolat(
            coords,
            values=dbH,
            colorbar_label="δbH (nT)",
            save_path=f"output/mlt_mcolat_{i}.png",
        )
