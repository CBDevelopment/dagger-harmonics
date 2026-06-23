from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm_y

from analysis.data_utils import DEFAULT_DATA_PATH, load_data
from analysis.plotting import supermag_col, supermag_dbH


def _extract_values(record, source, col):
    if source not in record:
        raise KeyError(f"Missing '{source}' in record")

    if col == "dbH":
        return supermag_dbH(record[source])
    return supermag_col(record[source], col)


def _fit_spherical_harmonics(theta, phi, values, l_max):
    mask = np.isfinite(theta) & np.isfinite(phi) & np.isfinite(values)
    theta = np.asarray(theta).ravel()[mask]
    phi = np.asarray(phi).ravel()[mask]
    values = np.asarray(values).ravel()[mask]

    if theta.size == 0:
        raise ValueError("No finite values to fit")

    lms = []
    cols = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            lms.append((l, m))
            cols.append(sph_harm_y(l, m, theta, phi))

    design = np.vstack(cols).T
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    return coeffs, lms


def _evaluate_spherical_harmonics(coeffs, lms, theta, phi):
    recon = np.zeros_like(theta, dtype=np.complex128)
    for coeff, (l, m) in zip(coeffs, lms):
        recon += coeff * sph_harm_y(l, m, theta, phi)
    return recon


def plot_entry_spherical_harmonics(
    entry_index,
    l_max=6,
    data_path=DEFAULT_DATA_PATH,
    source="past_supermag",
    col="dbH",
):
    data = load_data(Path(data_path))
    if entry_index < 0 or entry_index >= len(data):
        raise IndexError(f"entry_index out of range: {entry_index}")

    record = data[entry_index]
    coords = record["coords_radians"]
    phi = np.asarray(coords[0]).ravel()
    theta = np.asarray(coords[1]).ravel()

    values = _extract_values(record, source, col)
    coeffs, lms = _fit_spherical_harmonics(theta, phi, values, l_max)

    grid_theta, grid_phi = np.mgrid[0 : np.pi : 120j, 0 : 2 * np.pi : 120j]
    recon = _evaluate_spherical_harmonics(coeffs, lms, grid_theta, grid_phi)
    recon_real = np.real(recon)

    r = np.abs(recon_real)
    x_plot = r * np.sin(grid_theta) * np.cos(grid_phi)
    y_plot = r * np.sin(grid_theta) * np.sin(grid_phi)
    z_plot = r * np.cos(grid_theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_plot,
        y_plot,
        z_plot,
        cmap="viridis",
        rstride=1,
        cstride=1,
        facecolors=plt.cm.viridis(recon_real / np.nanmax(np.abs(recon_real))),
        linewidth=0,
        antialiased=False,
    )
    ax.set_title(
        f"Entry {entry_index} | {source}:{col} | l_max={l_max}",
        pad=12,
    )
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Fit and visualize a spherical harmonic surface for a record.",
    )
    parser.add_argument("--entry", type=int, default=0, help="Record index")
    parser.add_argument("--lmax", type=int, default=6, help="Max SH degree")
    parser.add_argument(
        "--source",
        type=str,
        default="past_supermag",
        choices=["past_supermag", "future_supermag"],
        help="Which array to use",
    )
    parser.add_argument(
        "--col",
        type=str,
        default="dbH",
        choices=["dbH", "maglat", "mlt", "dbe", "dbn", "ddbe_dt", "ddbn_dt"],
        help="Which SuperMAG column to map",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to val_data_2010.p",
    )
    args = parser.parse_args()

    plot_entry_spherical_harmonics(
        entry_index=args.entry,
        l_max=args.lmax,
        data_path=args.data,
        source=args.source,
        col=args.col,
    )


if __name__ == "__main__":
    main()
