import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.interpolate import griddata


# ---------------------------------------------------------------------------
# SuperMAG column definitions
# Confirmed by inspecting val_data_2010.p — see inspect_data.py
# ---------------------------------------------------------------------------

SUPERMAG_COLS = {
    "maglat": 0,  # magnetic latitude (degrees, always 40–90)
    "mlt": 1,  # magnetic local time (hours, 0–24)
    "dbe": 2,  # δbe — east perturbation (nT, signed)
    "dbn": 3,  # δbn — north perturbation (nT, signed)
    "ddbe_dt": 4,  # dδbe/dt — east time derivative (nT/min, signed)
    "ddbn_dt": 5,  # dδbn/dt — north time derivative (nT/min, signed)
}


def supermag_col(supermag, col):
    """
    Extract a named column from a supermag array of shape (T, N, 6) or (N, 6).
    Always returns the most recent timestep as shape (N,).

    Parameters
    ----------
    supermag : array-like — past_supermag or future_supermag from a data record
    col      : str — one of "maglat", "mlt", "dbe", "dbn", "ddbe_dt", "ddbn_dt"
    """
    arr = np.asarray(supermag)
    if arr.ndim == 3:
        arr = arr[-1]  # (T, N, 6) -> most recent timestep (N, 6)
    return arr[:, SUPERMAG_COLS[col]]


def supermag_dbH(supermag):
    """
    Compute horizontal perturbation magnitude δbH = √(δbe² + δbn²).
    Always positive.  Standard proxy for GIC risk.
    Returns shape (N,) using the most recent timestep.
    """
    dbe = supermag_col(supermag, "dbe")
    dbn = supermag_col(supermag, "dbn")
    return np.sqrt(dbe**2 + dbn**2)


# ---------------------------------------------------------------------------
# SqueezedNorm — power-law stretch around zero for diverging colormaps.
# Prevents large storm spikes from washing out smaller variations.
# Ported from the original DAGGER plotting utilities.
# ---------------------------------------------------------------------------


class SqueezedNorm(matplotlib.colors.Normalize):
    """
    Diverging normalisation with independent power-law compression on each
    side of `mid`.  Values near `mid` are expanded; extreme values are
    compressed.  s1 / s2 control the exponent above / below mid.
    """

    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.mid = mid
        self.s1 = s1
        self.s2 = s2
        f = lambda x, zero, vmax, s: (
            np.abs((x - zero) / (vmax - zero)) ** (1.0 / s) * 0.5
        )
        self.g = lambda x, zero, vmin, vmax, s1, s2: (
            f(x, zero, vmax, s1) * (x >= zero) - f(x, zero, vmin, s2) * (x < zero) + 0.5
        )
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid, self.vmin, self.vmax, self.s1, self.s2)
        r = np.clip(np.nan_to_num(r, nan=0.5), 0.0, 1.0)
        return np.ma.masked_array(r)


# ---------------------------------------------------------------------------
# Polar axes styling
# ---------------------------------------------------------------------------


def _style_polar_ax(ax, title=""):
    """
    Apply standard MLT/MCOLAT polar axes styling.
    Orientation: noon (12h) at top, midnight (00h) at bottom,
                 dawn (06h) right, dusk (18h) left.
    Uses set_theta_offset(-π/2) — same as original DAGGER plotting code.
    """
    ax.set_theta_offset(-np.pi / 2)
    ax.set_theta_direction(-1)  # clockwise

    # Co-latitude rings labelled in degrees
    colat_ticks = np.radians([10, 20, 30, 40, 50])
    ax.set_yticks(colat_ticks)
    ax.set_yticklabels(
        [f"{int(np.degrees(c))}°" for c in colat_ticks],
        fontsize=7,
        color="white",
        alpha=0.6,
    )
    ax.set_ylim(0, np.radians(55))

    # MLT cardinal labels
    mlts = [12, 15, 18, 21, 0, 3, 6, 9]
    mlt_labels = ["12\nNoon", "15", "18\nDusk", "21", "00\nMid", "03", "06\nDawn", "09"]
    ax.set_xticks(np.radians(np.array(mlts) * 15))
    ax.set_xticklabels(mlt_labels, fontsize=7, color="white")

    ax.grid(color="white", alpha=0.15, linewidth=0.5)
    ax.spines["polar"].set_color("#333355")
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color="white", fontsize=11, pad=16)


def _add_colorbar(fig, ax, sc, label):
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12)
    cb.set_label(label, color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")


def _format_time_label(value):
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    stamp = arr.ravel()[-1]
    if isinstance(stamp, np.datetime64):
        return np.datetime_as_string(stamp, unit="m")
    if np.issubdtype(np.asarray(stamp).dtype, np.number):
        # Dataset stores epoch seconds as floats.
        try:
            return np.datetime_as_string(np.datetime64(int(stamp), "s"), unit="m")
        except (ValueError, OverflowError):
            return str(stamp)
    return str(stamp)


def _build_title(title, timestamp):
    stamp = _format_time_label(timestamp)
    if not stamp:
        return title
    suffix = f"{stamp}".strip()
    if title:
        return f"{title} ({suffix})"
    return suffix


def _plot_scatter_panel(
    fig, ax, title, mlt_rad, mcolat_rad, values, cmap, norm, size, label
):
    _style_polar_ax(ax, title=title)
    sc = ax.scatter(
        mlt_rad,
        mcolat_rad,
        c=values,
        cmap=cmap,
        norm=norm,
        s=size,
        zorder=5,
        edgecolors="k",
        linewidths=0.2,
    )
    _add_colorbar(fig, ax, sc, label)


def _prepare_heatmap_grid(theta_min=0.0, theta_max=2 * np.pi, r_max=np.radians(55.0)):
    theta = np.linspace(theta_min, theta_max, 241)
    radius = np.linspace(0.0, r_max, 121)
    theta_g, radius_g = np.meshgrid(theta, radius)
    return theta_g, radius_g


def _interpolate_station_values_to_grid(mlt_rad, mcolat_rad, values, theta_g, radius_g):
    finite = np.isfinite(mlt_rad) & np.isfinite(mcolat_rad) & np.isfinite(values)
    theta = np.asarray(mlt_rad)[finite].ravel()
    radius = np.asarray(mcolat_rad)[finite].ravel()
    vals = np.asarray(values)[finite].ravel()

    # Duplicate points across 2pi boundaries to reduce interpolation seams.
    theta_wrap = np.concatenate([theta - 2 * np.pi, theta, theta + 2 * np.pi])
    radius_wrap = np.concatenate([radius, radius, radius])
    vals_wrap = np.concatenate([vals, vals, vals])

    grid = griddata(
        points=np.column_stack([theta_wrap, radius_wrap]),
        values=vals_wrap,
        xi=(theta_g, radius_g),
        method="linear",
    )

    # Fill sparse edge gaps with nearest-neighbor interpolation.
    if np.isnan(grid).any():
        grid_nn = griddata(
            points=np.column_stack([theta_wrap, radius_wrap]),
            values=vals_wrap,
            xi=(theta_g, radius_g),
            method="nearest",
        )
        grid = np.where(np.isnan(grid), grid_nn, grid)

    return grid


def _extract_channel(record, source, col):
    if source not in record:
        raise KeyError(f"Record missing source key: {source}")
    if col == "dbH":
        return supermag_dbH(record[source])
    return supermag_col(record[source], col)


def _select_min_mean_max_indices(mae):
    mae = np.asarray(mae)
    idx_min = int(np.nanargmin(mae))
    idx_max = int(np.nanargmax(mae))
    mean_val = float(np.nanmean(mae))
    idx_mean = int(np.nanargmin(np.abs(mae - mean_val)))
    return idx_min, idx_mean, idx_max


def plot_figure7_style(
    records,
    col="dbH",
    target_source="past_supermag",
    compare_source="future_supermag",
    save_path=None,
    storm_label="Dataset",
):
    """
    Build a Figure-7-style 2x3 MLT-MCOLAT heatmap figure from local records.

    Top row: target map at min / mean / max MAE records.
    Bottom row: comparison map (proxy forecast) for the same records.

    MAE is computed as station-wise mean absolute error between target_source and
    compare_source for each record.
    """
    if not records:
        raise ValueError("No records provided")

    maes = []
    for rec in records:
        target = _extract_channel(rec, target_source, col)
        comp = _extract_channel(rec, compare_source, col)
        finite = np.isfinite(target) & np.isfinite(comp)
        maes.append(float(np.nanmean(np.abs(target[finite] - comp[finite]))))

    idx_min, idx_mean, idx_max = _select_min_mean_max_indices(maes)
    chosen = [idx_min, idx_mean, idx_max]
    labels = ["min MAE", "mean MAE", "max MAE"]

    # Shared color scale across all six panels for direct comparison.
    all_vals = []
    for i in chosen:
        all_vals.append(_extract_channel(records[i], target_source, col))
        all_vals.append(_extract_channel(records[i], compare_source, col))
    stack = np.concatenate([np.asarray(v).ravel() for v in all_vals])
    finite_stack = stack[np.isfinite(stack)]
    if finite_stack.size == 0:
        raise ValueError("No finite values available for plotting")

    if col == "dbH":
        vmin = 0.0
        vmax = float(np.nanpercentile(finite_stack, 99))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = "plasma"
        cbar_label = "Horizontal magnetic field perturbation (dBH) (nT)"
    else:
        absmax = float(np.nanpercentile(np.abs(finite_stack), 99))
        norm = SqueezedNorm(vmin=-absmax, vmax=absmax, mid=0, s1=2, s2=2)
        cmap = "PuOr_r"
        cbar_label = f"{col} (nT)"

    theta_g, radius_g = _prepare_heatmap_grid()

    fig, axes = plt.subplots(
        2,
        3,
        subplot_kw={"projection": "polar"},
        figsize=(14, 8),
        facecolor="white",
        constrained_layout=True,
    )

    mappable = None
    for col_idx, (record_idx, label) in enumerate(zip(chosen, labels)):
        rec = records[record_idx]
        mlt_rad = np.asarray(rec["coords_radians"][0]).ravel()
        mcolat_rad = np.asarray(rec["coords_radians"][1]).ravel()

        target_vals = _extract_channel(rec, target_source, col)
        comp_vals = _extract_channel(rec, compare_source, col)

        for row_idx, values in enumerate([target_vals, comp_vals]):
            ax = axes[row_idx, col_idx]
            _style_polar_ax(ax, "")
            ax.set_facecolor("white")
            ax.grid(color="0.8", alpha=0.8, linewidth=0.6)

            grid = _interpolate_station_values_to_grid(
                mlt_rad, mcolat_rad, values, theta_g, radius_g
            )
            mappable = ax.pcolormesh(
                theta_g,
                radius_g,
                grid,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )
            ax.scatter(mlt_rad, mcolat_rad, s=5, c="k", alpha=0.35)

            base = "Target SuperMAG" if row_idx == 0 else "30-min forecast proxy"
            ax.set_title(
                f"{base}\n{label} (record {record_idx})", fontsize=10, color="black"
            )

    fig.suptitle(
        f"Figure-7-style MLT-MCOLAT heatmaps ({storm_label})",
        fontsize=14,
        y=0.98,
    )
    cb = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cb.set_label(cbar_label)

    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------


def plot_mlt_mcolat(
    coords,
    data=None,
    vmin=None,
    vmax=None,
    cmap="PuOr_r",
    point_size=20,
    colorbar_label="nT",
    title=None,
    timestamp=None,
    save_path=None,
):
    """
    MLT / Magnetic Co-Latitude polar plot.

    Renders up to three panels depending on what is provided:
      Panel 1 — target / observed station values          (always shown)
      Panel 2 — model predictions at the same stations    (if predictions given)
      Panel 3 — continuous SH reconstruction on a grid   (if sh_grid given)

    Parameters
    ----------
    coords : tuple of (mlt_rad, mcolat_rad)
        Directly from record["coords_radians"].
        mlt_rad    : 90 - (MLT_hours/24 * 360), in radians
        mcolat_rad : 90 - MAGLAT_degrees, in radians
    data : array-like, shape (N,), optional
        Observed values — use supermag_dbH() or supermag_col() to extract.
    vmin, vmax : float, optional
        Colour scale limits. Auto-computed from target if not given.
        For δbH (always positive) pass vmin=0.
        For signed components (δbe, δbn) leave as None for ±max centering.
    cmap : str
        "plasma" for δbH (unsigned); "PuOr_r" or "RdBu_r" for signed.
    colorbar_label : str
        Label on the colorbar.
    title : str, optional
        Overall figure title.
    timestamp : array-like or scalar, optional
        Used to annotate plots with the observation time. If array-like,
        the most recent timestamp is used.
    save_path : str or None
        Save PNG here; None opens an interactive window.
    """
    # Unpack and NaN-mask coords
    mlt_rad = np.asarray(coords[0]).ravel()
    mcolat_rad = np.asarray(coords[1]).ravel()
    mask = np.isfinite(mlt_rad) & np.isfinite(mcolat_rad)
    mlt_rad = mlt_rad[mask]
    mcolat_rad = mcolat_rad[mask]

    if data is not None:
        data = np.asarray(data).ravel()[mask]

    # Auto colour limits
    ref = data
    if ref is not None:
        finite = ref[np.isfinite(ref)]
        absmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    else:
        absmax = 1.0
    vmin = vmin if vmin is not None else -absmax
    vmax = vmax if vmax is not None else absmax

    # Norm: plain for positive-only data, squeezed diverging for signed
    if vmin >= 0:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = SqueezedNorm(vmin=vmin, vmax=vmax, mid=0, s1=2, s2=2)

    plt.style.use("default")
    fig, axes = plt.subplots(
        1,
        1,
        subplot_kw={"projection": "polar"},
        figsize=(5 * 1, 5.5),
        facecolor="#0d1117",
    )
    axes = [axes]

    panel = 0

    # --- Panel 1: observed target ---
    if data is not None:
        _plot_scatter_panel(
            fig,
            axes[panel],
            title="SuperMAG",
            mlt_rad=mlt_rad,
            mcolat_rad=mcolat_rad,
            values=data,
            cmap=cmap,
            norm=norm,
            size=point_size,
            label=colorbar_label,
        )
    panel += 1

    full_title = _build_title(title, timestamp)
    if full_title:
        fig.suptitle(full_title, color="white", fontsize=13, y=1.02)

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
    from analysis.data_utils import load_data, DEFAULT_DATA_PATH

    data = load_data(DEFAULT_DATA_PATH)
    record = data[0]
    coords = record["coords_radians"]
    past_time = record["past_dates"][-1]

    # δbH — unsigned magnitude, plasma colormap
    plot_mlt_mcolat(
        coords,
        data=supermag_dbH(record["past_supermag"]),
        vmin=0,
        cmap="plasma",
        colorbar_label="δbH (nT)",
        title="δbH",
        timestamp=past_time,
        save_path="mlt_mcolat_dbH.png",
    )

    # δbe - signed east component, diverging colormap
    plot_mlt_mcolat(
        coords,
        data=supermag_col(record["past_supermag"], "dbe"),
        cmap="PuOr_r",
        colorbar_label="δbe (nT)",
        title="δbe",
        timestamp=past_time,
        save_path="mlt_mcolat_dbe.png",
    )

    # δbn - signed north component, diverging colormap
    plot_mlt_mcolat(
        coords,
        data=supermag_col(record["past_supermag"], "dbn"),
        cmap="PuOr_r",
        colorbar_label="δbn (nT)",
        title="δbn",
        timestamp=past_time,
        save_path="mlt_mcolat_dbn.png",
    )

    # Figure-7-style 2x3 heatmaps using available data only.
    plot_figure7_style(
        records=data,
        col="dbH",
        target_source="past_supermag",
        compare_source="future_supermag",
        save_path="figure7_style_dbH.png",
        storm_label="val_data_2010",
    )
