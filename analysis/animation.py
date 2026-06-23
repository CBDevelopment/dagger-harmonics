from dagger_harmonics.utils import load_data
from dagger_harmonics.config import settings

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from datetime import datetime, timezone


def _supermag_last(rec):
    # use future_supermag only (do not fall back to past_supermag)
    if "future_supermag" not in rec:
        raise KeyError(
            "future_supermag missing from record; not falling back to past_supermag"
        )
    arr = np.asarray(rec["future_supermag"])
    # future_supermag contains readings for a single future date (axis 0 == 1)
    # return the first slice when 3D, else the array itself
    return arr[0] if arr.ndim == 3 else arr


def _dbH(sm):
    return np.sqrt(sm[:, 2] ** 2 + sm[:, 3] ** 2)


def _unix_ts(rec):
    # use future_dates (single date corresponding to all supermag readings)
    arr = np.asarray(rec.get("future_dates", [])).ravel()
    return float(arr[0]) if arr.size else 0.0


def _fmt_ts(unix_time):
    try:
        return datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    except Exception:
        return ""


def _setup_polar_ax(ax):
    # MLT frame: noon (mlt_rad=π) at top, clockwise progression
    ax.set_theta_offset(-np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, np.radians(55))
    ax.set_yticks(np.radians([10, 20, 30, 40, 50]))
    ax.set_yticklabels(["10°", "20°", "30°", "40°", "50°"], fontsize=7)
    # Ticks at mlt_rad = MLT_hours * π/12
    mlts = [0, 3, 6, 9, 12, 15, 18, 21]
    ax.set_xticks([h * np.pi / 12 for h in mlts])
    ax.set_xticklabels(["00\nMid", "03", "06", "09", "12\n☀", "15", "18", "21"])


def _compute_vmax(frames):
    sample = frames[:: max(1, len(frames) // 200)]
    vals = np.concatenate(
        [_dbH(_supermag_last(r)) for r in sample if "future_supermag" in r]
    )
    finite = vals[np.isfinite(vals)]
    return float(np.nanpercentile(finite, 99)) if finite.size else 100.0


def animate_stations(data, step=60, interval=80):
    frames = data[::step]
    vmax = _compute_vmax(frames)

    rec0 = frames[0]
    theta0 = np.asarray(rec0["coords_radians"][0]).ravel()
    r0 = np.asarray(rec0["coords_radians"][1]).ravel()
    dbh0 = _dbH(_supermag_last(rec0))
    show0 = np.isfinite(theta0) & np.isfinite(r0) & np.isfinite(dbh0)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    _setup_polar_ax(ax)
    title = ax.set_title("", pad=20)

    sc = ax.scatter(
        theta0[show0],
        r0[show0],
        c=dbh0[show0],
        s=20,
        cmap="plasma",
        vmin=0,
        vmax=vmax,
        zorder=5,
    )
    fig.colorbar(sc, ax=ax, label="δbH (nT)", shrink=0.7, pad=0.1)

    def update(i):
        rec = frames[i]
        theta = np.asarray(rec["coords_radians"][0]).ravel()
        r = np.asarray(rec["coords_radians"][1]).ravel()
        dbh = _dbH(_supermag_last(rec))
        show = np.isfinite(theta) & np.isfinite(r) & np.isfinite(dbh)
        sc.set_offsets(np.column_stack([theta[show], r[show]]))
        sc.set_array(dbh[show])
        title.set_text(_fmt_ts(_unix_ts(rec)))

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=False
    )
    # keep reference to avoid garbage collection before rendering
    plt.show()  # noqa: B018
    return anim


def main():
    data = load_data(settings.DATA_PATH / "val_data_2010.p")
    animate_stations(data, 60)


if __name__ == "__main__":
    main()
