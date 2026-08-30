import pickle

import numpy as np
import pytest

from dagger_harmonics.data.datasets import (
    OMNIDataset,
    SuperMAGDataset,
    load_dagger_data,
)

N_RECORDS = 3
T_PAST = 4
N_OMNI = 2
N_STATIONS = 5
OMNI_FEATURES = np.array(["bx", "by"])
SUPERMAG_FEATURES = np.array(
    ["MAGLAT", "MLT", "dbe_nez", "dbn_nez", "ddbe_dt", "ddbn_dt"]
)


def _write_pickle(path, obj) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


@pytest.fixture
def val_path(tmp_path):
    """Build a tiny fake val_data_2010.p plus its sidecar files in tmp_path."""
    records = []
    for i in range(N_RECORDS):
        records.append(
            {
                "past_omni": np.full((T_PAST, N_OMNI), float(i), dtype=np.float32),
                "past_dates": np.arange(T_PAST, dtype=float),
                "future_supermag": np.full(
                    (1, N_STATIONS, 6), float(i), dtype=np.float32
                ),
                "future_dates": np.array([[float(i)]]),
                "coords_radians": (np.zeros(N_STATIONS), np.zeros(N_STATIONS)),
            }
        )

    path = tmp_path / "val_data_2010.p"
    _write_pickle(path, records)
    _write_pickle(tmp_path / "omni_features.p", OMNI_FEATURES)
    _write_pickle(tmp_path / "supermag_features.p", SUPERMAG_FEATURES)
    _write_pickle(
        tmp_path / "scalers.p",
        {
            "omni": [np.zeros(N_OMNI), np.ones(N_OMNI)],
            "supermag": [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        },
    )
    return path


# ---------------------------------------------------------------------------
# load_dagger_data
# ---------------------------------------------------------------------------


def test_load_dagger_data_df_has_one_row_per_record(val_path):
    dd = load_dagger_data(val_path)
    assert len(dd.df) == N_RECORDS


def test_load_dagger_data_loads_omni_features_from_sidecar(val_path):
    dd = load_dagger_data(val_path)
    assert list(dd.omni_features) == ["bx", "by"]


def test_load_dagger_data_loads_supermag_features_from_sidecar(val_path):
    dd = load_dagger_data(val_path)
    assert list(dd.supermag_features) == list(SUPERMAG_FEATURES)


def test_load_dagger_data_omni_scalers(val_path):
    dd = load_dagger_data(val_path)
    np.testing.assert_array_equal(dd.omni_scalers["omni_mean"], [0.0, 0.0])
    np.testing.assert_array_equal(dd.omni_scalers["omni_std"], [1.0, 1.0])


def test_load_dagger_data_supermag_scalers(val_path):
    dd = load_dagger_data(val_path)
    assert dd.supermag_scalers["dbe_mean"] == 0.0
    assert dd.supermag_scalers["dbn_std"] == 1.0


def test_load_dagger_data_reads_sidecars_from_path_parent_not_settings(
    val_path, monkeypatch
):
    """Sidecars must be found next to `path`, regardless of settings.DATA_PATH."""
    from dagger_harmonics.config import settings

    monkeypatch.setattr(settings, "DATA_PATH", val_path.parent / "does-not-exist")
    dd = load_dagger_data(val_path)
    assert list(dd.omni_features) == ["bx", "by"]


# ---------------------------------------------------------------------------
# OMNIDataset
# ---------------------------------------------------------------------------


def test_omni_dataset_len_matches_record_count(val_path):
    ds = OMNIDataset(load_dagger_data(val_path))
    assert len(ds) == N_RECORDS


def test_omni_dataset_getitem_returns_past_omni_record(val_path):
    ds = OMNIDataset(load_dagger_data(val_path))
    np.testing.assert_array_equal(ds[1], np.full((T_PAST, N_OMNI), 1.0))


def test_omni_dataset_exposes_features_and_scalers(val_path):
    ds = OMNIDataset(load_dagger_data(val_path))
    assert list(ds.features) == ["bx", "by"]
    assert ds.scalers["omni_mean"].tolist() == [0.0, 0.0]


def test_omni_dataset_get_dates_returns_split_dates(val_path):
    ds = OMNIDataset(load_dagger_data(val_path))
    np.testing.assert_array_equal(ds.get_dates(0), np.arange(T_PAST, dtype=float))


# ---------------------------------------------------------------------------
# SuperMAGDataset
# ---------------------------------------------------------------------------


def test_supermag_dataset_len_matches_record_count(val_path):
    ds = SuperMAGDataset(load_dagger_data(val_path))
    assert len(ds) == N_RECORDS


def test_supermag_dataset_getitem_squeezes_single_future_timestep(val_path):
    ds = SuperMAGDataset(load_dagger_data(val_path))
    item = ds[2]
    assert item.shape == (N_STATIONS, 6)
    np.testing.assert_array_equal(item, np.full((N_STATIONS, 6), 2.0))


def test_supermag_dataset_exposes_features_and_scalers(val_path):
    ds = SuperMAGDataset(load_dagger_data(val_path))
    assert list(ds.features) == list(SUPERMAG_FEATURES)
    assert ds.scalers["dbn_mean"] == 0.0
