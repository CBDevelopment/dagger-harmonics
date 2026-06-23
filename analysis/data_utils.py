import pickle
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*numpy.core.numeric is deprecated.*",
)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "val_data_2010.p"

# Loaded 51060 records from D:\projects\research\SETI\dagger-harmonics\data\val_data_2010.p
#  - coords_radians: tuple (location of the measurement on the globe)
#  - future_dates: ndarray
#  - future_supermag: ndarray (our target)
#  - past_dates: ndarray
#  - past_omni: ndarray (input)
#  - past_supermag: ndarray


def load_data(file_path: Path) -> list[dict]:
    with file_path.open("rb") as file:
        data = pickle.load(file)

    if not data:
        raise ValueError(f"Dataset is empty: {file_path}")

    return data
