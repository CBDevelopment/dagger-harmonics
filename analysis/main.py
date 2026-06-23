from pathlib import Path
from pprint import pprint
from rich import print
import pandas as pd
import datetime

from analysis.data_utils import load_data, DEFAULT_DATA_PATH


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    data = load_data(DEFAULT_DATA_PATH)
    print(f"Loaded {len(data)} records from {DEFAULT_DATA_PATH}")

    for key in sorted(data[0].keys()):
        print(f" - {key}: {type(data[0][key]).__name__}")
        print(f"   Sample value: {repr(data[0][key])[:100]}")


if __name__ == "__main__":
    # main()
    data = load_data(DEFAULT_DATA_PATH)
    print(len(data))  # 51060 observations
    keys = data[0].keys()
    print(keys)
    for key in keys:
        print(f"{key}: {type(data[0][key])}")
        print(f"{key}: {len(data[0][key])}")

    print("--- Coords ---")
    # Coords structured like (array[0][0], array[0][0]), where coords are (MLT, Colat)
    mlt_coords = data[0]["coords_radians"][0][0]  # radians
    colat_coords = data[0]["coords_radians"][1][0]  # radians
    print(len(mlt_coords))  # 175
    print(len(colat_coords))  # 175
    # 175 stations in coords

    print("--- Future Dates ---")
    future_date = data[0]["future_dates"][0][0]
    print(future_date)
    formated_date = datetime.datetime.fromtimestamp(future_date, tz=datetime.UTC)
    print(formated_date)

    print("--- Past Dates ---")
    past_date = data[0]["past_dates"]
    print(len(past_date))

    print("--- Past SuperMAG ---")
    past_supermag = data[0]["past_supermag"]
    print(len(past_supermag))

    print("-- Pas OMNI ---")
    past_omni = data[0]["past_omni"]
    print(len(past_omni))
