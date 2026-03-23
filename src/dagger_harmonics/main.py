from pathlib import Path
from pprint import pprint

from dagger_harmonics.data_utils import load_data, DEFAULT_DATA_PATH


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
    pprint(data[0]["coords_radians"])
    pprint(data[0]["past_supermag"])
