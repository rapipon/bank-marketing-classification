from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_PATH = DATA_DIR / "bank-additional-full.csv"

# UCI dataset URL
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

EXPECTED_SHAPE = (41188, 21)


def download_data() -> None:
    """Download and extract the required dataset from UCI."""

    print("Downloading Bank Marketing dataset...")

    with urlopen(DATA_URL) as response:
        archive_data = response.read()

    # Open the main archive
    with ZipFile(BytesIO(archive_data)) as main_archive:

        # Find the archive containing the additional dataset
        nested_archive_name = next(
            name for name in main_archive.namelist()
            if name.endswith("bank-additional.zip")
        )

        nested_archive_data = main_archive.read(nested_archive_name)

    # Open the nested archive
    with ZipFile(BytesIO(nested_archive_data)) as nested_archive:

        # Find the full dataset
        dataset_name = next(
            name for name in nested_archive.namelist()
            if name.endswith("bank-additional-full.csv")
        )

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        DATA_PATH.write_bytes(nested_archive.read(dataset_name))

    print(f"Dataset saved to: {DATA_PATH}")


def load_data() -> pd.DataFrame:
    """Load Bank Marketing dataset from local storage or UCI."""

    if not DATA_PATH.exists():
        download_data()

    df = pd.read_csv(DATA_PATH, sep=";")

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected dataset shape: {df.shape}. "
            f"Expected: {EXPECTED_SHAPE}. "
            "Check the dataset file."
        )

    return df
