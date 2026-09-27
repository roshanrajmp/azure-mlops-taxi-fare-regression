from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

from config import RAW_DATA_DIR


def download_dataset() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_california_housing(as_frame=True)

    df = dataset.frame

    output_file = RAW_DATA_DIR / "california_housing.csv"

    df.to_csv(output_file, index=False)

    print(f"Dataset saved to: {output_file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    download_dataset()