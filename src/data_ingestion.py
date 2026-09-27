from pathlib import Path

import pandas as pd

from config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
)


COLUMN_MAPPING = {
    "CRIM": "crime_rate",
    "ZN": "residential_land_pct",
    "INDUS": "industrial_land_pct",
    "CHAS": "river_boundary",
    "NOX": "nitric_oxide_concentration",
    "RM": "avg_rooms",
    "AGE": "old_housing_pct",
    "DIS": "employment_distance",
    "RAD": "highway_access_index",
    "TAX": "property_tax_rate",
    "PTRATIO": "student_teacher_ratio",
    "B": "demographic_index",
    "LSTAT": "lower_status_pct",
    "MEDV": "median_house_value",
}


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """
    Load the original Kaggle dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {file_path}"
        )

    print(f"Loading dataset: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename source dataset columns to descriptive names.
    """

    missing_columns = [
        column
        for column in COLUMN_MAPPING
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Expected columns missing: {missing_columns}"
        )

    df = df.rename(columns=COLUMN_MAPPING)

    return df


def save_processed_data(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Save standardized dataset.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / "boston_processed.csv"

    df.to_csv(
        output_file,
        index=False,
    )

    print(f"Processed dataset saved to: {output_file}")

    return output_file


def run_ingestion() -> Path:

    print("=" * 60)
    print("DATA INGESTION")
    print("=" * 60)

    # Step 1: Load
    df = load_raw_data(RAW_DATA_FILE)

    # Step 2: Rename columns
    df = rename_columns(df)

    print("\nStandardized columns:")
    for column in df.columns:
        print(f"  - {column}")

    # Step 3: Save
    output_file = save_processed_data(
        df,
        PROCESSED_DATA_DIR,
    )

    print("\n" + "=" * 60)
    print("DATA INGESTION COMPLETED")
    print("=" * 60)

    return output_file


if __name__ == "__main__":
    run_ingestion()