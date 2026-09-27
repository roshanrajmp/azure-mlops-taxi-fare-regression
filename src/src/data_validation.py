from pathlib import Path

import pandas as pd

from config import (
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
)


EXPECTED_COLUMNS = [
    "crime_rate",
    "residential_land_pct",
    "industrial_land_pct",
    "river_boundary",
    "nitric_oxide_concentration",
    "avg_rooms",
    "old_housing_pct",
    "employment_distance",
    "highway_access_index",
    "property_tax_rate",
    "student_teacher_ratio",
    "demographic_index",
    "lower_status_pct",
    "median_house_value",
]


def validate_data(file_path: str) -> bool:
    """
    Validate the standardized Boston housing dataset.
    """

    path = Path(file_path)

    # ---------------------------------------------------------
    # 1. File existence
    # ---------------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    # ---------------------------------------------------------
    # 2. Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(path)

    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    print(f"Dataset: {path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # 3. Schema validation
    # ---------------------------------------------------------

    actual_columns = df.columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"\nSchema mismatch.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )

    print("\nSchema validation: PASS")

    # ---------------------------------------------------------
    # 4. Target validation
    # ---------------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column missing: {TARGET_COLUMN}"
        )

    print(
        f"Target column validation: PASS "
        f"({TARGET_COLUMN})"
    )

    # ---------------------------------------------------------
    # 5. Missing-value validation
    # ---------------------------------------------------------

    missing = df.isnull().sum()

    print("\nMissing values:")
    print(missing)

    if missing.sum() > 0:
        raise ValueError(
            "Dataset contains missing values."
        )

    print("Missing-value validation: PASS")

    # ---------------------------------------------------------
    # 6. Duplicate validation
    # ---------------------------------------------------------

    duplicates = df.duplicated().sum()

    print(
        f"\nDuplicate records: {duplicates}"
    )

    if duplicates > 0:
        print(
            "WARNING: Dataset contains duplicate records."
        )
    else:
        print("Duplicate validation: PASS")

    # ---------------------------------------------------------
    # 7. Numeric-column validation
    # ---------------------------------------------------------

    non_numeric = (
        df.select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    if non_numeric:
        raise ValueError(
            f"Unexpected non-numeric columns: "
            f"{non_numeric}"
        )

    print("Numeric-column validation: PASS")

    # ---------------------------------------------------------
    # 8. Target datatype validation
    # ---------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        df[TARGET_COLUMN]
    ):
        raise ValueError(
            "Target column must be numeric."
        )

    print("Target-type validation: PASS")

    # ---------------------------------------------------------
    # 9. Infinite-value validation
    # ---------------------------------------------------------

    numeric_df = df.select_dtypes(
        include="number"
    )

    infinite_values = (
        numeric_df
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(
        f"Infinite values: {infinite_values}"
    )

    if infinite_values > 0:
        raise ValueError(
            "Dataset contains infinite values."
        )

    print("Infinite-value validation: PASS")

    # ---------------------------------------------------------
    # 10. Target statistics
    # ---------------------------------------------------------

    print("\nTarget statistics:")

    print(
        df[TARGET_COLUMN].describe()
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":

    validate_data(PROCESSED_DATA_FILE)