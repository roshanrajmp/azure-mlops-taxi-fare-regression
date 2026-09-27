import pandas as pd

from src.config import PROCESSED_DATA_FILE, TARGET_COLUMN
from src.data_validation import validate_data


def test_processed_dataset_exists():
    assert PROCESSED_DATA_FILE.exists()


def test_processed_dataset_has_expected_shape():
    df = pd.read_csv(PROCESSED_DATA_FILE)

    assert df.shape == (506, 14)


def test_target_column_exists():
    df = pd.read_csv(PROCESSED_DATA_FILE)

    assert TARGET_COLUMN in df.columns


def test_target_column_is_numeric():
    df = pd.read_csv(PROCESSED_DATA_FILE)

    assert pd.api.types.is_numeric_dtype(
        df[TARGET_COLUMN]
    )


def test_validation_passes():
    assert validate_data(PROCESSED_DATA_FILE) is True