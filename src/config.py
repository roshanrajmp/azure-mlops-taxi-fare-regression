from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42

TARGET_COLUMN = "median_house_value"

RAW_DATA_FILE = RAW_DATA_DIR / "boston.csv"

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR / "boston_processed.csv"
)