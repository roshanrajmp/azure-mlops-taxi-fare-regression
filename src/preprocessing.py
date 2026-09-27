from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
    RANDOM_STATE,
    PROCESSED_DATA_DIR,
)


# ---------------------------------------------------------
# Output paths
# ---------------------------------------------------------

TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "X_train.csv"
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "X_test.csv"

TRAIN_TARGET_FILE = PROCESSED_DATA_DIR / "y_train.csv"
TEST_TARGET_FILE = PROCESSED_DATA_DIR / "y_test.csv"

SCALER_FILE = PROCESSED_DATA_DIR / "scaler.pkl"


# ---------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------

def preprocess_data():
    """
    Load the validated dataset, separate features and target,
    split into training and testing datasets, fit the scaler
    only on training data, transform both datasets, and save
    the resulting files.
    """

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------

    if not Path(PROCESSED_DATA_FILE).exists():
        raise FileNotFoundError(
            f"Dataset not found: {PROCESSED_DATA_FILE}"
        )

    df = pd.read_csv(PROCESSED_DATA_FILE)

    print(f"Input dataset: {PROCESSED_DATA_FILE}")
    print(f"Dataset shape: {df.shape}")

    # -----------------------------------------------------
    # 2. Separate features and target
    # -----------------------------------------------------

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print(f"Target column: {TARGET_COLUMN}")

    # -----------------------------------------------------
    # 3. Train/Test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    print("\nTrain/Test split:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    # -----------------------------------------------------
    # 4. Feature scaling
    # -----------------------------------------------------

    print("\nApplying StandardScaler...")

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit scaler ONLY on training data.
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using the scaler
    # learned from training data.
    X_test_scaled = scaler.transform(X_test)

    # Convert NumPy arrays back to DataFrames
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index,
    )

    print("Feature scaling: COMPLETE")

    # -----------------------------------------------------
    # 5. Save processed datasets
    # -----------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train_scaled.to_csv(
        TRAIN_FEATURES_FILE,
        index=False,
    )

    X_test_scaled.to_csv(
        TEST_FEATURES_FILE,
        index=False,
    )

    y_train.to_csv(
        TRAIN_TARGET_FILE,
        index=False,
    )

    y_test.to_csv(
        TEST_TARGET_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # 6. Save scaler
    # -----------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    # -----------------------------------------------------
    # 7. Final summary
    # -----------------------------------------------------

    print("\nSaved files:")

    print(f"X_train: {TRAIN_FEATURES_FILE}")
    print(f"X_test:  {TEST_FEATURES_FILE}")
    print(f"y_train: {TRAIN_TARGET_FILE}")
    print(f"y_test:  {TEST_TARGET_FILE}")
    print(f"Scaler:  {SCALER_FILE}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_data()