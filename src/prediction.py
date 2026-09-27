from pathlib import Path

import mlflow
import pandas as pd
import joblib


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

SCALER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scaler.pkl"
)


# ============================================================
# 2. MLFLOW CONFIGURATION
# ============================================================

TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

MODEL_NAME = "boston-house-price-xgboost"

MODEL_VERSION = "1"


# ============================================================
# 3. FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
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
]


# ============================================================
# 4. SAMPLE INPUT
# ============================================================

SAMPLE_DATA = {
    "crime_rate": 0.00632,
    "residential_land_pct": 18.00,
    "industrial_land_pct": 2.310,
    "river_boundary": 0,
    "nitric_oxide_concentration": 0.5380,
    "avg_rooms": 6.5750,
    "old_housing_pct": 65.20,
    "employment_distance": 4.0900,
    "highway_access_index": 1,
    "property_tax_rate": 296.0,
    "student_teacher_ratio": 15.30,
    "demographic_index": 396.90,
    "lower_status_pct": 4.98,
}


# ============================================================
# 5. LOAD SCALER
# ============================================================

def load_scaler():

    if not SCALER_FILE.exists():

        raise FileNotFoundError(
            f"Scaler not found: {SCALER_FILE}"
        )

    scaler = joblib.load(
        SCALER_FILE
    )

    print("Scaler loaded successfully.")

    return scaler


# ============================================================
# 6. LOAD MODEL
# ============================================================

def load_model():

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )

    print(
        f"Loading model: {model_uri}"
    )

    model = mlflow.pyfunc.load_model(
        model_uri
    )

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# 7. PREDICTION
# ============================================================

def predict():

    print("=" * 70)
    print("MODEL PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    scaler = load_scaler()

    # --------------------------------------------------------
    # Load MLflow model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Create raw input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [SAMPLE_DATA],
        columns=FEATURE_COLUMNS,
    )

    print("\nRaw input:")

    print(input_data)

    # --------------------------------------------------------
    # Apply same scaling used during training
    # --------------------------------------------------------

    scaled_values = scaler.transform(
        input_data
    )

    scaled_data = pd.DataFrame(
        scaled_values,
        columns=FEATURE_COLUMNS,
    )

    print("\nScaled input:")

    print(scaled_data)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        scaled_data
    )

    predicted_value = float(
        prediction[0]
    )

    print("\n" + "=" * 70)

    print(
        f"Predicted median house value: "
        f"{predicted_value:.2f}"
    )

    print("=" * 70)

    return predicted_value


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    predict()