from pathlib import Path

import joblib
import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


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

# Local packaged MLflow model.
#
# On the Mac:
#   project_root/model
#
# Inside Docker:
#   /app/model
#
MODEL_ARTIFACT_DIR = PROJECT_ROOT / "model"


# ============================================================
# 2. MLFLOW CONFIGURATION
# ============================================================

TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

MODEL_NAME = "boston-house-price-xgboost"

MODEL_VERSION = "1"


# ============================================================
# 3. FEATURE COLUMNS
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
# 4. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Boston House Price Prediction API",
    description="MLOps regression prediction API",
    version="1.0.0",
)


# ============================================================
# 5. REQUEST MODEL
# ============================================================

class HouseFeatures(BaseModel):

    crime_rate: float
    residential_land_pct: float
    industrial_land_pct: float
    river_boundary: float
    nitric_oxide_concentration: float
    avg_rooms: float
    old_housing_pct: float
    employment_distance: float
    highway_access_index: float
    property_tax_rate: float
    student_teacher_ratio: float
    demographic_index: float
    lower_status_pct: float


# ============================================================
# 6. LOAD SCALER
# ============================================================

def load_scaler():

    if not SCALER_FILE.exists():

        raise FileNotFoundError(
            f"Scaler not found: {SCALER_FILE}"
        )

    return joblib.load(
        SCALER_FILE
    )


# ============================================================
# 7. LOAD ML MODEL
# ============================================================

def load_model():

    # --------------------------------------------------------
    # Docker / packaged model
    # --------------------------------------------------------

    if MODEL_ARTIFACT_DIR.exists():

        print(
            f"Loading packaged MLflow model: "
            f"{MODEL_ARTIFACT_DIR}"
        )

        return mlflow.pyfunc.load_model(
            str(MODEL_ARTIFACT_DIR)
        )

    # --------------------------------------------------------
    # Local development fallback
    # --------------------------------------------------------

    print(
        "Packaged model not found."
    )

    print(
        f"Loading model from MLflow Registry: "
        f"{MODEL_NAME} version {MODEL_VERSION}"
    )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )

    return mlflow.pyfunc.load_model(
        model_uri
    )


# ============================================================
# 8. LOAD ARTIFACTS
# ============================================================

scaler = load_scaler()

model = load_model()


# ============================================================
# 9. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    }


# ============================================================
# 10. PREDICTION
# ============================================================

@app.post("/predict")
def predict(features: HouseFeatures):

    input_data = pd.DataFrame(
        [
            features.model_dump()
        ]
    )

    input_data = input_data[
        FEATURE_COLUMNS
    ]

    scaled_data = scaler.transform(
        input_data
    )

    scaled_data = pd.DataFrame(
        scaled_data,
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(
        scaled_data
    )

    return {
        "prediction": float(prediction[0]),
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }

