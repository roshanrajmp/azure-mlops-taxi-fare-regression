from pathlib import Path

import mlflow
from mlflow import MlflowClient


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


# ============================================================
# 2. MLflow CONFIGURATION
# ============================================================

TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

EXPERIMENT_NAME = "boston-house-price-regression"

RUN_NAME = "xgboost-tuned"

REGISTERED_MODEL_NAME = "boston-house-price-xgboost"


# ============================================================
# 3. CONNECT TO MLFLOW
# ============================================================

print("=" * 70)
print("MLFLOW MODEL REGISTRATION")
print("=" * 70)

mlflow.set_tracking_uri(TRACKING_URI)

print(
    f"Tracking URI: {TRACKING_URI}"
)


# ============================================================
# 4. FIND EXPERIMENT
# ============================================================

client = MlflowClient(
    tracking_uri=TRACKING_URI
)

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:

    raise RuntimeError(
        f"Experiment not found: "
        f"{EXPERIMENT_NAME}"
    )

print(
    f"Experiment ID: {experiment.experiment_id}"
)


# ============================================================
# 5. FIND TUNED RUN
# ============================================================

runs = client.search_runs(
    experiment_ids=[
        experiment.experiment_id
    ],
    filter_string=(
        "attributes.run_name= "
        f"'{RUN_NAME}'"
    ),
    order_by=[
        "attributes.start_time DESC"
    ],
)

if not runs:

    raise RuntimeError(
        f"Run not found: {RUN_NAME}"
    )


run = runs[0]

run_id = run.info.run_id

print(
    f"Found run: {RUN_NAME}"
)

print(
    f"Run ID: {run_id}"
)


# ============================================================
# 6. VERIFY MODEL ARTIFACT
# ============================================================

model_uri = (
    f"runs:/{run_id}/xgboost_model"
)

print(
    f"\nModel URI: {model_uri}"
)


# ============================================================
# 7. REGISTER MODEL
# ============================================================

print("\nRegistering model...")

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
)


# ============================================================
# 8. DISPLAY VERSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL REGISTERED")
print("=" * 70)

print(
    f"Model name: "
    f"{registered_model.name}"
)

print(
    f"Model version: "
    f"{registered_model.version}"
)

print(
    f"Source: "
    f"{registered_model.source}"
)

print(
    f"Run ID: "
    f"{run_id}"
)

print("\n" + "=" * 70)
print("MODEL REGISTRATION COMPLETED")
print("=" * 70)
