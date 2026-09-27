from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

RESULTS_FILE = MODEL_DIR / "model_results.csv"

TUNED_MODEL_FILE = (
    MODEL_DIR / "tuned_xgboost_model.pkl"
)

TUNED_PARAMS_FILE = (
    MODEL_DIR / "best_xgboost_params.txt"
)

CV_RESULTS_FILE = (
    MODEL_DIR / "xgboost_cv_results.csv"
)


# ============================================================
# 2. TRACKING FUNCTION
# ============================================================

def track_tuned_model():

    print("=" * 70)
    print("MLFLOW TUNED MODEL TRACKING")
    print("=" * 70)

    # --------------------------------------------------------
    # MLflow database
    # --------------------------------------------------------

    tracking_uri = f"sqlite:///{MLFLOW_DB}"

    mlflow.set_tracking_uri(tracking_uri)

    print(
        f"MLflow tracking URI: {tracking_uri}"
    )

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    experiment_name = (
        "boston-house-price-regression"
    )

    mlflow.set_experiment(
        experiment_name
    )

    print(
        f"Experiment: {experiment_name}"
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not TUNED_MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Tuned model not found: "
            f"{TUNED_MODEL_FILE}"
        )

    # --------------------------------------------------------
    # Load tuned model
    # --------------------------------------------------------

    model = joblib.load(
        TUNED_MODEL_FILE
    )

    print(
        f"\nLoaded model: "
        f"{TUNED_MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="xgboost-tuned"
    ):

        # ----------------------------------------------------
        # Log model parameters
        # ----------------------------------------------------

        params = model.get_params()

        parameters_to_log = {

            "n_estimators":
                params.get("n_estimators"),

            "max_depth":
                params.get("max_depth"),

            "learning_rate":
                params.get("learning_rate"),

            "subsample":
                params.get("subsample"),

            "min_child_weight":
                params.get("min_child_weight"),

            "gamma":
                params.get("gamma"),

            "colsample_bytree":
                params.get("colsample_bytree"),

        }

        mlflow.log_params(
            parameters_to_log
        )

        print("\nParameters logged:")

        for key, value in parameters_to_log.items():

            print(
                f"{key}: {value}"
            )

        # ----------------------------------------------------
        # Log tuned model metrics
        # ----------------------------------------------------

        metrics = {

            "mae": 1.8795,

            "mse": 7.4948,

            "rmse": 2.7377,

            "r2": 0.8978,

            "cv_rmse": 3.4844,

        }

        mlflow.log_metrics(
            metrics
        )

        print("\nMetrics logged:")

        for key, value in metrics.items():

            print(
                f"{key}: {value}"
            )

        # ----------------------------------------------------
        # Log model artifact
        # ----------------------------------------------------

        mlflow.xgboost.log_model(
            model,
            artifact_path="xgboost_model",
        )

        print(
            "\nXGBoost model artifact logged."
        )

        # ----------------------------------------------------
        # Log additional files
        # ----------------------------------------------------

        if TUNED_PARAMS_FILE.exists():

            mlflow.log_artifact(
                str(TUNED_PARAMS_FILE)
            )

        if CV_RESULTS_FILE.exists():

            mlflow.log_artifact(
                str(CV_RESULTS_FILE)
            )

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tag(
            "model_type",
            "XGBoost"
        )

        mlflow.set_tag(
            "stage",
            "tuned"
        )

        mlflow.set_tag(
            "dataset",
            "Boston Housing"
        )

        # ----------------------------------------------------
        # Run information
        # ----------------------------------------------------

        run_id = mlflow.active_run().info.run_id

        print(
            f"\nMLflow Run ID: {run_id}"
        )

    print("\n" + "=" * 70)
    print(
        "TUNED MODEL TRACKING COMPLETED"
    )
    print("=" * 70)


# ============================================================
# 3. MAIN
# ============================================================

if __name__ == "__main__":

    track_tuned_model()