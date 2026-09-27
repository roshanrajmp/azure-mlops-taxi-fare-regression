from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. DATA FILES
# ============================================================

X_TRAIN_FILE = (
    PROCESSED_DATA_DIR / "X_train.csv"
)

X_TEST_FILE = (
    PROCESSED_DATA_DIR / "X_test.csv"
)

Y_TRAIN_FILE = (
    PROCESSED_DATA_DIR / "y_train.csv"
)

Y_TEST_FILE = (
    PROCESSED_DATA_DIR / "y_test.csv"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

X_train = pd.read_csv(
    X_TRAIN_FILE
)

X_test = pd.read_csv(
    X_TEST_FILE
)

y_train = pd.read_csv(
    Y_TRAIN_FILE
).squeeze()

y_test = pd.read_csv(
    Y_TEST_FILE
).squeeze()

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 4. BASE XGBOOST MODEL
# ============================================================

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# 5. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    "n_estimators": [
        100,
        200,
        300,
        500,
    ],

    "max_depth": [
        3,
        4,
        5,
        6,
        8,
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1,
        0.2,
    ],

    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0,
    ],

    "min_child_weight": [
        1,
        3,
        5,
        7,
    ],

    "gamma": [
        0,
        0.1,
        0.2,
        0.5,
    ],
}


# ============================================================
# 6. RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(
    estimator=xgb_model,

    param_distributions=param_distributions,

    n_iter=20,

    scoring="neg_root_mean_squared_error",

    cv=5,

    verbose=1,

    random_state=42,

    n_jobs=-1,

    return_train_score=True,
)


# ============================================================
# 7. START TUNING
# ============================================================

print("\n" + "=" * 70)
print("STARTING RANDOMIZED SEARCH")
print("=" * 70)

search.fit(
    X_train,
    y_train
)


# ============================================================
# 8. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(
    search.best_params_
)


# ============================================================
# 9. BEST CROSS-VALIDATION SCORE
# ============================================================

best_cv_rmse = (
    -search.best_score_
)

print(
    f"\nBest CV RMSE: "
    f"{best_cv_rmse:.4f}"
)


# ============================================================
# 10. BEST MODEL
# ============================================================

best_model = search.best_estimator_


# ============================================================
# 11. TEST SET PREDICTION
# ============================================================

predictions = best_model.predict(
    X_test
)


# ============================================================
# 12. TEST METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 70)
print("TUNED MODEL TEST RESULTS")
print("=" * 70)

print(
    f"MAE : {mae:.4f}"
)

print(
    f"MSE : {mse:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R2  : {r2:.4f}"
)


# ============================================================
# 13. SAVE TUNED MODEL
# ============================================================

tuned_model_path = (
    MODEL_DIR / "tuned_xgboost_model.pkl"
)

joblib.dump(
    best_model,
    tuned_model_path
)

print(
    "\nTuned model saved to:"
)

print(
    tuned_model_path
)


# ============================================================
# 14. SAVE BEST PARAMETERS
# ============================================================

best_params_path = (
    MODEL_DIR / "best_xgboost_params.txt"
)

with open(
    best_params_path,
    "w"
) as file:

    for key, value in search.best_params_.items():

        file.write(
            f"{key}={value}\n"
        )


print(
    "Best parameters saved to:"
)

print(
    best_params_path
)


# ============================================================
# 15. SAVE TUNING RESULTS
# ============================================================

cv_results = pd.DataFrame(
    search.cv_results_
)

cv_results_path = (
    MODEL_DIR / "xgboost_cv_results.csv"
)

cv_results.to_csv(
    cv_results_path,
    index=False
)

print(
    "CV results saved to:"
)

print(
    cv_results_path
)


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)