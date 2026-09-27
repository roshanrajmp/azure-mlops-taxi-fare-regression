from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. DATA FILES
# ============================================================

X_TRAIN_FILE = (
    PROCESSED_DATA_DIR
    / "X_train.csv"
)

X_TEST_FILE = (
    PROCESSED_DATA_DIR
    / "X_test.csv"
)

Y_TRAIN_FILE = (
    PROCESSED_DATA_DIR
    / "y_train.csv"
)

Y_TEST_FILE = (
    PROCESSED_DATA_DIR
    / "y_test.csv"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL TRAINING")
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
# 4. DEFINE MODELS
# ============================================================

models = {

    "linear_regression": LinearRegression(),

    "ridge": Ridge(
        alpha=1.0
    ),

    "lasso": Lasso(
        alpha=0.01
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),

    "xgboost": XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    ),
}


# ============================================================
# 5. TRAIN MODELS
# ============================================================

results = {}

trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ========================================================
    # 6. EVALUATE MODEL
    # ========================================================

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

    # Store metrics
    results[model_name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

    # Store trained model
    trained_models[
        model_name
    ] = model

    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")


# ============================================================
# 7. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
).T

# Convert model names from the DataFrame index
# into a normal "model" column.
results_df.index.name = "model"

results_df = results_df.reset_index()


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.sort_values(
        by="RMSE"
    )
)


# ============================================================
# 8. SELECT BEST MODEL
# ============================================================

# Find the row containing the lowest RMSE.
best_model_row = results_df.loc[
    results_df["RMSE"].idxmin()
]

# Extract model name from the "model" column.
best_model_name = best_model_row["model"]

# Get the actual trained model object.
best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best model:",
    best_model_name
)

print(
    "Best RMSE:",
    best_model_row["RMSE"]
)

print(
    "Best R2:",
    best_model_row["R2"]
)


# ============================================================
# 9. SAVE BEST MODEL
# ============================================================

best_model_path = (
    MODEL_DIR
    / "best_model.pkl"
)

joblib.dump(
    best_model,
    best_model_path
)

print(
    "\nBest model saved to:",
    best_model_path
)


# ============================================================
# 10. SAVE MODEL RESULTS
# ============================================================

results_path = (
    MODEL_DIR
    / "model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    "Model comparison saved to:",
    results_path
)


# ============================================================
# 11. SAVE MODEL NAME
# ============================================================

best_model_name_path = (
    MODEL_DIR
    / "best_model_name.txt"
)

best_model_name_path.write_text(
    best_model_name
)

print(
    "Best model name saved to:",
    best_model_name_path
)


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)