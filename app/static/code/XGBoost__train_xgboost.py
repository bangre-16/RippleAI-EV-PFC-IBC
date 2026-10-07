import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset.csv")

print("\nDataset loaded successfully.")
print("Number of samples:", len(df))


# ============================================================
# 2. INPUT FEATURES AND TARGET
# ============================================================

features = [
    "Switching_Freq_Hz",
    "Load_Resistance_Ohm",
    "Vin_RMS_V",
    "Duty_Cycle_Pct",
    "Inductance_uH",
    "Vout_DC_V"
]

target = "Current_Ripple_A"

X = df[features]
y = df[target]


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 4. TRAIN XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 5. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 6. PERFORMANCE METRICS
# ============================================================

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

mape = np.mean(
    np.abs(
        (y_test.values - y_pred) /
        y_test.values
    )
) * 100

naive_error = np.mean(
    np.abs(np.diff(y_train.values))
)

mase = mae / naive_error


# ============================================================
# 7. DISPLAY METRICS IN TERMINAL
# ============================================================

print("\n========================================")
print("        XGBOOST MODEL PERFORMANCE")
print("========================================")

print(f"R²     : {r2:.6f}")
print(f"MSE    : {mse:.9f}")
print(f"RMSE   : {rmse:.6f} A")
print(f"MAE    : {mae:.6f} A")
print(f"MAPE   : {mape:.6f} %")
print(f"MASE   : {mase:.6f}")

print("========================================")


# ============================================================
# 8. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "XGBoost_model.pkl"
)


# ============================================================
# 9. SAVE ACTUAL VS PREDICTED VALUES
# ============================================================

results = pd.DataFrame({
    "Actual_Current_Ripple_A": y_test.values,
    "Predicted_Current_Ripple_A": y_pred
})

results.to_csv(
    "XGBoost_Actual_vs_Predicted.csv",
    index=False
)


# ============================================================
# 10. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

# Ideal prediction line
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Current Ripple (A)")
plt.ylabel("Predicted Current Ripple (A)")

plt.title(
    "XGBoost: Actual vs Predicted Current Ripple"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "XGBoost_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. SAVE PERFORMANCE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "R²",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "MASE"
    ],
    "Value": [
        r2,
        mse,
        rmse,
        mae,
        mape,
        mase
    ]
})

metrics.to_csv(
    "XGBoost_Metrics.csv",
    index=False
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "XGBoost_Feature_Importance.csv",
    index=False
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\nFiles generated successfully:")
print("----------------------------------------")
print("XGBoost_model.pkl")
print("XGBoost_Actual_vs_Predicted.csv")
print("XGBoost_Actual_vs_Predicted_Ripple.png")
print("XGBoost_Metrics.csv")
print("XGBoost_Feature_Importance.csv")
print("----------------------------------------")