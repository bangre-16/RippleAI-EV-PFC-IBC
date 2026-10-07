import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

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
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. TRAIN KNN MODEL
# ============================================================

model = KNeighborsRegressor(
    n_neighbors=5,
    weights="distance",
    metric="euclidean"
)

print("\nTraining KNN...")

model.fit(X_train_scaled, y_train)

print("Training completed.")


# ============================================================
# 6. PREDICTION
# ============================================================

y_pred = model.predict(X_test_scaled)


# ============================================================
# 7. PERFORMANCE METRICS
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
# 8. DISPLAY METRICS IN TERMINAL
# ============================================================

print("\n========================================")
print("          KNN MODEL PERFORMANCE")
print("========================================")

print(f"R²     : {r2:.6f}")
print(f"MSE    : {mse:.9f}")
print(f"RMSE   : {rmse:.6f} A")
print(f"MAE    : {mae:.6f} A")
print(f"MAPE   : {mape:.6f} %")
print(f"MASE   : {mase:.6f}")

print("========================================")


# ============================================================
# 9. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "KNN_model.pkl"
)

joblib.dump(
    scaler,
    "KNN_scaler.pkl"
)


# ============================================================
# 10. SAVE ACTUAL VS PREDICTED VALUES
# ============================================================

results = pd.DataFrame({
    "Actual_Current_Ripple_A": y_test.values,
    "Predicted_Current_Ripple_A": y_pred
})

results.to_csv(
    "KNN_Actual_vs_Predicted.csv",
    index=False
)


# ============================================================
# 11. ACTUAL VS PREDICTED GRAPH
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
    "KNN: Actual vs Predicted Current Ripple"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "KNN_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. SAVE PERFORMANCE METRICS
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
    "KNN_Metrics.csv",
    index=False
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\nFiles generated successfully:")
print("----------------------------------------")
print("KNN_model.pkl")
print("KNN_scaler.pkl")
print("KNN_Actual_vs_Predicted.csv")
print("KNN_Actual_vs_Predicted_Ripple.png")
print("KNN_Metrics.csv")
print("----------------------------------------")