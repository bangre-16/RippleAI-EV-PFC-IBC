import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
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


# ============================================================
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. GAUSSIAN PROCESS KERNEL
# ============================================================

kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
)


# ============================================================
# 6. TRAIN GAUSSIAN PROCESS MODEL
# ============================================================

model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=2,
    random_state=42
)

print("\nTraining Gaussian Process Regression...")
model.fit(X_train_scaled, y_train)

print("Training completed.")


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = model.predict(X_test_scaled)


# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

mape = np.mean(
    np.abs((y_test.values - y_pred) / y_test.values)
) * 100

naive_error = np.mean(
    np.abs(np.diff(y_train.values))
)

mase = mae / naive_error


# ============================================================
# 9. DISPLAY METRICS IN TERMINAL
# ============================================================

print("\n========================================")
print("   GAUSSIAN PROCESS MODEL PERFORMANCE")
print("========================================")

print(f"R²     : {r2:.6f}")
print(f"MSE    : {mse:.9f}")
print(f"RMSE   : {rmse:.6f} A")
print(f"MAE    : {mae:.6f} A")
print(f"MAPE   : {mape:.6f} %")
print(f"MASE   : {mase:.6f}")

print("========================================")


# ============================================================
# 10. SAVE TRAINED MODEL AND SCALER
# ============================================================

joblib.dump(model, "GPR_model.pkl")
joblib.dump(scaler, "GPR_scaler.pkl")


# ============================================================
# 11. SAVE ACTUAL VS PREDICTED VALUES
# ============================================================

results = pd.DataFrame({
    "Actual_Current_Ripple_A": y_test.values,
    "Predicted_Current_Ripple_A": y_pred
})

results.to_csv(
    "GPR_Actual_vs_Predicted.csv",
    index=False
)


# ============================================================
# 12. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Current Ripple (A)")
plt.ylabel("Predicted Current Ripple (A)")

plt.title(
    "Gaussian Process Regression: Actual vs Predicted Current Ripple"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "GPR_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. SAVE PERFORMANCE METRICS
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
    "GPR_Metrics.csv",
    index=False
)


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\nFiles generated successfully:")
print("----------------------------------------")
print("GPR_model.pkl")
print("GPR_scaler.pkl")
print("GPR_Actual_vs_Predicted.csv")
print("GPR_Actual_vs_Predicted_Ripple.png")
print("GPR_Metrics.csv")
print("----------------------------------------")