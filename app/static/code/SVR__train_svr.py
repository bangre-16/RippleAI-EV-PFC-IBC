import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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
print("Columns:", list(df.columns))


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
# 5. TRAIN SVR MODEL
# ============================================================

model = SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.001
)

model.fit(X_train_scaled, y_train)


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
    np.abs((y_test.values - y_pred) / y_test.values)
) * 100

# MASE
naive_error = np.mean(
    np.abs(np.diff(y_train.values))
)

mase = mae / naive_error


# ============================================================
# 8. DISPLAY PERFORMANCE IN TERMINAL
# ============================================================

print("\n========================================")
print("        SVR MODEL PERFORMANCE")
print("========================================")

print(f"R²     : {r2:.6f}")
print(f"MSE    : {mse:.10f}")
print(f"RMSE   : {rmse:.6f} A")
print(f"MAE    : {mae:.6f} A")
print(f"MAPE   : {mape:.6f} %")
print(f"MASE   : {mase:.6f}")

print("========================================")


# ============================================================
# 9. SAVE TRAINED SVR MODEL
# ============================================================

joblib.dump(model, "SVR_model.pkl")

# Save scaler because SVR was trained using scaled inputs
joblib.dump(scaler, "SVR_scaler.pkl")

print("\nTrained model saved:")
print("SVR_model.pkl")

print("\nScaler saved:")
print("SVR_scaler.pkl")


# ============================================================
# 10. SAVE ACTUAL VS PREDICTED VALUES
# ============================================================

results = pd.DataFrame({
    "Actual_Current_Ripple_A": y_test.values,
    "Predicted_Current_Ripple_A": y_pred
})

results.to_csv(
    "SVR_Actual_vs_Predicted.csv",
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

# Ideal prediction line
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

plt.title("SVR: Actual vs Predicted Current Ripple")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "SVR_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "R2",
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
    "SVR_Metrics.csv",
    index=False
)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\nAll SVR files generated successfully:")
print("----------------------------------------")
print("SVR_model.pkl")
print("SVR_scaler.pkl")
print("SVR_Actual_vs_Predicted.csv")
print("SVR_Actual_vs_Predicted_Ripple.png")
print("SVR_Metrics.csv")
print("----------------------------------------")