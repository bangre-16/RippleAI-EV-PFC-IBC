import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
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

data = pd.read_csv("dataset.csv")

# Input features
features = [
    "Switching_Freq_Hz",
    "Load_Resistance_Ohm",
    "Vin_RMS_V",
    "Duty_Cycle_Pct",
    "Inductance_uH",
    "Vout_DC_V"
]

# Target
target = "Current_Ripple_A"

X = data[features]
y = data[target]


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

# Keep the same split used in the original SVR model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Dataset loaded successfully.")
print("Total samples :", len(data))
print("Training samples :", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 3. CREATE SVR PIPELINE
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])


# ============================================================
# 4. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {
    "svr__kernel": ["rbf"],
    
    "svr__C": [
        10,
        50,
        100,
        200,
        500,
        1000
    ],
    
    "svr__epsilon": [
        0.001,
        0.005,
        0.01,
        0.02
    ],
    
    "svr__gamma": [
        "scale",
        0.01,
        0.05,
        0.1
    ]
}


# ============================================================
# 5. RANDOMIZED HYPERPARAMETER TUNING
# ============================================================

print("\nStarting SVR hyperparameter tuning...")
print("CPU workers : 2")
print("Cross-validation folds : 3")
print("Number of parameter combinations : 20")

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="neg_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=2,
    verbose=1
)

random_search.fit(X_train, y_train)


# ============================================================
# 6. BEST HYPERPARAMETERS
# ============================================================

print("\n==========================================")
print("BEST SVR HYPERPARAMETERS")
print("==========================================")

print(random_search.best_params_)

best_cv_mse = -random_search.best_score_

print("\nBest CV MSE :", best_cv_mse)


# ============================================================
# 7. GET BEST MODEL
# ============================================================

best_model = random_search.best_estimator_


# ============================================================
# 8. PREDICTION ON UNTOUCHED TEST DATA
# ============================================================

y_pred = best_model.predict(X_test)


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

# MAPE
mape = np.mean(
    np.abs((y_test - y_pred) / y_test)
) * 100

# MASE
# Naive baseline based on training data
naive_error = np.mean(
    np.abs(
        np.diff(y_train)
    )
)

model_error = np.mean(
    np.abs(y_test - y_pred)
)

if naive_error != 0:
    mase = model_error / naive_error
else:
    mase = np.nan


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("TUNED SVR TEST RESULTS")
print("==========================================")

print(f"R²   : {r2:.6f}")
print(f"MSE  : {mse:.9f}")
print(f"RMSE : {rmse:.6f} A")
print(f"MAE  : {mae:.6f} A")
print(f"MAPE : {mape:.6f} %")
print(f"MASE : {mase:.6f}")


# ============================================================
# 11. SAVE METRICS
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
    "SVR_Tuned_Metrics.csv",
    index=False
)


# ============================================================
# 12. SAVE BEST HYPERPARAMETERS
# ============================================================

hyperparameters = pd.DataFrame({
    "Parameter": list(random_search.best_params_.keys()),
    "Best_Value": list(random_search.best_params_.values())
})

hyperparameters.to_csv(
    "SVR_Tuned_Hyperparameters.csv",
    index=False
)


# ============================================================
# 13. SAVE ACTUAL VS PREDICTED DATA
# ============================================================

results = pd.DataFrame({
    "Actual_Current_Ripple_A": y_test.values,
    "Predicted_Current_Ripple_A": y_pred
})

results.to_csv(
    "SVR_Tuned_Actual_vs_Predicted.csv",
    index=False
)


# ============================================================
# 14. ACTUAL VS PREDICTED GRAPH
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
    linestyle="--"
)

plt.xlabel("Actual Current Ripple (A)")
plt.ylabel("Predicted Current Ripple (A)")

plt.title(
    "SVR Tuned - Actual vs Predicted Current Ripple"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "SVR_Tuned_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    best_model,
    "SVR_Tuned_model.pkl"
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("FILES SAVED SUCCESSFULLY")
print("==========================================")

print("1. SVR_Tuned_model.pkl")
print("2. SVR_Tuned_Hyperparameters.csv")
print("3. SVR_Tuned_Metrics.csv")
print("4. SVR_Tuned_Actual_vs_Predicted.csv")
print("5. SVR_Tuned_Actual_vs_Predicted_Ripple.png")

print("\nSVR hyperparameter tuning completed.")