import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

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
# 4. DEFINE BASE MODELS
# ============================================================

# ---------- SVR ----------
svr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(
        kernel="rbf",
        C=100,
        gamma="scale",
        epsilon=0.001
    ))
])


# ---------- Gaussian Process ----------
gpr_kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e2)
    )
    + WhiteKernel(
        noise_level=1e-5,
        noise_level_bounds=(1e-8, 1e-1)
    )
)

gpr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("gpr", GaussianProcessRegressor(
        kernel=gpr_kernel,
        n_restarts_optimizer=1,
        random_state=42
    ))
])


# ---------- Random Forest ----------
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

print("\nGenerating out-of-fold predictions...")

kf = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. GENERATE META FEATURES
# ============================================================

print("\nTraining SVR base model...")
svr_oof = cross_val_predict(
    svr_model,
    X_train,
    y_train,
    cv=kf,
    n_jobs=1
)

print("SVR completed.")


print("\nTraining Gaussian Process base model...")
gpr_oof = cross_val_predict(
    gpr_model,
    X_train,
    y_train,
    cv=kf,
    n_jobs=1
)

print("Gaussian Process completed.")


print("\nTraining Random Forest base model...")
rf_oof = cross_val_predict(
    rf_model,
    X_train,
    y_train,
    cv=kf,
    n_jobs=-1
)

print("Random Forest completed.")


# ============================================================
# 7. CREATE META-TRAINING DATA
# ============================================================

meta_X_train = np.column_stack([
    svr_oof,
    gpr_oof,
    rf_oof
])

print("\nMeta-training data created.")


# ============================================================
# 8. TRAIN META MODEL
# ============================================================

meta_model = Ridge(alpha=1.0)

meta_model.fit(
    meta_X_train,
    y_train
)

print("Meta-model training completed.")


# ============================================================
# 9. TRAIN BASE MODELS ON COMPLETE TRAINING DATA
# ============================================================

print("\nTraining final base models...")

svr_model.fit(X_train, y_train)

print("SVR final model completed.")

gpr_model.fit(X_train, y_train)

print("Gaussian Process final model completed.")

rf_model.fit(X_train, y_train)

print("Random Forest final model completed.")


# ============================================================
# 10. GENERATE TEST PREDICTIONS FROM BASE MODELS
# ============================================================

svr_test_pred = svr_model.predict(X_test)

gpr_test_pred = gpr_model.predict(X_test)

rf_test_pred = rf_model.predict(X_test)


# ============================================================
# 11. CREATE META-TEST DATA
# ============================================================

meta_X_test = np.column_stack([
    svr_test_pred,
    gpr_test_pred,
    rf_test_pred
])


# ============================================================
# 12. FINAL STACKED PREDICTION
# ============================================================

y_pred = meta_model.predict(meta_X_test)


# ============================================================
# 13. PERFORMANCE METRICS
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
# 14. DISPLAY PERFORMANCE
# ============================================================

print("\n========================================")
print("     STACKED META-ENSEMBLE PERFORMANCE")
print("========================================")

print(f"R²     : {r2:.6f}")
print(f"MSE    : {mse:.9f}")
print(f"RMSE   : {rmse:.6f} A")
print(f"MAE    : {mae:.6f} A")
print(f"MAPE   : {mape:.6f} %")
print(f"MASE   : {mase:.6f}")

print("========================================")


# ============================================================
# 15. SAVE COMPLETE STACKED MODEL
# ============================================================

stacked_model = {
    "SVR": svr_model,
    "GPR": gpr_model,
    "RandomForest": rf_model,
    "MetaModel": meta_model,
    "Features": features
}

joblib.dump(
    stacked_model,
    "Stacked_Ensemble_model.pkl"
)


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual_Current_Ripple_A": y_test.values,
    "Predicted_Current_Ripple_A": y_pred
})

results.to_csv(
    "Stacked_Ensemble_Actual_vs_Predicted.csv",
    index=False
)


# ============================================================
# 17. ACTUAL VS PREDICTED GRAPH
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

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Current Ripple (A)")
plt.ylabel("Predicted Current Ripple (A)")

plt.title(
    "Stacked Meta-Ensemble: Actual vs Predicted Current Ripple"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Stacked_Ensemble_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.close()


# ============================================================
# 18. SAVE PERFORMANCE METRICS
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
    "Stacked_Ensemble_Metrics.csv",
    index=False
)


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\nFiles generated successfully:")
print("----------------------------------------")
print("Stacked_Ensemble_model.pkl")
print("Stacked_Ensemble_Actual_vs_Predicted.csv")
print("Stacked_Ensemble_Actual_vs_Predicted_Ripple.png")
print("Stacked_Ensemble_Metrics.csv")
print("----------------------------------------")