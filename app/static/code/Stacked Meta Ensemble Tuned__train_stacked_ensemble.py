import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("dataset.csv")

features = [
    "Switching_Freq_Hz",
    "Load_Resistance_Ohm",
    "Vin_RMS_V",
    "Duty_Cycle_Pct",
    "Inductance_uH",
    "Vout_DC_V"
]

target = "Current_Ripple_A"

X = data[features]
y = data[target]


# ============================================================
# 2. SAME TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("==========================================")
print("STACKED META-ENSEMBLE")
print("==========================================")

print("Total samples    :", len(data))
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ============================================================
# 3. LOAD TUNED SVR
# ============================================================

print("\nLoading tuned SVR...")

tuned_svr = joblib.load(
    "SVR_Tuned_model.pkl"
)

svr_parameters = tuned_svr.named_steps[
    "svr"
].get_params()

print("Tuned SVR loaded.")


# ============================================================
# 4. LOAD ADVANCED TUNED GPR
# ============================================================

print("\nLoading advanced tuned GPR...")

tuned_gpr = joblib.load(
    "GPR_Tuned_model.pkl"
)

# Extract the optimized kernel found during GPR tuning
optimized_gpr_kernel = tuned_gpr.kernel_

print("Optimized GPR kernel:")
print(optimized_gpr_kernel)

print("Tuned GPR loaded.")


# ============================================================
# 5. RANDOM FOREST PARAMETERS
# ============================================================

print("\nPreparing Random Forest...")

rf_parameters = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": 2
}


# ============================================================
# 6. CREATE 3-FOLD OOF SETUP
# ============================================================

kf = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# OOF predictions
oof_predictions = np.zeros(
    (len(X_train), 3)
)

print("\n==========================================")
print("GENERATING OUT-OF-FOLD PREDICTIONS")
print("==========================================")

print("3-fold cross-validation")
print("Base models:")
print("1. Tuned SVR")
print("2. Advanced tuned GPR")
print("3. Random Forest")


# ============================================================
# 7. OOF TRAINING
# ============================================================

for fold, (train_index, val_index) in enumerate(
    kf.split(X_train),
    start=1
):

    print("\n------------------------------------------")
    print(f"FOLD {fold}/3")
    print("------------------------------------------")

    X_fold_train = X_train.iloc[train_index]
    X_fold_val = X_train.iloc[val_index]

    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]


    # --------------------------------------------------------
    # SVR
    # --------------------------------------------------------

    print("Training SVR...")

    fold_svr = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svr",
            SVR(
                kernel=svr_parameters["kernel"],
                C=svr_parameters["C"],
                epsilon=svr_parameters["epsilon"],
                gamma=svr_parameters["gamma"]
            )
        )
    ])

    fold_svr.fit(
        X_fold_train,
        y_fold_train
    )

    svr_pred = fold_svr.predict(
        X_fold_val
    )

    oof_predictions[
        val_index,
        0
    ] = svr_pred


    # --------------------------------------------------------
    # GPR
    # --------------------------------------------------------

    print("Training GPR...")

    fold_scaler = StandardScaler()

    X_fold_train_scaled = fold_scaler.fit_transform(
        X_fold_train
    )

    X_fold_val_scaled = fold_scaler.transform(
        X_fold_val
    )

    fold_gpr = GaussianProcessRegressor(

        kernel=optimized_gpr_kernel,

        # IMPORTANT:
        # Kernel is already optimized.
        # Do not repeat expensive optimization.
        optimizer=None,

        normalize_y=True,

        random_state=42
    )

    fold_gpr.fit(
        X_fold_train_scaled,
        y_fold_train.values
    )

    gpr_pred = fold_gpr.predict(
        X_fold_val_scaled
    )

    oof_predictions[
        val_index,
        1
    ] = gpr_pred


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print("Training Random Forest...")

    fold_rf = RandomForestRegressor(
        **rf_parameters
    )

    fold_rf.fit(
        X_fold_train,
        y_fold_train
    )

    rf_pred = fold_rf.predict(
        X_fold_val
    )

    oof_predictions[
        val_index,
        2
    ] = rf_pred


    print("Fold completed.")


# ============================================================
# 8. CREATE META-TRAINING DATA
# ============================================================

meta_X = pd.DataFrame({

    "SVR_Prediction":
        oof_predictions[:, 0],

    "GPR_Prediction":
        oof_predictions[:, 1],

    "RandomForest_Prediction":
        oof_predictions[:, 2]

})

meta_y = y_train.values


print("\n==========================================")
print("META-TRAINING DATA CREATED")
print("==========================================")

print(
    "Meta features :",
    meta_X.shape[1]
)

print(
    "Meta samples  :",
    meta_X.shape[0]
)


# ============================================================
# 9. TRAIN RIDGE META-MODEL
# ============================================================

print("\nTraining Ridge Meta-Model...")

ridge_meta = RidgeCV(
    alphas=[
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ],
    cv=3
)

ridge_meta.fit(
    meta_X,
    meta_y
)

print(
    "Best Ridge Alpha :",
    ridge_meta.alpha_
)

print(
    "Meta-model coefficients :"
)

print(
    ridge_meta.coef_
)


# ============================================================
# 10. TRAIN BASE MODELS ON COMPLETE TRAINING DATA
# ============================================================

print("\n==========================================")
print("TRAINING FINAL BASE MODELS")
print("==========================================")

# ------------------------------------------------------------
# FINAL SVR
# ------------------------------------------------------------

print("Training final SVR...")

final_svr = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svr",
        SVR(
            kernel=svr_parameters["kernel"],
            C=svr_parameters["C"],
            epsilon=svr_parameters["epsilon"],
            gamma=svr_parameters["gamma"]
        )
    )
])

final_svr.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# FINAL GPR
# ------------------------------------------------------------

print("Training final GPR...")

final_gpr_scaler = StandardScaler()

X_train_scaled = final_gpr_scaler.fit_transform(
    X_train
)

X_test_scaled = final_gpr_scaler.transform(
    X_test
)

final_gpr = GaussianProcessRegressor(

    kernel=optimized_gpr_kernel,

    optimizer=None,

    normalize_y=True,

    random_state=42
)

final_gpr.fit(
    X_train_scaled,
    y_train.values
)


# ------------------------------------------------------------
# FINAL RANDOM FOREST
# ------------------------------------------------------------

print("Training final Random Forest...")

final_rf = RandomForestRegressor(
    **rf_parameters
)

final_rf.fit(
    X_train,
    y_train
)


print("\nAll final base models trained.")


# ============================================================
# 11. GENERATE TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")


# SVR prediction
test_svr_pred = final_svr.predict(
    X_test
)


# GPR prediction
test_gpr_pred = final_gpr.predict(
    X_test_scaled
)


# Random Forest prediction
test_rf_pred = final_rf.predict(
    X_test
)


# ============================================================
# 12. CREATE META TEST DATA
# ============================================================

meta_test_X = pd.DataFrame({

    "SVR_Prediction":
        test_svr_pred,

    "GPR_Prediction":
        test_gpr_pred,

    "RandomForest_Prediction":
        test_rf_pred

})


# ============================================================
# 13. FINAL ENSEMBLE PREDICTION
# ============================================================

ensemble_pred = ridge_meta.predict(
    meta_test_X
)


# ============================================================
# 14. CALCULATE METRICS
# ============================================================

r2 = r2_score(
    y_test,
    ensemble_pred
)

mse = mean_squared_error(
    y_test,
    ensemble_pred
)

rmse = np.sqrt(
    mse
)

mae = mean_absolute_error(
    y_test,
    ensemble_pred
)

mape = np.mean(
    np.abs(
        (y_test.values - ensemble_pred)
        / y_test.values
    )
) * 100


# ============================================================
# 15. MASE
# ============================================================

naive_error = np.mean(
    np.abs(
        np.diff(y_train.values)
    )
)

model_error = np.mean(
    np.abs(
        y_test.values - ensemble_pred
    )
)

if naive_error != 0:

    mase = model_error / naive_error

else:

    mase = np.nan


# ============================================================
# 16. FINAL RESULTS
# ============================================================

print("\n==========================================")
print("STACKED META-ENSEMBLE TEST RESULTS")
print("==========================================")

print(f"R²   : {r2:.6f}")
print(f"MSE  : {mse:.9f}")
print(f"RMSE : {rmse:.6f} A")
print(f"MAE  : {mae:.6f} A")
print(f"MAPE : {mape:.6f} %")
print(f"MASE : {mase:.6f}")


# ============================================================
# 17. SAVE METRICS
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
    "Stacked_Ensemble_Metrics.csv",
    index=False
)


# ============================================================
# 18. SAVE META-MODEL INFORMATION
# ============================================================

meta_information = pd.DataFrame({

    "Parameter": [
        "Meta_Model",
        "Best_Ridge_Alpha",
        "SVR_Coefficient",
        "GPR_Coefficient",
        "RandomForest_Coefficient"
    ],

    "Value": [
        "RidgeCV",
        ridge_meta.alpha_,
        ridge_meta.coef_[0],
        ridge_meta.coef_[1],
        ridge_meta.coef_[2]
    ]

})

meta_information.to_csv(
    "Stacked_Ensemble_Hyperparameters.csv",
    index=False
)


# ============================================================
# 19. SAVE ACTUAL VS PREDICTED
# ============================================================

results = pd.DataFrame({

    "Actual_Current_Ripple_A":
        y_test.values,

    "Predicted_Current_Ripple_A":
        ensemble_pred

})

results.to_csv(
    "Stacked_Ensemble_Actual_vs_Predicted.csv",
    index=False
)


# ============================================================
# 20. SAVE BASE MODEL PREDICTIONS
# ============================================================

base_predictions = pd.DataFrame({

    "Actual_Current_Ripple_A":
        y_test.values,

    "SVR_Prediction_A":
        test_svr_pred,

    "GPR_Prediction_A":
        test_gpr_pred,

    "RandomForest_Prediction_A":
        test_rf_pred,

    "Stacked_Ensemble_Prediction_A":
        ensemble_pred

})

base_predictions.to_csv(
    "Stacked_Ensemble_Base_Predictions.csv",
    index=False
)


# ============================================================
# 21. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    ensemble_pred,
    alpha=0.7
)

min_value = min(
    y_test.min(),
    ensemble_pred.min()
)

max_value = max(
    y_test.max(),
    ensemble_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Current Ripple (A)"
)

plt.ylabel(
    "Predicted Current Ripple (A)"
)

plt.title(
    "Stacked Meta-Ensemble - Actual vs Predicted"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Stacked_Ensemble_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.show()


# ============================================================
# 22. SAVE FINAL ENSEMBLE
# ============================================================

ensemble_package = {

    "svr": final_svr,

    "gpr": final_gpr,

    "gpr_scaler": final_gpr_scaler,

    "random_forest": final_rf,

    "meta_model": ridge_meta,

    "features": features

}

joblib.dump(
    ensemble_package,
    "Stacked_Ensemble_model.pkl"
)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("FILES SAVED SUCCESSFULLY")
print("==========================================")

print("1. Stacked_Ensemble_model.pkl")
print("2. Stacked_Ensemble_Metrics.csv")
print("3. Stacked_Ensemble_Hyperparameters.csv")
print("4. Stacked_Ensemble_Actual_vs_Predicted.csv")
print("5. Stacked_Ensemble_Base_Predictions.csv")
print("6. Stacked_Ensemble_Actual_vs_Predicted_Ripple.png")

print("\nStacked Meta-Ensemble completed successfully.")