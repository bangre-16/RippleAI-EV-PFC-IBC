import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor


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
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("==========================================")
print("XGBOOST HYPERPARAMETER TUNING")
print("==========================================")

print("Total samples    :", len(data))
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ============================================================
# 3. BASE XGBOOST MODEL
# ============================================================

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=2,
    tree_method="hist"
)


# ============================================================
# 4. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    "n_estimators": [
        100,
        200,
        300,
        400,
        500
    ],

    "max_depth": [
        2,
        3,
        4,
        5,
        6,
        8
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
        0.08,
        0.1,
        0.15
    ],

    "subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "min_child_weight": [
        1,
        3,
        5
    ],

    "gamma": [
        0,
        0.01,
        0.05,
        0.1
    ],

    "reg_alpha": [
        0,
        0.001,
        0.01,
        0.1
    ],

    "reg_lambda": [
        0.5,
        1.0,
        2.0,
        5.0
    ]
}


# ============================================================
# 5. RANDOMIZED SEARCH
# ============================================================

print("\nStarting XGBoost tuning...")

print("Random combinations : 40")
print("Cross-validation    : 3-fold")
print("CPU workers         : 2")


random_search = RandomizedSearchCV(

    estimator=xgb,

    param_distributions=param_distributions,

    n_iter=40,

    scoring="neg_mean_squared_error",

    cv=3,

    random_state=42,

    n_jobs=2,

    verbose=1
)


# ============================================================
# 6. RUN SEARCH
# ============================================================

random_search.fit(
    X_train,
    y_train
)


# ============================================================
# 7. BEST PARAMETERS
# ============================================================

print("\n==========================================")
print("BEST XGBOOST HYPERPARAMETERS")
print("==========================================")

print(
    random_search.best_params_
)

best_cv_mse = -random_search.best_score_

print(
    "\nBest CV MSE :",
    best_cv_mse
)


# ============================================================
# 8. BEST MODEL
# ============================================================

best_xgb = random_search.best_estimator_


# ============================================================
# 9. TEST PREDICTION
# ============================================================

y_pred = best_xgb.predict(
    X_test
)


# ============================================================
# 10. METRICS
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mse
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mape = np.mean(
    np.abs(
        (y_test.values - y_pred)
        / y_test.values
    )
) * 100


# ============================================================
# 11. MASE
# ============================================================

naive_error = np.mean(
    np.abs(
        np.diff(y_train.values)
    )
)

model_error = np.mean(
    np.abs(
        y_test.values - y_pred
    )
)

if naive_error != 0:

    mase = model_error / naive_error

else:

    mase = np.nan


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("TUNED XGBOOST TEST RESULTS")
print("==========================================")

print(f"R²   : {r2:.6f}")
print(f"MSE  : {mse:.9f}")
print(f"RMSE : {rmse:.6f} A")
print(f"MAE  : {mae:.6f} A")
print(f"MAPE : {mape:.6f} %")
print(f"MASE : {mase:.6f}")


# ============================================================
# 13. SAVE METRICS
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
    "XGBoost_Tuned_Metrics.csv",
    index=False
)


# ============================================================
# 14. SAVE HYPERPARAMETERS
# ============================================================

hyperparameters = pd.DataFrame({

    "Parameter":
        list(
            random_search.best_params_.keys()
        ),

    "Best_Value":
        list(
            random_search.best_params_.values()
        )

})

hyperparameters.to_csv(
    "XGBoost_Tuned_Hyperparameters.csv",
    index=False
)


# ============================================================
# 15. SAVE ACTUAL VS PREDICTED
# ============================================================

results = pd.DataFrame({

    "Actual_Current_Ripple_A":
        y_test.values,

    "Predicted_Current_Ripple_A":
        y_pred

})

results.to_csv(
    "XGBoost_Tuned_Actual_vs_Predicted.csv",
    index=False
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature":
        features,

    "Importance":
        best_xgb.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "XGBoost_Tuned_Feature_Importance.csv",
    index=False
)


print("\n==========================================")
print("XGBOOST FEATURE IMPORTANCE")
print("==========================================")

print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 17. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

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
    linestyle="--"
)

plt.xlabel(
    "Actual Current Ripple (A)"
)

plt.ylabel(
    "Predicted Current Ripple (A)"
)

plt.title(
    "Tuned XGBoost - Actual vs Predicted"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "XGBoost_Tuned_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Input Parameter"
)

plt.title(
    "XGBoost - Feature Importance"
)

plt.gca().invert_yaxis()

plt.grid(
    axis="x"
)

plt.tight_layout()

plt.savefig(
    "XGBoost_Tuned_Feature_Importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. SAVE MODEL
# ============================================================

joblib.dump(
    best_xgb,
    "XGBoost_Tuned_model.pkl"
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("FILES SAVED SUCCESSFULLY")
print("==========================================")

print("1. XGBoost_Tuned_model.pkl")
print("2. XGBoost_Tuned_Hyperparameters.csv")
print("3. XGBoost_Tuned_Metrics.csv")
print("4. XGBoost_Tuned_Actual_vs_Predicted.csv")
print("5. XGBoost_Tuned_Feature_Importance.csv")
print("6. XGBoost_Tuned_Actual_vs_Predicted_Ripple.png")
print("7. XGBoost_Tuned_Feature_Importance.png")

print("\nXGBoost tuning completed successfully.")