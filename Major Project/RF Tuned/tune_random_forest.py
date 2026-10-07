import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
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
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("==========================================")
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("==========================================")

print("Total samples    :", len(data))
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ============================================================
# 3. BASE RANDOM FOREST
# ============================================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=2
)


# ============================================================
# 4. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    "n_estimators": [
        100,
        200,
        300,
        400
    ],

    "max_depth": [
        None,
        10,
        15,
        20,
        25,
        30
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        1.0,
        "sqrt",
        "log2"
    ],

    "bootstrap": [
        True,
        False
    ]
}


# ============================================================
# 5. RANDOMIZED SEARCH
# ============================================================

print("\nStarting Random Forest tuning...")

print("Random combinations : 30")
print("Cross-validation    : 3-fold")
print("CPU workers         : 2")

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_distributions,

    n_iter=30,

    scoring="neg_mean_squared_error",

    cv=3,

    random_state=42,

    n_jobs=2,

    verbose=1
)


# ============================================================
# 6. TRAIN SEARCH
# ============================================================

random_search.fit(
    X_train,
    y_train
)


# ============================================================
# 7. BEST PARAMETERS
# ============================================================

print("\n==========================================")
print("BEST RANDOM FOREST HYPERPARAMETERS")
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
# 8. GET BEST MODEL
# ============================================================

best_rf = random_search.best_estimator_


# ============================================================
# 9. TEST PREDICTION
# ============================================================

y_pred = best_rf.predict(
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
print("TUNED RANDOM FOREST TEST RESULTS")
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
    "RandomForest_Tuned_Metrics.csv",
    index=False
)


# ============================================================
# 14. SAVE HYPERPARAMETERS
# ============================================================

hyperparameters = pd.DataFrame({

    "Parameter": list(
        random_search.best_params_.keys()
    ),

    "Best_Value": list(
        random_search.best_params_.values()
    )
})

hyperparameters.to_csv(
    "RandomForest_Tuned_Hyperparameters.csv",
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
    "RandomForest_Tuned_Actual_vs_Predicted.csv",
    index=False
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        best_rf.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "RandomForest_Tuned_Feature_Importance.csv",
    index=False
)


print("\n==========================================")
print("FEATURE IMPORTANCE")
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
    "Tuned Random Forest - Actual vs Predicted"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "RandomForest_Tuned_Actual_vs_Predicted_Ripple.png",
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
    "Random Forest - Feature Importance"
)

plt.gca().invert_yaxis()

plt.grid(
    axis="x"
)

plt.tight_layout()

plt.savefig(
    "RandomForest_Tuned_Feature_Importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. SAVE MODEL
# ============================================================

joblib.dump(
    best_rf,
    "RandomForest_Tuned_model.pkl"
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("FILES SAVED SUCCESSFULLY")
print("==========================================")

print("1. RandomForest_Tuned_model.pkl")
print("2. RandomForest_Tuned_Hyperparameters.csv")
print("3. RandomForest_Tuned_Metrics.csv")
print("4. RandomForest_Tuned_Actual_vs_Predicted.csv")
print("5. RandomForest_Tuned_Feature_Importance.csv")
print("6. RandomForest_Tuned_Actual_vs_Predicted_Ripple.png")
print("7. RandomForest_Tuned_Feature_Importance.png")

print("\nRandom Forest tuning completed successfully.")