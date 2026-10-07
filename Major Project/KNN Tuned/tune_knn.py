import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
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
print("KNN HYPERPARAMETER TUNING")
print("==========================================")

print("Total samples    :", len(data))
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ============================================================
# 3. KNN PIPELINE
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])


# ============================================================
# 4. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    "knn__n_neighbors": list(range(2, 31)),

    "knn__weights": [
        "uniform",
        "distance"
    ],

    "knn__algorithm": [
        "auto",
        "ball_tree",
        "kd_tree"
    ],

    "knn__leaf_size": [
        10,
        20,
        30,
        40,
        50,
        60
    ],

    "knn__p": [
        1,
        2
    ]
}


# ============================================================
# 5. RANDOMIZED SEARCH
# ============================================================

print("\nStarting KNN tuning...")

print("Random combinations : 50")
print("Cross-validation    : 5-fold")

random_search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_distributions,

    n_iter=50,

    scoring="neg_mean_squared_error",

    cv=5,

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
print("BEST KNN HYPERPARAMETERS")
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

best_knn = random_search.best_estimator_


# ============================================================
# 9. TEST PREDICTION
# ============================================================

y_pred = best_knn.predict(
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
print("TUNED KNN TEST RESULTS")
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
    "KNN_Tuned_Metrics.csv",
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
    "KNN_Tuned_Hyperparameters.csv",
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
    "KNN_Tuned_Actual_vs_Predicted.csv",
    index=False
)


# ============================================================
# 16. ACTUAL VS PREDICTED GRAPH
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
    "Tuned KNN - Actual vs Predicted Current Ripple"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "KNN_Tuned_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. SAVE MODEL
# ============================================================

joblib.dump(
    best_knn,
    "KNN_Tuned_model.pkl"
)


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("FILES SAVED SUCCESSFULLY")
print("==========================================")

print("1. KNN_Tuned_model.pkl")
print("2. KNN_Tuned_Hyperparameters.csv")
print("3. KNN_Tuned_Metrics.csv")
print("4. KNN_Tuned_Actual_vs_Predicted.csv")
print("5. KNN_Tuned_Actual_vs_Predicted_Ripple.png")

print("\nKNN tuning completed successfully.")