import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ConstantKernel,
    WhiteKernel
)
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

print("Dataset loaded successfully.")
print("Total samples    :", len(data))
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# ============================================================
# 3. SCALE DATA
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. CREATE LARGER TUNING DATASET
# ============================================================

# More samples than previous version.
# This improves kernel selection but increases computation.

tuning_samples = 1000

rng = np.random.default_rng(42)

if len(X_train_scaled) > tuning_samples:

    indices = rng.choice(
        len(X_train_scaled),
        size=tuning_samples,
        replace=False
    )

    X_tune = X_train_scaled[indices]
    y_tune = y_train.iloc[indices].values

else:

    X_tune = X_train_scaled
    y_tune = y_train.values


print("\nGPR tuning samples :", len(X_tune))


# ============================================================
# 5. TUNING / VALIDATION SPLIT
# ============================================================

X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
    X_tune,
    y_tune,
    test_size=0.20,
    random_state=42
)

print("Tuning training samples   :", len(X_tune_train))
print("Tuning validation samples :", len(X_tune_val))


# ============================================================
# 6. DEFINE MORE POWERFUL KERNEL SEARCH
# ============================================================

kernel_options = [

    # --------------------------------------------------------
    # KERNEL 1 - RBF
    # --------------------------------------------------------

    ConstantKernel(
        1.0,
        (1e-3, 1e3)
    )
    * RBF(
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e2)
    )
    + WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-6, 1e-1)
    ),


    # --------------------------------------------------------
    # KERNEL 2 - RBF with smaller length scale
    # --------------------------------------------------------

    ConstantKernel(
        1.0,
        (1e-3, 1e3)
    )
    * RBF(
        length_scale=0.5,
        length_scale_bounds=(1e-2, 1e2)
    )
    + WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-6, 1e-1)
    ),


    # --------------------------------------------------------
    # KERNEL 3 - MATERN NU = 1.5
    # --------------------------------------------------------

    ConstantKernel(
        1.0,
        (1e-3, 1e3)
    )
    * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e2),
        nu=1.5
    )
    + WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-6, 1e-1)
    ),


    # --------------------------------------------------------
    # KERNEL 4 - MATERN NU = 2.5
    # --------------------------------------------------------

    ConstantKernel(
        1.0,
        (1e-3, 1e3)
    )
    * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5
    )
    + WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-6, 1e-1)
    ),


    # --------------------------------------------------------
    # KERNEL 5 - RATIONAL QUADRATIC
    # --------------------------------------------------------

    ConstantKernel(
        1.0,
        (1e-3, 1e3)
    )
    * RationalQuadratic(
        length_scale=1.0,
        alpha=1.0,
        length_scale_bounds=(1e-2, 1e2),
        alpha_bounds=(1e-2, 1e2)
    )
    + WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-6, 1e-1)
    ),


    # --------------------------------------------------------
    # KERNEL 6 - RBF + MATERN
    # --------------------------------------------------------

    ConstantKernel(
        1.0,
        (1e-3, 1e3)
    )
    * (
        RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2)
        )
        +
        Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        )
    )
    + WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-6, 1e-1)
    ),


    # --------------------------------------------------------
    # KERNEL 7 - RBF + RATIONAL QUADRATIC
    # --------------------------------------------------------

    ConstantKernel(
        1.0,
        (1e-3, 1e3)
    )
    * (
        RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2)
        )
        +
        RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-2, 1e2),
            alpha_bounds=(1e-2, 1e2)
        )
    )
    + WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-6, 1e-1)
    ),


    # --------------------------------------------------------
    # KERNEL 8 - MATERN + RATIONAL QUADRATIC
    # --------------------------------------------------------

    ConstantKernel(
        1.0,
        (1e-3, 1e3)
    )
    * (
        Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        )
        +
        RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-2, 1e2),
            alpha_bounds=(1e-2, 1e2)
        )
    )
    + WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=(1e-6, 1e-1)
    )
]


# ============================================================
# 7. GPR KERNEL TUNING
# ============================================================

print("\n==========================================")
print("STARTING ADVANCED GPR TUNING")
print("==========================================")

print("Number of kernels       :", len(kernel_options))
print("Tuning samples          :", len(X_tune))
print("Optimizer restarts     : 3")
print("Kernel optimizer        : L-BFGS-B")
print("\nWARNING:")
print("This is computationally heavy.")
print("CPU usage may become high.")


best_validation_mse = float("inf")
best_kernel_number = None
best_optimized_kernel = None


for kernel_number, kernel in enumerate(
    kernel_options,
    start=1
):

    print("\n==========================================")
    print(
        f"TESTING KERNEL {kernel_number}"
        f"/{len(kernel_options)}"
    )
    print("==========================================")

    print("Initial Kernel:")
    print(kernel)

    gpr = GaussianProcessRegressor(

        kernel=kernel,

        # Enable kernel optimization
        optimizer="fmin_l_bfgs_b",

        # MORE COMPUTATION
        n_restarts_optimizer=3,

        normalize_y=True,

        random_state=42
    )

    gpr.fit(
        X_tune_train,
        y_tune_train
    )

    y_val_pred = gpr.predict(
        X_tune_val
    )

    validation_mse = mean_squared_error(
        y_tune_val,
        y_val_pred
    )

    print("\nOptimized Kernel:")
    print(gpr.kernel_)

    print(
        f"\nValidation MSE : "
        f"{validation_mse:.9f}"
    )

    if validation_mse < best_validation_mse:

        best_validation_mse = validation_mse

        best_kernel_number = kernel_number

        best_optimized_kernel = gpr.kernel_

        print("\n*** NEW BEST KERNEL ***")


# ============================================================
# 8. DISPLAY BEST KERNEL
# ============================================================

print("\n==========================================")
print("BEST GPR HYPERPARAMETERS")
print("==========================================")

print(
    "Best Kernel Number :",
    best_kernel_number
)

print("\nBest Optimized Kernel:")

print(
    best_optimized_kernel
)

print(
    "\nBest Validation MSE :",
    best_validation_mse
)


# ============================================================
# 9. FINAL GPR TRAINING
# ============================================================

print("\n==========================================")
print("TRAINING FINAL GPR")
print("==========================================")

print(
    "Training using all",
    len(X_train_scaled),
    "training samples..."
)

print(
    "The final training may take significant time."
)


final_gpr = GaussianProcessRegressor(

    # Use optimized kernel found during tuning
    kernel=best_optimized_kernel,

    # Kernel has already been optimized.
    # Avoid repeating optimization on 1600 samples.
    optimizer=None,

    normalize_y=True,

    random_state=42
)


final_gpr.fit(
    X_train_scaled,
    y_train.values
)


print("\nFinal GPR training completed.")


# ============================================================
# 10. TEST PREDICTION
# ============================================================

print("\nGenerating test predictions...")

y_pred = final_gpr.predict(
    X_test_scaled
)


# ============================================================
# 11. METRICS
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

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
# 12. MASE
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
# 13. FINAL RESULTS
# ============================================================

print("\n==========================================")
print("ADVANCED TUNED GPR TEST RESULTS")
print("==========================================")

print(f"R²   : {r2:.6f}")
print(f"MSE  : {mse:.9f}")
print(f"RMSE : {rmse:.6f} A")
print(f"MAE  : {mae:.6f} A")
print(f"MAPE : {mape:.6f} %")
print(f"MASE : {mase:.6f}")


# ============================================================
# 14. SAVE METRICS
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
    "GPR_Tuned_Metrics.csv",
    index=False
)


# ============================================================
# 15. SAVE HYPERPARAMETERS
# ============================================================

hyperparameters = pd.DataFrame({

    "Parameter": [
        "Best_Kernel_Number",
        "Optimized_Kernel",
        "Best_Validation_MSE",
        "Tuning_Samples",
        "Optimizer_Restarts"
    ],

    "Value": [
        best_kernel_number,
        str(best_optimized_kernel),
        best_validation_mse,
        len(X_tune),
        3
    ]
})

hyperparameters.to_csv(
    "GPR_Tuned_Hyperparameters.csv",
    index=False
)


# ============================================================
# 16. SAVE ACTUAL VS PREDICTED
# ============================================================

results = pd.DataFrame({

    "Actual_Current_Ripple_A":
        y_test.values,

    "Predicted_Current_Ripple_A":
        y_pred

})

results.to_csv(
    "GPR_Tuned_Actual_vs_Predicted.csv",
    index=False
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
    "Advanced Tuned GPR - Actual vs Predicted"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "GPR_Tuned_Actual_vs_Predicted_Ripple.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. SAVE MODEL
# ============================================================

joblib.dump(
    final_gpr,
    "GPR_Tuned_model.pkl"
)

joblib.dump(
    scaler,
    "GPR_Tuned_scaler.pkl"
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("FILES SAVED SUCCESSFULLY")
print("==========================================")

print("1. GPR_Tuned_model.pkl")
print("2. GPR_Tuned_scaler.pkl")
print("3. GPR_Tuned_Hyperparameters.csv")
print("4. GPR_Tuned_Metrics.csv")
print("5. GPR_Tuned_Actual_vs_Predicted.csv")
print("6. GPR_Tuned_Actual_vs_Predicted_Ripple.png")

print("\nAdvanced GPR tuning completed.")