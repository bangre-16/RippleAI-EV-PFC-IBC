# RippleAI -- EV PFC-IBC Current Ripple Intelligence

## 1. Project Overview

RippleAI is a simulation-driven machine-learning project for predicting
the **output current ripple of a PFC Interleaved Boost Converter
(PFC-IBC)** used in EV battery-charging applications.

The project combines MATLAB/Simulink-based converter simulation,
automatic dataset generation, machine-learning prediction,
hyperparameter tuning, feature-influence analysis, and a web-based
prediction platform.

The system is designed to reduce the need for repeated converter
simulations by using trained ML models to estimate current ripple for
new operating conditions.

## 2. Problem Statement

The PFC Interleaved Boost Converter used in EV charging applications
produces output current ripple that varies with its operating
conditions. Evaluating current ripple through repeated MATLAB/Simulink
simulations can be time-consuming when many operating points must be
studied.

This project develops a machine-learning-based system that learns the
relationship between important PFC-IBC operating parameters and output
current ripple. The trained models provide rapid ripple predictions for
new operating points.

## 3. Objectives

1.  Develop and analyze a PFC Interleaved Boost Converter using
    MATLAB/Simulink.
2.  Generate a dataset by varying important converter operating
    parameters.
3.  Use simulation results to build a current-ripple prediction dataset.
4.  Develop multiple regression-based machine-learning models.
5.  Optimize selected models using hyperparameter tuning.
6.  Compare models using R², MSE, RMSE, MAE, MAPE, and MASE.
7.  Analyze the influence of input parameters on current ripple.
8.  Provide an interactive web-based prediction system.
9.  Support batch prediction using CSV input.
10. Generate automatic prediction reports in PDF format.

## 4. Converter and Dataset

The six input parameters are:

  Parameter             Description
  --------------------- -------------------------------
  Switching_Freq_Hz     Converter switching frequency
  Load_Resistance_Ohm   Load resistance
  Vin_RMS_V             Input RMS voltage
  Duty_Cycle_Pct        Converter duty cycle
  Inductance_uH         Boost inductance
  Vout_DC_V             DC output voltage

### Target Variable

``` text
Current_Ripple_A
```

The ML task is:

``` text
Six converter operating parameters
              ↓
        ML prediction
              ↓
       Current Ripple (A)
```

The project dataset contains **2,000 operating points** generated from
the simulation workflow.

## 5. Simulation

The PFC-IBC converter is modeled and analyzed using MATLAB/Simulink.

Workflow:

``` text
PFC-IBC Model
      ↓
Parameter Variation
      ↓
Simulation
      ↓
Output Measurement
      ↓
Current Ripple Extraction
      ↓
Dataset Generation
```

The project includes the Simulink model and converter output waveforms
for current and voltage.

## 6. Machine Learning Models

The project implements six regression approaches:

-   **Support Vector Regression (SVR)**
-   **Gaussian Process Regression (GPR)**
-   **Random Forest**
-   **XGBoost**
-   **K-Nearest Neighbors (KNN)**
-   **Stacked Meta-Ensemble**

Both baseline and tuned model versions are maintained where available.

## 7. Hyperparameter Tuning

Hyperparameter tuning is used to search for improved model
configurations. The tuned models are evaluated using validation data and
compared with their corresponding baseline models.

Examples of tuned parameters include kernel configuration,
regularization, epsilon, gamma, tree parameters, number of neighbors,
weighting strategy, and GPR kernel structures.

## 8. Model Evaluation

The models are evaluated using:

### R² -- Coefficient of Determination

Indicates how well the model explains variation in the target data.
Higher is generally better.

### MSE -- Mean Squared Error

Measures average squared prediction error. Lower is better.

### RMSE -- Root Mean Squared Error

Square root of MSE and expressed in amperes for this project. Lower is
better.

### MAE -- Mean Absolute Error

Average absolute difference between actual and predicted current ripple.
Lower is better.

### MAPE -- Mean Absolute Percentage Error

Expresses average prediction error as a percentage. Lower is generally
better.

### MASE -- Mean Absolute Scaled Error

Compares model error against a naive reference. Values below 1 generally
indicate improvement over that reference.

## 9. Feature Influence Analysis

The project analyzes the influence of the six input parameters on
current ripple.

For tree-based models, native feature importance is used. For models
without native feature importance, permutation-based analysis is used.

This answers:

> **Which converter parameter has the greatest influence on current
> ripple?**

The website presents the influence information for the implemented
models.

## 10. Prediction Studio

RippleAI provides an interactive Prediction Studio where users can
enter:

-   Switching frequency
-   Load resistance
-   Input RMS voltage
-   Duty cycle
-   Inductance
-   DC output voltage

The backend loads saved trained models rather than retraining a model
for every request.

Workflow:

``` text
User Input
    ↓
Training-Range Check
    ↓
Saved ML Models
    ↓
Current-Ripple Prediction
    ↓
Model Comparison
    ↓
Prediction Result
```

Out-of-range values generate a warning because prediction reliability
may be reduced outside the training region.

## 11. Batch Prediction

The Prediction Studio supports CSV-based batch prediction.

The system:

1.  Accepts a CSV containing the six required input columns.
2.  Checks the input format and training ranges.
3.  Processes multiple operating points automatically.
4.  Generates predictions using the saved models.
5.  Displays the results in the website.
6.  Provides the results for further analysis.

## 12. Automatic Report Generation

The website can generate a PDF prediction report containing relevant
prediction information such as:

-   Operating-point inputs
-   Model predictions
-   Range/warning status
-   Model-performance context
-   Feature-influence information
-   Prediction comparison graph

## 13. Web Application

The RippleAI website contains:

-   **Overview**
-   **Dataset & Simulation**
-   **Models**
-   **Features**
-   **Results**
-   **Files**
-   **Prediction Studio**

The interface includes simulation graphs, actual-vs-predicted graphs,
model comparisons, feature influence, single prediction, batch
prediction, and report generation.

## 14. Project Workflow

``` text
PFC-IBC Design
      ↓
MATLAB/Simulink Simulation
      ↓
Parameter Variation
      ↓
Dataset Generation
      ↓
Data Processing
      ↓
ML Model Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Feature Influence Analysis
      ↓
Saved Model Deployment
      ↓
Web-Based Prediction
      ↓
Batch Prediction
      ↓
Automatic Report Generation
```

## 15. Project Structure

``` text
EV_Ripple_Website/
│
├── app.py
├── requirements.txt
├── README.md
├── website_data.json
│
├── frontend/
│   ├── package.json
│   ├── src/
│   ├── public/
│   └── ...
│
├── Major Project/
│   ├── Dataset/
│   ├── Simulation/
│   ├── SVR/
│   ├── SVR Tuned/
│   ├── Gaussian Process/
│   ├── Gaussian Process Tuned/
│   ├── Random Forest/
│   ├── RF Tuned/
│   ├── XGBoost/
│   ├── XGBoost Tuned/
│   ├── KNN/
│   ├── KNN Tuned/
│   ├── Stacked Meta Ensemble/
│   └── Stacked Meta Ensemble Tuned/
│
└── .venv/
```

## 16. Technologies Used

### Simulation

-   MATLAB
-   Simulink

### Machine Learning

-   Python
-   NumPy
-   Pandas
-   Scikit-learn
-   XGBoost
-   Joblib

### Backend

-   Flask
-   Python

### Frontend

-   React
-   TypeScript
-   Vite
-   Tailwind CSS
-   Framer Motion
-   Spline

### Visualization and Reporting

-   Matplotlib
-   ReportLab

## 17. Installation

Activate the existing Python environment:

### PowerShell

``` powershell
.\.venv\Scripts\Activate.ps1
```

### Command Prompt

``` cmd
.venv\Scripts\activate
```

Install Python dependencies:

``` cmd
python -m pip install -r requirements.txt
```

Install frontend dependencies:

``` cmd
cd frontend
npm install
```

## 18. Running the Website

### Terminal 1 -- Flask Backend

``` cmd
cd "C:\path o\EV_Ripple_Website_BATCH_REPORT_FIXED"
.venv\Scripts\activate
python app.py
```

### Terminal 2 -- React Frontend

``` cmd
cd "C:\path\EV_Ripple_Website_BATCH_REPORT_FIXED_frontend"
npm run dev
```

The frontend normally runs at:

``` text
http://localhost:5173
```

The Flask backend normally runs at:

``` text
http://127.0.0.1:5000
```

## 19. Important Notes

-   Trained models are loaded from saved model files.
-   Individual prediction requests do not retrain the models.
-   Predictions are most reliable inside the training-data ranges.
-   An out-of-range warning indicates that the model is being used
    outside the region represented by its training data.
-   Model performance should be interpreted using multiple evaluation
    metrics rather than a single metric.
-   The project is simulation-based and does not represent a physical
    hardware prototype.

## 20. Future Engineering Extensions

### Operating Region / Ripple Map

Visualize predicted current ripple across combinations of converter
operating parameters to identify low-ripple and high-ripple operating
regions.

### Engineering Recommendation

Provide practical recommendations based on the operating point and
predicted ripple to help identify parameter changes that may reduce
ripple.

## 21. Expected Outcome

The completed system provides an end-to-end workflow from **PFC-IBC
simulation to machine-learning prediction and web deployment**. It
demonstrates how simulation-generated engineering data can be combined
with machine learning to reduce repeated simulation effort and provide a
faster interactive method for analyzing converter current ripple.

## 22. Project Highlights

-   PFC Interleaved Boost Converter for EV charging
-   MATLAB/Simulink simulation
-   2,000-point simulation-generated dataset
-   Six converter input parameters
-   Six ML regression approaches
-   Baseline and tuned model comparison
-   Feature influence analysis
-   Actual-vs-predicted visualization
-   Interactive prediction system
-   Training-range warning
-   Batch CSV prediction
-   Automatic PDF report generation
-   Web-based deployment
-   Engineering-oriented visualization

## 23. Authors

**Project:** EV PFC-IBC Current Ripple Prediction Using Machine Learning

**Department:** Electrical and Electronics Engineering

**Institution:** Dayananda Sagar Academy of Technology and Management,
Bengaluru

## 24. License

This project is developed for academic and research purposes. Refer to
the project repository and institutional requirements for redistribution
and usage conditions.
