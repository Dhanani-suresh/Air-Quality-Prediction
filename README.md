# Air Quality Index (AQI) Prediction Using Ensemble Machine Learning

A machine learning project for predicting the **Air Quality Index (AQI)** from pollutant measurements using an ensemble of **XGBoost** and **Random Forest** regression models.

The project focuses on data preprocessing, feature selection, feature engineering, hyperparameter optimization, ensemble learning, and model evaluation using historical air-quality data collected across Indian cities.

---

## Overview

Air Quality Index prediction is a regression problem where multiple pollutant measurements are used to estimate a continuous AQI value.

This project explores how ensemble machine learning can be used to model the complex, non-linear relationships between different air pollutants and overall air quality.

The final approach combines predictions from:

* **XGBoost Regressor**
* **Random Forest Regressor**

using a weighted ensemble to improve prediction performance and robustness.

---

## Dataset

The project uses the **Air Quality Data in India** dataset available on Kaggle.

**Dataset:** [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The dataset contains historical daily air-quality measurements collected from Indian cities between **2015 and 2020**.

### Dataset characteristics

* **29,531** original records
* **16** columns
* Multiple pollutant measurements
* AQI as the prediction target
* Data collected across multiple Indian cities

### Main pollutant features

The dataset includes measurements such as:

* PM2.5
* PM10
* NO
* NO₂
* NOx
* NH₃
* CO
* SO₂
* O₃
* Benzene
* Toluene
* Xylene

---

## Machine Learning Approach

The workflow consists of several stages:

```text
Raw Air Quality Data
        │
        ▼
Data Cleaning & Missing Value Handling
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Correlation Analysis
        │
        ▼
Feature Selection
        │
        ▼
Log Transformation
        │
        ▼
Polynomial Feature Engineering
        │
        ▼
Robust Scaling
        │
        ▼
Optuna Hyperparameter Optimization
        │
        ▼
┌───────────────────┐
│                   │
▼                   ▼
XGBoost        Random Forest
│                   │
└─────────┬─────────┘
          ▼
   Weighted Ensemble
          │
          ▼
     AQI Prediction
          │
          ▼
       Evaluation
```

---

## Data Preprocessing

The following preprocessing steps were applied:

### Missing Value Handling

Rows with missing AQI values were removed because AQI is the target variable.

Missing numeric feature values were then filled using the corresponding means calculated from the training data.

### Train / Validation / Test Split

The dataset was divided into:

* **80% Training**
* **10% Validation**
* **10% Testing**

The validation set was used during model development and hyperparameter optimization, while the test set was reserved for final evaluation.

### Feature Removal

Non-numeric/non-feature columns such as:

* `City`
* `Date`
* `AQI_Bucket`

were excluded from the regression model.

---

## Feature Selection

Correlation analysis was performed to examine the relationship between the available numeric features and AQI.

Features with an absolute correlation greater than **0.1** with AQI were selected for the initial model input.

Correlation visualizations were also generated to analyse relationships between pollutant variables.

---

## Feature Engineering

Several feature engineering techniques were explored.

### Log Transformation

Highly skewed features were identified dynamically and transformed using:

```python
np.log1p()
```

This helps reduce the effect of highly skewed pollutant distributions.

### Polynomial Interaction Features

Polynomial interaction features of degree 2 were generated using:

```python
PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)
```

The interactions were generated from the most strongly correlated features.

### Robust Scaling

The engineered features were scaled using:

```python
RobustScaler()
```

This provides greater resistance to the influence of extreme values and outliers.

---

## Models

### XGBoost

An **XGBoost Regressor** was trained to capture complex non-linear relationships between pollutant measurements and AQI.

Hyperparameters were optimized using **Optuna**.

The optimization explored parameters including:

* Number of estimators
* Maximum tree depth
* Learning rate
* Subsample ratio
* L1 regularization
* L2 regularization

### Random Forest

A **Random Forest Regressor** was also trained using the engineered feature set.

Random Forest provides an alternative tree-based learning approach and helps introduce model diversity into the final ensemble.

---

## Weighted Ensemble

Predictions from XGBoost and Random Forest were combined using a weighted ensemble.

```text
XGBoost Prediction ─────┐
                        ├──► Weighted Ensemble ───► Final AQI
Random Forest Prediction┘
```

The validation performance of each model was used to determine the relative ensemble weights.

The final ensemble was then evaluated on the unseen test dataset.

---

## Model Evaluation

The models were evaluated using three regression metrics:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual AQI values.

### Root Mean Squared Error (RMSE)

Penalizes larger prediction errors more strongly and provides an indication of the overall prediction error.

### R² Score

Measures how much of the variation in AQI is explained by the model.

### Model Comparison

The notebook compares:

| Model                 | MAE | RMSE | R² |
| --------------------- | --- | ---- | -- |
| XGBoost               | —   | —    | —  |
| Random Forest         | —   | —    | —  |
| **Weighted Ensemble** | —   | —    | —  |

> The exact values are generated when the notebook is executed and may vary depending on the environment and library versions.

---

## Analysis & Visualizations

The notebook includes several analyses to evaluate model behaviour:

* Feature correlation analysis
* Correlation bar plots
* Feature importance analysis
* Training vs. testing error comparison
* Actual vs. predicted AQI scatter plot
* Sample-level actual vs. predicted AQI comparison

These visualizations provide additional insight into model performance beyond numerical evaluation metrics.

---

## Feature Importance

Feature importance was analysed using the trained XGBoost and Random Forest models.

The importance values were combined to identify the most influential engineered features used by the ensemble.

This provides an additional level of interpretability and helps identify pollutant relationships that contribute strongly to AQI predictions.

---

## Repository Structure

```text
AQI-Prediction/
│
├── AQI_Prediction_Model.ipynb
├── README.md
└── .gitignore
```

The main notebook contains the complete workflow, including:

* Dataset loading
* Preprocessing
* Feature selection
* Feature engineering
* Hyperparameter tuning
* Model training
* Ensemble prediction
* Evaluation
* Visual analysis

---

## Running the Project

### Requirements

The notebook was developed using Python and requires libraries including:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
optuna
```

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna
```

### Dataset Setup

Download the dataset from Kaggle:

**Air Quality Data in India**

Place the `city_day.csv` file in the location expected by the notebook.

The notebook currently loads the dataset using:

```python
pd.read_csv("/city_day.csv")
```

If running locally, update the path as required for your environment.

### Run the Notebook

Open:

```text
AQI_Prediction_Model.ipynb
```

using Jupyter Notebook, JupyterLab, Google Colab, or another compatible environment and execute the cells sequentially.

---

## Limitations

The project has several limitations:

* The dataset consists of historical measurements from Indian cities between 2015 and 2020.
* Model performance may not generalize directly to other geographic regions or environmental conditions.
* Rare extreme-AQI events are less represented in the dataset.
* Temporal factors such as seasonality and wind patterns were not explicitly modelled.
* Combining multiple models increases computational overhead compared with using a single model.


