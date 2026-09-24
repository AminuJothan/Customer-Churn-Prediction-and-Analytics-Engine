# Customer-Churn-Prediction-and-Analytics-Engine

An end-to-end machine learning and analytics pipeline designed to forecast customer turnover, evaluate classification models, and segment users into actionable business risk tiers using interpretable linear modeling.

---

## 🚀 Project Overview
Customer retention is critical for subscription and service-based businesses. This project builds a complete analytics engine that goes beyond binary predictions (will churn / will not churn) by generating **probability scores, risk classifications, and coefficient-based insights** to help stakeholders understand *why* customers leave.

After comparing a non-linear ensemble model (**Random Forest**) against a linear baseline (**Logistic Regression**), **Logistic Regression** was selected for the final production engine to maximize interpretability and surface direct behavioral risk drivers.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing & Manipulation:** Pandas, NumPy
* **Database / Extraction:** SQL
* **Modeling & Evaluation:** Scikit-Learn (Logistic Regression, RandomForestClassifier, StandardScaler, Metrics)
* **Visualization:** Matplotlib, Seaborn (ROC Curves, Feature Coefficients)

---

## 🔄 Project Workflow

1. **Data Extraction & Exploration (SQL & Pandas):** Extracted user behavioral data, cleaned missing values, and conducted exploratory data analysis (EDA) to find initial turnover trends.
2. **Feature Engineering:** Transformed raw user activity metrics into robust predictors ready for scaling.
3. **Model Training & Comparative Evaluation:** 
   * Trained a **Logistic Regression** baseline and a **Random Forest Classifier**.
   * Evaluated both models using **ROC-AUC scores**, classification reports (Precision, Recall, F1-Score), and visual ROC curve comparisons.
4. **Prediction & Analytics Engine:** 
   * Implemented a custom pipeline wrapping `StandardScaler` and the trained Logistic Regression model.
   * Converted continuous probability outputs into **Low, Medium, and High Risk** tiers.
   * Extracted model coefficients to expose top features increasing or mitigating churn risk.

---

## 📊 Model Performance & Selection
* **Logistic Regression:** Chosen as the final production model due to its high transparency and explicit coefficients, allowing direct business interpretation of log-odds.
* **Evaluation Metric:** Prioritized **ROC-AUC** and **Precision-Recall** trade-offs to handle class imbalance effectively, ensuring high-risk churners are successfully captured while minimizing false alarms.

---

## ⚙️ How to Use the Analytics Engine

### 1. Load Dependencies & Artifacts
```python
import joblib
import numpy as np
import pandas as pd

# Load your trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
