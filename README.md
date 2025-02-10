# **Fraud-Shield-AI**
An AI-powered fraud detection system for e-commerce and banking, leveraging machine learning, geolocation analysis, and transaction patterns. Features real-time data processing, Flask and Docker deployment, and interactive Dash dashboards for enhanced financial security.


## Tasks Completed

### 1. Handle Missing Values
- **Imputation or Removal**: Addressed missing values by either imputing them with appropriate methods or dropping them as necessary.

### 2. Data Cleaning
- **Remove Duplicates**: Ensured the dataset is free of duplicate records to maintain integrity.
- **Correct Data Types**: Adjusted data types for various features to ensure accurate analysis.

### 3. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Examined individual feature distributions to understand their characteristics.
- **Bivariate Analysis**: Analyzed relationships between pairs of features to uncover potential correlations.

### 4. Merge Datasets for Geolocation Analysis
- **IP Address Conversion**: Converted IP addresses to integer format for easier processing.
- **Data Merging**: Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` to enrich the dataset with geographical information.

### 5. Feature Engineering
- **Transaction Features**: Analyzed transaction frequency and velocity from `Fraud_Data.csv`.
- **Time-Based Features**: Created features like `hour_of_day` and `day_of_week` to capture temporal patterns in transactions.

### 6. Normalization and Scaling
- **Categorical Encoding**: Encoded categorical features to prepare them for modeling.

## Dataset Overview

### Short Summary

The dataset comprises **151,112** records with the following key features:

- **user_id**: Ranges from 2 to 400,000; Mean: 200,171.04.
- **purchase_value**: Ranges from 9 to 154; Mean: 36.94; Standard Deviation: 18.32.
- **age**: Ranges from 18 to 76; Mean: 33.14; Standard Deviation: 8.62.
- **ip_address**: Indicates a wide geographical distribution.
- **class**: Binary variable with a mean of 0.09, reflecting a low prevalence of fraud.

### Additional Data Features

- **device_id**: 137,956 unique values; most frequent: CQTUVBYIWWWBC (20 occurrences).
- **source**: 3 unique sources; most common: SEO (60,615 records).
- **browser**: 5 unique browsers; most used: Chrome (61,432 records).
- **sex**: 2 categories (M/F); predominant: M (88,293 occurrences).

### Distribution Insights

- **Purchase Value**: Right-skewed distribution, mainly between 10 and 50.
- **Age**: Right-skewed, mostly individuals aged 20 to 40.
- **Source and Browser**: SEO and Chrome dominate, with notable differences in usage patterns between fraud and non-fraud classes.

### Key Insights from EDA

- **Fraudulent Behavior**: Significant overlap in purchase value distributions between classes indicates that neither purchase value nor age is a strong predictor of fraud.
- **Categorical Features**: Distribution differences in `sex`, `source`, and `browser` suggest potential predictive power, particularly for `sex` and `browser`.

### Purchase Analysis Summary

- **Delays**: Most purchases are completed quickly, with a long-tail distribution for delays.
- **Hourly and Weekly Trends**: Steady purchasing behavior throughout the day and week, indicating stable consumer activity.
- **Fraud Prediction**: Current features are insufficient for reliable fraud classification; additional features are needed.

## **Fraud Detection Model Building and Training**

### Task Overview
The goal of this task is to build and train multiple machine learning models to detect fraudulent transactions in two datasets:
1. **Credit Card Bank Dataset**
2. **Fraud Ecommerce Dataset**

This involves data preparation, model selection, training, evaluation, and MLOps implementation for versioning and tracking experiments.

---

### **1. Data Preparation**
1. **Feature and Target Separation**
For both datasets:
- **Credit Card Dataset:** Target column is `Class`
- **Fraud Dataset:** Target column is `class`

2. **Handling Class Imbalance**
Class imbalance is significant in both datasets:
- **Credit Card Dataset:**
  - Class 0: 284,315 instances
  - Class 1: 492 instances
- **Fraud Dataset:**
  - Class 0: 116,878 instances
  - Class 1: 12,268 instances

To handle this, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to balance the classes.

3. **Train-Test Split**
- **Credit Card Dataset (after SMOTE):**
  - Training Size: `(227,845, 30)`
  - Testing Size: `(56,962, 30)`
- **Fraud Dataset (after SMOTE):**
  - Training Size: `(103,316, 17)`
  - Testing Size: `(25,830, 17)`

---

### **2. Model Selection**
The following models are implemented for performance comparison:
1. **Random Forest**
2. **Gradient Boosting**
3. **Convolutional Neural Network (CNN)**
4. **Long Short-Term Memory (LSTM)**

---

### **3. Model Training and Evaluation**
Each model is trained separately for the two datasets, and their performance is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC Score**

### **Best Model Selection**
| Model                | Accuracy | Precision | Recall  | F1 Score | ROC AUC |
|---------------------|----------|------------|---------|----------|---------|
| Random Forest      | 0.9569    | 1.0000     | 0.5378  | 0.6994   | 0.8387  |
| Gradient Boosting  | 0.9569    | 1.0000     | 0.5378  | 0.6994   | 0.8344  |
| LSTM               | 0.9569    | 1.0000     | 0.5378  | 0.6994   | 0.8332  |
| CNN                | 0.9552    | 0.9563     | 0.5448  | 0.6942   | 0.7779  |

🏆 **Selected Best Model:** `Random Forest`

---

### **4. MLOps - Experiment Tracking & Versioning**
To ensure reproducibility and model version control, **MLflow** is used for tracking:

🔹 **Logging Model Performance**
- Best parameters recorded for `Random Forest` & `Gradient Boosting`
- Training time for each model logged
- Metrics stored in MLflow

🔹 **Model Versioning**
- Registered `Random Forest`, `Gradient Boosting`, `LSTM`, and `CNN` models in MLflow
- Best model (`Random Forest`) stored as `random_forest_fraud_best_model.pkl`

🔹 **Experiment Tracking**
- View experiment runs at: `http://127.0.0.1:5000/`

---
### Conclusion

The analysis reveals critical insights into transaction behaviors and fraud patterns, highlighting the need for enhanced security measures and fraud detection strategies, especially in regions with high fraud rates. The results underscore the importance of integrating geolocation data for a comprehensive understanding of fraudulent activities.


# **Project Structure**

```
+---.github
| └── workflows
| └── blank.yml
+---.vscode
| └── settings.json
+---api
| ├── init.py
| └── README.md
+---notebooks
| ├── init.ipynb
| ├── analysis_preprocessing.ipynb
| ├── fraud_detection_pipeline.ipynb
| └── README.md
+---screenshots
| ├── cnn_run_model_metrics.png
| ├── experiments_mlflow.png
| ├── fraud_detection_experiments.png
| ├── gradient_boosting_run_model_metrics.png
| ├── registered_models.png
+---scripts
| ├── init.py
| ├── data_loader.py
| ├── data_preparation.py
| ├── feature_engineering.py
| ├── fraud_data_visualizer.py
| ├── fraud_detection_pipeline.py
| ├── fraud_geo_analyzer.py
| ├── fraud_ip_analysis.py
| ├── logger.py
| └── README.md
+---src
| ├── init.py
| └── README.md
+---tests
| ├── init.py
| ├── README.md
| ├── .gitignore
| ├── LICENSE
| ├── README.md
| └── requirements.txt
```