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
| └── README.md
+---scripts
| ├── init.py
| ├── data_loader.py
| ├── feature_engineering.py
| ├── fraud_data_visualizer.py
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