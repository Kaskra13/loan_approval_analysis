# Loan Approval Analysis with PySpark

A comprehensive machine learning project analyzing loan approval patterns using PySpark's distributed computing framework. This project demonstrates end-to-end data processing, exploratory analysis, feature engineering, and predictive modeling at scale.

## Overview

This project analyzes a loan approval dataset to predict whether a loan application will be approved or rejected based on applicant characteristics and financial metrics. Built with PySpark, it showcases distributed data processing capabilities and machine learning pipelines suitable for large-scale datasets.

## Features
- Distributed Data Processing: Leverages PySpark for scalable data manipulation and analysis

- Data Quality Assessment: Comprehensive schema validation and missing value analysis

- Feature Engineering: Creates derived features including loan-to-income ratio

- Exploratory Analysis: Aggregations, window functions, and statistical breakdowns by applicant segments

- Machine Learning Models:

  - Logistic Regression (baseline model)

  - Random Forest Classifier (advanced ensemble method)

- Model Optimization: Cross-validation with hyperparameter tuning

- Performance Evaluation: Multiple metrics including AUC, accuracy, precision, recall, and F1-score


## Dataset
Source: [Kaggle - Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data)

Size: 4,269 records

### Features:

- Applicant demographics (dependents, education, employment status)

- Financial metrics (annual income, loan amount, loan term)

- Credit score (CIBIL score)

- Asset values (residential, commercial, luxury, bank assets)

- Target variable: loan approval status

### Technologies used
- PySpark 3.x: Distributed data processing and machine learning

- PySpark SQL: Data manipulation and transformations

- PySpark MLlib: Machine learning algorithms and evaluation metrics

- Python 3.x: Primary programming language

## Installation
```bash
# Install PySpark
pip install pyspark

# Or using conda
conda install -c conda-forge pyspark
```


## Project workflow
1. Data Loading and Quality Check
= Initialize Spark session

- Load CSV dataset with schema inference

- Validate data types and structure

- Check for missing values (none found in critical columns)

2. Data Preprocessing
- Trim and normalize column names

- Standardize text values (lowercase, trimmed)

- Encode categorical variables (education, self_employed)

- Map loan status to binary labels (Approved=1, Rejected=0)

- Drop rows with missing critical values

3. Feature Engineering
- Loan-to-Income Ratio: Calculated as loan_amount / income_annum

- Numerical encoding of categorical features

- Vector assembly of 11 feature columns

4. Exploratory Analysis
- Aggregated statistics by loan status and education level

- Window functions for income ranking within employment groups

- Approval rate analysis by CIBIL score ranges

5. Machine Learning Pipeline
- 70/30 train-test split

- Feature vectorization using VectorAssembler

- Model training and evaluation

- Hyperparameter tuning with cross-validation

## Model performance

### Logistic regression
| Metric    | Score  |
| --------- | ------ |
| AUC       | 0.9705 |
| Accuracy  | 91.09% |
| Precision | 91.08% |
| Recall    | 91.09% |
| F1-Score  | 91.09% |

### Random Forest (100 trees, max depth 8)
| Metric    | Score  |
| --------- | ------ |
| AUC       | 0.9985 |
| Accuracy  | 98.10% |
| Precision | 98.14% |
| Recall    | 98.10% |
| F1-Score  | 98.09% |

### Cross-Validated Random Forest
- Best Parameters: Optimized through 3-fold cross-validation

- Test AUC: 0.9985

- Test Accuracy: 98.10%

## Key knsights
Feature Importance (Random Forest)
1. CIBIL Score (81.19%) - Most critical factor

2. Loan Term (7.85%)

3. Loan-to-Income Ratio (5.66%)

4. Loan Amount (1.02%)

5. Annual Income (1.01%)

Approval Patterns
- Applicants with CIBIL score ≥750: 99.43% approval rate

- Applicants with CIBIL score <500: 10.57% approval rate

- Average CIBIL score: Approved (703) vs Rejected (429)

## Usage
```python
# Load and run the notebook/script
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.appName("LoanApprovalAnalysis").getOrCreate()

# Load your dataset
df = spark.read.option("header", True).option("inferSchema", True).csv("loan_approval_dataset.csv")

# Follow the preprocessing and modeling steps
# ...

# Load the saved model for predictions
from pyspark.ml.classification import RandomForestClassificationModel
model = RandomForestClassificationModel.load("best_rf_model_spark")
```

## Requirements
```text
pyspark>=3.0.0
```
