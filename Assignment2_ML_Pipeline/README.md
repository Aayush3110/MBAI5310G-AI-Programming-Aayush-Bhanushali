# Assignment 2 - End-to-End ML Pipeline

**Course:** MBAI 5310G: AI Programming - Ontario Tech University

## Overview

End-to-end machine learning pipeline that predicts e-commerce customer churn from customer attributes and behaviour.

- **Target:** `Churned` (Yes / No - 1 / 0)
- **Task:** Supervised binary classification
- **Baseline model:** Logistic Regression (Decision Tree also included for comparison)

## Dataset

`ecommerce_customer_churn_dataset.csv` - 325 customer records, 14 columns.

| Column | Type | Description |
|---|---|---|
| `Customer_ID` | int | Identifier (not used as a feature) |
| `Age` | numeric | Customer age |
| `Gender` | categorical | Female / Male |
| `Region` | categorical | Canadian province |
| `Membership_Type` | categorical | Basic / Silver / Gold / Platinum |
| `Monthly_Spending` | numeric | Average monthly spend ($) |
| `Number_of_Orders` | numeric | Total orders placed |
| `Days_Since_Last_Purchase` | numeric | Days since last purchase |
| `Customer_Support_Calls` | numeric | Number of support calls |
| `Average_Rating` | numeric | Average rating given |
| `Used_Coupon` | categorical | Yes / No |
| `Newsletter_Subscribed` | categorical | Yes / No |
| `Device_Type` | categorical | Mobile / Desktop / Tablet |
| **`Churned`** | categorical | **Target** — Yes / No |

## Pipeline Steps

1. Load the dataset
2. Inspect the dataset
3. Understand the problem
4. Define features and target
5. Data cleaning (median/mode imputation, drop duplicates, fix dtypes)
6. Preprocessing (`StandardScaler` + `OneHotEncoder` via `ColumnTransformer`)
7. Train/test split (80/20, stratified)
8. Baseline model - Logistic Regression
9. Evaluate the baseline model
10. Interpret results
11. Responsible AI reflection

A short written summary answering the assignment questions (what the model predicts, dataset, features/target, result, one limitation) is included at the end of the notebook under **Final Explanation**.

## Files

```
Assignment2_ML_Pipeline/
├── Assignment2_Aayush_B.ipynb
├── ecommerce_customer_churn_dataset.csv
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook Assignment2_Aayush_B.ipynb
```

Run all cells. The CSV must be in the same folder as the notebook.

## Results

| Metric | Score |
|---|---:|
| Accuracy | 0.844 |
| Precision | 0.733 |
| Recall | 0.647 |
| F1-score | 0.688 |

Due to class imbalance (≈73% No / ≈27% Yes), F1-score is a more honest measure than accuracy.
