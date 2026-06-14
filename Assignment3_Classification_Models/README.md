# Assignment 3 - Classification Models and Evaluation

**Course:** MBAI 5310G: AI Programming - Ontario Tech University

## Overview

Train and compare two classification models - **Logistic Regression** and **SVM** - on the e-commerce customer churn dataset, and evaluate them using standard classification metrics.

- **Target:** `Churned` (Yes / No - 1 / 0)
- **Task:** Supervised binary classification

## Dataset

`ecommerce_customer_churn_dataset.csv` - 325 customer records, 14 columns. See the dataset description in the notebook.

## Pipeline Steps

1. Understand the ML problem
2. Initial data inspection
3. Data cleaning (drop duplicates, median/mode imputation)
4. Define features and target
5. Train/test split (80/20, stratified)
6. Preprocessing (`StandardScaler` + `OneHotEncoder` via `ColumnTransformer`)
7. Baseline model - Logistic Regression
8. Evaluate Logistic Regression
9. Comparison model - SVM
10. Evaluate SVM and compare with Logistic Regression
11. Interpret classification outputs (predicted probabilities)
12. Final interpretation and business conclusion
13. Responsible AI reflection

A short written summary answering the assignment questions (better model, most important metric, false positives/negatives, limitation/bias, role of human judgment) is included at the end of the notebook under **Final Explanation**.

## Models Compared

| Model | Library |
|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` |
| SVM | `sklearn.svm.SVC` (with `probability=True`) |

## Evaluation Metrics

Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report, Predicted Probabilities.

## Files

```
Assignment3_Classification_Models/
├── Assignment3_Aayush_B.ipynb
├── ecommerce_customer_churn_dataset.csv
├── outputs/
│   ├── cleaned_ecommerce_customer_churn_dataset.csv
│   ├── model_comparison_results.csv
│   └── classification_outputs.csv
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook Assignment3_Aayush_B.ipynb
```

Run all cells. The notebook creates the `outputs/` folder automatically.

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.844 | 0.733 | 0.647 | 0.688 |
| **SVM** | **0.875** | **0.909** | 0.588 | **0.714** |

**Selected model:** SVM (highest F1-score). The trade-off is higher precision and accuracy at the cost of slightly lower recall.
