# Assignment 4 — Decision Tree Model and Business Interpretation

**Course:** MBAI 5310G-001: AI Programming — Ontario Tech University
**Business:** FreshBasket Subscription Grocery Delivery
**Dataset:** FreshBasket Customer Churn Dataset (`freshbasket_customer_churn_dataset.xlsx`)
**Model:** Decision Tree Classifier (scikit-learn)

## Business Problem

FreshBasket is a subscription-based grocery delivery service operating in the Greater Toronto Area. Revenue depends on **repeat customers**, so customer retention is critical: when subscribers cancel or fail to renew, the business loses recurring revenue and must spend more on acquisition to replace them.

This project builds a Decision Tree classifier that predicts whether a customer will churn (`Yes` / `No`). The intended use is **retention decision-support** — identify at-risk customers so the team can intervene early with service recovery, targeted offers, or operational fixes.

## Dataset

`freshbasket_customer_churn_dataset.xlsx` — 425 customer records, 16 columns. Includes 5 intentional duplicate rows and small numbers of missing values across four columns (`Region`, `Monthly_Spend`, `App_Engagement_Score`, `Satisfaction_Score`) for data-preparation practice.

| Group | Columns |
|---|---|
| Identifier (excluded) | `Customer_ID` |
| Demographics | `Age_Group`, `Region` |
| Subscription | `Subscription_Type`, `Payment_Method`, `Delivery_Window`, `Membership_Length_Months`, `Discount_Used` |
| Spending | `Monthly_Spend`, `Average_Order_Value`, `Orders_Last_3_Months` |
| Service quality / engagement | `Delivery_Issues_Last_3_Months`, `Customer_Service_Tickets`, `App_Engagement_Score`, `Satisfaction_Score` |
| **Target** | **`Churned`** (`Yes` / `No`) |

The target is **imbalanced**: ≈72% `No` / ≈28% `Yes`.

## Machine Learning Model

A **Decision Tree Classifier** with `max_depth=4` is the main model. A second unrestricted Decision Tree (`max_depth=None`) is also trained for direct overfitting comparison.

> **Note on scope:** Week 4's lecture material also covers Random Forest, but the assignment brief only requires a Decision Tree. Random Forest is intentionally not included in this submission.

Pipeline steps:
1. Load, inspect, and clean the dataset (drop duplicates, median/mode imputation)
2. Define features and target; encode target `Yes`/`No` → `1`/`0`
3. Train/test split (80/20, stratified, `random_state=42`)
4. Preprocess (`StandardScaler` + `OneHotEncoder` via `ColumnTransformer`)
5. Train a Decision Tree (`max_depth=4`) and predict
6. Evaluate (confusion matrix, accuracy, precision, recall, F1, classification report)
7. Compare training vs. testing accuracy for overfitting
8. Train an unrestricted tree to show overfitting in contrast
9. Plot the tree and compute feature importance
10. Business interpretation

A short written summary answering the assignment questions (performance, overfitting, important metric, false positives/negatives, feature importance, limitations, role of human judgment) is at the end of the notebook under **Final Business Interpretation**.

## Main Evaluation Results

| Metric (test set) | Score |
|---|---:|
| Accuracy | 0.762 |
| Precision | 0.714 |
| Recall | 0.217 |
| F1-score | 0.333 |

| Tree variant | Training accuracy | Testing accuracy | Gap |
|---|---:|---:|---:|
| Controlled (`max_depth=4`) | 0.816 | 0.762 | 0.054 |
| Unrestricted (`max_depth=None`) | 1.000 | 0.702 | 0.298 |

**Overfitting verdict:** the controlled tree shows only mild overfitting (≈5-point gap). The unrestricted tree clearly overfits (≈30-point gap) — it memorises the training set and loses predictive power on new customers.

**Important caveat:** the accuracy figure is partly inflated by class imbalance — a naive "predict everyone stays" model would already hit ≈0.72. The model's true weakness is **recall: only about 22% of actual churners are caught**, which is exactly the wrong trade-off for a retention business.

## Key Business Insights

- **Top features:** `Satisfaction_Score` (clearly dominant), `Monthly_Spend`, `Delivery_Issues_Last_3_Months`, `App_Engagement_Score`, `Membership_Length_Months`. Service-quality and engagement signals dominate demographics.
- **Recommended primary metric: Recall** (with F1 as a close second). Missing a churner means lost recurring revenue, which is typically more expensive than spending a retention offer on someone who would have stayed anyway.
- **False positives** = wasted retention spend (unnecessary discount or support call). **False negatives** = missed at-risk customer who then cancels → lost subscription revenue.
- **Actionable use:** trigger retention outreach when satisfaction drops below threshold; investigate delivery issues in regions where they cluster; design app re-engagement nudges for low-activity users; give long-tenured / high-spend customers the highest-touch treatment.

## One Limitation

The dataset is **synthetic, small (≈420 rows after cleaning, only ≈118 churners), and class-imbalanced**. With so few churn examples, the tree has very limited material to learn from on the minority class — the most likely reason recall is so poor. The model also uses `Age_Group` and `Region` directly, which introduces a fairness consideration: if certain customer groups are under-represented in the training data, predictions for those groups may be unreliable or unfair. The model should be used as **decision support**, not as an automated cancel-prevention system.

## Files

```
Assignment4_FreshBasket_Decision_Tree/
├── Assignment4_FreshBasket_Aayush_Bhanushali.ipynb
├── freshbasket_customer_churn_dataset.xlsx
├── business_plan_freshbasket_customer_churn.docx
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl jupyter
jupyter notebook Assignment4_FreshBasket_Aayush_Bhanushali.ipynb
```

Run all cells. The dataset XLSX must be in the same folder as the notebook.
