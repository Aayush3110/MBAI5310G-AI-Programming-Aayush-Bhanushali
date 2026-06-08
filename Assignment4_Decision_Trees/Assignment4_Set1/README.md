# Assignment 4 - Decision Tree Model and Business Interpretation

**Course:** MBAI 5310G: AI Programming - Ontario Tech University
**Business:** AdVantage Growth Studio (marketing campaign analytics)
**Dataset:** Marketing Campaign Decision Tree Dataset (`marketing_campaign_decision_tree_dataset.csv`)
**Model:** Decision Tree Classifier (scikit-learn)

## Business Problem

AdVantage Growth Studio is a data-driven marketing agency that helps small and medium-sized businesses run campaigns across channels such as Email, Social Media, Search Ads, Display Ads, and SMS. The agency's core problem is **campaign inefficiency**: budget is wasted on customers who won't convert, while converters who could have been targeted more effectively are sometimes missed.

This project builds a Decision Tree classifier that predicts whether a customer will convert (`Yes` / `No`) after receiving a campaign. The model's outputs - predicted classes plus feature importance - are intended as a **decision-support tool** to prioritise follow-up marketing.

## Dataset

`marketing_campaign_decision_tree_dataset.csv` - 605 customer campaign records, 21 columns. Includes 5 intentional duplicate rows and small numbers of missing values across four columns (`Annual_Income`, `Customer_Segment`, `Average_Order_Value`, `Customer_Satisfaction`) for data-preparation practice.

| Group | Columns |
|---|---|
| Identifier / metadata (excluded) | `Customer_ID`, `Campaign_Date` |
| Customer demographics | `Age`, `Annual_Income`, `Region` |
| Customer relationship | `Customer_Segment`, `Loyalty_Score`, `Customer_Satisfaction` |
| Campaign information | `Campaign_Channel`, `Campaign_Type`, `Discount_Offered_Percent`, `Ad_Exposure_Count` |
| Engagement behaviour | `Email_Opened`, `Ad_Clicked`, `Website_Visits_Last_30_Days`, `Social_Media_Engagement_Score` |
| Purchase history | `Previous_Purchases`, `Average_Order_Value`, `Days_Since_Last_Purchase`, `Prior_Campaign_Response` |
| **Target** | **`Converted`** (`Yes` / `No`) |

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

A short written summary answering the assignment questions (model performance, overfitting, important metric, false positives/negatives, feature importance, limitations, role of human judgment) is at the end of the notebook under **Final Business Interpretation**.

## Main Evaluation Results

| Metric (test set) | Score |
|---|---:|
| Accuracy | 0.583 |
| Precision | 0.600 |
| Recall | 0.414 |
| F1-score | 0.490 |

| Tree variant | Training accuracy | Testing accuracy | Gap |
|---|---:|---:|---:|
| Controlled (`max_depth=4`) | 0.729 | 0.583 | 0.146 |
| Unrestricted (`max_depth=None`) | 1.000 | 0.525 | 0.475 |

**Overfitting verdict:** the controlled tree shows mild-to-moderate overfitting (≈15-point gap); the unrestricted tree clearly overfits (≈47-point gap). Capping `max_depth` is what stops the controlled tree from memorising the training set.

## Key Business Insights

- **Top features:** `Loyalty_Score`, `Previous_Purchases`, `Ad_Clicked`, `Days_Since_Last_Purchase`, `Annual_Income`. Customer-relationship and engagement signals beat raw demographics - consistent with the business plan's expectations.
- **Recommended primary metric: F1-score**, because the business cares about both wasting budget (precision) and missing converters (recall).
- **False positives** = wasted ad budget on customers who won't convert. **False negatives** = missed revenue from customers who would have.
- **Actionable suggestions** from the feature ranking: retarget customers who clicked an ad, build retention/reactivation campaigns based on loyalty and recency, A/B-test discount levels rather than maxing them out.

## One Limitation

The dataset is **synthetic and modestly sized (≈600 rows)**, which limits how robust the patterns the model picks up really are. Demographic features (`Age`, `Annual_Income`, `Region`) directly enter the model, so any under-representation of customer groups in the training data could produce unfair or unreliable predictions for those groups. The model should be used as decision-support, not as an automated decision-maker.

## Files

```
Assignment4_Decision_Trees/Assignment4_Set1/
├── Assignment4_Aayush_B.ipynb
├── marketing_campaign_decision_tree_dataset.csv
├── business_plan_marketing_campaign_decision_tree.pdf
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl jupyter
jupyter notebook Assignment4_Aayush_Bhanushali.ipynb
```

Run all cells. The dataset XLSX must be in the same folder as the notebook.
