# Assignment 6: Model Evaluation, Explainability, and Fairness Reflection

**Course:** MBAI 5310G: AI Programming - Ontario Tech University
**Dataset:** Digital Subscription Retention Dataset (`digital_subscription_retention_dataset.csv`)
**Model:** Decision Tree Classifier (scikit-learn)

## Dataset Description

The dataset describes customers of a digital subscription service. It contains 385 records and 20 columns covering demographics (age, age group, gender, region, income), subscription behaviour (tenure, monthly spend, contract type, payment method, membership level), engagement (website visits, marketing emails opened, days since last purchase, primary device), and service signals (support tickets, satisfaction score, complaint in the last month). The data includes 5 duplicate records and small numbers of missing values across several columns, which are handled during preprocessing (median imputation for numerical columns, mode imputation for categorical columns). After removing duplicates, 380 records remain.

| Group | Columns |
|---|---|
| Identifier (removed) | `customer_id` |
| Demographics | `age`, `age_group`, `gender`, `region`, `income` |
| Subscription | `tenure_months`, `monthly_spend`, `contract_type`, `payment_method`, `membership_level`, `discount_used` |
| Engagement | `website_visits`, `marketing_emails_opened`, `days_since_last_purchase`, `primary_device` |
| Service signals | `support_tickets`, `satisfaction_score`, `complained_last_month` |
| **Target** | **`churn`** (1 = left, 0 = stayed) |
| Kept only for fairness reflection | `gender`, `age_group`, `region` |

## Target Variable

The target variable is **`churn`**: 1 means the customer left the service and 0 means the customer stayed. The target is **imbalanced**, with about 66% not churned and about 34% churned. Because of this imbalance, accuracy alone is not a reliable measure, since predicting that everyone stays would already score about 0.66.

## Model Used

A simple **Decision Tree Classifier** (`max_depth=4`, `random_state=42`) trained on an 80/20 stratified train/test split. The identifier column is removed, and the group columns (`gender`, `age_group`, `region`) are held aside for fairness analysis and are **not** used as model inputs. Categorical features are one-hot encoded, giving 21 numerical features. The notebook also evaluates the model with cross-validation, tests tree depths from 1 to 10 for overfitting, and explains predictions with feature importance, SHAP, and LIME.

## Main Evaluation Results

| Metric (test set) | Score |
|---|---:|
| Accuracy | 0.671 |
| Precision | 0.600 |
| Recall | 0.115 |
| F1-Score | 0.194 |

| Confusion matrix | Predicted Not Churn | Predicted Churn |
|---|---:|---:|
| Actual Not Churn | 48 (TN) | 2 (FP) |
| Actual Churn | 23 (FN) | 3 (TP) |

| Tree variant | Training accuracy | Testing accuracy | Gap |
|---|---:|---:|---:|
| Controlled (`max_depth=4`) | 0.793 | 0.671 | 0.122 |
| Deep (`max_depth=10`) | 0.977 | 0.553 | 0.424 |

- **Cross-validation:** 5-fold accuracy averaged about 0.65 with a standard deviation of about 0.05, so the model is fairly stable but weak.
- **Overfitting:** mild to moderate at depth 4 and strong at higher depths, where training accuracy climbs toward 0.98 while testing accuracy falls toward 0.55.
- **Most important metric:** Recall, because this is a retention business and the costly mistake is missing a customer who will leave. The model's recall is only about 0.12.
- **Top features:** `support_tickets`, `satisfaction_score`, `tenure_months`, `days_since_last_purchase`, `marketing_emails_opened`, `monthly_spend`, and `income`. Service quality and engagement signals dominate. SHAP and LIME confirm these drivers and allow the model to be explained for individual customers.

## Main Business Interpretation

The model predicts whether a digital subscription customer will churn. Its accuracy (about 0.67) is only slightly above the baseline of predicting that everyone stays (about 0.66), and its recall is very low (about 0.12), so it catches only a small share of real churners. The main error type is **false negatives** (23 customers): at-risk customers the model expects to stay, who then leave without any retention action, causing lost recurring revenue. False positives are rare (2 customers), so the error mix is heavily weighted toward the more expensive mistake for a subscription business. Feature importance, SHAP, and LIME all point to service tickets, satisfaction, tenure, and recency of purchase as the strongest churn signals, which the business can watch as early warnings. A fairness check shows the model under-predicts churn for every group and has uneven accuracy across gender, region, and age group (some groups are very small), so it should be reviewed before being used for targeting.

**Final recommendation:** this model is **not ready for real business use**. It is suitable only as an **early prototype**. Before real use it needs more churn examples, stronger features, handling of the class imbalance (class weights or resampling), threshold tuning to raise recall, and comparison with other models such as Logistic Regression, Random Forest, or Gradient Boosting. It should support human judgment, not replace it.

## One Limitation of the Model

The dataset is small and imbalanced (380 records after cleaning, with only about 131 churners), so the model has very few churn examples to learn from, which is the most likely reason recall is so poor. The model also relies on behavioural features that can be correlated with sensitive groups, so even though gender, age group, and region are not used as inputs, biased or uneven predictions can still appear and must be checked by people before the model influences retention decisions.

## Files

```
Assignment6_DigitalSubscription_Model_Evaluation/
├── Assignment6_Aayush_B.ipynb
├── digital_subscription_retention_dataset.csv
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib shap lime jupyter
jupyter notebook Assignment6_Aayush_B.ipynb
```

Run all cells in order. The dataset CSV must be in the same folder as the notebook.
