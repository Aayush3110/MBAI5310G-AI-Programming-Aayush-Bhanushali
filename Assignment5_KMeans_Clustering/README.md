# Assignment 5 - K-means Clustering and Customer Segmentation

**Course:** MBAI 5310G-001: AI Programming - Ontario Tech University
**Business:** BankWise Retail Banking - Credit Card Offer Targeting
**Dataset:** BankWise Credit Card Offer Acceptance Dataset (`bankwise_credit_card_offer_acceptance_dataset.csv`)
**Model:** K-means Clustering (scikit-learn)

## Business Problem

BankWise is a mid-sized retail bank that promotes credit-card products (cashback, travel rewards, low-interest, balance-transfer) to its existing customers. Today it sends broadly similar offers to a wide list, which produces **low acceptance rates**, wasted marketing spend, and extra load on branch and call-centre teams.

This project uses **unsupervised learning** to discover natural groups within the customer base. The intended use is **marketing decision-support** - segment customers by behaviour so the team can match the right offer and channel to each group, prioritise spend on the most valuable and responsive segments, and treat high-value or at-risk customers differently from low-engagement ones. This is a segmentation task, so there is **no target variable** to predict.

## Dataset

`bankwise_credit_card_offer_acceptance_dataset.csv` - 302 customer records, 16 columns. Includes 2 intentional duplicate rows and small numbers of missing values across six columns (`Age`, `Income_Level`, `Average_Monthly_Balance`, `Digital_Banking_Usage`, `Credit_Score_Band`, `Previous_Campaign_Response`) for data-preparation practice. After cleaning: ≈300 rows.

| Group | Columns |
|---|---|
| Identifier (excluded) | `Customer_ID` |
| Demographics | `Age`, `Income_Level` |
| Account / relationship | `Account_Tenure_Years`, `Credit_Score_Band`, `Existing_Loan`, `Has_Savings_Account` |
| Spending / activity | `Average_Monthly_Balance`, `Monthly_Transactions`, `Digital_Banking_Usage` |
| Service quality / engagement | `Service_Complaints_6M`, `Branch_Visits_6M` |
| Campaign / offer | `Previous_Campaign_Response`, `Marketing_Channel`, `Offer_Type` |
| Not used (unsupervised) | `Accepted_Offer` |

**Features used for clustering (numerical only):** `Age`, `Account_Tenure_Years`, `Average_Monthly_Balance`, `Monthly_Transactions`, `Service_Complaints_6M`, `Branch_Visits_6M`. `Accepted_Offer` exists in the data but is **not** used, because segmentation is unsupervised.

## Machine Learning Model

A **K-means Clustering** model with `n_clusters=3` is the main model. The number of clusters is chosen with the **Elbow Method** and confirmed with the **silhouette score**.

> **Note on scope:** Week 5's lecture material also covers PCA (Principal Component Analysis). PCA is used here **only to visualise** the clusters in 2-D - it does not create the clusters. K-means creates the segments; PCA simply helps display them. Categorical fields and `Accepted_Offer` are intentionally excluded from clustering.

Pipeline steps:
1. Load, inspect, and clean the dataset (strip column names, drop duplicates, median-impute the numerical features)
2. Select the six numerical features for clustering
3. Scale the features with `StandardScaler`
4. Choose K with the Elbow Method, confirmed by the silhouette score
5. Train the final K-means model (`n_clusters=3`, `random_state=42`, `n_init=10`) and assign cluster labels
6. Add cluster labels back to the dataset and profile each cluster (group means)
7. Name and interpret the customer segments
8. Visualise the clusters (two-feature scatter plot and a 2-D PCA projection)
9. Business interpretation, limitations, and Responsible AI reflection

Short written summaries answering the assignment questions (segment characteristics, most valuable segment, segment needing attention, marketing strategy, limitations, Responsible AI) are embedded as markdown throughout the notebook under each task heading.

## Main Results

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1538.5 | 0.145 |
| **3** | **1352.3** | **0.158** |
| 4 | 1210.8 | 0.147 |
| 5 | 1080.9 | 0.158 |
| 6 | 964.7 | 0.166 |

**K-selection verdict:** the elbow sits at **K = 3** (the largest single drop in inertia, ≈186, after which the curve flattens). The silhouette score confirms it - K = 3 scores 0.158 versus 0.147 for K = 4, so adding a fourth cluster *reduces* separation quality rather than improving it. K = 3 is therefore both the statistically and practically stronger choice.

| Segment | Customers | Defining profile |
|---|---:|---|
| Active Engaged Customers | 94 | Longest tenure (≈10y), most transactions (≈37), very few complaints |
| High-Value Premium Customers | 38 | Highest balance (≈$6,400), highest complaint rate (≈2.5) |
| Low-Engagement Customers | 168 | Moderate balance, short tenure (≈3y), low transaction activity |

**Important caveat:** the silhouette scores are modest (≈0.16) because this synthetic data has **overlapping, continuous behaviour** rather than crisply separated groups (the 2-D PCA view captures ≈42% of total variance). The segments are reliable as **directional marketing guides**, not as hard boundaries - borderline customers can sit between two segments.

## Key Business Insights

- **Segments separate along three axes:** *activity* (transactions, tenure), *value* (average balance), and *satisfaction* (complaints).
- **Most valuable segment: High-Value Premium Customers** - they hold the largest balances and the greatest growth potential, but their high complaint rate means **service recovery should come before cross-sell**.
- **Segment needing the most marketing attention: Low-Engagement Customers** - the largest, mostly passive group; the priority is reactivation. (The Premium group needs *service* attention rather than marketing.)
- **Recommended strategy per segment:** loyalty rewards and activity-based cashback for **Active Engaged Customers**; premium / travel-rewards offers plus complaint resolution for **High-Value Premium Customers**; reactivation offers (low-interest / balance-transfer) and digital nudges for **Low-Engagement Customers**.
- **Actionable use:** replace one-size-fits-all campaigns with segment-specific targeting, match channel and offer type to each group, prioritise budget by segment value, and measure campaign performance segment by segment.

## One Limitation

The dataset is **synthetic, small (≈300 rows after cleaning), and clustered on only six numerical features** - including the categorical fields (income band, credit score, channel) could shift the groupings. K-means also assumes roughly spherical, similarly sized clusters and needs K fixed in advance, so on continuous real-world behaviour the boundaries are fuzzy (visible in the overlap in the PCA plot). Features tied to `Income_Level` or `Credit_Score_Band` can encode existing social and economic inequalities, so the segments should be used as **decision support**, not as an automated system for deciding who receives financial products - every targeting rule should be reviewed by people for fairness first.

## Files

```
Assignment5_KMeans_Clustering/
├── Assignment5_Aayush_B.ipynb
├── bankwise_credit_card_offer_acceptance_dataset.csv
├── business_plan_bankwise_credit_card_offer.pdf
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook Assignment5_BankWise_Aayush_Bhanushali.ipynb
```

Run all cells. The dataset CSV must be in the same folder as the notebook.
