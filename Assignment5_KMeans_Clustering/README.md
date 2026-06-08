# Assignment 5 - Unsupervised Learning for Customer Segmentation

**Course:** MBAI 5310G: AI Programming - Ontario Tech University
**Business:** BankWise Retail Banking
**Dataset:** BankWise Credit Card Offer Acceptance Dataset (`bankwise_credit_card_offer_acceptance_dataset.csv`)
**Technique:** K-means clustering

## Business Problem
BankWise, a mid-sized retail bank, sends the same credit-card offers to a broad customer list, producing low acceptance rates and wasted marketing spend. The goal is to discover natural customer segments (an unsupervised learning problem - no target variable) so outreach can be targeted by offer type and channel instead of being uniform.

## Dataset
**BankWise Credit Card Offer Acceptance Dataset** (synthetic, educational) - 302 rows × 16 columns of customer profile, account behaviour, and campaign-response fields. The file `bankwise_credit_card_offer_acceptance_dataset.csv` is the raw data; the notebook strips column names, removes 2 duplicate rows, and median-fills a small number of missing numeric values.

## Clustering Method
Six numerical features were selected - `Age`, `Account_Tenure_Years`, `Average_Monthly_Balance`, `Monthly_Transactions`, `Service_Complaints_6M`, `Branch_Visits_6M` - and standardised with `StandardScaler`. The Elbow Method indicated **K = 3**, which was confirmed by the silhouette score (K = 3 scored 0.158 versus 0.147 for K = 4, so adding a fourth cluster reduced separation quality). A final K-means model was trained, the cluster labels added back, and PCA used to visualise the segments in 2-D.

## Main Results
Three customer segments were found:

| Cluster | Segment | Defining trait |
|---|---|---|
| Active Engaged Customers (94) | Loyal frequent users | Longest tenure (≈10y) + most transactions (≈37) |
| High-Value Premium Customers (38) | Strong purchasing power | Highest balance (≈6,400) but highest complaints (≈2.5) |
| Low-Engagement Customers (168) | Large passive majority | Moderate balance, short tenure, low activity |

## Business Recommendation
Replace one-size-fits-all campaigns with segment-specific targeting: loyalty and cashback offers for Active Engaged Customers; premium/travel-rewards plus service recovery for High-Value Premium Customers; reactivation and low-interest offers for the Low-Engagement majority. This should raise acceptance rates, lower cost-per-contact, and let BankWise measure performance segment by segment.

## Limitation & Responsible AI Note
The data is synthetic and only six numeric features were used, so segments are approximate. K-means assumes circular, balanced clusters and produces fuzzy, overlapping boundaries on real customer behaviour. Features tied to income or credit standing can encode social inequality, so segments must inform - not replace - human marketing decisions, and any targeting rule should be reviewed for fairness before reaching real customers.

## Files
- `Assignment5_Aayush_B` - completed, executed notebook
- `bankwise_credit_card_offer_acceptance_dataset.csv` - dataset
- `business_plan_bankwise_credit_card_offer.pdf` - business plan
- `README.md` - this file
