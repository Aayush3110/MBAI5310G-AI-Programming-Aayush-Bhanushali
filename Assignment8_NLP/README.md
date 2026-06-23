# Assignment 8 - NLP Pipeline and Text Classification

**Course:** MBAI 5310G: AI Programming - Ontario Tech University
**Business:** NorthStar Telecom - Customer Support Operations
**Dataset:** NorthStar Service Complaint Dataset (`northstar_service_complaints.csv`)
**Model:** TF-IDF features + Multinomial Naive Bayes (scikit-learn)

## Business Problem

NorthStar is a telecom and internet provider whose support desk receives complaint tickets through five channels (Email, Call Center, Mobile App, Web Form, Chat). Each ticket arrives as a short block of free text, and today a human agent reads every message and manually decides which team it belongs to. That manual triage is **slow, inconsistent between agents, and a bottleneck at peak hours**.

This project builds a **supervised text classifier** that reads the complaint text and predicts its issue category automatically. The intended use is **support-desk decision-support**: route each new ticket to the right team instantly, keep routing consistent regardless of which agent handles it, and measure how complaint volume splits across categories for staffing. The input is the complaint text and the output is one of six issue categories.

## Dataset

`northstar_service_complaints.csv` - 120 support tickets, 9 columns, no missing values. Converted from the provided `.xlsx` file to CSV. The target `IssueCategory` is perfectly balanced at 20 tickets per class.

| Column group | Columns | Role |
|---|---|---|
| Identifier | `TicketID` | Not used for modelling |
| Metadata | `TicketDate`, `Source`, `City`, `PlanType`, `CustomerType`, `Priority` | Context only, not used as features here |
| Text input | `MessageText` | The customer complaint (model input) |
| Target | `IssueCategory` | One of 6 classes (model output) |

**Classes (20 each):** `Mobile_App`, `Cancellation`, `Billing`, `Technical_Support`, `Account_Access`, `Internet_Service`.

## NLP Pipeline and Model

> **Scope:** This is a multi-class text classification task. Only `MessageText` is used as input; the ticket metadata columns are left aside. The pipeline follows the Week 6 NLP reference idioms (NLTK for language processing, scikit-learn for modelling).

Pipeline steps:

1. Load and inspect the dataset (shape, columns, missing values, target balance).
2. Preprocess text: lowercase, remove punctuation and digits, tokenize, remove stopwords, lemmatize, into a `clean_text` column.
3. Exploratory text analysis: frequency distribution and a top-15 word bar chart.
4. POS tagging and Named Entity Recognition on three example complaints.
5. Feature extraction with TF-IDF.
6. Train a Multinomial Naive Bayes classifier on a stratified 75 / 25 split.
7. Evaluate with accuracy, confusion matrix, and classification report.
8. Business interpretation.

## Main Results

| Metric | Value |
|---|---|
| Test accuracy (30 held-out tickets) | 1.00 |
| 5-fold cross-validation accuracy | ≈1.00 (+/- 0.00) |
| Confusion matrix | Perfectly diagonal (no errors on the split) |
| Per-class precision / recall / F1 | 1.00 across all 6 classes |

**Verdict:** The approach works and the six complaint categories are cleanly separable from text alone.

**Caveat:** The 100% score is partly inflated. The 120 tickets are built from only **49 unique message templates** (≈2.4 copies each), so a random split places near-identical messages in both train and test. The model is partly recognising templates it has already seen rather than generalising to new wording. The exact 100% number should be treated with caution until tested on distinct, real complaints.

## Key Business Insights

- Each complaint type is driven by a small set of distinctive keywords (*modem* / *outage* for Internet_Service, *cancel* / *renewal* for Cancellation, *app* / *crash* for Mobile_App), so even a simple model can take over first-line triage.
- TF-IDF correctly downweights the brand word *northstar* and boilerplate agent notes that appear across all categories, letting genuine category words drive the prediction.
- NER pulls structured facts (locations, dates, the organization name) out of free text, which could later flag regional outage clusters or incident dates.

## One Limitation

The dataset is small and heavily templated (120 tickets from 49 unique templates), so the train/test split leaks near-duplicate text and overstates real-world performance. A larger set of distinct, real complaints, plus a deduplicated or template-grouped split, would give a more honest production estimate.

## Files

```
Assignment8_NLP/
├── Assignment8_Aayush_B.ipynb   
├── northstar_service_complaints.csv                
└── README.md                                       
```

## How to Run

1. Install dependencies:
   ```bash
   pip install nltk scikit-learn pandas matplotlib
   ```
2. Open `Assignment8_Aayush_B.ipynb` in Jupyter.
3. Run all cells top to bottom. The first code cell downloads the required NLTK data packages automatically.
