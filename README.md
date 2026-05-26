# Mastercard Data Quest 2026 — Hidden Entrepreneur Detection

Detect self-employed individuals using personal consumer cards for business purposes, using transaction data and a Random Forest classifier.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [Quick Start](#4-quick-start)
5. [How to Run](#5-how-to-run)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Training & Evaluation](#7-model-training--evaluation)
8. [Results & Metrics](#8-results--metrics)
9. [Score Interpretation & Action Tiers](#9-score-interpretation--action-tiers)
10. [Hidden Entrepreneur Personas](#10-hidden-entrepreneur-personas)
11. [Explainability](#11-explainability)
12. [Dashboard Guide](#12-dashboard-guide)
13. [Synthetic Data Warning](#13-synthetic-data-warning)
14. [Business Recommendations](#14-business-recommendations)
15. [Files & Outputs](#15-files--outputs)
16. [Requirements](#16-requirements)
17. [Deployment](#17-deployment)
18. [Limitations & Future Work](#18-limitations--future-work)

---

## 1. Project Overview

**Author:** Assem Kadirova & Aiganym Tyshkanbayeva

**Competition:** Mastercard Data Quest 2026  
**Market:** Kazakhstan  
**Task:** Binary classification — identify consumer cardholders who are actually running commercial activity through their personal card

---

## 2. Problem Statement

A segment of small business owners and self-employed people in Kazakhstan run their commercial activity through regular **consumer cards** instead of business cards. Mastercard cannot identify or offer them B2B products (acquiring terminals, working capital loans, payroll solutions, trade finance) because they appear as ordinary consumers in transaction data.

**Goal:** Build an ML model that flags these "hidden entrepreneurs" based purely on their transaction behaviour, so Mastercard's commercial team can reach out with tailored offers.

---

## 3. Dataset

Three parquet files (stored in Git LFS):

| File | Rows | Description |
|---|---|---|
| `business_cards_MDQ.parquet` | ~3M transactions | 25,000 confirmed business cardholders |
| `consumer_cards_MDQ.parquet` | ~10M transactions | 80,000 consumer cardholders |
| `merchants_reference.parquet` | 2,165 merchants | Merchant name, MCC, country, recurring capability |

**Transaction date range:** 1 October 2025 – 31 March 2026  
**Data is fully synthetic and for educational purposes only.**

### Key columns

| Column | Description |
|---|---|
| `transaction_date` | Calendar date |
| `transaction_timestamp` | Full datetime (used for hour/day-of-week features) |
| `transaction_amount_kzt` | Amount in Kazakhstani Tenge |
| `mcc` | Merchant Category Code (ISO 18245) |
| `merchant_id` | Links to `merchants_reference.parquet` |
| `channel` | `online` or `POS` |
| `card_number` | 16-digit card identifier |
| `card_tier` | Product level (`Business` / `Standard` / `Affluent` / `Premium`) |
| `tokenized` | Apple Pay / Samsung Pay used |
| `is_recurring` | Subscription or auto-billing charge |

---

## 4. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run ML pipeline (produces scores + charts, ~3–5 min)
python3 solution.py

# 3. Launch interactive dashboard
python3 dashboard.py
# Open http://127.0.0.1:8050
```

---

## 5. How to Run

### ML Pipeline (`solution.py`)

Runs the full 13-step pipeline:

```bash
python3 solution.py
```

Steps executed:
1. Load data (3 parquet files)
2. Data quality checks & outlier analysis
3. Feature engineering (26 features per card)
4. Exploratory visualisation
5. Train/test split (80/20, stratified)
6. Logistic Regression baseline
7. Random Forest with RandomizedSearchCV (12 combos × 5 folds)
8. Cross-validation comparison
9. Evaluation on held-out test set
10. Evaluation charts
11. Threshold optimisation (F1-maximised)
12. Score all consumer cards
13. Feature importance & explainability summary

**Runtime:** ~3–5 minutes on a modern laptop (feature engineering on 13M rows + RF with 300 trees).

### Dashboard (`dashboard.py`)

```bash
python3 dashboard.py
```

Open **http://127.0.0.1:8050** in your browser. The dashboard re-trains the model on startup (~2–3 minutes for the first load).

---

## 6. Feature Engineering

26 features are engineered at the card level (one row per cardholder) across five categories:

### Volume & Amount
| Feature | Description |
|---|---|
| `txn_count` | Total number of transactions |
| `total_spend_kzt` | Sum of all transaction amounts |
| `avg_amount` | Mean transaction amount |
| `std_amount` | Standard deviation of amounts |
| `max_amount` | Largest single transaction |
| `amount_cv` | Coefficient of variation (std / mean) |
| `large_txn_ratio` | Share of transactions > 100,000 KZT |

### Merchant & Category Diversity
| Feature | Description |
|---|---|
| `unique_merchants` | Number of distinct merchants |
| `unique_mccs` | Number of distinct MCCs |
| `unique_countries` | Number of distinct merchant countries |
| `txn_per_merchant` | Average transactions per merchant |
| `business_mcc_ratio` | Share of transactions at B2B MCCs (advertising, SaaS, consulting) |
| `foreign_merchant_ratio` | Share of transactions at non-Kazakh merchants |

### Channel & Payment Method
| Feature | Description |
|---|---|
| `online_ratio` | Share of online transactions |
| `pos_ratio` | Share of POS terminal transactions |
| `recurring_ratio` | Share of recurring / subscription charges |
| `tokenized_ratio` | Share using Apple Pay / Samsung Pay |
| `recurring_capable_ratio` | Share at merchants with recurring capability |

### Temporal Patterns
| Feature | Description |
|---|---|
| `business_hours_ratio` | Share of transactions 09:00–18:00 |
| `night_ratio` | Share of transactions 22:00–05:59 |
| `weekday_ratio` | Share Mon–Fri |
| `weekend_ratio` | Share Sat–Sun |
| `evening_ratio` | Share 18:00–21:59 (consumer peak) |

### Activity Consistency
| Feature | Description |
|---|---|
| `active_months` | Number of months with at least one transaction |
| `monthly_txn_mean` | Average monthly transaction count |

---

## 7. Model Training & Evaluation

### Approach

- **Labelling:** Business cardholders → label `1`, consumer cardholders → label `0`
- **Split:** 80/20 stratified train/test
- **Class imbalance:** SMOTE oversampling on training set (80K consumers vs 25K business)
- **Baseline:** Logistic Regression with StandardScaler
- **Main model:** Random Forest with RandomizedSearchCV (12 random combinations, 5-fold CV, scoring=ROC-AUC)

### Hyperparameter search space

```python
param_dist = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [8, 10, 12, None],
    "min_samples_leaf": [3, 5, 10],
    "max_features":     ["sqrt", "log2"],
}
```

### Results

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | 1.0000 | 1.0000 |
| **Random Forest (tuned)** | **1.0000** | **1.0000** |

> AUC = 1.000 is expected on this synthetic dataset. See [Section 13](#13-synthetic-data-warning).

### Threshold optimisation

Optimal threshold (maximising F1 on test set): **0.41**

| Metric | Value |
|---|---|
| Precision | 0.999 |
| Recall | 1.000 |
| F1-score | 1.000 |

---

## 8. Results & Metrics

- **29 hidden entrepreneurs** identified out of 80,000 consumer cardholders (0.036%)

### Top discriminating features

| Rank | Feature | Importance | Insight |
|---|---|---|---|
| 1 | `online_ratio` | 15.8% | Businesses transact online (SaaS, ads); consumers use POS |
| 2 | `evening_ratio` | 14.8% | Consumers peak 18–21h; businesses do not |
| 3 | `pos_ratio` | 13.2% | Businesses rarely swipe at POS terminals |
| 4 | `weekend_ratio` | 10.9% | Consumer spending is weekend-heavy |
| 5 | `weekday_ratio` | 9.9% | Business activity is weekday-concentrated |
| 6 | `tokenized_ratio` | 6.0% | Business cards more frequently tokenised |
| 7 | `business_mcc_ratio` | 5.9% | Advertising, SaaS, consulting MCCs dominate business spend |

---

## 9. Score Interpretation & Action Tiers

Every consumer card receives a **business score** (0–1): the Random Forest's probability of being a business cardholder.

| Score Range | Tier | Recommended Action |
|---|---|---|
| ≥ 0.75 | **Direct Outreach** | Personal call or direct mail with business card upgrade offer |
| 0.50–0.75 | **Campaign Target** | Include in targeted B2B product campaign |
| 0.30–0.50 | **Monitor** | Add to broader prospecting watchlist; re-score monthly |
| < 0.30 | **No Action** | Insufficient business signals at current threshold |

**Threshold guidance:**
- Use **≥ 0.41** for high-precision conversion campaigns (current default)
- Lower to **≥ 0.25** for broader prospecting with slightly more false positives
- Raise to **≥ 0.75** for direct outreach where false positives carry a cost

---

## 10. Hidden Entrepreneur Personas

Five archetypes identified from the data:

### Digital Seller
Runs e-commerce or drop-shipping. Heavy spend on Meta/Google Ads, logistics platforms, and SaaS tools — entirely online, concentrated on weekdays.  
**Signals:** High `online_ratio`, advertising & SaaS MCCs, low `weekend_ratio`  
**Offer:** Business card + merchant acquiring + working capital loan

### International Freelancer
Software developer or designer serving international clients. Pays for AWS, GitHub, Figma; receives Stripe or Wise payouts.  
**Signals:** High `foreign_merchant_ratio`, developer tool MCCs, consulting categories  
**Offer:** Multi-currency business card + FX fee waiver

### Traditional Merchant
Brick-and-mortar shop owner buying inventory from wholesalers. Large individual transactions, consistent monthly patterns.  
**Signals:** Higher `pos_ratio`, wholesale MCCs (5045, 5085), high `large_txn_ratio`  
**Offer:** POS acquiring terminal + trade finance + inventory credit

### Platform Operator
Operates a subscription service or app. Recurring charges dominate — hosting, CRM, payment processors.  
**Signals:** High `recurring_ratio`, SaaS MCCs, stable weekday pattern  
**Offer:** Business card + payroll solution + B2B credit line

### Emerging Side Hustle
Growing side business alongside regular employment. Mixed consumer and business signals — early stage.  
**Signals:** Moderate score (0.30–0.50), mixed channel and timing patterns  
**Offer:** Monitor + educational B2B onboarding content

---

## 11. Explainability

The dashboard's **Candidates** tab shows a "Why Flagged?" panel for the top 5 candidates. For each card, the top 5 signals are computed by comparing the card's feature values to the **average consumer card**:

- **▲ above average** on a business-like feature (e.g., `online_ratio` higher than typical consumer) → strengthens the business classification
- **▼ below average** on a consumer-like feature (e.g., `evening_ratio` lower than typical consumer) → also strengthens the business classification

This is a feature-deviation approach rather than SHAP (which requires an additional library). It produces non-technical, actionable explanations suitable for a relationship manager.

Example output:
```
****9846  Score: 0.756  [Direct Outreach]
▲ Online channel usage:   88% vs avg 52%
▲ Business MCC spend:     42% vs avg 8%
▲ Weekday concentration:  92% vs avg 71%
▲ Recurring payments:     23% vs avg 11%
▼ Evening activity:        6% vs avg 22%
```

---

## 12. Dashboard Guide

Run `python3 dashboard.py` and open **http://127.0.0.1:8050**.

### Tab: Overview
- Synthetic data warning banner
- KPI row: business cards, consumer cards, hidden entrepreneurs, detection rate, ROC-AUC
- Business score distribution histogram with threshold line
- Monthly spend volume trend (Oct 2025 – Mar 2026)
- Transaction count vs average amount scatter by segment
- Five customer persona cards

### Tab: Behaviour
- Hourly transaction activity (Business vs Consumer) with business-hours highlight
- Day-of-week transaction patterns
- Top 10 merchant categories per segment
- Behavioural radar chart (7 dimensions: Online, Recurring, B2B MCC, Weekday, Large Txn, Tokenized, Foreign)

### Tab: Model
- Synthetic data warning banner
- Top 15 feature importances (Random Forest)
- ROC curve (labelled as synthetic dataset)
- Precision / Recall / F1 vs threshold with optimal threshold marked
- Feature correlation heatmap (top features + label)

### Tab: Candidates
- Score tier summary: Direct Outreach / Campaign Target / Monitor counts with recommended actions
- Top 20 candidates bar chart (colour-scaled by score)
- "Why Flagged?" explanation cards for top 5 candidates
- Sortable, filterable table of top 50 consumer cards with tier column

---

## 13. Synthetic Data Warning

> **Important:** The dataset used in this project is **fully synthetic** and was generated specifically for the Mastercard Data Quest 2026 competition. It is for educational and demonstration purposes only.

The model achieves **ROC-AUC = 1.000** because the synthetic data generator creates very clean and exaggerated behavioural differences between business and consumer card segments. This is expected and is **not** a sign of overfitting or data leakage.

In a real-world production deployment:
- AUC would typically be **0.75–0.90** — a good result, but not perfect
- Threshold tuning would be significantly more important
- Feature distributions would overlap more, requiring richer features (incoming payment data, KYC attributes, device fingerprints)
- Monthly re-scoring and drift monitoring would be essential

All business metrics (29 hidden entrepreneurs, score distributions, threshold of 0.41) apply to this synthetic dataset only.

---

## 14. Business Recommendations

1. **Convert top-scored cardholders** — reach out to the 29 flagged cards (score ≥ 0.41) with a tailored business card upgrade offer. Direct Outreach tier (score ≥ 0.75) should receive personalised contact.

2. **Cross-sell B2B products** — working capital loans, POS acquiring terminals, payroll solutions, trade finance, multi-currency accounts.

3. **Targeting threshold** — use score ≥ 0.41 for high-precision campaigns; lower to 0.25 for broader prospecting with slightly more false positives.

4. **Refresh monthly** — re-score the consumer portfolio each month as transaction patterns evolve. A side hustle in month 1 may be a full business by month 6.

5. **Enrich with incoming data** — if payroll or incoming transfer data is available, add as features. Self-employed individuals often receive irregular large incoming transfers.

6. **Segment-specific outreach** — match the offer to the persona (e.g., multi-currency card for Freelancers, POS terminal for Traditional Merchants).

---

## 15. Files & Outputs

```
mastercard-data-quest/
├── solution.py                      # Full reproducible ML pipeline (13 steps)
├── dashboard.py                     # Interactive Dash web dashboard
├── requirements.txt                 # Python dependencies
├── business_cards_MDQ.parquet       # Training data – business cards (Git LFS)
├── consumer_cards_MDQ.parquet       # Training data – consumer cards (Git LFS)
├── merchants_reference.parquet      # Merchant reference table (Git LFS)
├── hidden_entrepreneur_scores.csv   # All 80K consumer cards ranked by score
├── eda_distributions.png            # Feature distributions by segment
├── model_evaluation.png             # Confusion matrix, ROC, feature importance
├── threshold_analysis.png           # Precision/Recall/F1 vs threshold
└── consumer_score_distribution.png  # Business score histogram for consumers
```

The four PNG files and the CSV are generated by running `solution.py`.

---

## 16. Requirements

```
pandas>=2.0.0
pyarrow>=12.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
dash>=2.11.0
```

Install with:
```bash
pip install -r requirements.txt
```

Tested with Python 3.10 and 3.11.

---

## 17. Deployment

### Local

```bash
pip install -r requirements.txt
python3 dashboard.py
# Open http://127.0.0.1:8050
```

### Streamlit Cloud / Render / Railway

The dashboard uses **Dash** (not Streamlit), but can be deployed to any platform that supports Python web apps.

**Entry point:** `dashboard.py`  
**Port:** `8050` (configurable via environment variable or argument)  
**Data folder:** Place the three parquet files in the same directory as `dashboard.py`, or set `DATA_DIR` to their location.

For cloud deployment, ensure the parquet files are either:
- Included in the deployment bundle (they are ~200MB, which may require Git LFS support)
- Hosted in object storage (S3, GCS) and loaded via URL at startup

### Docker (example)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["python3", "dashboard.py"]
```

---

## 18. Limitations & Future Work

### Current Limitations

- **Synthetic data** — real-world AUC will be lower; the 29 flagged cards are an artefact of the data generator
- **Spending-side only** — no incoming payment data; self-employed income signals are not captured
- **No demographic or KYC features** — the model is purely transaction-based
- **Static threshold** — the 0.41 threshold is optimised for this dataset; a real deployment needs periodic recalibration
- **No temporal model** — treats the 6-month window as a static snapshot; a time-series approach could detect emerging businesses earlier

### Potential Improvements

- Add **SHAP values** for richer per-card explainability (requires `shap` library)
- Train on **rolling windows** to detect businesses as they emerge
- Add **incoming transfer features** (payroll, B2B payments received)
- Use **LightGBM or XGBoost** for faster training with similar accuracy
- Implement **online learning** to update the model as new transactions arrive
- Add **fairness checks** across card tiers and banks
- Build a **monitoring dashboard** for score drift over time
