# Mastercard Data Quest 2026 — Hidden Entrepreneur Detection

Detect self-employed individuals using personal consumer cards for business purposes, using transaction data and a Random Forest classifier.

---

## Problem

A segment of small business owners and self-employed people in Kazakhstan run their commercial activity through regular **consumer cards** instead of business cards. Mastercard can't identify or offer them B2B products (acquiring, working capital loans, payroll) because they look like ordinary consumers.

**Goal:** build an ML model that flags these "hidden entrepreneurs" based purely on their transaction behaviour.

---

## Data

Three parquet files (stored in Git LFS):

| File | Rows | Description |
|---|---|---|
| `business_cards_MDQ.parquet` | ~3M transactions | 25,000 confirmed business cardholders |
| `consumer_cards_MDQ.parquet` | ~10M transactions | 80,000 consumer cardholders |
| `merchants_reference.parquet` | 2,165 merchants | Merchant name, MCC, country, recurring capability |

Transaction date range: **1 October 2025 – 31 March 2026**

> Data is fully synthetic and for educational purposes only.

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

## Approach

### 1. Labelling
Business cardholders → label `1`, consumer cardholders → label `0`. The model learns to distinguish the two groups, then scores all consumer cards to surface those behaving like businesses.

### 2. Feature engineering (card-level aggregation)
33 features per card covering:

| Category | Features |
|---|---|
| **Volume** | `txn_count`, `total_spend_kzt`, `avg_amount`, `median_amount`, `std_amount`, `max_amount`, `amount_cv`, `large_txn_ratio` |
| **Diversity** | `unique_merchants`, `unique_mccs`, `unique_countries`, `txn_per_merchant` |
| **Channel** | `online_ratio`, `pos_ratio`, `recurring_ratio`, `tokenized_ratio` |
| **Timing** | `business_hours_ratio`, `night_ratio`, `weekday_ratio`, `weekend_ratio`, `morning_peak_ratio`, `evening_ratio` |
| **Activity** | `active_months`, `monthly_txn_mean`, `monthly_txn_std`, `monthly_txn_cv`, `monthly_growth` |
| **Merchant metadata** | `business_mcc_ratio`, `foreign_merchant_ratio`, `recurring_capable_ratio` |

### 3. Class imbalance
80K consumers vs 25K business cards (3:1 ratio). Handled with **SMOTE** oversampling on the training set before fitting.

### 4. Models trained
| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | 1.0000 | 1.0000 |
| **Random Forest (final)** | **1.0000** | **1.0000** |

Perfect separation is expected on this synthetic dataset — the behavioural signals are very strong. In production data the separation would be softer, making feature engineering and threshold tuning more important.

### 5. Threshold
Optimal threshold (maximising F1): **0.41**
- Precision: 0.999 | Recall: 1.000 | F1: 1.000

---

## Results

- **29 hidden entrepreneurs** identified out of 80,000 consumer cardholders
- Top discriminating features:

| Rank | Feature | Importance | Insight |
|---|---|---|---|
| 1 | `online_ratio` | 15.8% | Businesses transact mostly online (SaaS, ads) |
| 2 | `evening_ratio` | 14.8% | Consumers peak 18–21h; businesses don't |
| 3 | `pos_ratio` | 13.2% | Businesses rarely use POS terminals |
| 4 | `weekend_ratio` | 10.9% | Consumers spend heavily on weekends |
| 5 | `weekday_ratio` | 9.9% | Business activity is weekday-concentrated |
| 6 | `tokenized_ratio` | 6.0% | Business cards more frequently tokenised |
| 7 | `business_mcc_ratio` | 5.9% | Advertising, SaaS, consulting MCCs |

---

## Files

```
mastercard-data-quest/
├── solution.py                      # Full reproducible ML pipeline
├── dashboard.py                     # Interactive Dash web dashboard
├── business_cards_MDQ.parquet       # Training data – business cards (LFS)
├── consumer_cards_MDQ.parquet       # Training data – consumer cards (LFS)
├── merchants_reference.parquet      # Merchant reference table (LFS)
├── hidden_entrepreneur_scores.csv   # All 80K consumer cards ranked by score
├── eda_distributions.png            # Feature distributions by segment
├── model_evaluation.png             # Confusion matrix, ROC, feature importance
├── threshold_analysis.png           # Precision/Recall/F1 vs threshold
└── consumer_score_distribution.png  # Business score histogram for consumers
```

---

## How to run

### Requirements

```bash
pip install pandas pyarrow scikit-learn imbalanced-learn matplotlib seaborn plotly dash
```

### Run the ML pipeline

```bash
python3 solution.py
```

Reproduces all model training, evaluation metrics, and output charts. Takes ~3–5 minutes (feature engineering on 13M rows + Random Forest with 300 trees).

### Run the dashboard

```bash
python3 dashboard.py
```

Open **http://127.0.0.1:8050** in your browser.

The dashboard has four tabs:

| Tab | Contents |
|---|---|
| **Overview** | KPI cards, business score distribution, monthly spend trend, segment scatter |
| **Behaviour** | Hourly activity, day-of-week patterns, top MCCs, behavioural radar chart |
| **Model** | Feature importances, ROC curve, threshold analysis, feature correlation heatmap |
| **Candidates** | Ranked bar chart + sortable/filterable table of top 50 scored consumer cards |

---

## Business recommendations

1. **Convert top-scored cardholders** — reach out to the 29 flagged cards with an offer to upgrade to a business card product.
2. **Cross-sell B2B products** — working capital loans, POS-acquiring terminals, payroll projects, trade finance.
3. **Targeting threshold** — use score ≥ 0.41 for high-precision campaigns; lower the threshold (e.g. 0.25) for broader prospecting with slightly more false positives.
4. **Refresh monthly** — re-score the consumer portfolio each month as transaction patterns evolve.

---

## Limitations

- Data is synthetic — real-world AUC will be lower and threshold tuning will matter more.
- Model is trained on spending behaviour only; incoming payment data (if available) would add strong signal.
- No demographic or KYC features used — the model is purely transaction-based.
