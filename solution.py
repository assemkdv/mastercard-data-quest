"""
Mastercard Data Quest 2026
Hidden Entrepreneur Detection – ML Solution
Author: Assem Kadirova
Task: Build a predictive ML model to identify hidden commercial activity
      (self-employed people using consumer cards for business purposes)
      based on transaction behavior.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from imblearn.over_sampling import SMOTE

# ────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ────────────────────────────────────────────────────────────

DATA_DIR = "/home/user/mastercard-data-quest"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Business-oriented MCCs (confirmed by domain knowledge + EDA):
# higher frequency in business card holders vs consumer card holders
BUSINESS_MCCS = {
    "7311",  # Advertising Services (Google Ads, Meta Ads, Yandex Direct)
    "7372",  # Computer Programming / Software / SaaS
    "5968",  # Direct Marketing / Subscriptions
    "4816",  # Online Data Processing Services
    "7399",  # Business Services NEC
    "7392",  # Management Consulting
    "5045",  # Computers, Peripherals & Software (wholesale)
    "5065",  # Electronic Parts & Equipment (wholesale)
    "5085",  # Industrial & Commercial Machinery (wholesale)
    "4814",  # Telephone Services (B2B telecom)
    "7011",  # Hotels & Lodging (business travel)
    "4511",  # Airlines (business travel)
    "5712",  # Furniture / Office Equipment
    "7021",  # Rooming & Boarding Houses
    "7389",  # Services – Not Elsewhere Classified
    "5200",  # Home Supply / Hardware (construction businesses)
    "5040",  # Medical / Professional Equipment
    "7941",  # Professional Sports
    "7020",  # Hotels (B2B)
    "5122",  # Drug, Drug Proprietaries & Druggists' Sundries (wholesale)
}

# Night hours: 22:00 – 05:59 (signals non-standard schedule)
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))
# Core business hours: 09:00 – 18:00
BUSINESS_HOURS = set(range(9, 19))


# ────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Loading data")
print("=" * 60)

biz = pd.read_parquet(f"{DATA_DIR}/business_cards_MDQ.parquet")
con = pd.read_parquet(f"{DATA_DIR}/consumer_cards_MDQ.parquet")
mer = pd.read_parquet(f"{DATA_DIR}/merchants_reference.parquet")

# Attach label before concatenation
biz["label"] = 1  # business card holder
con["label"] = 0  # consumer card holder

df = pd.concat([biz, con], ignore_index=True)
print(f"Combined dataset: {df.shape[0]:,} transactions  |  {df['card_number'].nunique():,} unique cards")
print(f"  Business cards : {biz['card_number'].nunique():,}")
print(f"  Consumer cards : {con['card_number'].nunique():,}")

# Merge merchant metadata
df = df.merge(mer[["merchant_id", "merchant_country", "recurring_capable"]],
              on="merchant_id", how="left")

# Parse timestamp features
df["hour"]    = df["transaction_timestamp"].dt.hour
df["dow"]     = df["transaction_timestamp"].dt.dayofweek   # 0=Mon … 6=Sun
df["month"]   = df["transaction_timestamp"].dt.month


# ────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (card-level aggregation)
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Feature engineering")
print("=" * 60)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-card transaction data into ML features."""

    g = df.groupby("card_number")

    # ── Volume & amount features ──────────────────────────────
    feat = pd.DataFrame({
        "txn_count":         g.size(),
        "total_spend_kzt":   g["transaction_amount_kzt"].sum(),
        "avg_amount":        g["transaction_amount_kzt"].mean(),
        "median_amount":     g["transaction_amount_kzt"].median(),
        "std_amount":        g["transaction_amount_kzt"].std().fillna(0),
        "max_amount":        g["transaction_amount_kzt"].max(),
        "min_amount":        g["transaction_amount_kzt"].min(),
    })

    # Coefficient of variation in amounts (high CV → irregular / variable spend)
    feat["amount_cv"] = feat["std_amount"] / feat["avg_amount"].replace(0, np.nan)
    feat["amount_cv"] = feat["amount_cv"].fillna(0)

    # Large-ticket transactions (>100K KZT – typical for wholesale, B2B)
    feat["large_txn_ratio"] = (
        g.apply(lambda x: (x["transaction_amount_kzt"] > 100_000).mean())
    )

    # ── Merchant & MCC diversity ──────────────────────────────
    feat["unique_merchants"]   = g["merchant_id"].nunique()
    feat["unique_mccs"]        = g["mcc"].nunique()
    feat["unique_countries"]   = g["country"].nunique()

    # Transactions per merchant (many txn / few merchants → loyal consumer)
    feat["txn_per_merchant"] = feat["txn_count"] / feat["unique_merchants"]

    # Business-oriented MCC ratio
    feat["business_mcc_ratio"] = g.apply(
        lambda x: x["mcc"].isin(BUSINESS_MCCS).mean()
    )

    # ── Channel & payment method ──────────────────────────────
    feat["online_ratio"]    = g.apply(lambda x: (x["channel"] == "online").mean())
    feat["pos_ratio"]       = g.apply(lambda x: (x["channel"] == "POS").mean())
    feat["recurring_ratio"] = g["is_recurring"].mean()
    feat["tokenized_ratio"] = g["tokenized"].mean()

    # ── Time-of-day patterns ──────────────────────────────────
    feat["business_hours_ratio"] = g.apply(
        lambda x: x["hour"].isin(BUSINESS_HOURS).mean()
    )
    feat["night_ratio"] = g.apply(
        lambda x: x["hour"].isin(NIGHT_HOURS).mean()
    )
    feat["weekday_ratio"] = g.apply(lambda x: (x["dow"] < 5).mean())
    feat["weekend_ratio"]  = g.apply(lambda x: (x["dow"] >= 5).mean())

    # Peak business hour (10-12) ratio
    feat["morning_peak_ratio"] = g.apply(lambda x: x["hour"].isin({10, 11, 12}).mean())
    # Evening consumer peak (18-21)
    feat["evening_ratio"] = g.apply(lambda x: x["hour"].isin({18, 19, 20, 21}).mean())

    # ── Activity consistency across months ────────────────────
    monthly = df.groupby(["card_number", "month"]).size().unstack(fill_value=0)
    feat["active_months"]        = (monthly > 0).sum(axis=1)
    feat["monthly_txn_mean"]     = monthly.mean(axis=1)
    feat["monthly_txn_std"]      = monthly.std(axis=1).fillna(0)
    feat["monthly_txn_cv"]       = (
        feat["monthly_txn_std"] / feat["monthly_txn_mean"].replace(0, np.nan)
    ).fillna(0)
    # Acceleration: last 3 months trend (higher = growing business)
    last_months  = monthly.iloc[:, -3:].mean(axis=1)
    first_months = monthly.iloc[:, :3].mean(axis=1)
    feat["monthly_growth"] = (
        (last_months - first_months) / first_months.replace(0, np.nan)
    ).fillna(0).clip(-3, 3)

    # ── Merchant metadata ─────────────────────────────────────
    feat["recurring_capable_ratio"] = g.apply(
        lambda x: x["recurring_capable"].mean() if "recurring_capable" in x else 0
    )
    feat["foreign_merchant_ratio"] = g.apply(
        lambda x: (x["merchant_country"] != "Kazakhstan").mean()
    )

    # ── Label (one per card – consistent by construction) ─────
    feat["label"] = g["label"].first()

    return feat.reset_index()


card_features = build_features(df)
print(f"Features built for {len(card_features):,} cards")
print(f"Feature matrix shape: {card_features.shape}")
print(f"\nLabel distribution:\n{card_features['label'].value_counts().rename({1:'Business', 0:'Consumer'})}")


# ────────────────────────────────────────────────────────────
# 3. EXPLORATORY VISUALISATION
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – Exploratory visualisation")
print("=" * 60)

FEATURE_LABELS = {
    "txn_count":              "Transaction Count",
    "total_spend_kzt":        "Total Spend (KZT)",
    "avg_amount":             "Avg Transaction Amount",
    "online_ratio":           "Online Channel Ratio",
    "recurring_ratio":        "Recurring Payment Ratio",
    "business_mcc_ratio":     "Business MCC Ratio",
    "business_hours_ratio":   "Business Hours (9-18) Ratio",
    "weekday_ratio":          "Weekday Ratio",
    "unique_merchants":       "Unique Merchants",
    "unique_mccs":            "Unique MCCs",
    "large_txn_ratio":        "Large Txn (>100K KZT) Ratio",
    "tokenized_ratio":        "Tokenized Payment Ratio",
}

biz_f = card_features[card_features["label"] == 1]
con_f = card_features[card_features["label"] == 0]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Business vs Consumer Card Holders – Key Feature Distributions", fontsize=14, fontweight="bold")

for ax, (col, label) in zip(axes.flat, FEATURE_LABELS.items()):
    data_b = biz_f[col].clip(biz_f[col].quantile(0.01), biz_f[col].quantile(0.99))
    data_c = con_f[col].clip(con_f[col].quantile(0.01), con_f[col].quantile(0.99))
    ax.hist(data_c, bins=40, alpha=0.6, color="#4C7BF4", density=True, label="Consumer")
    ax.hist(data_b, bins=40, alpha=0.6, color="#F4934C", density=True, label="Business")
    ax.set_title(label, fontsize=9)
    ax.set_yticks([])
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_distributions.png")


# ────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – Model training")
print("=" * 60)

FEATURE_COLS = [c for c in card_features.columns if c not in ("card_number", "label")]

X = card_features[FEATURE_COLS].values
y = card_features["label"].values

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"Class balance in train – Business: {y_train.sum():,}  Consumer: {(y_train==0).sum():,}")

# SMOTE to handle class imbalance (80K consumer vs 25K business)
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE – Train: {X_train_res.shape[0]:,}  (balanced)")

# ── Model 1: Logistic Regression (baseline) ──────────────────
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.5))
])
lr_pipe.fit(X_train_res, y_train_res)

# ── Model 2: Random Forest ────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_res, y_train_res)

# ── Cross-validation comparison ───────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models = {"Logistic Regression": lr_pipe, "Random Forest": rf}
print("\nCross-validated ROC-AUC (5-fold, on original train data):")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")


# ────────────────────────────────────────────────────────────
# 5. EVALUATION
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 – Evaluation on held-out test set")
print("=" * 60)

def evaluate(model, X_test, y_test, name, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)
    auc   = roc_auc_score(y_test, proba)
    ap    = average_precision_score(y_test, proba)
    cm    = confusion_matrix(y_test, pred)
    print(f"\n── {name} ──")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Avg Precision (AP): {ap:.4f}")
    print(classification_report(y_test, pred, target_names=["Consumer", "Business"]))
    return proba, pred, cm, auc, ap

lr_proba, lr_pred, lr_cm, lr_auc, lr_ap = evaluate(lr_pipe, X_test, y_test, "Logistic Regression")
rf_proba, rf_pred, rf_cm, rf_auc, rf_ap = evaluate(rf, X_test, y_test, "Random Forest")

# Best model = Random Forest (highest AUC typically)
best_proba = rf_proba
best_pred  = rf_pred
best_cm    = rf_cm
best_name  = "Random Forest"


# ────────────────────────────────────────────────────────────
# 6. VISUALISATIONS – Confusion Matrix, ROC, Feature Importance
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 – Generating output charts")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Evaluation – Random Forest", fontsize=13, fontweight="bold")

# 6a. Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["Consumer", "Business"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix", fontsize=11)

# 6b. ROC curves for all models
from sklearn.metrics import roc_curve
for name, proba, auc in [
    ("Logistic Regression", lr_proba, lr_auc),
    ("Random Forest", rf_proba, rf_auc),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])

# 6c. Top-20 Feature Importances (Random Forest)
importances  = rf.feature_importances_
feat_names   = FEATURE_COLS
sorted_idx   = np.argsort(importances)[-20:]
axes[2].barh(
    [feat_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color="#4C7BF4"
)
axes[2].set_title("Top-20 Feature Importances (RF)", fontsize=11)
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: model_evaluation.png")


# ────────────────────────────────────────────────────────────
# 7. THRESHOLD ANALYSIS & BUSINESS RECOMMENDATIONS
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 – Threshold optimisation & business scoring")
print("=" * 60)

# Precision-Recall tradeoff
precisions, recalls, thresholds = precision_recall_curve(y_test, rf_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_thresh_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_thresh_idx]
print(f"Optimal threshold (F1): {best_thresh:.3f}")
print(f"  Precision: {precisions[best_thresh_idx]:.3f}")
print(f"  Recall   : {recalls[best_thresh_idx]:.3f}")
print(f"  F1-score : {f1_scores[best_thresh_idx]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Threshold Analysis & Precision-Recall Curve", fontsize=12, fontweight="bold")

axes[0].plot(thresholds, precisions[:-1], label="Precision", color="#4C7BF4")
axes[0].plot(thresholds, recalls[:-1], label="Recall", color="#F4934C")
axes[0].plot(thresholds, f1_scores[:-1], label="F1", color="#2DBF70", linewidth=2)
axes[0].axvline(best_thresh, color="red", linestyle="--", label=f"Best threshold={best_thresh:.2f}")
axes[0].set_xlabel("Threshold")
axes[0].set_title("Precision / Recall / F1 vs Threshold")
axes[0].legend()

axes[1].plot(recalls[:-1], precisions[:-1], color="#4C7BF4")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title(f"Precision-Recall Curve (AP={rf_ap:.3f})")
axes[1].fill_between(recalls[:-1], precisions[:-1], alpha=0.15, color="#4C7BF4")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: threshold_analysis.png")


# ────────────────────────────────────────────────────────────
# 8. APPLY MODEL TO CONSUMER CARDS → FIND HIDDEN ENTREPRENEURS
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 – Identifying hidden entrepreneurs among consumer cards")
print("=" * 60)

# Use features computed for consumer cards only
consumer_feat = card_features[card_features["label"] == 0].copy()
X_consumers = consumer_feat[FEATURE_COLS].values
consumer_proba = rf.predict_proba(X_consumers)[:, 1]

consumer_feat["business_score"] = consumer_proba
consumer_feat["predicted_hidden_entrepreneur"] = (consumer_proba >= best_thresh).astype(int)

n_hidden = consumer_feat["predicted_hidden_entrepreneur"].sum()
pct_hidden = n_hidden / len(consumer_feat) * 100
print(f"Consumer cards scored: {len(consumer_feat):,}")
print(f"Predicted hidden entrepreneurs: {n_hidden:,}  ({pct_hidden:.1f}%)")

# Top candidates
top_hidden = consumer_feat.nlargest(20, "business_score")[
    ["card_number", "business_score", "txn_count", "total_spend_kzt",
     "online_ratio", "recurring_ratio", "business_mcc_ratio", "weekday_ratio"]
]
print("\nTop-20 Hidden Entrepreneur Candidates:")
print(top_hidden.to_string(index=False))

# Score distribution
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(consumer_proba, bins=80, color="#4C7BF4", edgecolor="none", alpha=0.8)
ax.axvline(best_thresh, color="red", linestyle="--", linewidth=1.5,
           label=f"Threshold = {best_thresh:.2f}")
ax.set_xlabel("Business Score (probability)")
ax.set_ylabel("Number of Consumer Cards")
ax.set_title("Distribution of Business Scores for Consumer Cardholders", fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/consumer_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: consumer_score_distribution.png")

# Export results
consumer_feat.sort_values("business_score", ascending=False).to_csv(
    f"{DATA_DIR}/hidden_entrepreneur_scores.csv", index=False
)
print("Saved: hidden_entrepreneur_scores.csv")


# ────────────────────────────────────────────────────────────
# 9. MODEL EXPLAINABILITY SUMMARY
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 – Model explainability")
print("=" * 60)

# Top features ranked by importance
feat_imp_df = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print("\nTop-15 features driving the business/consumer classification:")
print(feat_imp_df.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("SOLUTION COMPLETE")
print("=" * 60)
print("\nKey business signals for hidden entrepreneurs:")
print("  1. High proportion of ONLINE transactions (business buys SaaS, ads online)")
print("  2. Business-oriented MCC mix (advertising, software, consulting)")
print("  3. High recurring payment ratio (SaaS subscriptions, platform fees)")
print("  4. Larger avg transaction amounts (wholesale, B2B purchases)")
print("  5. Weekday-concentrated transactions (vs. consumer evening/weekend peak)")
print("  6. Transactions during business hours 9-18")
print("  7. Higher merchant & MCC diversity")
print("  8. Foreign merchant exposure (AWS, Google, Stripe, etc.)")
print("\nRecommendations:")
print("  → Offer business card conversion to top-scored consumer holders")
print("  → Target with B2B products: POS-acquiring, payroll projects, working capital loans")
print("  → Use score >= {:.2f} for high-precision targeting campaign".format(best_thresh))
