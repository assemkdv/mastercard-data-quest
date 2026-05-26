"""
Mastercard Data Quest 2026 – Hidden Entrepreneur Dashboard
Run: python3 dashboard.py  →  open http://127.0.0.1:8050

First run trains the model (~3 min) and saves dashboard_cache.pkl.gz.
Every subsequent start loads from cache in < 10 seconds.
Commit dashboard_cache.pkl.gz so deployed instances start instantly too.
"""

import os
import sys
import gzip
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, dash_table
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
THRESHOLD  = 0.41
CACHE_PATH = f"{DATA_DIR}/dashboard_cache.pkl.gz"

BUSINESS_MCCS  = {"7311","7372","5968","4816","7399","7392","5045","5065","5085",
                   "4814","7011","4511","5712","7389","5200","5040"}
NIGHT_HOURS    = set(range(22, 24)) | set(range(0, 6))
BUSINESS_HOURS = set(range(9, 19))

# ── Cache helpers ─────────────────────────────────────────────
def _save_cache(data):
    with gzip.open(CACHE_PATH, "wb") as fh:
        pickle.dump(data, fh, protocol=4)
    mb = os.path.getsize(CACHE_PATH) / 1e6
    print(f"Cache saved ({mb:.1f} MB) — next startup will be instant.")

def _load_cache():
    with gzip.open(CACHE_PATH, "rb") as fh:
        return pickle.load(fh)

# ── Load / train ──────────────────────────────────────────────
if os.path.exists(CACHE_PATH):
    print("Loading from cache…")
    c              = _load_cache()
    feat           = c["feat"]
    FEATURE_COLS   = c["feature_cols"]
    hour_biz       = c["hour_biz"]
    hour_con       = c["hour_con"]
    dow_biz        = c["dow_biz"]
    dow_con        = c["dow_con"]
    mcc_biz        = c["mcc_biz"]
    mcc_con        = c["mcc_con"]
    monthly_biz    = c["monthly_biz"]
    monthly_con    = c["monthly_con"]
    feat_importances = c["feat_importances"]
    X_test         = c["X_test"]
    y_test         = c["y_test"]
    rf_proba       = c["rf_proba"]

else:
    print("First run — loading data and training (~3 min)…")
    try:
        biz = pd.read_parquet(f"{DATA_DIR}/business_cards_MDQ.parquet")
        con = pd.read_parquet(f"{DATA_DIR}/consumer_cards_MDQ.parquet")
        mer = pd.read_parquet(f"{DATA_DIR}/merchants_reference.parquet")
    except FileNotFoundError as e:
        sys.exit(f"Data file not found: {e}\n"
                 "Place the three parquet files in the same directory as dashboard.py.")

    biz["label"] = 1
    con["label"] = 0
    df = pd.concat([biz, con], ignore_index=True)
    df = df.merge(mer[["merchant_id","merchant_country","recurring_capable"]],
                  on="merchant_id", how="left")
    df["hour"]  = df["transaction_timestamp"].dt.hour
    df["dow"]   = df["transaction_timestamp"].dt.dayofweek
    df["month"] = df["transaction_timestamp"].dt.month

    print("Building features…")
    g       = df.groupby("card_number")
    monthly = df.groupby(["card_number","month"]).size().unstack(fill_value=0)

    feat = pd.DataFrame({
        "txn_count":               g.size(),
        "total_spend_kzt":         g["transaction_amount_kzt"].sum(),
        "avg_amount":              g["transaction_amount_kzt"].mean(),
        "std_amount":              g["transaction_amount_kzt"].std().fillna(0),
        "max_amount":              g["transaction_amount_kzt"].max(),
        "unique_merchants":        g["merchant_id"].nunique(),
        "unique_mccs":             g["mcc"].nunique(),
        "unique_countries":        g["country"].nunique(),
        "online_ratio":            g.apply(lambda x: (x["channel"]=="online").mean()),
        "pos_ratio":               g.apply(lambda x: (x["channel"]=="POS").mean()),
        "recurring_ratio":         g["is_recurring"].mean(),
        "tokenized_ratio":         g["tokenized"].mean(),
        "business_hours_ratio":    g.apply(lambda x: x["hour"].isin(BUSINESS_HOURS).mean()),
        "night_ratio":             g.apply(lambda x: x["hour"].isin(NIGHT_HOURS).mean()),
        "weekday_ratio":           g.apply(lambda x: (x["dow"]<5).mean()),
        "weekend_ratio":           g.apply(lambda x: (x["dow"]>=5).mean()),
        "evening_ratio":           g.apply(lambda x: x["hour"].isin({18,19,20,21}).mean()),
        "business_mcc_ratio":      g.apply(lambda x: x["mcc"].isin(BUSINESS_MCCS).mean()),
        "large_txn_ratio":         g.apply(lambda x: (x["transaction_amount_kzt"]>100_000).mean()),
        "foreign_merchant_ratio":  g.apply(lambda x: (x["merchant_country"]!="Kazakhstan").mean()),
        "recurring_capable_ratio": g.apply(lambda x: x["recurring_capable"].mean()),
        "active_months":           (monthly>0).sum(axis=1),
        "monthly_txn_mean":        monthly.mean(axis=1),
        "bank_name":               g["bank_name"].first(),
        "card_tier":               g["card_tier"].first(),
        "label":                   g["label"].first(),
    })
    feat["amount_cv"]        = (feat["std_amount"] / feat["avg_amount"].replace(0,np.nan)).fillna(0)
    feat["txn_per_merchant"] = feat["txn_count"] / feat["unique_merchants"]
    feat = feat.reset_index()

    FEATURE_COLS = [c for c in feat.columns
                    if c not in ("card_number","label","bank_name","card_tier")]

    print("Training model…")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    X = feat[FEATURE_COLS].values
    y = feat["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_res, y_res)

    feat["score"]     = rf.predict_proba(X)[:, 1]
    feat["predicted"] = (feat["score"] >= THRESHOLD).astype(int)
    feat["segment"]   = feat.apply(
        lambda r: "Business Card" if r["label"]==1
        else ("Hidden Entrepreneur" if r["predicted"]==1 else "Consumer Card"), axis=1)
    rf_proba         = rf.predict_proba(X_test)[:, 1]
    feat_importances = rf.feature_importances_.copy()

    print("Precomputing aggregates…")
    hour_biz = df[df["label"]==1].groupby("hour").size().reset_index(name="count")
    hour_biz["pct"] = hour_biz["count"] / hour_biz["count"].sum() * 100
    hour_con = df[df["label"]==0].groupby("hour").size().reset_index(name="count")
    hour_con["pct"] = hour_con["count"] / hour_con["count"].sum() * 100
    dow_biz = df[df["label"]==1].groupby("dow").size().reset_index(name="count")
    dow_biz["pct"] = dow_biz["count"] / dow_biz["count"].sum() * 100
    dow_con = df[df["label"]==0].groupby("dow").size().reset_index(name="count")
    dow_con["pct"] = dow_con["count"] / dow_con["count"].sum() * 100
    mcc_biz = df[df["label"]==1]["mcc"].value_counts().head(10).reset_index()
    mcc_biz.columns = ["mcc","count"]
    mcc_con = df[df["label"]==0]["mcc"].value_counts().head(10).reset_index()
    mcc_con.columns = ["mcc","count"]
    monthly_biz = df[df["label"]==1].groupby("month")["transaction_amount_kzt"].sum() / 1e9
    monthly_con = df[df["label"]==0].groupby("month")["transaction_amount_kzt"].sum() / 1e9

    _save_cache({
        "feat": feat, "feature_cols": FEATURE_COLS,
        "hour_biz": hour_biz, "hour_con": hour_con,
        "dow_biz": dow_biz, "dow_con": dow_con,
        "mcc_biz": mcc_biz, "mcc_con": mcc_con,
        "monthly_biz": monthly_biz, "monthly_con": monthly_con,
        "feat_importances": feat_importances,
        "X_test": X_test, "y_test": y_test, "rf_proba": rf_proba,
    })
    del df, rf, X, y, X_train, y_train, X_res, y_res

# ── Derived variables (always fast) ──────────────────────────
biz_f    = feat[feat["label"]==1]
con_f    = feat[feat["label"]==0]
hidden_f = feat[(feat["label"]==0) & (feat["predicted"]==1)]
print(f"Ready — {len(hidden_f)} hidden entrepreneurs identified")

MCC_NAMES = {
    "7311":"Advertising",    "7372":"Software/SaaS",    "5968":"Direct Marketing",
    "4816":"Online Services","7399":"Business Services","7392":"Consulting",
    "5045":"Computers/HW",  "4814":"Telecom",           "7011":"Hotels",
    "4511":"Airlines",       "5812":"Restaurants",       "4121":"Taxi",
    "5541":"Gas Stations",   "4111":"Transit",           "5814":"Fast Food",
    "4899":"Cable/Satellite","5411":"Grocery",           "5311":"Dept Stores",
}
mcc_biz["name"] = mcc_biz["mcc"].map(MCC_NAMES).fillna(mcc_biz["mcc"])
mcc_con["name"] = mcc_con["mcc"].map(MCC_NAMES).fillna(mcc_con["mcc"])

feat_imp = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": feat_importances,
}).sort_values("importance", ascending=True).tail(15)

consumer_scores = feat[feat["label"]==0]["score"].values
month_names     = {10:"Oct",11:"Nov",12:"Dec",1:"Jan",2:"Feb",3:"Mar"}

tier_counts = {
    "direct":   int((con_f["score"] >= 0.75).sum()),
    "campaign": int(((con_f["score"] >= 0.50) & (con_f["score"] < 0.75)).sum()),
    "monitor":  int(((con_f["score"] >= 0.30) & (con_f["score"] < 0.50)).sum()),
}

# ── MASTERCARD BRAND COLOURS ──────────────────────────────────
C_RED    = "#EB001B"
C_YELLOW = "#F79E1B"
C_ORANGE = "#FF5F00"
C_BIZ    = C_RED
C_CON    = C_YELLOW
C_HIDDEN = C_ORANGE
C_BG     = "#0D0D0D"
C_CARD   = "#1A1A1A"
C_BORDER = "#2E2E2E"
C_TEXT   = "#FFFFFF"
C_MUTED  = "#999999"
C_GRID   = "#2E2E2E"

BASE = dict(
    paper_bgcolor=C_BG, plot_bgcolor=C_CARD,
    font=dict(color=C_TEXT, family="'Helvetica Neue', Arial, sans-serif"),
    margin=dict(l=10, r=10, t=44, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    xaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID),
    yaxis=dict(gridcolor=C_GRID, zerolinecolor=C_GRID),
)

def L(**kw):
    """Merge BASE layout with per-chart overrides, no duplicate-key conflicts."""
    d = {k: v for k, v in BASE.items() if k not in kw}
    d.update(kw)
    return d

# ── Explainability helpers ────────────────────────────────────
con_means = con_f[FEATURE_COLS].mean()

EXPLAIN_FEATURES = [
    ("online_ratio",           "Online channel usage",      True),
    ("business_mcc_ratio",     "Business MCC spend",        True),
    ("weekday_ratio",          "Weekday concentration",     True),
    ("recurring_ratio",        "Recurring payments",        True),
    ("tokenized_ratio",        "Tokenized payments",        True),
    ("foreign_merchant_ratio", "Foreign merchants",         True),
    ("large_txn_ratio",        "Large transactions",        True),
    ("evening_ratio",          "Evening activity (18–21h)", False),
    ("pos_ratio",              "POS terminal usage",        False),
    ("weekend_ratio",          "Weekend spending",          False),
]

def get_top_signals(card_row, n=5):
    """Return top n features explaining why a card looks like a business."""
    signals = []
    for feat_name, label, higher_is_biz in EXPLAIN_FEATURES:
        if feat_name not in card_row.index:
            continue
        card_val     = float(card_row[feat_name])
        avg_val      = float(con_means[feat_name])
        deviation    = card_val - avg_val
        contribution = deviation if higher_is_biz else -deviation
        signals.append({"feature": label, "card_val": card_val,
                         "avg_val": avg_val, "contribution": contribution})
    signals.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return signals[:n]

def tier_label(score):
    if score >= 0.75:
        return ("Direct Outreach", C_RED,
                "Priority conversion — strong business signals across multiple features.")
    if score >= 0.50:
        return ("Campaign Target", C_ORANGE,
                "Clear business behaviour — include in targeted B2B product campaign.")
    if score >= 0.30:
        return ("Monitor", C_YELLOW,
                "Emerging signals — add to broader prospecting watchlist.")
    return ("No Action", C_MUTED,
            "Insufficient business signals at current threshold.")

# ── App constants ─────────────────────────────────────────────
app = Dash(__name__, title="MDQ 2026 – Hidden Entrepreneurs")

CARD = {
    "background": C_CARD, "borderRadius": "8px", "padding": "20px",
    "marginBottom": "16px", "border": f"1px solid {C_BORDER}",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.4)",
}
KPI  = {**CARD, "textAlign": "center", "padding": "28px 16px"}
WARN = {
    "background": "#1a1200", "border": f"1px solid {C_YELLOW}",
    "borderRadius": "6px", "padding": "12px 16px", "marginBottom": "16px",
    "fontSize": "0.8rem", "color": C_YELLOW, "lineHeight": "1.6",
}
TAB_STYLE = {
    "background": C_CARD, "color": C_MUTED,
    "border": f"1px solid {C_BORDER}", "borderRadius": "6px 6px 0 0",
    "padding": "10px 20px", "fontWeight": "500",
    "fontFamily": "'Helvetica Neue', Arial, sans-serif",
}
TAB_SELECTED = {**TAB_STYLE, "background": C_RED, "color": C_TEXT,
                "borderBottom": f"2px solid {C_YELLOW}"}

def kpi(value, label, color=C_TEXT):
    return html.Div([
        html.Div(value, style={"fontSize":"2.2rem","fontWeight":"700",
                               "color":color,"letterSpacing":"-0.5px"}),
        html.Div(label, style={"fontSize":"0.75rem","color":C_MUTED,"marginTop":"6px",
                               "textTransform":"uppercase","letterSpacing":"0.8px"}),
    ], style=KPI)

mc_logo = html.Div([
    html.Div(style={"width":"36px","height":"36px","borderRadius":"50%",
                    "background":C_RED,"position":"absolute","left":"0"}),
    html.Div(style={"width":"36px","height":"36px","borderRadius":"50%",
                    "background":C_YELLOW,"position":"absolute","left":"22px","opacity":"0.9"}),
], style={"position":"relative","width":"62px","height":"36px",
          "marginRight":"18px","flexShrink":"0"})

SYNTHETIC_WARNING = html.Div([
    html.B("Synthetic Data Notice: "),
    "ROC-AUC = 1.000 is expected — the synthetic generator creates very clean behavioural "
    "separation between card segments. In real-world deployment AUC would typically be "
    "0.75–0.90 and threshold tuning would matter significantly more.",
], style=WARN)

PERSONAS = [
    {"title":"Digital Seller",         "icon":"🛒", "color":C_RED,
     "signals":"High online ratio · Advertising & SaaS MCCs · Weekday spend",
     "desc":"Runs e-commerce or drop-shipping. Heavy spend on Meta/Google Ads, logistics and SaaS — entirely online, concentrated on weekdays.",
     "action":"Offer: Business card + merchant acquiring + working capital loan"},
    {"title":"International Freelancer","icon":"💻", "color":C_ORANGE,
     "signals":"High foreign merchant ratio · GitHub/AWS/Stripe MCCs",
     "desc":"Software developer or designer serving international clients. Pays for cloud tools; receives Stripe or Wise payouts.",
     "action":"Offer: Multi-currency business card + FX fee waiver"},
    {"title":"Traditional Merchant",   "icon":"🏪", "color":C_YELLOW,
     "signals":"Higher POS ratio · Wholesale MCCs (5045, 5085) · Large txns",
     "desc":"Brick-and-mortar shop owner buying inventory from wholesalers. Large individual transactions, consistent monthly patterns.",
     "action":"Offer: POS acquiring terminal + trade finance + inventory credit"},
    {"title":"Platform Operator",      "icon":"📱", "color":C_RED,
     "signals":"High recurring ratio · SaaS MCCs · Stable weekday pattern",
     "desc":"Operates a subscription service. Recurring charges dominate — hosting, CRM, payment processors.",
     "action":"Offer: Business card + payroll solution + B2B credit line"},
    {"title":"Emerging Side Hustle",   "icon":"🚀", "color":C_MUTED,
     "signals":"Mixed signals · Moderate score · Growing monthly trend",
     "desc":"Growing side business alongside regular employment. Mixed consumer and business patterns — early stage.",
     "action":"Monitor: Broader prospecting campaign + educational B2B content"},
]

app.layout = html.Div(
    style={"background":C_BG,"minHeight":"100vh","padding":"28px 32px",
           "fontFamily":"'Helvetica Neue', Arial, sans-serif","color":C_TEXT},
    children=[
        html.Div([
            mc_logo,
            html.Div([
                html.H1("Hidden Entrepreneur Detection",
                        style={"margin":"0","fontSize":"1.5rem","fontWeight":"700",
                               "letterSpacing":"-0.3px"}),
                html.Div("Mastercard Data Quest 2026  •  Kazakhstan",
                         style={"color":C_MUTED,"fontSize":"0.8rem","marginTop":"3px"}),
            ]),
            html.Div(style={"flex":"1"}),
            html.Div("MDQ 2026", style={
                "background":f"linear-gradient(135deg,{C_RED},{C_YELLOW})",
                "color":C_TEXT,"padding":"6px 16px","borderRadius":"20px",
                "fontSize":"0.75rem","fontWeight":"700","letterSpacing":"1px",
            }),
        ], style={"display":"flex","alignItems":"center","marginBottom":"28px",
                  "borderBottom":f"1px solid {C_BORDER}","paddingBottom":"20px"}),
        dcc.Tabs(id="tabs", value="overview",
                 style={"marginBottom":"20px","borderBottom":f"2px solid {C_BORDER}"},
                 children=[
            dcc.Tab(label="Overview",   value="overview",   style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Behaviour",  value="behaviour",  style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Model",      value="model",      style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Candidates", value="candidates", style=TAB_STYLE, selected_style=TAB_SELECTED),
        ]),
        html.Div(id="tab-content"),
    ]
)


@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):

    # ── OVERVIEW ─────────────────────────────────────────────
    if tab == "overview":
        kpi_row = html.Div([
            kpi(f"{biz_f.shape[0]:,}",                  "Business Cardholders",        C_BIZ),
            kpi(f"{con_f.shape[0]:,}",                  "Consumer Cardholders",        C_CON),
            kpi(f"{len(hidden_f):,}",                   "Hidden Entrepreneurs",        C_HIDDEN),
            kpi(f"{len(hidden_f)/len(con_f)*100:.2f}%", "Detection Rate",              C_HIDDEN),
            kpi("1.0000",                               "ROC-AUC (synthetic dataset)", C_YELLOW),
        ], style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)",
                  "gap":"12px","marginBottom":"16px"})

        fig_score = go.Figure(go.Histogram(x=consumer_scores, nbinsx=80,
                                           marker_color=C_RED, opacity=0.8))
        fig_score.add_vline(x=THRESHOLD, line_color="red", line_dash="dash",
                            annotation_text=f"Threshold {THRESHOLD}",
                            annotation_font_color="red")
        fig_score.update_layout(**L(title="Business Score Distribution – Consumer Cardholders",
                                    xaxis_title="Business Score", yaxis_title="Number of Cards"))

        sample = feat.sample(min(5000,len(feat)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="txn_count", y="avg_amount", color="segment",
            opacity=0.6, log_y=True,
            color_discrete_map={"Business Card":C_BIZ,"Consumer Card":C_CON,
                                "Hidden Entrepreneur":C_HIDDEN},
            labels={"txn_count":"Transaction Count","avg_amount":"Avg Amount (KZT)"},
            title="Transaction Count vs Avg Amount by Segment",
        )
        fig_scatter.update_layout(**L())
        fig_scatter.update_traces(marker_size=4)

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=[month_names[m] for m in monthly_biz.index], y=monthly_biz.values,
            name="Business", line=dict(color=C_BIZ,width=2), mode="lines+markers"))
        fig_monthly.add_trace(go.Scatter(
            x=[month_names[m] for m in monthly_con.index], y=monthly_con.values,
            name="Consumer", line=dict(color=C_CON,width=2), mode="lines+markers"))
        fig_monthly.update_layout(**L(title="Monthly Spend Volume (Billion KZT)",
                                      xaxis_title="Month", yaxis_title="Total Spend (B KZT)"))

        persona_section = html.Div([
            html.Div("Hidden Entrepreneur Personas",
                     style={"fontWeight":"600","fontSize":"1rem","marginBottom":"12px"}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span(p["icon"], style={"fontSize":"1.3rem","marginRight":"8px"}),
                        html.Span(p["title"], style={"fontWeight":"700","color":p["color"],
                                                      "fontSize":"0.9rem"}),
                    ], style={"display":"flex","alignItems":"center","marginBottom":"6px"}),
                    html.Div(p["signals"],  style={"fontSize":"0.68rem","color":C_MUTED,
                                                    "fontStyle":"italic","marginBottom":"6px"}),
                    html.Div(p["desc"],     style={"fontSize":"0.76rem","color":C_TEXT,
                                                    "lineHeight":"1.4","marginBottom":"8px"}),
                    html.Div(p["action"],   style={"fontSize":"0.7rem","color":p["color"],
                                                    "fontWeight":"500",
                                                    "borderTop":f"1px solid {C_BORDER}",
                                                    "paddingTop":"6px"}),
                ], style={**CARD,"flex":"1","minWidth":"175px","marginBottom":"0",
                           "borderLeft":f"3px solid {p['color']}"})
                for p in PERSONAS
            ], style={"display":"flex","gap":"10px","flexWrap":"wrap"}),
        ], style=CARD)

        return html.Div([
            SYNTHETIC_WARNING,
            kpi_row,
            html.Div([
                html.Div(dcc.Graph(figure=fig_score),   style={**CARD,"flex":"1.5"}),
                html.Div(dcc.Graph(figure=fig_monthly), style={**CARD,"flex":"1"}),
            ], style={"display":"flex","gap":"16px"}),
            html.Div(dcc.Graph(figure=fig_scatter), style=CARD),
            persona_section,
        ])

    # ── BEHAVIOUR ────────────────────────────────────────────
    elif tab == "behaviour":
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(x=hour_biz["hour"], y=hour_biz["pct"], name="Business",
            line=dict(color=C_BIZ,width=2), fill="tozeroy", fillcolor="rgba(235,0,27,0.15)"))
        fig_hour.add_trace(go.Scatter(x=hour_con["hour"], y=hour_con["pct"], name="Consumer",
            line=dict(color=C_CON,width=2), fill="tozeroy", fillcolor="rgba(247,158,27,0.15)"))
        fig_hour.add_vrect(x0=9, x1=18, fillcolor="rgba(255,255,255,0.04)", line_width=0,
                           annotation_text="Business hours 9–18",
                           annotation_font_color=C_MUTED, annotation_position="top left")
        fig_hour.update_layout(**L(
            title="Transaction Activity by Hour of Day",
            xaxis_title="Hour", yaxis_title="% of Transactions",
            xaxis=dict(tickvals=list(range(0,24)), gridcolor=C_GRID, zerolinecolor=C_GRID),
        ))

        dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(x=dow_labels, y=dow_biz["pct"],
                                 name="Business", marker_color=C_BIZ, opacity=0.85))
        fig_dow.add_trace(go.Bar(x=dow_labels, y=dow_con["pct"],
                                 name="Consumer", marker_color=C_CON, opacity=0.85))
        fig_dow.update_layout(**L(title="Transaction Activity by Day of Week",
                                  xaxis_title="Day", yaxis_title="% of Transactions",
                                  barmode="group"))

        fig_mcc = make_subplots(rows=1, cols=2,
            subplot_titles=["Top 10 MCCs – Business","Top 10 MCCs – Consumer"])
        fig_mcc.add_trace(go.Bar(x=mcc_biz["count"], y=mcc_biz["name"],
                                 orientation="h", marker_color=C_BIZ, name="Business"), row=1, col=1)
        fig_mcc.add_trace(go.Bar(x=mcc_con["count"], y=mcc_con["name"],
                                 orientation="h", marker_color=C_CON, name="Consumer"), row=1, col=2)
        fig_mcc.update_layout(**L(showlegend=False, title="Top Merchant Categories by Segment",
                                  height=380))

        radar_feats  = ["online_ratio","recurring_ratio","business_mcc_ratio",
                        "weekday_ratio","large_txn_ratio","tokenized_ratio",
                        "foreign_merchant_ratio"]
        radar_labels = ["Online","Recurring","B2B MCC","Weekday","Large Txn","Tokenized","Foreign"]
        biz_vals = [biz_f[f].mean() for f in radar_feats]
        con_vals = [con_f[f].mean() for f in radar_feats]
        hid_vals = [hidden_f[f].mean() for f in radar_feats] if len(hidden_f) else con_vals
        FILL = {"#EB001B":"rgba(235,0,27,0.15)","#F79E1B":"rgba(247,158,27,0.15)",
                "#FF5F00":"rgba(255,95,0,0.15)"}
        fig_radar = go.Figure()
        for vals, name, color in [(biz_vals,"Business",C_BIZ),(con_vals,"Consumer",C_CON),
                                   (hid_vals,"Hidden",C_HIDDEN)]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=radar_labels+[radar_labels[0]],
                name=name, line_color=color, fill="toself", fillcolor=FILL[color]))
        fig_radar.update_layout(**L(
            title="Behavioural Profile Radar",
            polar=dict(bgcolor=C_CARD,
                       radialaxis=dict(visible=True,range=[0,1],gridcolor=C_GRID,color=C_MUTED),
                       angularaxis=dict(gridcolor=C_GRID,color=C_TEXT)),
        ))

        return html.Div([
            html.Div([
                html.Div(dcc.Graph(figure=fig_hour), style={**CARD,"flex":"1"}),
                html.Div(dcc.Graph(figure=fig_dow),  style={**CARD,"flex":"1"}),
            ], style={"display":"flex","gap":"16px"}),
            html.Div([
                html.Div(dcc.Graph(figure=fig_mcc),   style={**CARD,"flex":"2"}),
                html.Div(dcc.Graph(figure=fig_radar), style={**CARD,"flex":"1"}),
            ], style={"display":"flex","gap":"16px"}),
        ])

    # ── MODEL ────────────────────────────────────────────────
    elif tab == "model":
        fig_fi = go.Figure(go.Bar(x=feat_imp["importance"], y=feat_imp["feature"],
                                  orientation="h", marker_color=C_BIZ))
        fig_fi.update_layout(**L(title="Top 15 Feature Importances (Random Forest)",
                                 xaxis_title="Importance", height=450))

        prec, rec, thresh = precision_recall_curve(y_test, rf_proba)
        f1 = 2*prec*rec/(prec+rec+1e-8)
        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(x=thresh, y=prec[:-1], name="Precision",
                                        line_color=C_BIZ))
        fig_thresh.add_trace(go.Scatter(x=thresh, y=rec[:-1],  name="Recall",
                                        line_color=C_CON))
        fig_thresh.add_trace(go.Scatter(x=thresh, y=f1[:-1],   name="F1",
                                        line_color=C_HIDDEN, line_width=2))
        fig_thresh.add_vline(x=THRESHOLD, line_dash="dash", line_color="red",
                             annotation_text=f"Threshold={THRESHOLD}",
                             annotation_font_color="red")
        fig_thresh.update_layout(**L(title="Precision / Recall / F1 vs Threshold",
                                     xaxis_title="Threshold", yaxis_title="Score"))

        fpr, tpr, _ = roc_curve(y_test, rf_proba)
        auc = roc_auc_score(y_test, rf_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"RF (AUC={auc:.3f})",
                                     line=dict(color=C_BIZ,width=2), fill="tozeroy",
                                     fillcolor="rgba(235,0,27,0.1)"))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],
                                     line=dict(color=C_MUTED,dash="dash"), showlegend=False))
        fig_roc.update_layout(**L(title="ROC Curve — Synthetic Dataset (AUC=1.000 expected)",
                                  xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate"))

        top_feats = feat_imp["feature"].tail(10).tolist()
        corr = feat[top_feats+["label"]].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=np.round(corr.values,2), texttemplate="%{text}", textfont_size=9,
        ))
        fig_corr.update_layout(**L(title="Feature Correlation (top features + label)", height=420))

        return html.Div([
            SYNTHETIC_WARNING,
            html.Div([
                html.Div(dcc.Graph(figure=fig_fi),  style={**CARD,"flex":"1"}),
                html.Div(dcc.Graph(figure=fig_roc), style={**CARD,"flex":"1"}),
            ], style={"display":"flex","gap":"16px"}),
            html.Div([
                html.Div(dcc.Graph(figure=fig_thresh), style={**CARD,"flex":"1"}),
                html.Div(dcc.Graph(figure=fig_corr),   style={**CARD,"flex":"1"}),
            ], style={"display":"flex","gap":"16px"}),
        ])

    # ── CANDIDATES ───────────────────────────────────────────
    elif tab == "candidates":
        tier_row = html.Div([
            html.Div([
                html.Div(str(tier_counts["direct"]),
                         style={"fontSize":"2rem","fontWeight":"700","color":C_RED}),
                html.Div("Direct Outreach",
                         style={"fontSize":"0.7rem","color":C_MUTED,
                                "textTransform":"uppercase","marginTop":"4px"}),
                html.Div("Score ≥ 0.75",
                         style={"fontSize":"0.65rem","color":C_RED,"marginTop":"2px"}),
            ], style={**KPI,"borderLeft":f"3px solid {C_RED}"}),
            html.Div([
                html.Div(str(tier_counts["campaign"]),
                         style={"fontSize":"2rem","fontWeight":"700","color":C_ORANGE}),
                html.Div("Campaign Target",
                         style={"fontSize":"0.7rem","color":C_MUTED,
                                "textTransform":"uppercase","marginTop":"4px"}),
                html.Div("Score 0.50–0.75",
                         style={"fontSize":"0.65rem","color":C_ORANGE,"marginTop":"2px"}),
            ], style={**KPI,"borderLeft":f"3px solid {C_ORANGE}"}),
            html.Div([
                html.Div(str(tier_counts["monitor"]),
                         style={"fontSize":"2rem","fontWeight":"700","color":C_YELLOW}),
                html.Div("Monitor",
                         style={"fontSize":"0.7rem","color":C_MUTED,
                                "textTransform":"uppercase","marginTop":"4px"}),
                html.Div("Score 0.30–0.50",
                         style={"fontSize":"0.65rem","color":C_YELLOW,"marginTop":"2px"}),
            ], style={**KPI,"borderLeft":f"3px solid {C_YELLOW}"}),
            html.Div([
                html.Div("Recommended Actions",
                         style={"fontWeight":"700","fontSize":"0.85rem","marginBottom":"8px"}),
                html.Div("Reach out to Direct Outreach tier with a business card upgrade offer.",
                         style={"fontSize":"0.76rem","color":C_MUTED,"marginBottom":"4px"}),
                html.Div("Cross-sell: POS Acquiring · Working Capital · Payroll · Trade Finance",
                         style={"fontSize":"0.73rem","color":C_ORANGE,"fontWeight":"500"}),
                html.Div("Lower threshold to 0.25 for broader prospecting campaigns.",
                         style={"fontSize":"0.72rem","color":C_MUTED,"marginTop":"4px"}),
            ], style={**CARD,"flex":"2","marginBottom":"0","alignSelf":"stretch"}),
        ], style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr 2fr",
                  "gap":"12px","marginBottom":"16px"})

        table_src = con_f.sort_values("score", ascending=False).head(50).copy()
        table_src["tier"] = table_src["score"].apply(lambda s: tier_label(s)[0])
        table_df = table_src[[
            "card_number","score","tier","txn_count","total_spend_kzt",
            "online_ratio","recurring_ratio","business_mcc_ratio",
            "weekday_ratio","bank_name",
        ]].copy()
        table_df["score"]              = table_df["score"].round(3)
        table_df["total_spend_kzt"]    = (table_df["total_spend_kzt"]/1e6).round(2)
        table_df["online_ratio"]       = table_df["online_ratio"].round(2)
        table_df["recurring_ratio"]    = table_df["recurring_ratio"].round(2)
        table_df["business_mcc_ratio"] = table_df["business_mcc_ratio"].round(2)
        table_df["weekday_ratio"]      = table_df["weekday_ratio"].round(2)
        table_df.columns = ["Card Number","Score","Tier","Txn Count","Spend (M KZT)",
                             "Online","Recurring","B2B MCC","Weekday","Bank"]

        fig_top = px.bar(
            table_df.head(20), x="Score", y="Card Number",
            orientation="h", color="Score",
            color_continuous_scale=[[0,C_YELLOW],[0.5,C_ORANGE],[1,C_RED]],
            title="Top 20 Hidden Entrepreneur Candidates by Business Score",
        )
        fig_top.update_layout(**L(yaxis={"categoryorder":"total ascending"},
                                  coloraxis_showscale=False, height=500))

        top5 = con_f.sort_values("score", ascending=False).head(5)
        why_cards = []
        for _, row in top5.iterrows():
            signals = get_top_signals(row)
            t_label, t_color, t_desc = tier_label(float(row["score"]))
            items = []
            for sig in signals:
                direction = "▲" if sig["contribution"] > 0 else "▼"
                d_color   = C_RED if sig["contribution"] > 0 else C_MUTED
                items.append(html.Div([
                    html.Span(direction+" ", style={"color":d_color,"fontWeight":"700"}),
                    html.Span(sig["feature"]+": ",
                              style={"fontWeight":"600","fontSize":"0.77rem"}),
                    html.Span(f"{sig['card_val']:.0%} vs avg {sig['avg_val']:.0%}",
                              style={"color":C_MUTED,"fontSize":"0.73rem"}),
                ], style={"marginBottom":"4px","lineHeight":"1.3"}))

            why_cards.append(html.Div([
                html.Div([
                    html.Span(f"****{str(row['card_number'])[-4:]}",
                              style={"fontWeight":"700","fontSize":"0.88rem"}),
                    html.Span(f"  {row['score']:.3f}",
                              style={"color":t_color,"fontWeight":"600","fontSize":"0.85rem"}),
                ], style={"marginBottom":"6px"}),
                html.Div(t_label, style={
                    "display":"inline-block","fontSize":"0.65rem","fontWeight":"700",
                    "background":t_color,
                    "color":"#000" if t_color==C_YELLOW else C_TEXT,
                    "padding":"2px 8px","borderRadius":"10px","marginBottom":"8px",
                }),
                html.Div(items),
                html.Div(t_desc, style={"fontSize":"0.7rem","color":C_MUTED,"marginTop":"8px",
                                         "borderTop":f"1px solid {C_BORDER}","paddingTop":"6px",
                                         "lineHeight":"1.4"}),
            ], style={**CARD,"flex":"1","minWidth":"180px","marginBottom":"0",
                      "borderLeft":f"3px solid {t_color}"}))

        why_section = html.Div([
            html.Div("Why Flagged? — Top 5 Candidates",
                     style={"fontWeight":"600","fontSize":"1rem","marginBottom":"6px"}),
            html.Div("▲ above average consumer (business-like)  "
                     "▼ below average (also business-like for signals like evening activity)",
                     style={"fontSize":"0.73rem","color":C_MUTED,"marginBottom":"12px"}),
            html.Div(why_cards, style={"display":"flex","gap":"12px","flexWrap":"wrap"}),
        ], style=CARD)

        return html.Div([
            tier_row,
            html.Div(dcc.Graph(figure=fig_top), style=CARD),
            why_section,
            html.Div([
                html.Div("Top 50 Consumer Cards Ranked by Business Score",
                         style={"fontWeight":"600","marginBottom":"12px","fontSize":"1rem"}),
                dash_table.DataTable(
                    data=table_df.to_dict("records"),
                    columns=[{"name":c,"id":c} for c in table_df.columns],
                    style_table={"overflowX":"auto"},
                    style_cell={"background":C_CARD,"color":C_TEXT,
                                "border":f"1px solid {C_GRID}",
                                "fontFamily":"Inter, sans-serif",
                                "fontSize":"13px","padding":"8px 12px"},
                    style_header={"background":C_GRID,"fontWeight":"600","color":C_TEXT},
                    style_data_conditional=[
                        {"if":{"filter_query":'{Tier} = "Direct Outreach"'},
                         "color":C_RED,"fontWeight":"700"},
                        {"if":{"filter_query":'{Tier} = "Campaign Target"'},
                         "color":C_ORANGE,"fontWeight":"600"},
                        {"if":{"filter_query":'{Tier} = "Monitor"'},"color":C_YELLOW},
                    ],
                    page_size=15, sort_action="native", filter_action="native",
                ),
            ], style=CARD),
        ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"\nDashboard ready at http://127.0.0.1:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
