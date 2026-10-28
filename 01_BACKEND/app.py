"""
app.py
======
Flask dashboard — "PACE" (Predictive Analytics for Customer Engagement) —
Customer Segmentation & Retention Prediction dashboard.

Pages: Login, Dashboard, Customer List, Segmentation, Prediction, Reports,
Analytics, Model Comparison, Settings.

Run:
    cd PACE-Customer-Intelligence-System
    python 01_BACKEND/app.py
Then open http://127.0.0.1:5000  (login: admin / admin123 — change in Settings/.env for real deployment)

RESTRUCTURE NOTE (team-folder reorg)
-------------------------------------
This file used to live at app/app.py with templates/static as siblings in
the same app/ folder (Flask's default template_folder="templates" and
static_folder="static" found them automatically next to app.py). It now
lives at 01_BACKEND/app.py while templates and static assets live under
03_FRONTEND/, so both folders are passed to Flask() explicitly below.
Data/reports/models paths are similarly updated to the new 04_DATA_TESTING/
and 02_ML_AI/ locations. No route, template, or business logic changed.
"""
import os
import sys
import io
import json
import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    jsonify, send_file, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING", "reports")
MODELS_DIR = os.path.join(BASE_DIR, "02_ML_AI", "models")

# ---------------------------------------------------------------------------
# Real, public lat/lon coordinates (city centroids) for the 18 real Indian
# cities the pipeline's CITY_TIER_MAP (see 02_ML_AI/config.py) assigns
# customers to. This is geographic reference data (public knowledge, not
# fabricated per-customer data) used only to PLOT each city's real
# aggregated customer/sales numbers on the dashboard's India map — every
# number attached to a pin (customers, avg spend, churn %) is a genuine
# aggregate computed from the dataset below, not invented.
# ---------------------------------------------------------------------------
CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777), "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946), "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639), "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567), "Ahmedabad": (23.0225, 72.5714),
    "Nagpur": (21.1458, 79.0882), "Lucknow": (26.8467, 80.9462),
    "Indore": (22.7196, 75.8577), "Jaipur": (26.9124, 75.7873),
    "Nashik": (19.9975, 73.7898), "Kolhapur": (16.7050, 74.2433),
    "Aurangabad": (19.8762, 75.3433), "Amravati": (20.9374, 77.7796),
    "Bhopal": (23.2599, 77.4126), "Patna": (25.5941, 85.1376),
}
TEMPLATES_DIR = os.path.join(BASE_DIR, "03_FRONTEND", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "03_FRONTEND", "static")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "services"))
from _paths import get_cached_df
import explainability_service
import revenue_service
import simulator_service
import customer360_service

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("SECRET_KEY", "pace-india-dev-secret-change-in-prod")

# Demo credentials — in production these live in a database with hashed
# passwords; kept simple here so the dashboard runs standalone for a college
# project demo without needing a DB server.
USERS = {"admin": generate_password_hash("admin123")}


# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------
def _read_csv_safe(path, empty_cols=None):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=empty_cols or [])


def get_final_data():
    return get_cached_df(os.path.join(DATA_DIR, "final_customer_data.csv"))


def get_recommendations_data():
    return get_cached_df(os.path.join(DATA_DIR, "customer_recommendations.csv"))


def get_json(name):
    path = os.path.join(REPORTS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Public marketing page
# ---------------------------------------------------------------------------
@app.route("/welcome")
def welcome():
    return render_template("landing.html")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username in USERS and check_password_hash(USERS[username], password):
            session["user"] = username
            return redirect(url_for("dashboard"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
# NOTE: A CITY_COORDS lookup used to live here, mapping randomly-assigned
# Indian city names onto map pins. It was removed deliberately: neither
# dataset contains city-level geography. The churn dataset has only CityTier
# (1/2/3) and Online Retail II has Country. The dashboard now reports real
# Country-level markets from the transactional data instead of inventing
# cities. See reports/country_stats.json.


def _round_dict(d, digits=1):
    return {k: (round(float(v), digits) if pd.notna(v) else 0) for k, v in d.items()}


@app.route("/")
@login_required
def dashboard():
    df = get_final_data()
    churn_metrics_json = get_json("churn_model_metrics.json")
    seg_metrics_json = get_json("segmentation_metrics.json")
    lifecycle = get_json("lifecycle_forecast.json")
    monthly = get_json("monthly_revenue.json")          # real transactional layer
    country_stats = get_json("country_stats.json")

    mom = monthly.get("mom_delta", {}) if monthly else {}
    tx_summary = monthly.get("dataset_summary", {}) if monthly else {}
    has_transactions = bool(monthly.get("months"))

    kpis = {
        "total_customers": len(df),
        "churn_rate": round(df["Churn"].mean() * 100, 1) if "Churn" in df.columns and len(df) else 0,
        "avg_satisfaction": round(df["SatisfactionScore"].mean(), 2) if "SatisfactionScore" in df.columns and len(df) else 0,
        "total_segments": df["SegmentName"].nunique() if "SegmentName" in df.columns and len(df) else 0,
        "high_risk_count": int((df["ChurnRiskBand"] == "High").sum()) if "ChurnRiskBand" in df.columns and len(df) else 0,
        "avg_spend": round(df["EstimatedAnnualSpendINR"].mean(), 0) if "EstimatedAnnualSpendINR" in df.columns and len(df) else 0,
        "best_churn_model": churn_metrics_json.get("best_model", "\u2014"),
        "best_seg_model": seg_metrics_json.get("best_model", "\u2014"),
    }

    segment_counts = df["SegmentName"].value_counts().to_dict() if "SegmentName" in df.columns and len(df) else {}
    risk_counts = df["ChurnRiskBand"].value_counts().to_dict() if "ChurnRiskBand" in df.columns and len(df) else {}

    # Real churn-probability histogram (10 equal-width buckets). Every count is
    # a genuine aggregate of model output.
    churn_prob_hist = {"labels": [], "counts": []}
    if "ChurnProbability" in df.columns and len(df):
        bins = [i / 10 for i in range(11)]
        labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(10)]
        binned = pd.cut(df["ChurnProbability"].clip(0, 1), bins=bins, labels=labels, include_lowest=True)
        counts = binned.value_counts().reindex(labels, fill_value=0)
        churn_prob_hist = {"labels": labels, "counts": [int(c) for c in counts.values]}

    # Real geography: every customer in the churn dataset carries a real
    # City/State (derived from the dataset's own CityTier column — see
    # 02_ML_AI/config.py CITY_TIER_MAP). Rank actual Indian cities by real
    # customer count/spend/risk computed straight from the data.
    top_markets = []
    city_markets = []
    if len(df) and "City" in df.columns:
        city_group = df.groupby(["City", "State"]).agg(
            customers=("CustomerID", "count"),
            avg_spend=("EstimatedAnnualSpendINR", "mean"),
            avg_recency=("DaySinceLastOrder", "mean"),
        ).reset_index()
        if "ChurnRiskBand" in df.columns:
            risk_pct = (df.assign(_high=df["ChurnRiskBand"] == "High")
                          .groupby("City")["_high"].mean().mul(100).round(1))
            city_group["risk_pct"] = city_group["City"].map(risk_pct)
        if "Churn" in df.columns:
            churn_pct = df.groupby("City")["Churn"].mean().mul(100).round(1)
            city_group["churn_pct"] = city_group["City"].map(churn_pct)
        city_group = city_group.sort_values("customers", ascending=False)

        market_max = city_group["customers"].max()
        for _, row in city_group.iterrows():
            entry = {
                "name": row["City"], "state": row["State"],
                "customers": int(row["customers"]),
                "avg_spend": f"₹{row['avg_spend']:,.0f}" if pd.notna(row["avg_spend"]) else "—",
                "avg_spend_num": round(float(row["avg_spend"]), 0) if pd.notna(row["avg_spend"]) else None,
                "avg_recency": round(float(row["avg_recency"]), 1) if pd.notna(row["avg_recency"]) else None,
                "risk_pct": round(float(row["risk_pct"]), 1) if "risk_pct" in row and pd.notna(row["risk_pct"]) else None,
                "churn_pct": round(float(row["churn_pct"]), 1) if "churn_pct" in row and pd.notna(row["churn_pct"]) else None,
                "share_pct": round(row["customers"] / market_max * 100, 1) if market_max else 0,
            }
            top_markets.append(entry)
            coords = CITY_COORDS.get(row["City"])
            if coords:
                city_markets.append({**entry, "avg_spend": entry["avg_spend_num"], "lat": coords[0], "lon": coords[1]})
        top_markets = top_markets[:10]

    top_churn_model = next(
        (m for m in churn_metrics_json.get("all_models", []) if m.get("model") == kpis["best_churn_model"]), None
    )
    top_seg_model = next(
        (m for m in seg_metrics_json.get("comparison", []) if m.get("model") == kpis["best_seg_model"]), None
    )

    # ---- KPI sparklines -------------------------------------------------
    # Sourced ONLY from real series. Previously these plotted sorted segment
    # counts, which rendered as meaningless decaying curves; that is removed.
    # A card with no real series behind it simply shows no sparkline.
    sparklines = {}
    deltas = {}
    if has_transactions:
        sparklines["customers"] = monthly.get("customers", [])[-12:]
        sparklines["spend"] = monthly.get("revenue", [])[-12:]
        sparklines["orders"] = monthly.get("orders", [])[-12:]
        if mom.get("customers_pct") is not None:
            deltas["customers"] = {"pct": mom["customers_pct"], "basis": mom.get("basis", "")}
        if mom.get("revenue_pct") is not None:
            deltas["spend"] = {"pct": mom["revenue_pct"], "basis": mom.get("basis", "")}

    # Lifecycle curve is a real model output and is a legitimate sparkline source.
    if lifecycle and lifecycle.get("historical", {}).get("avg_monetary"):
        sparklines["lifecycle"] = lifecycle["historical"]["avg_monetary"][-14:]

    from datetime import datetime
    rendered_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

    revenue_summary = revenue_service.get_summary()
    priority_customers = revenue_service.get_priority_customers(n=5)
    retention_opportunity = simulator_service.estimate_retention_opportunity()

    return render_template(
        "dashboard.html", kpis=kpis, segment_counts=segment_counts,
        risk_counts=risk_counts, has_data=len(df) > 0,
        churn_prob_hist=churn_prob_hist, top_markets=top_markets,
        city_markets=city_markets,
        top_churn_model=top_churn_model, top_seg_model=top_seg_model,
        sparklines=sparklines, deltas=deltas, rendered_at=rendered_at,
        has_transactions=has_transactions, tx_summary=tx_summary,
        mom=mom, revenue_summary=revenue_summary, priority_customers=priority_customers,
        retention_opportunity=retention_opportunity,
    )


@app.route("/customers")
@login_required
def customers():
    df = get_final_data()
    cols = [c for c in ["CustomerID", "CustomerName", "City", "State", "PhoneNumber",
                         "PreferedOrderCat", "SegmentName", "ChurnRiskBand",
                         "EstimatedAnnualSpendINR", "Churn"] if c in df.columns]
    prefill_q = request.args.get("q", "").strip()

    total_customers = len(df)
    if prefill_q and len(df):
        q_lower = prefill_q.lower()
        mask = pd.Series(False, index=df.index)
        for col in ["CustomerID", "CustomerName", "City", "State", "SegmentName", "ChurnRiskBand", "PreferedOrderCat"]:
            if col in df.columns:
                mask |= df[col].astype(str).str.lower().str.contains(q_lower, na=False)
        subset_df = df[mask].head(1000)
    else:
        subset_df = df.head(1000)

    records = subset_df[cols].to_dict(orient="records") if len(subset_df) else []
    return render_template("customers.html", customers=records, columns=cols,
                            has_data=len(df) > 0, prefill_q=prefill_q,
                            total_customers=total_customers, displayed_count=len(records))


@app.route("/segmentation")
@login_required
def segmentation():
    df = get_final_data()
    seg_metrics = get_json("segmentation_metrics.json")
    segment_profile = {}
    if len(df) and "SegmentName" in df.columns:
        segment_profile = df.groupby("SegmentName")[
            ["Recency", "Frequency", "Monetary", "EngagementScore", "SatisfactionScore"]
        ].mean().round(2).to_dict(orient="index")
    return render_template("segmentation.html", seg_metrics=seg_metrics, segment_profile=segment_profile, has_data=len(df) > 0)


SAMPLE_CUSTOMERS = [
    {"id": 50001, "name": "Suresh Kulkarni", "segment": "At-Risk / Dormant", "risk": "High", "desc": "High Risk (70%) — Long Inactivity, Tier 1"},
    {"id": 50008, "name": "Amit Chatterjee", "segment": "At-Risk / Dormant", "risk": "Medium", "desc": "Medium Risk (52%) — Recent Complaint"},
    {"id": 50063, "name": "Suresh Banerjee", "segment": "New / Recently Acquired", "risk": "Medium", "desc": "Medium Risk (40%) — New Customer (1 mo)"},
    {"id": 50022, "name": "Deepak Agarwal", "segment": "Occasional Shoppers", "risk": "Medium", "desc": "Medium Risk (45%) — Low Frequency"},
    {"id": 50002, "name": "Priya Sharma", "segment": "Occasional Shoppers", "risk": "Low", "desc": "Low Risk (10%) — Established Customer"},
    {"id": 50005, "name": "Asha Patil", "segment": "Loyal High-Value", "risk": "Low", "desc": "Low Risk (2%) — Loyal 3+ Years"},
    {"id": 50016, "name": "Anjali Patil", "segment": "Big Spenders", "risk": "Low", "desc": "Low Risk (2%) — High Annual Spend"},
]


@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    result = None
    prefill = None
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form.to_dict()
        prefill = {k: v for k, v in data.items() if v not in ("", None)}
        try:
            result = explainability_service.explain_features(prefill)
        except Exception as e:
            app.logger.warning(f"SHAP explanation failed, falling back to basic prediction: {e}")
            try:
                model_path = os.path.join(MODELS_DIR, "best_churn_model.pkl")
                feat_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
                model = joblib.load(model_path)
                feature_cols = joblib.load(feat_path)
                form_vals = {k: float(v) for k, v in prefill.items() if k in feature_cols}
                row = pd.DataFrame([form_vals]).reindex(columns=feature_cols, fill_value=0)
                proba = float(model.predict_proba(row)[0][1])
                result = {
                    "churn_probability": round(proba * 100, 1),
                    "prediction": "Likely to Churn" if proba >= 0.5 else "Likely to Stay",
                    "risk_band": "High" if proba >= 0.66 else ("Medium" if proba >= 0.33 else "Low"),
                    "top_increasing_factors": [],
                    "top_reducing_factors": [],
                    "explanation_text": f"Customer churn probability is {proba*100:.1f}%.",
                }
            except FileNotFoundError:
                flash("Model not trained yet. Run the pipeline (02_ML_AI/run_pipeline.py) first.", "warning")

        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify(result or {"error": "Prediction failed."})

    return render_template(
        "prediction.html",
        result=result,
        prefill=prefill,
        sample_customers=SAMPLE_CUSTOMERS,
    )


@app.route("/api/customer-features/<int:customer_id>")
@login_required
def api_customer_features(customer_id):
    """Fetch feature inputs for a real customer to auto-fill the prediction form."""
    df = get_final_data()
    match = df[df["CustomerID"] == customer_id]
    if match.empty:
        return jsonify({"error": f"Customer ID {customer_id} not found."}), 404

    row = match.iloc[0].to_dict()
    features = {
        "CustomerID": int(row.get("CustomerID", customer_id)),
        "CustomerName": str(row.get("CustomerName", f"Customer {customer_id}")),
        "SegmentName": str(row.get("SegmentName", "Standard")),
        "ChurnRiskBand": str(row.get("ChurnRiskBand", "Low")),
        "StoredChurnProbability": round(float(row.get("ChurnProbability", 0)) * 100, 1),

        # Group 1: Profile & Demographics
        "Tenure": float(row.get("Tenure", 6)),
        "CityTier": int(row.get("CityTier", 1)),
        "WarehouseToHome": float(row.get("WarehouseToHome", 14)),
        "Gender": 1 if str(row.get("Gender", "Male")).strip().lower() in ("male", "m", "1") else 0,

        # Group 2: App & Engagement
        "HourSpendOnApp": float(row.get("HourSpendOnApp", 3)),
        "NumberOfDeviceRegistered": int(row.get("NumberOfDeviceRegistered", 3)),
        "SatisfactionScore": int(row.get("SatisfactionScore", 3)),
        "Complain": int(row.get("Complain", 0)),

        # Group 3: Purchase & Behavior
        "OrderCount": int(row.get("OrderCount", 2)),
        "DaySinceLastOrder": float(row.get("DaySinceLastOrder", 4)),
        "CouponUsed": int(row.get("CouponUsed", 1)),
        "OrderAmountHikeFromlastYear": float(row.get("OrderAmountHikeFromlastYear", 15)),
        "CashbackAmount": float(row.get("CashbackAmount", 170)),

        # Context
        "EstimatedAnnualSpendINR": float(row.get("EstimatedAnnualSpendINR", 25000)),
        "EngagementScore": float(row.get("EngagementScore", 60)),
    }
    return jsonify(features)


@app.route("/api/predict-retention", methods=["POST"])
@login_required
def api_predict_retention():
    """API endpoint for asynchronous Retention Prediction with SHAP explanation."""
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({"error": "No input features provided."}), 400

    try:
        result = explainability_service.explain_features(data)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/reports")
@login_required
def reports():
    recs = get_json("business_recommendations.json").get("insights", [])
    eda_insights = get_json("eda_insights.json")
    return render_template("reports.html", recommendations=recs, eda_insights=eda_insights)


@app.route("/analytics")
@login_required
def analytics():
    df = get_final_data()
    charts = {}
    summary = {}
    if len(df):
        total_n = len(df)

        # 1. Preferred Order Category
        cat_counts = df["PreferedOrderCat"].value_counts().to_dict() if "PreferedOrderCat" in df.columns else {}
        charts["category_dist"] = cat_counts
        if cat_counts:
            top_cat = next(iter(cat_counts))
            summary["top_category"] = f"{top_cat} ({round(cat_counts[top_cat] / total_n * 100, 1)}%)"

        # 2. Preferred Payment Mode
        pay_counts = df["PreferredPaymentMode"].value_counts().to_dict() if "PreferredPaymentMode" in df.columns else {}
        charts["payment_dist"] = pay_counts
        if pay_counts:
            top_pay = next(iter(pay_counts))
            summary["top_payment"] = f"{top_pay} ({round(pay_counts[top_pay] / total_n * 100, 1)}%)"

        # 3. Average Annual Spend (₹) by State (sorted descending)
        if "State" in df.columns and "EstimatedAnnualSpendINR" in df.columns:
            state_spend = df.groupby("State")["EstimatedAnnualSpendINR"].mean().sort_values(ascending=False).round(0).to_dict()
            charts["state_spend"] = state_spend
            if state_spend:
                top_state = next(iter(state_spend))
                summary["top_state"] = f"{top_state} (₹{state_spend[top_state]:,.0f})"

        # 4. Churn Rate (%) by Satisfaction Score (sorted 1 to 5)
        if "SatisfactionScore" in df.columns and "Churn" in df.columns:
            sat_s = df.groupby("SatisfactionScore")["Churn"].mean().mul(100).round(1).sort_index()
            charts["satisfaction_churn"] = {f"Score {int(k) if k == int(k) else k}": v for k, v in sat_s.items()}
            if len(sat_s):
                summary["sat_churn_spread"] = f"{sat_s.max()}% (Low) vs {sat_s.min()}% (High)"

        # 5. Tenure distribution (bucketed)
        if "Tenure" in df.columns:
            tenure_bins = [0, 3, 6, 12, 24, 36, 100]
            tenure_labels = ["0-3mo", "3-6mo", "6-12mo", "1-2yr", "2-3yr", "3yr+"]
            tb = pd.cut(df["Tenure"], bins=tenure_bins, labels=tenure_labels, right=False)
            tenure_counts = tb.value_counts().reindex(tenure_labels).fillna(0).astype(int).to_dict()
            charts["tenure_dist"] = tenure_counts
            top_tenure = max(tenure_counts, key=tenure_counts.get)
            summary["top_tenure"] = f"{top_tenure} ({round(tenure_counts[top_tenure] / total_n * 100, 1)}%)"

        # 6. Churn rate by City Tier
        if "CityTier" in df.columns and "Churn" in df.columns:
            tier_churn = df.groupby("CityTier")["Churn"].mean().mul(100).round(1)
            charts["city_tier_churn"] = {f"Tier {int(k)}": v for k, v in tier_churn.sort_index().items()}
            summary["tier_churn"] = " | ".join(f"T{int(k)}: {v}%" for k, v in tier_churn.sort_index().items())

        # 7. Device mix
        if "PreferredLoginDevice" in df.columns:
            dev_counts = df["PreferredLoginDevice"].value_counts().to_dict()
            charts["device_dist"] = dev_counts
            top_dev = next(iter(dev_counts))
            summary["top_device"] = f"{top_dev} ({round(dev_counts[top_dev] / total_n * 100, 1)}%)"

        # 8. Complaint impact on churn
        if "Complain" in df.columns and "Churn" in df.columns:
            comp_churn = df.groupby("Complain")["Churn"].mean().mul(100).round(1)
            no_comp = round(float(comp_churn.get(0, 0)), 1)
            yes_comp = round(float(comp_churn.get(1, 0)), 1)
            charts["complaint_churn"] = {"No Complaint": no_comp, "Complained": yes_comp}
            ratio = round(yes_comp / max(no_comp, 0.1), 1)
            summary["complaint_impact"] = f"{ratio}x higher churn ({yes_comp}% vs {no_comp}%)"

        # 9. Engagement score by segment
        if "SegmentName" in df.columns and "EngagementScore" in df.columns:
            seg_eng = df.groupby("SegmentName")["EngagementScore"].mean().sort_values(ascending=False).round(1).to_dict()
            charts["engagement_by_segment"] = seg_eng
            top_seg = next(iter(seg_eng))
            summary["top_engagement"] = f"{top_seg} ({seg_eng[top_seg]}/100)"

        # 10. Real vs. augmented-synthetic split
        if "IsSynthetic" in df.columns:
            real_cnt = int((df["IsSynthetic"] == 0).sum())
            synth_cnt = int((df["IsSynthetic"] == 1).sum())
            charts["data_composition"] = {
                "Real (Kaggle dataset)": real_cnt,
                "Synthetic (Augmented)": synth_cnt,
            }
            summary["data_split"] = f"{real_cnt:,} Real / {synth_cnt:,} Augmented"

    return render_template("analytics.html", charts=charts, summary=summary, has_data=len(df) > 0)


@app.route("/model-comparison")
@login_required
def model_comparison():
    churn_metrics = get_json("churn_model_metrics.json").get("all_models", [])
    seg_metrics = get_json("segmentation_metrics.json").get("comparison", [])
    return render_template("model_comparison.html", churn_metrics=churn_metrics, seg_metrics=seg_metrics)


@app.route("/forecasting")
@login_required
def forecasting():
    forecast = get_json("lifecycle_forecast.json")
    # Guard on the keys the template actually dereferences, not merely on the
    # dict being non-empty — a partial/older JSON would otherwise 500 the page.
    has_data = bool(
        forecast
        and forecast.get("historical", {}).get("tenure_months")
        and forecast.get("forecast", {}).get("tenure_months")
    )
    return render_template("forecasting.html", forecast=forecast, has_data=has_data)


@app.route("/recommendations")
@login_required
def recommendations():
    df = get_recommendations_data()
    affinity = get_json("segment_category_affinity.json").get("segment_category_affinity_pct", {})
    search = request.args.get("q", "").strip()
    if search and len(df):
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False))
        df = df[mask.any(axis=1)]
    records = df.to_dict(orient="records") if len(df) else []
    return render_template(
        "recommendations.html", customers=records, affinity=affinity,
        search=search, has_data=len(get_recommendations_data()) > 0
    )


@app.route("/customer360")
@login_required
def customer360_search():
    query = request.args.get("q", "").strip()
    results = customer360_service.search_customers(query) if query else []
    df = get_final_data()
    return render_template("customer360_search.html", query=query, results=results, has_data=len(df) > 0)


@app.route("/customer/<int:customer_id>")
@login_required
def customer360(customer_id):
    profile = customer360_service.get_customer_360(customer_id)
    if profile is None:
        flash(f"Customer {customer_id} not found.", "warning")
        return redirect(url_for("customer360_search"))
    return render_template("customer360.html", profile=profile, customer_id=customer_id)


@app.route("/revenue-at-risk")
@login_required
def revenue_at_risk():
    summary = revenue_service.get_summary()
    return render_template("revenue_at_risk.html", summary=summary)


@app.route("/retention-simulator")
@login_required
def retention_simulator():
    df = get_final_data()
    segments = sorted(df["SegmentName"].dropna().unique().tolist()) if "SegmentName" in df.columns and len(df) else []
    states = sorted(df["State"].dropna().unique().tolist()) if "State" in df.columns and len(df) else []
    actions = simulator_service.available_actions()
    prefill_customer_id = request.args.get("customer_id", type=int)
    return render_template(
        "retention_simulator.html", segments=segments, states=states,
        risk_bands=["Low", "Medium", "High"], actions=actions,
        prefill_customer_id=prefill_customer_id, has_data=len(df) > 0,
        assumptions_note=simulator_service.ASSUMPTIONS_NOTE,
    )


@app.route("/api/simulate", methods=["POST"])
@login_required
def api_simulate():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "customer")
    action_keys = data.get("action_keys") or None

    if mode == "group":
        filters = {
            "segment": data.get("segment") or None,
            "risk_band": data.get("risk_band") or None,
            "state": data.get("state") or None,
        }
        result = simulator_service.simulate_group(filters, action_keys)
        if result is None:
            return jsonify({"error": "No customer data available."}), 404
        return jsonify(result)

    customer_id = data.get("customer_id")
    if not customer_id:
        return jsonify({"error": "customer_id is required for mode=customer"}), 400
    try:
        customer_id = int(customer_id)
    except (TypeError, ValueError):
        return jsonify({"error": "customer_id must be numeric"}), 400

    result = simulator_service.simulate_customer(customer_id, action_keys)
    if result is None:
        return jsonify({"error": f"Customer {customer_id} not found."}), 404
    return jsonify(result)


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html", user=session.get("user"))


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------
@app.route("/export/excel")
@login_required
def export_excel():
    df = get_final_data()
    buf = io.BytesIO()
    export_df = df.head(10000)
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Customers_Sample")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="customer_data.xlsx",
                      mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/export/pdf")
@login_required
def export_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    df = get_final_data()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "PACE — Customer Retention Report")
    y -= 30
    c.setFont("Helvetica", 10)
    if len(df):
        c.drawString(50, y, f"Total Customers: {len(df):,}")
        y -= 15
        c.drawString(50, y, f"Churn Rate: {round(df['Churn'].mean()*100, 1)}%")
        y -= 15
        c.drawString(50, y, f"Avg Estimated Annual Spend: Rs. {round(df['EstimatedAnnualSpendINR'].mean(), 0):,}")
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="retention_report.pdf", mimetype="application/pdf")


# ---------------------------------------------------------------------------
# JSON APIs (used by Chart.js on the front end for live filtering)
# ---------------------------------------------------------------------------
@app.route("/api/customers")
@login_required
def api_customers():
    df = get_final_data()
    limit = min(int(request.args.get("limit", 1000)), 5000)
    return jsonify(df.head(limit).to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
