"""
===============================================================================
RENTAL BUSINESS PROFIT OPTIMIZATION PLATFORM
Script 04: Machine Learning Models
===============================================================================
Trains 5 XGBoost models:

MODEL 1: Predictive Maintenance
  - Predicts asset failure probability in next 30 days
  - Business Output: Failure risk tier, estimated downtime cost

MODEL 2: Revenue Forecasting
  - Predicts monthly rental revenue per asset
  - Business Output: 12-month revenue forecast

MODEL 3: Customer Churn Prediction
  - Predicts churn probability for each customer
  - Business Output: Revenue at risk, retention priority

MODEL 4: Asset Utilization Prediction
  - Predicts next-month utilization rate
  - Business Output: Idle risk flag, revenue opportunity

MODEL 5: Profit Prediction
  - Predicts rental contract profitability
  - Business Output: Expected profit tier
===============================================================================
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load(name):
    path = os.path.join(PROC_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {name}  [{len(df):,} rows]")
    return df


def encode_categoricals(df):
    """Label-encode all object columns."""
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def save_model_report(name, metrics):
    path = os.path.join(REPORT_DIR, f"model_{name}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  ✓ Model report saved: model_{name}.json")


def print_model_box(name, metrics):
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  MODEL: {name:<40} │")
    print(f"  ├─────────────────────────────────────────────────┤")
    for k, v in metrics.items():
        if k not in ("name", "feature_importance"):
            print(f"  │  {k:<28}: {str(v):<16}  │")
    print(f"  └─────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: PREDICTIVE MAINTENANCE
# ═══════════════════════════════════════════════════════════════════════════════
def train_predictive_maintenance():
    print("\n[MODEL 1] Predictive Maintenance...")
    df = load("ml_maintenance_features")

    feature_cols = [
        "Asset_Age_Years", "Purchase_Cost_INR", "Daily_Rental_Rate_INR",
        "Book_Value_Pct", "Overdue_Service_Flag",
        "Total_Breakdowns", "Avg_Maintenance_Cost", "Total_Downtime_Hours",
        "Avg_Failure_Prob", "Maintenance_Count",
        "Asset_Category", "Country"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target = "Failure_Target"

    df_model = df[feature_cols + [target]].dropna()
    df_model = encode_categoricals(df_model)

    X = df_model[feature_cols]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, eval_metric="logloss", verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = round(roc_auc_score(y_test, y_prob), 4)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Apply to full dataset for scoring
    df_score = df[feature_cols].dropna().copy()
    df_score_enc = encode_categoricals(df_score.copy())
    df["Failure_Probability_Predicted"] = model.predict_proba(df_score_enc)[:, 1]
    df["Failure_Risk_Tier"] = pd.cut(
        df["Failure_Probability_Predicted"],
        bins=[0, 0.30, 0.60, 0.80, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk", "Critical"]
    ).astype(str)
    df["Estimated_Downtime_Cost_INR"] = (
        df["Failure_Probability_Predicted"] * df["Avg_Maintenance_Cost"] * 2.5
    ).fillna(0).astype(int)
    df["Preventive_Savings_Potential_INR"] = (df["Estimated_Downtime_Cost_INR"] * 0.60).astype(int)

    # Save scored dataset
    output_path = os.path.join(PROC_DIR, "predictive_maintenance_scores.csv")
    df.to_csv(output_path, index=False)

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, "model_predictive_maintenance.pkl"))

    feature_imp = dict(zip(feature_cols, model.feature_importances_.round(4)))
    metrics = {
        "model": "Predictive Maintenance",
        "algorithm": "XGBoost Classifier",
        "auc_roc": auc,
        "precision_class1": round(report["1"]["precision"], 4),
        "recall_class1": round(report["1"]["recall"], 4),
        "f1_class1": round(report["1"]["f1-score"], 4),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "business_kpi": "Preventive Maintenance Savings",
        "estimated_annual_savings_INR": int(df["Preventive_Savings_Potential_INR"].sum()),
        "feature_importance": feature_imp,
    }
    print_model_box("Predictive Maintenance", metrics)
    save_model_report("predictive_maintenance", metrics)
    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: REVENUE FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
def train_revenue_forecast():
    print("\n[MODEL 2] Revenue Forecasting...")
    df = load("ml_revenue_features")

    feature_cols = [
        "Month_Num", "Year", "Asset_Age_Years", "Daily_Rental_Rate_INR",
        "Monthly_Rentals", "Avg_Rental_Days", "Avg_Margin",
        "Asset_Category", "Country"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target = "Monthly_Revenue"

    df_model = df[feature_cols + [target]].dropna()
    df_model = encode_categoricals(df_model)

    X = df_model[feature_cols]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 4)

    # Generate 12-month forecast
    future_months = []
    for month in range(1, 13):
        row = {
            "Month_Num": month,
            "Year": 2025,
            "Asset_Age_Years": df["Asset_Age_Years"].mean() if "Asset_Age_Years" in df.columns else 5,
            "Daily_Rental_Rate_INR": df["Daily_Rental_Rate_INR"].mean() if "Daily_Rental_Rate_INR" in df.columns else 15000,
            "Monthly_Rentals": df["Monthly_Rentals"].mean(),
            "Avg_Rental_Days": df["Avg_Rental_Days"].mean(),
            "Avg_Margin": df["Avg_Margin"].mean(),
            "Asset_Category": 0,
            "Country": 0,
        }
        future_months.append(row)

    future_df = pd.DataFrame(future_months)[feature_cols]
    forecast = model.predict(future_df)

    forecast_df = pd.DataFrame({
        "Month": pd.date_range("2025-01", periods=12, freq="MS").strftime("%Y-%m").tolist(),
        "Forecasted_Revenue_INR": forecast.astype(int),
        "Lower_Bound_INR": (forecast * 0.85).astype(int),
        "Upper_Bound_INR": (forecast * 1.15).astype(int),
    })
    forecast_path = os.path.join(PROC_DIR, "revenue_forecast_2025.csv")
    forecast_df.to_csv(forecast_path, index=False)

    joblib.dump(model, os.path.join(MODEL_DIR, "model_revenue_forecast.pkl"))

    metrics = {
        "model": "Revenue Forecasting",
        "algorithm": "XGBoost Regressor",
        "mean_absolute_error_INR": mae,
        "r2_score": r2,
        "training_samples": len(X_train),
        "forecast_year": 2025,
        "forecasted_annual_revenue_INR": int(forecast_df["Forecasted_Revenue_INR"].sum()),
        "business_kpi": "Expected Revenue Growth",
    }
    print_model_box("Revenue Forecasting", metrics)
    save_model_report("revenue_forecast", metrics)
    return model, metrics, forecast_df


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: CUSTOMER CHURN
# ═══════════════════════════════════════════════════════════════════════════════
def train_churn_prediction():
    print("\n[MODEL 3] Customer Churn Prediction...")
    df = load("ml_churn_features")

    feature_cols = [
        "Days_Since_Last_Rental", "Lifetime_Value_INR", "Total_Rentals_x",
        "RFM_Score", "Avg_Rental_Days", "Avg_Margin",
        "Customer_Category", "Customer_Segment", "Country"
    ]
    # handle column name variations
    if "Total_Rentals_x" not in df.columns and "Total_Rentals" in df.columns:
        df["Total_Rentals_x"] = df["Total_Rentals"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target = "Churn_Target"

    df_model = df[feature_cols + [target]].dropna()
    df_model = encode_categoricals(df_model)

    X = df_model[feature_cols]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42, eval_metric="logloss", verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = round(roc_auc_score(y_test, y_prob), 4)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Score all customers
    df_score = df[feature_cols].fillna(0).copy()
    df_score_enc = encode_categoricals(df_score.copy())
    df["Churn_Score_Predicted"] = model.predict_proba(df_score_enc)[:, 1]
    df["Churn_Risk_Label"] = pd.cut(
        df["Churn_Score_Predicted"],
        bins=[0, 0.30, 0.60, 0.80, 1.01],
        labels=["Safe", "Watch", "At Risk", "Critical"]
    ).astype(str)
    df["Revenue_At_Risk_INR"] = (df["Lifetime_Value_INR"] * df["Churn_Score_Predicted"] * 0.5).fillna(0).astype(int)
    df["Recommended_Action"] = df["Churn_Risk_Label"].map({
        "Critical": "Immediate Retention Call + Special Offer",
        "At Risk": "Account Manager Engagement + Discount",
        "Watch": "Monthly Check-In",
        "Safe": "Standard Service",
    }).fillna("Standard Service")

    output_path = os.path.join(PROC_DIR, "churn_predictions.csv")
    df.to_csv(output_path, index=False)

    joblib.dump(model, os.path.join(MODEL_DIR, "model_churn_prediction.pkl"))

    metrics = {
        "model": "Customer Churn Prediction",
        "algorithm": "XGBoost Classifier",
        "auc_roc": auc,
        "precision_class1": round(report["1"]["precision"], 4),
        "recall_class1": round(report["1"]["recall"], 4),
        "f1_class1": round(report["1"]["f1-score"], 4),
        "total_revenue_at_risk_INR": int(df["Revenue_At_Risk_INR"].sum()),
        "critical_customers": int((df["Churn_Risk_Label"] == "Critical").sum()),
        "business_kpi": "Revenue Retention Opportunity",
    }
    print_model_box("Customer Churn Prediction", metrics)
    save_model_report("churn_prediction", metrics)
    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4: ASSET UTILIZATION PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
def train_utilization_prediction():
    print("\n[MODEL 4] Asset Utilization Prediction...")
    df = load("ml_utilization_features")

    feature_cols = [
        "Month_Num", "Asset_Age_Years", "Daily_Rental_Rate_INR",
        "Asset_Category", "Country"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target = "Utilization_Rate"

    df_model = df[feature_cols + [target]].dropna()
    df_model = encode_categoricals(df_model)

    X = df_model[feature_cols]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)

    # Score full dataset
    df_score = df[feature_cols].fillna(0).copy()
    df_score_enc = encode_categoricals(df_score.copy())
    df["Predicted_Utilization_Pct"] = model.predict(df_score_enc)
    df["Idle_Risk_Predicted"] = (df["Predicted_Utilization_Pct"] < 45).astype(int)
    df["Revenue_Opportunity_INR"] = (
        (72 - df["Predicted_Utilization_Pct"]).clip(0) / 100 * 365 * df["Daily_Rental_Rate_INR"]
    ).fillna(0).astype(int)

    output_path = os.path.join(PROC_DIR, "utilization_predictions.csv")
    df.to_csv(output_path, index=False)

    joblib.dump(model, os.path.join(MODEL_DIR, "model_utilization_prediction.pkl"))

    metrics = {
        "model": "Asset Utilization Prediction",
        "algorithm": "XGBoost Regressor",
        "mean_absolute_error_pct": mae,
        "r2_score": r2,
        "assets_with_idle_risk": int(df["Idle_Risk_Predicted"].sum()),
        "total_revenue_opportunity_INR": int(df["Revenue_Opportunity_INR"].sum()),
        "business_kpi": "Additional Rental Revenue Opportunity",
    }
    print_model_box("Asset Utilization Prediction", metrics)
    save_model_report("utilization_prediction", metrics)
    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 5: PROFIT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
def train_profit_prediction():
    print("\n[MODEL 5] Profit Prediction...")
    df = load("rental_transactions_processed")

    feature_cols = [
        "Rental_Days", "Base_Daily_Rate_INR", "Discount_Pct",
        "Effective_Daily_Rate_INR", "Operating_Cost_INR",
        "Mobilization_Cost_INR", "Is_Long_Term",
        "Asset_Category", "Customer_Category", "Customer_Segment",
        "Rental_Mode", "Project_Type", "Country"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target = "Net_Profit_INR"

    df_model = df[feature_cols + [target]].dropna().head(100000)
    df_model = encode_categoricals(df_model)

    X = df_model[feature_cols]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 4)

    # Score + profit tier
    df_score = df[feature_cols].fillna(0).head(100000).copy()
    df_score_enc = encode_categoricals(df_score.copy())
    df_out = df.head(100000).copy()
    df_out["Predicted_Profit_INR"] = model.predict(df_score_enc).astype(int)
    df_out["Profit_Tier"] = pd.cut(
        df_out["Predicted_Profit_INR"],
        bins=[-999999999, 0, 50000, 150000, 999999999],
        labels=["Loss", "Low Profit", "Medium Profit", "High Profit"]
    ).astype(str)

    output_path = os.path.join(PROC_DIR, "profit_predictions.csv")
    df_out.to_csv(output_path, index=False)

    joblib.dump(model, os.path.join(MODEL_DIR, "model_profit_prediction.pkl"))

    metrics = {
        "model": "Profit Prediction",
        "algorithm": "XGBoost Regressor",
        "mean_absolute_error_INR": mae,
        "r2_score": r2,
        "pct_high_profit": round((df_out["Profit_Tier"] == "High Profit").mean() * 100, 2),
        "pct_loss_making": round((df_out["Profit_Tier"] == "Loss").mean() * 100, 2),
        "business_kpi": "Profit Improvement Potential",
    }
    print_model_box("Profit Prediction", metrics)
    save_model_report("profit_prediction", metrics)
    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  RENTAL PROFIT OPTIMIZATION — ML MODEL TRAINING")
    print("=" * 70)

    all_metrics = {}

    _, m1 = train_predictive_maintenance()
    all_metrics["predictive_maintenance"] = m1

    _, m2, forecast_df = train_revenue_forecast()
    all_metrics["revenue_forecast"] = m2
    print(f"\n  2025 Revenue Forecast Preview:")
    print(forecast_df[["Month", "Forecasted_Revenue_INR"]].to_string(index=False))

    _, m3 = train_churn_prediction()
    all_metrics["churn_prediction"] = m3

    _, m4 = train_utilization_prediction()
    all_metrics["utilization_prediction"] = m4

    _, m5 = train_profit_prediction()
    all_metrics["profit_prediction"] = m5

    # Save consolidated model summary
    summary_path = os.path.join(REPORT_DIR, "ml_models_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  ✅ ALL ML MODELS TRAINED SUCCESSFULLY")
    print(f"  Models saved in: {MODEL_DIR}")
    print("  Next: python scripts/05_recommendation_engine.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
