"""
===============================================================================
RENTAL BUSINESS PROFIT OPTIMIZATION PLATFORM
Script 02: Data Preprocessing Pipeline
===============================================================================
Performs:
  - Missing value imputation
  - Outlier detection and capping (IQR method)
  - Feature engineering (20+ business features)
  - Data quality validation
  - Saves processed datasets to data/processed/
===============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ── Utilities ────────────────────────────────────────────────────────────────
def load_csv(name):
    path = os.path.join(RAW_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {name}.csv  [{len(df):,} rows, {df.shape[1]} cols]")
    return df


def cap_outliers(df, cols, low=0.01, high=0.99):
    """Cap outliers at low/high quantile boundaries."""
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            lo = df[col].quantile(low)
            hi = df[col].quantile(high)
            df[col] = df[col].clip(lo, hi)
    return df


def fill_missing(df, strategy="median"):
    """Fill numeric nulls with median, categorical with mode."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def quality_report(name, df_raw, df_proc):
    return {
        "dataset": name,
        "raw_rows": len(df_raw),
        "processed_rows": len(df_proc),
        "raw_missing_pct": round(df_raw.isnull().mean().mean() * 100, 2),
        "processed_missing_pct": round(df_proc.isnull().mean().mean() * 100, 2),
        "raw_columns": df_raw.shape[1],
        "processed_columns": df_proc.shape[1],
        "new_features": df_proc.shape[1] - df_raw.shape[1],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET MASTER — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def process_assets(df_raw):
    print("\n  Processing asset_master...")
    df = df_raw.copy()
    df = fill_missing(df)
    df = cap_outliers(df, ["Purchase_Cost_INR", "Book_Value_INR", "Daily_Rental_Rate_INR"])

    # Feature engineering
    df["Revenue_Per_Day_INR"] = df["Daily_Rental_Rate_INR"]
    df["Asset_Age_Band"] = pd.cut(
        df["Asset_Age_Years"],
        bins=[0, 2, 5, 9, 100],
        labels=["New (0-2y)", "Mid-Life (2-5y)", "Mature (5-9y)", "Old (9y+)"]
    ).astype(str)
    df["Depreciation_To_Date_INR"] = df["Purchase_Cost_INR"] - df["Book_Value_INR"]
    df["Book_Value_Pct"] = (df["Book_Value_INR"] / df["Purchase_Cost_INR"] * 100).round(2)
    df["Days_Since_Last_Service"] = (
        pd.to_datetime("today") - pd.to_datetime(df["Last_Service_Date"])
    ).dt.days
    df["Overdue_Service_Flag"] = (df["Days_Since_Last_Service"] > df["Service_Interval_Days"]).astype(int)
    df["Estimated_Annual_Revenue_INR"] = df["Daily_Rental_Rate_INR"] * 365 * 0.68  # 68% avg utilization
    df["Estimated_Annual_Cost_INR"] = df["Estimated_Annual_Revenue_INR"] * 0.45
    df["Estimated_Annual_Profit_INR"] = (df["Estimated_Annual_Revenue_INR"] - df["Estimated_Annual_Cost_INR"]).astype(int)
    df["Payback_Remaining_Years"] = (df["Book_Value_INR"] / df["Estimated_Annual_Profit_INR"].clip(1)).round(2)
    df["Replacement_Priority_Score"] = (
        (df["Asset_Age_Years"] / 15 * 40) +
        (df["Days_Since_Last_Service"] / df["Service_Interval_Days"] * 30) +
        (df["Book_Value_Pct"].rsub(100) / 100 * 30)
    ).round(2)
    df["Replacement_Recommended"] = (df["Replacement_Priority_Score"] > 70).astype(int)

    path = os.path.join(PROC_DIR, "asset_master_processed.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ asset_master_processed.csv  [{len(df):,} rows, {df.shape[1]} cols]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOMERS — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def process_customers(df_raw):
    print("\n  Processing customers...")
    df = df_raw.copy()
    df = fill_missing(df)

    df["Churn_Risk_Tier"] = pd.cut(
        df["Churn_Probability"],
        bins=[0, 0.30, 0.60, 0.80, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    ).astype(str)
    df["Avg_Revenue_Per_Rental_INR"] = (df["Lifetime_Value_INR"] / df["Total_Rentals"].clip(1)).round(0)
    df["Revenue_At_Risk_INR"] = (df["Lifetime_Value_INR"] * df["Churn_Probability"] * 0.5).astype(int)
    df["Retention_Priority"] = (df["Revenue_At_Risk_INR"] > 500000).astype(int)
    df["Days_Since_Last_Rental"] = df["Days_Since_Last_Rental"].clip(0, 730)
    df["Recency_Score"] = 100 - (df["Days_Since_Last_Rental"] / 730 * 100).round(0)
    df["Frequency_Score"] = (df["Total_Rentals"] / df["Total_Rentals"].quantile(0.95) * 100).clip(0, 100).round(0)
    df["Monetary_Score"] = (df["Lifetime_Value_INR"] / df["Lifetime_Value_INR"].quantile(0.95) * 100).clip(0, 100).round(0)
    df["RFM_Score"] = (df["Recency_Score"] * 0.30 + df["Frequency_Score"] * 0.35 + df["Monetary_Score"] * 0.35).round(2)
    df["Customer_Health"] = pd.cut(
        df["RFM_Score"],
        bins=[0, 30, 50, 70, 100],
        labels=["Churning", "At Risk", "Active", "Champion"]
    ).astype(str)

    path = os.path.join(PROC_DIR, "customers_processed.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ customers_processed.csv  [{len(df):,} rows, {df.shape[1]} cols]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTIONS — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def process_transactions(df_raw):
    print("\n  Processing rental_transactions...")
    df = df_raw.copy()
    df = fill_missing(df)
    df = cap_outliers(df, ["Gross_Revenue_INR", "Net_Profit_INR", "Total_Cost_INR"])

    df["Rental_Start_Date"] = pd.to_datetime(df["Rental_Start_Date"])
    df["Rental_End_Date"] = pd.to_datetime(df["Rental_End_Date"])
    df["Month"] = df["Rental_Start_Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Rental_Start_Date"].dt.to_period("Q").astype(str)
    df["Year"] = df["Rental_Start_Date"].dt.year
    df["Month_Num"] = df["Rental_Start_Date"].dt.month

    df["Revenue_Vs_Competitor_INR"] = df["Effective_Daily_Rate_INR"] - df["Competitor_Rate_INR"]
    df["Underpriced_Flag"] = (df["Effective_Daily_Rate_INR"] < df["Competitor_Rate_INR"] * 0.95).astype(int)
    df["Overpriced_Flag"] = (df["Effective_Daily_Rate_INR"] > df["Competitor_Rate_INR"] * 1.05).astype(int)
    df["Pricing_Opportunity_INR"] = (
        (df["Competitor_Rate_INR"] - df["Effective_Daily_Rate_INR"]).clip(0) * df["Rental_Days"]
    ).astype(int)

    df["Is_Long_Term"] = (df["Rental_Days"] >= 30).astype(int)
    df["Is_Profitable"] = (df["Net_Profit_INR"] > 0).astype(int)
    df["Margin_Band"] = pd.cut(
        df["Rental_Margin_Pct"],
        bins=[-999, 0, 15, 30, 50, 999],
        labels=["Loss", "Low (<15%)", "Medium (15-30%)", "Good (30-50%)", "Excellent (>50%)"]
    ).astype(str)

    path = os.path.join(PROC_DIR, "rental_transactions_processed.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ rental_transactions_processed.csv  [{len(df):,} rows, {df.shape[1]} cols]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def process_maintenance(df_raw):
    print("\n  Processing maintenance_records...")
    df = df_raw.copy()
    df = fill_missing(df)
    df = cap_outliers(df, ["Maintenance_Cost_INR", "Downtime_Hours"])

    df["Maintenance_Date"] = pd.to_datetime(df["Maintenance_Date"])
    df["Month"] = df["Maintenance_Date"].dt.to_period("M").astype(str)
    df["Year"] = df["Maintenance_Date"].dt.year
    df["Revenue_Lost_From_Downtime_INR"] = (df["Downtime_Hours"] / 24 * 8000).astype(int)  # ₹8000/day avg
    df["Total_Impact_INR"] = df["Maintenance_Cost_INR"] + df["Revenue_Lost_From_Downtime_INR"]
    df["High_Risk_Asset"] = (df["Failure_Probability"] > 0.70).astype(int)
    df["Downtime_Days"] = (df["Downtime_Hours"] / 24).round(1)
    df["Cost_Band"] = pd.cut(
        df["Maintenance_Cost_INR"],
        bins=[0, 25000, 75000, 200000, 999999999],
        labels=["Low (<₹25K)", "Medium (₹25K-75K)", "High (₹75K-2L)", "Critical (>₹2L)"]
    ).astype(str)

    path = os.path.join(PROC_DIR, "maintenance_records_processed.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ maintenance_records_processed.csv  [{len(df):,} rows, {df.shape[1]} cols]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIZATION — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def process_utilization(df_raw):
    print("\n  Processing asset_utilization...")
    df = df_raw.copy()
    df = fill_missing(df)

    df["Below_Industry_Avg"] = (df["Utilization_Rate_Pct"] < df["Industry_Avg_Util_Pct"]).astype(int)
    df["Utilization_Gap_Pct"] = (df["Industry_Avg_Util_Pct"] - df["Utilization_Rate_Pct"]).clip(0).round(2)
    df["Idle_Risk_Flag"] = (df["Utilization_Rate_Pct"] < 40).astype(int)
    df["High_Performer_Flag"] = (df["Utilization_Rate_Pct"] > 85).astype(int)
    df["Annual_Revenue_Opportunity_INR"] = (df["Utilization_Gap_Pct"] / 100 * 365 * df["Daily_Rate_INR"]).astype(int)

    path = os.path.join(PROC_DIR, "asset_utilization_processed.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ asset_utilization_processed.csv  [{len(df):,} rows, {df.shape[1]} cols]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def process_financial(df_raw):
    print("\n  Processing financial_performance...")
    df = df_raw.copy()
    df = fill_missing(df)
    df = cap_outliers(df, ["Gross_Revenue_INR", "Net_Profit_INR"])

    df["Revenue_Growth_Pct"] = df.groupby(["Asset_ID"])["Gross_Revenue_INR"].pct_change() * 100
    df["Revenue_Growth_Pct"] = df["Revenue_Growth_Pct"].fillna(0).round(2)
    df["Is_Growth_Month"] = (df["Revenue_Growth_Pct"] > 0).astype(int)
    df["EBITDA_INR"] = df["Net_Profit_INR"] + df["Depreciation_INR"]
    df["EBITDA_Margin_Pct"] = (df["EBITDA_INR"] / df["Gross_Revenue_INR"].clip(1) * 100).round(2)

    path = os.path.join(PROC_DIR, "financial_performance_processed.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ financial_performance_processed.csv  [{len(df):,} rows, {df.shape[1]} cols]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD ML FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════════
def build_ml_features(assets, customers, transactions, maintenance, utilization):
    print("\n  Building ML feature sets...")

    # --- Maintenance ML features (per asset) ---
    maint_agg = maintenance.groupby("Asset_ID").agg(
        Total_Breakdowns=("Failure_Flag", "sum"),
        Avg_Maintenance_Cost=("Maintenance_Cost_INR", "mean"),
        Total_Downtime_Hours=("Downtime_Hours", "sum"),
        Avg_Failure_Prob=("Failure_Probability", "mean"),
        Maintenance_Count=("Maintenance_ID", "count"),
    ).reset_index()

    ml_maintenance = assets[[
        "Asset_ID", "Asset_Category", "Asset_Age_Years", "Purchase_Cost_INR",
        "Daily_Rental_Rate_INR", "Country", "Overdue_Service_Flag", "Book_Value_Pct"
    ]].merge(maint_agg, on="Asset_ID", how="left").fillna(0)
    ml_maintenance["Failure_Target"] = (ml_maintenance["Avg_Failure_Prob"] > 0.65).astype(int)

    path = os.path.join(PROC_DIR, "ml_maintenance_features.csv")
    ml_maintenance.to_csv(path, index=False)
    print(f"    ✓ ml_maintenance_features.csv  [{len(ml_maintenance):,} rows]")

    # --- Revenue ML features (monthly) ---
    rev_monthly = transactions.groupby(["Asset_ID", "Month"]).agg(
        Monthly_Revenue=("Gross_Revenue_INR", "sum"),
        Monthly_Rentals=("Transaction_ID", "count"),
        Avg_Rental_Days=("Rental_Days", "mean"),
        Avg_Margin=("Rental_Margin_Pct", "mean"),
    ).reset_index()
    rev_monthly["Month_Num"] = pd.to_datetime(rev_monthly["Month"]).dt.month
    rev_monthly["Year"] = pd.to_datetime(rev_monthly["Month"]).dt.year
    rev_monthly = rev_monthly.merge(
        assets[["Asset_ID", "Asset_Category", "Country", "Asset_Age_Years", "Daily_Rental_Rate_INR"]],
        on="Asset_ID", how="left"
    )

    path = os.path.join(PROC_DIR, "ml_revenue_features.csv")
    rev_monthly.to_csv(path, index=False)
    print(f"    ✓ ml_revenue_features.csv  [{len(rev_monthly):,} rows]")

    # --- Churn ML features (per customer) ---
    cust_txn = transactions.groupby("Customer_ID").agg(
        Total_Revenue=("Gross_Revenue_INR", "sum"),
        Total_Rentals=("Transaction_ID", "count"),
        Avg_Rental_Days=("Rental_Days", "mean"),
        Avg_Margin=("Rental_Margin_Pct", "mean"),
        Last_Rental_Month=("Month", "max"),
    ).reset_index()

    ml_churn = customers[[
        "Customer_ID", "Customer_Category", "Customer_Segment", "Country",
        "Days_Since_Last_Rental", "Lifetime_Value_INR", "Total_Rentals",
        "Churn_Probability", "RFM_Score"
    ]].merge(cust_txn, on="Customer_ID", how="left").fillna(0)
    ml_churn["Churn_Target"] = (ml_churn["Churn_Probability"] > 0.60).astype(int)

    path = os.path.join(PROC_DIR, "ml_churn_features.csv")
    ml_churn.to_csv(path, index=False)
    print(f"    ✓ ml_churn_features.csv  [{len(ml_churn):,} rows]")

    # --- Utilization ML features (per asset per month) ---
    util_agg = utilization.groupby(["Asset_ID", "Month"]).agg(
        Utilization_Rate=("Utilization_Rate_Pct", "mean"),
        Idle_Days=("Idle_Days", "sum"),
        Revenue_Loss=("Revenue_Loss_From_Idle_INR", "sum"),
    ).reset_index()
    util_agg["Month_Num"] = pd.to_datetime(util_agg["Month"]).dt.month
    util_agg = util_agg.merge(
        assets[["Asset_ID", "Asset_Category", "Country", "Asset_Age_Years", "Daily_Rental_Rate_INR"]],
        on="Asset_ID", how="left"
    )
    util_agg["Idle_Risk_Target"] = (util_agg["Utilization_Rate"] < 45).astype(int)

    path = os.path.join(PROC_DIR, "ml_utilization_features.csv")
    util_agg.to_csv(path, index=False)
    print(f"    ✓ ml_utilization_features.csv  [{len(util_agg):,} rows]")

    return ml_maintenance, rev_monthly, ml_churn, util_agg


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  RENTAL PROFIT OPTIMIZATION — DATA PREPROCESSING")
    print("=" * 70)

    print("\n[LOADING] Raw datasets...")
    assets_raw = load_csv("asset_master")
    customers_raw = load_csv("customers")
    txn_raw = load_csv("rental_transactions")
    maint_raw = load_csv("maintenance_records")
    util_raw = load_csv("asset_utilization")
    fin_raw = load_csv("financial_performance")

    print("\n[PROCESSING] Cleaning and engineering features...")
    assets = process_assets(assets_raw)
    customers = process_customers(customers_raw)
    transactions = process_transactions(txn_raw)
    maintenance = process_maintenance(maint_raw)
    utilization = process_utilization(util_raw)
    financial = process_financial(fin_raw)

    build_ml_features(assets, customers, transactions, maintenance, utilization)

    # Data quality report
    reports = [
        quality_report("asset_master", assets_raw, assets),
        quality_report("customers", customers_raw, customers),
        quality_report("rental_transactions", txn_raw, transactions),
        quality_report("maintenance_records", maint_raw, maintenance),
        quality_report("asset_utilization", util_raw, utilization),
        quality_report("financial_performance", fin_raw, financial),
    ]
    report_path = os.path.join(REPORT_DIR, "data_quality_report.json")
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\n  ✓ Data quality report saved: {report_path}")

    print("\n" + "=" * 70)
    print("  ✅ PREPROCESSING COMPLETE")
    print("  Next: python scripts/03_business_analytics.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
