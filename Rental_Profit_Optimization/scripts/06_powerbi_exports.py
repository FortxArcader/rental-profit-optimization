"""
===============================================================================
RENTAL BUSINESS PROFIT OPTIMIZATION PLATFORM
Script 06: Power BI Export Generator
===============================================================================
Creates 11 optimized flat-file CSVs for Power BI data model:

  1. revenue_dashboard.csv
  2. maintenance_dashboard.csv
  3. profit_dashboard.csv
  4. customer_dashboard.csv
  5. utilization_dashboard.csv
  6. executive_summary.csv
  7. pricing_optimization.csv
  8. fleet_expansion.csv
  9. asset_retirement.csv
 10. country_performance.csv
 11. executive_profit_summary.csv
===============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
PBI_DIR = os.path.join(BASE_DIR, "powerbi_exports")
os.makedirs(PBI_DIR, exist_ok=True)


def load(name):
    path = os.path.join(PROC_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"  ⚠ Skipping missing file: {name}.csv")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def save_pbi(df, name):
    path = os.path.join(PBI_DIR, name)
    df.to_csv(path, index=False)
    print(f"  ✓ {name}  [{len(df):,} rows, {df.shape[1]} cols]")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. REVENUE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def export_revenue_dashboard():
    txn = load("rental_transactions_processed")
    if txn.empty:
        return

    txn["Rental_Start_Date"] = pd.to_datetime(txn["Rental_Start_Date"], errors="coerce")
    txn["Month"] = txn["Rental_Start_Date"].dt.to_period("M").astype(str)
    txn["Year"] = txn["Rental_Start_Date"].dt.year
    txn["Quarter"] = txn["Rental_Start_Date"].dt.to_period("Q").astype(str)
    txn["Month_Name"] = txn["Rental_Start_Date"].dt.strftime("%b %Y")

    df = txn[[
        "Transaction_ID", "Asset_ID", "Customer_ID", "Asset_Category",
        "Customer_Category", "Customer_Segment", "Country", "Project_Type",
        "Rental_Mode", "Month", "Quarter", "Year", "Month_Name",
        "Rental_Days", "Gross_Revenue_INR", "Net_Profit_INR",
        "Rental_Margin_Pct", "Discount_Pct",
        "Underpriced_Flag", "Pricing_Opportunity_INR"
    ]].copy()

    # Add YoY comparison columns
    monthly = df.groupby(["Year", "Month"]).agg(
        Monthly_Revenue=("Gross_Revenue_INR", "sum"),
        Monthly_Profit=("Net_Profit_INR", "sum"),
        Monthly_Rentals=("Transaction_ID", "count"),
    ).reset_index()
    monthly["MoM_Revenue_Growth_Pct"] = monthly["Monthly_Revenue"].pct_change() * 100

    save_pbi(df, "revenue_dashboard.csv")
    save_pbi(monthly, "revenue_monthly_trend.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MAINTENANCE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def export_maintenance_dashboard():
    maint = load("maintenance_records_processed")
    if maint.empty:
        return

    df = maint[[
        "Maintenance_ID", "Asset_ID", "Asset_Category", "Country",
        "Maintenance_Type", "Maintenance_Date", "Month", "Year",
        "Maintenance_Cost_INR", "Downtime_Hours", "Downtime_Days",
        "Failure_Flag", "Failure_Probability", "High_Risk_Asset",
        "Revenue_Lost_From_Downtime_INR", "Total_Impact_INR",
        "Spare_Parts_Cost_INR", "Labour_Cost_INR", "Cost_Band"
    ]].copy()

    save_pbi(df, "maintenance_dashboard.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROFIT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def export_profit_dashboard():
    profit_preds = load("profit_predictions")
    fin = load("financial_performance_processed")

    if not profit_preds.empty:
        df = profit_preds[[
            "Transaction_ID", "Asset_ID", "Asset_Category", "Customer_Category",
            "Country", "Project_Type", "Rental_Mode", "Month", "Year",
            "Gross_Revenue_INR", "Net_Profit_INR", "Rental_Margin_Pct",
            "Predicted_Profit_INR", "Profit_Tier"
        ]].copy()
        save_pbi(df, "profit_dashboard.csv")

    if not fin.empty:
        df2 = fin[[
            "Asset_ID", "Asset_Category", "Country", "Branch",
            "Month", "Year", "Quarter",
            "Gross_Revenue_INR", "Operating_Cost_INR", "Depreciation_INR",
            "Gross_Profit_INR", "Net_Profit_INR", "Profit_Margin_Pct",
            "EBITDA_INR", "EBITDA_Margin_Pct"
        ]].copy()
        save_pbi(df2, "profit_financial_performance.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CUSTOMER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def export_customer_dashboard():
    customers = load("customers_processed")
    churn = load("churn_predictions")

    if not customers.empty:
        df = customers[[
            "Customer_ID", "Customer_Name", "Customer_Category",
            "Customer_Segment", "Country", "City",
            "Total_Rentals", "Lifetime_Value_INR", "Days_Since_Last_Rental",
            "Churn_Probability", "Churn_Risk_Tier", "RFM_Score",
            "Customer_Health", "Recency_Score", "Frequency_Score",
            "Monetary_Score", "Revenue_At_Risk_INR", "Retention_Priority"
        ]].copy()

        if not churn.empty and "Churn_Score_Predicted" in churn.columns:
            merge_cols = ["Customer_ID", "Churn_Score_Predicted", "Churn_Risk_Label",
                          "Revenue_At_Risk_INR", "Recommended_Action"]
            merge_cols = [c for c in merge_cols if c in churn.columns]
            df = df.merge(churn[merge_cols], on="Customer_ID", how="left", suffixes=("", "_ml"))

        save_pbi(df, "customer_dashboard.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. UTILIZATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def export_utilization_dashboard():
    util = load("asset_utilization_processed")
    util_preds = load("utilization_predictions")

    if not util.empty:
        df = util[[
            "Utilization_ID", "Asset_ID", "Asset_Category", "Country", "Branch",
            "Month", "Year", "Quarter",
            "Days_In_Month", "Days_Rented", "Idle_Days",
            "Utilization_Rate_Pct", "Industry_Avg_Util_Pct",
            "Revenue_Loss_From_Idle_INR", "Daily_Rate_INR",
            "Below_Industry_Avg", "Utilization_Gap_Pct",
            "Idle_Risk_Flag", "High_Performer_Flag",
            "Annual_Revenue_Opportunity_INR"
        ]].copy()

        if not util_preds.empty and "Revenue_Opportunity_INR" in util_preds.columns:
            pred_cols = ["Asset_ID", "Month", "Predicted_Utilization_Pct",
                         "Idle_Risk_Predicted", "Revenue_Opportunity_INR"]
            pred_cols = [c for c in pred_cols if c in util_preds.columns]
            df = df.merge(util_preds[pred_cols], on=["Asset_ID", "Month"], how="left", suffixes=("", "_ml"))

        save_pbi(df, "utilization_dashboard.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def export_executive_summary():
    txn = load("rental_transactions_processed")
    util = load("asset_utilization_processed")
    maint = load("maintenance_records_processed")
    customers = load("customers_processed")
    churn = load("churn_predictions")
    impact = load("financial_impact_summary")
    assets = load("asset_master_processed")
    forecast = load("revenue_forecast_2025")

    summary = {}

    if not txn.empty:
        summary["Total_Revenue_INR"] = int(txn["Gross_Revenue_INR"].sum())
        summary["Total_Profit_INR"] = int(txn["Net_Profit_INR"].sum())
        summary["Overall_Margin_Pct"] = round(txn["Rental_Margin_Pct"].mean(), 2)
        summary["Total_Rentals"] = len(txn)
        summary["Avg_Rental_Days"] = round(txn["Rental_Days"].mean(), 1)

    if not util.empty:
        summary["Overall_Utilization_Pct"] = round(util["Utilization_Rate_Pct"].mean(), 2)
        summary["Total_Revenue_Loss_Idle_INR"] = int(util["Revenue_Loss_From_Idle_INR"].sum())
        summary["High_Util_Assets"] = int((util["Utilization_Rate_Pct"] > 75).sum())

    if not maint.empty:
        summary["Total_Maintenance_Cost_INR"] = int(maint["Maintenance_Cost_INR"].sum())
        summary["Total_Downtime_Hours"] = round(maint["Downtime_Hours"].sum(), 1)
        summary["Total_Breakdowns"] = int(maint["Failure_Flag"].sum())

    if not customers.empty:
        summary["Total_Customers"] = len(customers)
        summary["High_Churn_Risk_Customers"] = int((customers["Churn_Probability"] > 0.70).sum())
        summary["Total_Revenue_At_Risk_INR"] = int(customers["Revenue_At_Risk_INR"].sum())

    if not assets.empty:
        summary["Total_Assets"] = len(assets)
        summary["Active_Assets"] = int((assets["Asset_Status"] == "Active").sum())
        summary["Idle_Assets"] = int((assets["Asset_Status"] == "Idle").sum())
        summary["Under_Maintenance_Assets"] = int((assets["Asset_Status"] == "Under Maintenance").sum())

    if not impact.empty:
        for col in impact.columns:
            summary[col] = impact[col].iloc[0]

    if not forecast.empty:
        summary["Forecasted_Annual_Revenue_2025_INR"] = int(forecast["Forecasted_Revenue_INR"].sum())

    df = pd.DataFrame([summary])
    save_pbi(df, "executive_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PRICING OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def export_pricing_optimization():
    pricing = load("pricing_optimization")
    if not pricing.empty:
        save_pbi(pricing, "pricing_optimization.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FLEET EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════
def export_fleet_expansion():
    fleet = load("fleet_expansion")
    if not fleet.empty:
        save_pbi(fleet, "fleet_expansion.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ASSET RETIREMENT
# ═══════════════════════════════════════════════════════════════════════════════
def export_asset_retirement():
    retire = load("asset_retirement_candidates")
    if not retire.empty:
        df = retire[[
            "Asset_ID", "Asset_Category", "Asset_Subcategory", "Country", "Branch",
            "Asset_Age_Years", "Purchase_Cost_INR", "Book_Value_INR",
            "Total_Maint_Cost_INR", "Total_Breakdowns", "Total_Downtime_Hours",
            "Maint_Cost_To_Purchase_Ratio", "Retire_Flag",
            "Annual_Savings_If_Replaced_INR", "Replacement_Cost_INR"
        ]].copy()
        save_pbi(df, "asset_retirement.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. COUNTRY PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
def export_country_performance():
    country = load("country_performance")
    if not country.empty:
        save_pbi(country, "country_performance.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. EXECUTIVE PROFIT SUMMARY (CEO Command Center)
# ═══════════════════════════════════════════════════════════════════════════════
def export_executive_profit_summary():
    recs = load("recommendations")
    impact = load("financial_impact_summary")
    forecast = load("revenue_forecast_2025")

    rows = []
    if not impact.empty:
        for col in impact.columns:
            rows.append({
                "KPI_Name": col.replace("_", " ").title(),
                "KPI_Value_INR": int(impact[col].iloc[0]) if pd.api.types.is_numeric_dtype(impact[col]) else 0,
                "KPI_Category": "Financial Impact",
            })

    if not recs.empty:
        rows.append({
            "KPI_Name": "Total Profit Opportunity",
            "KPI_Value_INR": int(recs["Expected_Annual_Gain_INR"].sum()),
            "KPI_Category": "Executive Summary",
        })
        rows.append({
            "KPI_Name": "Total Recommendations",
            "KPI_Value_INR": len(recs),
            "KPI_Category": "Executive Summary",
        })
        rows.append({
            "KPI_Name": "Critical Actions Required",
            "KPI_Value_INR": int((recs["Priority"] == "Critical").sum()),
            "KPI_Category": "Executive Summary",
        })

    if not forecast.empty:
        rows.append({
            "KPI_Name": "Forecasted 2025 Annual Revenue",
            "KPI_Value_INR": int(forecast["Forecasted_Revenue_INR"].sum()),
            "KPI_Category": "Revenue Forecast",
        })

    df = pd.DataFrame(rows)
    save_pbi(df, "executive_profit_summary.csv")

    if not recs.empty:
        save_pbi(recs, "branch_profitability.csv")
        top_recs = recs[[
            "Rank", "Category", "Asset_Category", "Country",
            "Action", "Detail", "Expected_Annual_Gain_INR",
            "Expected_Annual_Gain_Label", "Priority", "Action_Owner",
            "Implementation_Timeline"
        ]].copy()
        save_pbi(top_recs, "ai_recommendations.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  RENTAL PROFIT OPTIMIZATION — POWER BI EXPORT GENERATION")
    print("=" * 70)

    print("\n[EXPORTING] Generating Power BI optimized files...")
    export_revenue_dashboard()
    export_maintenance_dashboard()
    export_profit_dashboard()
    export_customer_dashboard()
    export_utilization_dashboard()
    export_executive_summary()
    export_pricing_optimization()
    export_fleet_expansion()
    export_asset_retirement()
    export_country_performance()
    export_executive_profit_summary()

    # List all exported files
    files = os.listdir(PBI_DIR)
    total_size = sum(os.path.getsize(os.path.join(PBI_DIR, f)) for f in files) / (1024 * 1024)

    print(f"\n  📁 Power BI Export Directory: {PBI_DIR}")
    print(f"  📊 Total Files: {len(files)}")
    print(f"  💾 Total Size: {total_size:.1f} MB")

    print("\n" + "=" * 70)
    print("  ✅ POWER BI EXPORTS COMPLETE")
    print("  Next: streamlit run dashboard/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
