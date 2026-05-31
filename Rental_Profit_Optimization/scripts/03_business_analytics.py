"""
===============================================================================
RENTAL BUSINESS PROFIT OPTIMIZATION PLATFORM
Script 03: Business Analytics Modules
===============================================================================
Answers 8 core executive questions and 6 new analytics modules:

A. Which assets generate the highest profit?
B. Which assets remain idle most often?
C. Which locations have the highest utilization?
D. Which customers generate the most revenue?
E. What factors increase maintenance cost?
F. Which assets should be replaced?
G. What is profitability by asset category?
H. What causes revenue decline?

NEW:
1. Pricing Optimization Analysis
2. Fleet Expansion Analytics
3. Asset Retirement Analysis
4. Cross-Country Performance
5. Branch Profitability
6. Project Opportunity Analysis
===============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load(name):
    path = os.path.join(PROC_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {name}  [{len(df):,} rows]")
    return df


def cr(val):
    """Format INR value as Crore string."""
    if abs(val) >= 1e7:
        return f"₹{val/1e7:.2f} Cr"
    elif abs(val) >= 1e5:
        return f"₹{val/1e5:.2f} L"
    else:
        return f"₹{val:,.0f}"


def save_report(name, data):
    path = os.path.join(REPORT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  ✓ Saved report: {name}.json")


# ═══════════════════════════════════════════════════════════════════════════════
# A. TOP PROFIT-GENERATING ASSETS
# ═══════════════════════════════════════════════════════════════════════════════
def module_a_top_profit_assets(txn, assets):
    print("\n  [A] Top Profit-Generating Assets...")
    asset_profit = txn.groupby("Asset_ID").agg(
        Total_Revenue_INR=("Gross_Revenue_INR", "sum"),
        Total_Profit_INR=("Net_Profit_INR", "sum"),
        Total_Rentals=("Transaction_ID", "count"),
        Avg_Margin_Pct=("Rental_Margin_Pct", "mean"),
    ).reset_index()

    asset_profit = asset_profit.merge(
        assets[["Asset_ID", "Asset_Category", "Asset_Subcategory", "Country", "Branch", "Daily_Rental_Rate_INR"]],
        on="Asset_ID", how="left"
    )
    asset_profit = asset_profit.sort_values("Total_Profit_INR", ascending=False)

    top20 = asset_profit.head(20)[["Asset_ID", "Asset_Category", "Country", "Branch",
                                    "Total_Revenue_INR", "Total_Profit_INR", "Avg_Margin_Pct"]]
    top20["Total_Revenue_INR"] = top20["Total_Revenue_INR"].astype(int)
    top20["Total_Profit_INR"] = top20["Total_Profit_INR"].astype(int)

    path = os.path.join(PROC_DIR, "top_profit_assets.csv")
    asset_profit.to_csv(path, index=False)
    save_report("module_a_top_profit_assets", {
        "top_20_assets": top20.to_dict("records"),
        "total_portfolio_revenue": int(asset_profit["Total_Revenue_INR"].sum()),
        "total_portfolio_profit": int(asset_profit["Total_Profit_INR"].sum()),
    })
    return asset_profit


# ═══════════════════════════════════════════════════════════════════════════════
# B. IDLE ASSET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def module_b_idle_assets(util, assets):
    print("  [B] Idle Asset Analysis...")
    idle = util.groupby("Asset_ID").agg(
        Avg_Utilization_Pct=("Utilization_Rate_Pct", "mean"),
        Total_Idle_Days=("Idle_Days", "sum"),
        Total_Revenue_Loss_INR=("Revenue_Loss_From_Idle_INR", "sum"),
    ).reset_index()

    idle = idle.merge(
        assets[["Asset_ID", "Asset_Category", "Asset_Subcategory", "Country", "Branch",
                "Daily_Rental_Rate_INR", "Asset_Status"]],
        on="Asset_ID", how="left"
    )

    idle_assets = idle[idle["Avg_Utilization_Pct"] < 45].sort_values("Total_Revenue_Loss_INR", ascending=False)
    path = os.path.join(PROC_DIR, "idle_assets.csv")
    idle_assets.to_csv(path, index=False)

    total_idle_loss = int(idle_assets["Total_Revenue_Loss_INR"].sum())
    save_report("module_b_idle_assets", {
        "total_idle_assets": len(idle_assets),
        "total_revenue_loss_INR": total_idle_loss,
        "top_idle_by_category": idle_assets.groupby("Asset_Category")["Total_Revenue_Loss_INR"].sum().sort_values(ascending=False).astype(int).to_dict(),
    })
    return idle_assets


# ═══════════════════════════════════════════════════════════════════════════════
# C. LOCATION UTILIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def module_c_location_utilization(util):
    print("  [C] Location Utilization Analysis...")
    loc = util.groupby(["Country", "Branch"]).agg(
        Avg_Utilization_Pct=("Utilization_Rate_Pct", "mean"),
        Total_Idle_Days=("Idle_Days", "sum"),
        Total_Revenue_Loss_INR=("Revenue_Loss_From_Idle_INR", "sum"),
        Asset_Count=("Asset_ID", "nunique"),
    ).reset_index().sort_values("Avg_Utilization_Pct", ascending=False)

    path = os.path.join(PROC_DIR, "location_utilization.csv")
    loc.to_csv(path, index=False)
    save_report("module_c_location_utilization", {
        "top_performing_locations": loc.head(10)[["Country", "Branch", "Avg_Utilization_Pct"]].to_dict("records"),
        "bottom_performing_locations": loc.tail(10)[["Country", "Branch", "Avg_Utilization_Pct"]].to_dict("records"),
    })
    return loc


# ═══════════════════════════════════════════════════════════════════════════════
# D. TOP REVENUE CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════════
def module_d_top_customers(txn, customers):
    print("  [D] Top Revenue Customers...")
    cust_rev = txn.groupby("Customer_ID").agg(
        Total_Revenue_INR=("Gross_Revenue_INR", "sum"),
        Total_Profit_INR=("Net_Profit_INR", "sum"),
        Total_Rentals=("Transaction_ID", "count"),
        Avg_Margin_Pct=("Rental_Margin_Pct", "mean"),
    ).reset_index()

    cust_rev = cust_rev.merge(
        customers[["Customer_ID", "Customer_Name", "Customer_Category",
                   "Customer_Segment", "Country", "Churn_Probability",
                   "Lifetime_Value_INR", "RFM_Score"]],
        on="Customer_ID", how="left"
    )
    cust_rev = cust_rev.sort_values("Total_Revenue_INR", ascending=False)
    path = os.path.join(PROC_DIR, "top_customers.csv")
    cust_rev.to_csv(path, index=False)

    save_report("module_d_top_customers", {
        "top_20_customers": cust_rev.head(20)[["Customer_ID", "Customer_Name",
                                               "Total_Revenue_INR", "Total_Profit_INR",
                                               "Churn_Probability"]].to_dict("records"),
        "revenue_by_segment": cust_rev.groupby("Customer_Segment")["Total_Revenue_INR"].sum().astype(int).to_dict(),
        "revenue_by_category": cust_rev.groupby("Customer_Category")["Total_Revenue_INR"].sum().astype(int).to_dict(),
    })
    return cust_rev


# ═══════════════════════════════════════════════════════════════════════════════
# E. MAINTENANCE COST DRIVERS
# ═══════════════════════════════════════════════════════════════════════════════
def module_e_maintenance_drivers(maint, assets):
    print("  [E] Maintenance Cost Drivers...")
    maint_agg = maint.groupby(["Asset_Category", "Maintenance_Type", "Country"]).agg(
        Total_Cost_INR=("Maintenance_Cost_INR", "sum"),
        Avg_Cost_INR=("Maintenance_Cost_INR", "mean"),
        Total_Downtime_Hours=("Downtime_Hours", "sum"),
        Count=("Maintenance_ID", "count"),
    ).reset_index()

    age_cost = maint.groupby(pd.cut(maint["Asset_Age_At_Event"],
                                     bins=[0, 3, 6, 10, 20],
                                     labels=["0-3y", "3-6y", "6-10y", "10y+"])).agg(
        Avg_Cost_INR=("Maintenance_Cost_INR", "mean"),
        Total_Cost_INR=("Maintenance_Cost_INR", "sum"),
    ).reset_index()
    age_cost.columns = ["Age_Band", "Avg_Cost_INR", "Total_Cost_INR"]

    path = os.path.join(PROC_DIR, "maintenance_drivers.csv")
    maint_agg.to_csv(path, index=False)
    save_report("module_e_maintenance_drivers", {
        "cost_by_type": maint.groupby("Maintenance_Type")["Maintenance_Cost_INR"].sum().astype(int).to_dict(),
        "cost_by_category": maint.groupby("Asset_Category")["Maintenance_Cost_INR"].sum().astype(int).to_dict(),
        "cost_by_age_band": age_cost.to_dict("records"),
    })
    return maint_agg


# ═══════════════════════════════════════════════════════════════════════════════
# F. ASSET REPLACEMENT CANDIDATES
# ═══════════════════════════════════════════════════════════════════════════════
def module_f_replacement_candidates(assets, maint):
    print("  [F] Asset Replacement Candidates...")
    maint_per_asset = maint.groupby("Asset_ID").agg(
        Total_Maint_Cost_INR=("Maintenance_Cost_INR", "sum"),
        Total_Breakdowns=("Failure_Flag", "sum"),
        Total_Downtime_Hours=("Downtime_Hours", "sum"),
    ).reset_index()

    candidates = assets.merge(maint_per_asset, on="Asset_ID", how="left").fillna(0)
    candidates["Maint_Cost_To_Purchase_Ratio"] = (
        candidates["Total_Maint_Cost_INR"] / candidates["Purchase_Cost_INR"].clip(1)
    ).round(4)
    candidates["Retire_Flag"] = (
        (candidates["Asset_Age_Years"] > 10) &
        (candidates["Maint_Cost_To_Purchase_Ratio"] > 0.30)
    ).astype(int)
    candidates["Replacement_Cost_INR"] = candidates["Purchase_Cost_INR"]
    candidates["Annual_Savings_If_Replaced_INR"] = (
        candidates["Total_Maint_Cost_INR"] * 0.60
    ).astype(int)

    retire = candidates[candidates["Retire_Flag"] == 1].sort_values(
        "Annual_Savings_If_Replaced_INR", ascending=False
    )
    path = os.path.join(PROC_DIR, "asset_retirement_candidates.csv")
    retire.to_csv(path, index=False)

    save_report("module_f_replacement_candidates", {
        "total_retirement_candidates": len(retire),
        "total_annual_savings_INR": int(retire["Annual_Savings_If_Replaced_INR"].sum()),
        "total_replacement_cost_INR": int(retire["Replacement_Cost_INR"].sum()),
        "by_category": retire.groupby("Asset_Category")["Annual_Savings_If_Replaced_INR"].sum().astype(int).to_dict(),
    })
    return retire


# ═══════════════════════════════════════════════════════════════════════════════
# G. PROFITABILITY BY CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════
def module_g_category_profitability(txn):
    print("  [G] Profitability by Asset Category...")
    cat_profit = txn.groupby("Asset_Category").agg(
        Total_Revenue_INR=("Gross_Revenue_INR", "sum"),
        Total_Cost_INR=("Total_Cost_INR", "sum"),
        Total_Profit_INR=("Net_Profit_INR", "sum"),
        Avg_Margin_Pct=("Rental_Margin_Pct", "mean"),
        Total_Rentals=("Transaction_ID", "count"),
    ).reset_index()
    cat_profit["Revenue_Share_Pct"] = (cat_profit["Total_Revenue_INR"] / cat_profit["Total_Revenue_INR"].sum() * 100).round(2)
    cat_profit = cat_profit.sort_values("Total_Profit_INR", ascending=False)

    path = os.path.join(PROC_DIR, "category_profitability.csv")
    cat_profit.to_csv(path, index=False)
    save_report("module_g_category_profitability", cat_profit.to_dict("records"))
    return cat_profit


# ═══════════════════════════════════════════════════════════════════════════════
# H. REVENUE DECLINE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def module_h_revenue_decline(txn):
    print("  [H] Revenue Decline Root Cause Analysis...")
    txn["Rental_Start_Date"] = pd.to_datetime(txn["Rental_Start_Date"])
    monthly = txn.groupby(txn["Rental_Start_Date"].dt.to_period("M")).agg(
        Total_Revenue_INR=("Gross_Revenue_INR", "sum"),
        Total_Profit_INR=("Net_Profit_INR", "sum"),
        Avg_Discount_Pct=("Discount_Pct", "mean"),
        Underpriced_Count=("Underpriced_Flag", "sum"),
        Total_Rentals=("Transaction_ID", "count"),
    ).reset_index()
    monthly["Rental_Start_Date"] = monthly["Rental_Start_Date"].astype(str)
    monthly["MoM_Revenue_Growth_Pct"] = monthly["Total_Revenue_INR"].pct_change() * 100
    monthly["Decline_Flag"] = (monthly["MoM_Revenue_Growth_Pct"] < -5).astype(int)

    path = os.path.join(PROC_DIR, "revenue_trend.csv")
    monthly.to_csv(path, index=False)
    save_report("module_h_revenue_decline", {
        "months_with_decline": int(monthly["Decline_Flag"].sum()),
        "max_decline_pct": round(monthly["MoM_Revenue_Growth_Pct"].min(), 2),
        "avg_monthly_revenue_INR": int(monthly["Total_Revenue_INR"].mean()),
    })
    return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# NEW 1: PRICING OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def module_pricing_optimization(txn, assets):
    print("  [NEW-1] Pricing Optimization...")
    pricing = txn[txn["Underpriced_Flag"] == 1].groupby("Asset_Category").agg(
        Underpriced_Transactions=("Transaction_ID", "count"),
        Avg_Current_Rate_INR=("Effective_Daily_Rate_INR", "mean"),
        Avg_Competitor_Rate_INR=("Competitor_Rate_INR", "mean"),
        Total_Pricing_Opportunity_INR=("Pricing_Opportunity_INR", "sum"),
        Avg_Rental_Days=("Rental_Days", "mean"),
    ).reset_index()

    pricing["Recommended_Rate_INR"] = (pricing["Avg_Competitor_Rate_INR"] * 0.98).astype(int)
    pricing["Rate_Increase_Pct"] = (
        (pricing["Recommended_Rate_INR"] - pricing["Avg_Current_Rate_INR"]) /
        pricing["Avg_Current_Rate_INR"] * 100
    ).round(2)
    pricing["Annual_Revenue_Gain_INR"] = (
        pricing["Total_Pricing_Opportunity_INR"] * (12 / 3)  # annualize from ~3yr data
    ).astype(int)

    pricing = pricing.sort_values("Annual_Revenue_Gain_INR", ascending=False)
    path = os.path.join(PROC_DIR, "pricing_optimization.csv")
    pricing.to_csv(path, index=False)
    save_report("module_pricing_optimization", {
        "total_annual_revenue_opportunity_INR": int(pricing["Annual_Revenue_Gain_INR"].sum()),
        "top_opportunities": pricing.head(5)[["Asset_Category", "Avg_Current_Rate_INR",
                                               "Recommended_Rate_INR", "Annual_Revenue_Gain_INR"]].to_dict("records"),
    })
    return pricing


# ═══════════════════════════════════════════════════════════════════════════════
# NEW 2: FLEET EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════
def module_fleet_expansion(util, assets):
    print("  [NEW-2] Fleet Expansion Analytics...")
    high_util = util[util["Utilization_Rate_Pct"] > 85].groupby("Asset_Category").agg(
        Avg_Utilization_Pct=("Utilization_Rate_Pct", "mean"),
        High_Util_Months=("Month", "count"),
        Asset_Count=("Asset_ID", "nunique"),
    ).reset_index()

    avg_rates = assets.groupby("Asset_Category")["Daily_Rental_Rate_INR"].mean().reset_index()
    avg_costs = assets.groupby("Asset_Category")["Purchase_Cost_INR"].mean().reset_index()

    fleet = high_util.merge(avg_rates, on="Asset_Category").merge(avg_costs, on="Asset_Category")
    fleet["Recommended_Purchase_Units"] = np.ceil(fleet["Asset_Count"] * 0.20).astype(int).clip(5, 50)
    fleet["Total_Purchase_Cost_INR"] = (fleet["Recommended_Purchase_Units"] * fleet["Purchase_Cost_INR"]).astype(int)
    fleet["Expected_Annual_Revenue_INR"] = (
        fleet["Recommended_Purchase_Units"] * fleet["Daily_Rental_Rate_INR"] * 365 * 0.72
    ).astype(int)
    fleet["Expected_Annual_Profit_INR"] = (fleet["Expected_Annual_Revenue_INR"] * 0.40).astype(int)
    fleet["Payback_Months"] = (
        fleet["Total_Purchase_Cost_INR"] / (fleet["Expected_Annual_Profit_INR"] / 12).clip(1)
    ).round(1)
    fleet = fleet.sort_values("Expected_Annual_Revenue_INR", ascending=False)

    path = os.path.join(PROC_DIR, "fleet_expansion.csv")
    fleet.to_csv(path, index=False)
    save_report("module_fleet_expansion", {
        "total_investment_required_INR": int(fleet["Total_Purchase_Cost_INR"].sum()),
        "total_expected_revenue_INR": int(fleet["Expected_Annual_Revenue_INR"].sum()),
        "top_categories": fleet.head(5)[["Asset_Category", "Recommended_Purchase_Units",
                                          "Expected_Annual_Revenue_INR", "Payback_Months"]].to_dict("records"),
    })
    return fleet


# ═══════════════════════════════════════════════════════════════════════════════
# NEW 3: CROSS-COUNTRY PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
def module_country_performance(txn, maint, util):
    print("  [NEW-3] Cross-Country Performance Analysis...")
    country_rev = txn.groupby("Country").agg(
        Total_Revenue_INR=("Gross_Revenue_INR", "sum"),
        Total_Profit_INR=("Net_Profit_INR", "sum"),
        Avg_Margin_Pct=("Rental_Margin_Pct", "mean"),
        Total_Rentals=("Transaction_ID", "count"),
        Avg_Discount_Pct=("Discount_Pct", "mean"),
    ).reset_index()

    country_maint = maint.groupby("Country").agg(
        Total_Maint_Cost_INR=("Maintenance_Cost_INR", "sum"),
        Total_Downtime_Hours=("Downtime_Hours", "sum"),
        Avg_Failure_Prob=("Failure_Probability", "mean"),
    ).reset_index()

    country_util = util.groupby("Country").agg(
        Avg_Utilization_Pct=("Utilization_Rate_Pct", "mean"),
        Total_Revenue_Loss_INR=("Revenue_Loss_From_Idle_INR", "sum"),
    ).reset_index()

    country = country_rev.merge(country_maint, on="Country", how="left")
    country = country.merge(country_util, on="Country", how="left")
    country["Revenue_Share_Pct"] = (country["Total_Revenue_INR"] / country["Total_Revenue_INR"].sum() * 100).round(2)
    country["Growth_Potential"] = pd.cut(
        country["Avg_Utilization_Pct"],
        bins=[0, 55, 70, 100],
        labels=["High Opportunity", "Moderate", "Saturated"]
    ).astype(str)

    path = os.path.join(PROC_DIR, "country_performance.csv")
    country.to_csv(path, index=False)
    save_report("module_country_performance", country.to_dict("records"))
    return country


# ═══════════════════════════════════════════════════════════════════════════════
# NEW 4: BRANCH PROFITABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def module_branch_profitability(txn, util, assets):
    print("  [NEW-4] Branch Profitability Analysis...")
    branch = txn.groupby(["Country", "Asset_Category"]).agg(
        Total_Revenue_INR=("Gross_Revenue_INR", "sum"),
        Total_Profit_INR=("Net_Profit_INR", "sum"),
        Avg_Margin_Pct=("Rental_Margin_Pct", "mean"),
        Total_Rentals=("Transaction_ID", "count"),
    ).reset_index()

    branch_util = util.groupby(["Country", "Asset_Category"]).agg(
        Avg_Utilization_Pct=("Utilization_Rate_Pct", "mean"),
        Total_Idle_Revenue_Loss_INR=("Revenue_Loss_From_Idle_INR", "sum"),
    ).reset_index()

    branch = branch.merge(branch_util, on=["Country", "Asset_Category"], how="left")
    branch["Profit_Rank"] = branch["Total_Profit_INR"].rank(ascending=False, method="min").astype(int)
    branch["Needs_Fleet_Transfer"] = (branch["Avg_Utilization_Pct"] < 45).astype(int)

    path = os.path.join(PROC_DIR, "branch_profitability.csv")
    branch.to_csv(path, index=False)
    save_report("module_branch_profitability", {
        "total_branches_analyzed": len(branch),
        "branches_needing_fleet_transfer": int(branch["Needs_Fleet_Transfer"].sum()),
        "top_10_branches": branch.sort_values("Total_Profit_INR", ascending=False).head(10)[
            ["Country", "Asset_Category", "Total_Revenue_INR", "Total_Profit_INR", "Avg_Utilization_Pct"]
        ].to_dict("records"),
    })
    return branch


# ═══════════════════════════════════════════════════════════════════════════════
# NEW 5: PROJECT OPPORTUNITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def module_project_opportunity(txn):
    print("  [NEW-5] Project Opportunity Analysis...")
    proj = txn.groupby("Project_Type").agg(
        Total_Revenue_INR=("Gross_Revenue_INR", "sum"),
        Total_Profit_INR=("Net_Profit_INR", "sum"),
        Avg_Margin_Pct=("Rental_Margin_Pct", "mean"),
        Total_Rentals=("Transaction_ID", "count"),
        Avg_Rental_Days=("Rental_Days", "mean"),
        Avg_Discount_Pct=("Discount_Pct", "mean"),
    ).reset_index()

    proj["ROI_Score"] = (proj["Avg_Margin_Pct"] * 0.4 + proj["Avg_Rental_Days"] * 0.3 +
                         proj["Total_Rentals"] / proj["Total_Rentals"].max() * 100 * 0.3).round(2)
    proj = proj.sort_values("Total_Profit_INR", ascending=False)

    path = os.path.join(PROC_DIR, "project_opportunity.csv")
    proj.to_csv(path, index=False)
    save_report("module_project_opportunity", {
        "top_5_sectors": proj.head(5)[["Project_Type", "Total_Revenue_INR",
                                        "Avg_Margin_Pct", "ROI_Score"]].to_dict("records"),
        "total_revenue_INR": int(proj["Total_Revenue_INR"].sum()),
    })
    return proj


# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL IMPACT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def compute_financial_impact(pricing, fleet, retire, idle, cust):
    print("\n  [IMPACT] Computing Financial Impact Summary...")
    pricing_gain = int(abs(pricing["Annual_Revenue_Gain_INR"].sum())) if len(pricing) else 0
    fleet_gain = int(fleet["Expected_Annual_Revenue_INR"].sum()) if len(fleet) else 0
    retirement_savings = int(retire["Annual_Savings_If_Replaced_INR"].sum()) if len(retire) else 0
    idle_recovery = int(idle["Total_Revenue_Loss_INR"].sum() * 0.35) if len(idle) else 0
    churn_retention = int(cust["Revenue_At_Risk_INR"].sum() * 0.40) if "Revenue_At_Risk_INR" in cust.columns else 0
    downtime_savings = int(fleet_gain * 0.12)  # 12% of fleet expansion revenue as downtime proxy

    total = pricing_gain + fleet_gain + retirement_savings + idle_recovery + churn_retention + downtime_savings

    impact = {
        "Revenue_Increase_From_Pricing_INR": pricing_gain,
        "Revenue_From_Fleet_Expansion_INR": fleet_gain,
        "Savings_From_Asset_Retirement_INR": retirement_savings,
        "Revenue_Recovery_From_Idle_Assets_INR": idle_recovery,
        "Revenue_Retention_From_Churn_Prevention_INR": churn_retention,
        "Savings_From_Downtime_Reduction_INR": downtime_savings,
        "Total_Annual_Profit_Improvement_INR": total,
        "Total_Annual_Profit_Improvement_Crore": round(total / 1e7, 2),
    }

    path = os.path.join(PROC_DIR, "financial_impact_summary.csv")
    pd.DataFrame([impact]).to_csv(path, index=False)
    save_report("financial_impact_summary", impact)

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │          FINANCIAL IMPACT SUMMARY                   │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Pricing Optimization:        {cr(pricing_gain):>20}  │")
    print(f"  │  Fleet Expansion Revenue:     {cr(fleet_gain):>20}  │")
    print(f"  │  Asset Retirement Savings:    {cr(retirement_savings):>20}  │")
    print(f"  │  Idle Asset Recovery:         {cr(idle_recovery):>20}  │")
    print(f"  │  Customer Retention:          {cr(churn_retention):>20}  │")
    print(f"  │  Downtime Reduction:          {cr(downtime_savings):>20}  │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  TOTAL ANNUAL OPPORTUNITY:    {cr(total):>20}  │")
    print(f"  └─────────────────────────────────────────────────────┘")
    return impact


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  RENTAL PROFIT OPTIMIZATION — BUSINESS ANALYTICS")
    print("=" * 70)

    assets = load("asset_master_processed")
    customers = load("customers_processed")
    txn = load("rental_transactions_processed")
    maint = load("maintenance_records_processed")
    util = load("asset_utilization_processed")

    print("\n[CORE MODULES]")
    asset_profit = module_a_top_profit_assets(txn, assets)
    idle_assets = module_b_idle_assets(util, assets)
    module_c_location_utilization(util)
    cust_rev = module_d_top_customers(txn, customers)
    module_e_maintenance_drivers(maint, assets)
    retire = module_f_replacement_candidates(assets, maint)
    module_g_category_profitability(txn)
    module_h_revenue_decline(txn)

    print("\n[NEW MODULES]")
    pricing = module_pricing_optimization(txn, assets)
    fleet = module_fleet_expansion(util, assets)
    module_country_performance(txn, maint, util)
    module_branch_profitability(txn, util, assets)
    module_project_opportunity(txn)

    customers_processed = load("customers_processed")
    compute_financial_impact(pricing, fleet, retire, idle_assets, customers_processed)

    print("\n" + "=" * 70)
    print("  ✅ BUSINESS ANALYTICS COMPLETE")
    print("  Next: python scripts/04_ml_models.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
