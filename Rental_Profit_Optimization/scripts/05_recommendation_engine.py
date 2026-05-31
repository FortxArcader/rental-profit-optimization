"""
===============================================================================
RENTAL BUSINESS PROFIT OPTIMIZATION PLATFORM
Script 05: Business Rule Engine & AI Recommendation System
===============================================================================
Generates ranked executive recommendations with ₹ impact estimates.

Recommendation Types:
  1. Pricing Optimization (underpriced assets)
  2. Fleet Relocation (idle assets to high-demand locations)
  3. Fleet Expansion (high-ROI purchase recommendations)
  4. Customer Retention (churn prevention actions)
  5. Preventive Maintenance (high-risk assets)
  6. Asset Retirement (loss-making assets)
  7. Cross-Country Expansion (growth opportunities)
  8. Project Sector Focus (high-ROI project types)
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
    if not os.path.exists(path):
        print(f"  ⚠ File not found: {name}.csv — skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Loaded {name}  [{len(df):,} rows]")
    return df


def cr(val):
    if val >= 1e7:
        return f"₹{val/1e7:.2f} Crore"
    elif val >= 1e5:
        return f"₹{val/1e5:.2f} Lakh"
    else:
        return f"₹{val:,.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 1: PRICING OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def rule_pricing_optimization(txn, assets):
    recs = []
    if txn.empty or assets.empty:
        return recs

    underpriced = txn[txn["Underpriced_Flag"] == 1].copy()
    by_cat_country = underpriced.groupby(["Asset_Category", "Country"]).agg(
        Transactions=("Transaction_ID", "count"),
        Avg_Current_Rate=("Effective_Daily_Rate_INR", "mean"),
        Avg_Competitor_Rate=("Competitor_Rate_INR", "mean"),
        Total_Opportunity=("Pricing_Opportunity_INR", "sum"),
    ).reset_index()

    by_cat_country = by_cat_country[by_cat_country["Total_Opportunity"] > 100000]
    by_cat_country = by_cat_country.sort_values("Total_Opportunity", ascending=False)

    for _, row in by_cat_country.head(8).iterrows():
        annual_gain = int(row["Total_Opportunity"] * 4)
        recommended_rate = int(row["Avg_Competitor_Rate"] * 0.97)
        recs.append({
            "Recommendation_ID": f"REC-PRICE-{len(recs)+1:03d}",
            "Category": "💰 Pricing Optimization",
            "Asset_Category": row["Asset_Category"],
            "Country": row["Country"],
            "Action": f"Increase {row['Asset_Category']} daily rate in {row['Country']}",
            "Detail": (
                f"Current Rate: ₹{int(row['Avg_Current_Rate']):,}/day | "
                f"Recommended Rate: ₹{recommended_rate:,}/day | "
                f"Rate Increase: {((recommended_rate/row['Avg_Current_Rate'])-1)*100:.1f}%"
            ),
            "Expected_Annual_Gain_INR": annual_gain,
            "Expected_Annual_Gain_Label": cr(annual_gain),
            "Priority": "High" if annual_gain > 5000000 else "Medium",
            "Action_Owner": "Sales & Pricing Team",
            "Implementation_Timeline": "30 Days",
        })
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 2: FLEET RELOCATION (Idle → High Demand)
# ═══════════════════════════════════════════════════════════════════════════════
def rule_fleet_relocation(util, assets):
    recs = []
    if util.empty or assets.empty:
        return recs

    idle_by_loc = util.groupby(["Asset_Category", "Country"]).agg(
        Avg_Util=("Utilization_Rate_Pct", "mean"),
        Total_Assets=("Asset_ID", "nunique"),
        Total_Idle_Loss=("Revenue_Loss_From_Idle_INR", "sum"),
    ).reset_index()

    idle_locs = idle_by_loc[idle_by_loc["Avg_Util"] < 45].sort_values("Total_Idle_Loss", ascending=False)
    high_demand = idle_by_loc[idle_by_loc["Avg_Util"] > 78].sort_values("Avg_Util", ascending=False)

    for _, idle_row in idle_locs.head(5).iterrows():
        if high_demand.empty:
            continue
        target = high_demand[high_demand["Asset_Category"] == idle_row["Asset_Category"]]
        if target.empty:
            target = high_demand.head(1)
        target_row = target.iloc[0]

        units = max(int(idle_row["Total_Assets"] * 0.30), 5)
        gain = int(idle_row["Total_Idle_Loss"] * 0.40)

        recs.append({
            "Recommendation_ID": f"REC-RELOC-{len(recs)+1:03d}",
            "Category": "🚛 Fleet Relocation",
            "Asset_Category": idle_row["Asset_Category"],
            "Country": idle_row["Country"],
            "Action": f"Relocate idle {idle_row['Asset_Category']}s from {idle_row['Country']}",
            "Detail": (
                f"Relocate ~{units} units from {idle_row['Country']} "
                f"(Util: {idle_row['Avg_Util']:.1f}%) to {target_row['Country']} "
                f"(Util: {target_row['Avg_Util']:.1f}%)"
            ),
            "Expected_Annual_Gain_INR": gain,
            "Expected_Annual_Gain_Label": cr(gain),
            "Priority": "High" if gain > 3000000 else "Medium",
            "Action_Owner": "Fleet & Operations Manager",
            "Implementation_Timeline": "45 Days",
        })
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 3: FLEET EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════
def rule_fleet_expansion(fleet_df):
    recs = []
    if fleet_df.empty:
        return recs

    for _, row in fleet_df.head(6).iterrows():
        gain = int(row["Expected_Annual_Revenue_INR"]) if "Expected_Annual_Revenue_INR" in row else 0
        recs.append({
            "Recommendation_ID": f"REC-FLEET-{len(recs)+1:03d}",
            "Category": "📈 Fleet Expansion",
            "Asset_Category": row["Asset_Category"],
            "Country": "All Countries",
            "Action": f"Purchase {int(row['Recommended_Purchase_Units'])} additional {row['Asset_Category']}s",
            "Detail": (
                f"Avg Utilization: {row['Avg_Utilization_Pct']:.1f}% | "
                f"Investment: {cr(int(row['Total_Purchase_Cost_INR']))} | "
                f"Payback: {row['Payback_Months']:.0f} months"
            ),
            "Expected_Annual_Gain_INR": gain,
            "Expected_Annual_Gain_Label": cr(gain),
            "Priority": "High" if gain > 10000000 else "Medium",
            "Action_Owner": "CEO / CFO / Procurement",
            "Implementation_Timeline": "90 Days",
        })
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 4: CUSTOMER RETENTION
# ═══════════════════════════════════════════════════════════════════════════════
def rule_customer_retention(churn_df, customers_df):
    recs = []
    if churn_df.empty:
        return recs

    critical = churn_df[churn_df["Churn_Risk_Label"] == "Critical"].sort_values(
        "Revenue_At_Risk_INR", ascending=False
    ).head(50)

    if len(critical) == 0:
        critical = churn_df[churn_df["Churn_Score_Predicted"] > 0.70].sort_values(
            "Revenue_At_Risk_INR", ascending=False
        ).head(50)

    total_at_risk = int(critical["Revenue_At_Risk_INR"].sum()) if "Revenue_At_Risk_INR" in critical.columns else 0
    gain = int(total_at_risk * 0.55)

    if gain > 0:
        recs.append({
            "Recommendation_ID": "REC-CHURN-001",
            "Category": "🤝 Customer Retention",
            "Asset_Category": "All Categories",
            "Country": "All Countries",
            "Action": f"Launch retention program for top {len(critical)} at-risk customers",
            "Detail": (
                f"Critical churn risk customers: {len(critical)} | "
                f"Total revenue at risk: {cr(total_at_risk)} | "
                f"Recommended: Priority calls + 5-10% loyalty discount"
            ),
            "Expected_Annual_Gain_INR": gain,
            "Expected_Annual_Gain_Label": cr(gain),
            "Priority": "Critical",
            "Action_Owner": "Sales Director / Account Managers",
            "Implementation_Timeline": "7 Days",
        })

    # By country
    if "Country" in churn_df.columns and "Revenue_At_Risk_INR" in churn_df.columns:
        country_risk = churn_df[churn_df["Churn_Score_Predicted"] > 0.60].groupby("Country").agg(
            Customers_At_Risk=("Customer_ID", "count"),
            Total_Revenue_At_Risk=("Revenue_At_Risk_INR", "sum"),
        ).reset_index().sort_values("Total_Revenue_At_Risk", ascending=False)

        for _, crow in country_risk.head(3).iterrows():
            gain_c = int(crow["Total_Revenue_At_Risk"] * 0.50)
            if gain_c > 500000:
                recs.append({
                    "Recommendation_ID": f"REC-CHURN-{len(recs)+1:03d}",
                    "Category": "🤝 Customer Retention",
                    "Asset_Category": "All Categories",
                    "Country": crow["Country"],
                    "Action": f"Retention drive for {crow['Country']} market",
                    "Detail": f"{crow['Customers_At_Risk']:,} customers at risk | Revenue at risk: {cr(crow['Total_Revenue_At_Risk'])}",
                    "Expected_Annual_Gain_INR": gain_c,
                    "Expected_Annual_Gain_Label": cr(gain_c),
                    "Priority": "High",
                    "Action_Owner": "Country Sales Manager",
                    "Implementation_Timeline": "14 Days",
                })
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 5: PREVENTIVE MAINTENANCE
# ═══════════════════════════════════════════════════════════════════════════════
def rule_preventive_maintenance(maint_scores):
    recs = []
    if maint_scores.empty:
        return recs

    critical = maint_scores[
        maint_scores.get("Failure_Risk_Tier", pd.Series(dtype=str)) == "Critical"
    ] if "Failure_Risk_Tier" in maint_scores.columns else pd.DataFrame()

    if critical.empty and "Failure_Probability_Predicted" in maint_scores.columns:
        critical = maint_scores[maint_scores["Failure_Probability_Predicted"] > 0.80]

    if len(critical) == 0:
        return recs

    total_savings = int(critical["Preventive_Savings_Potential_INR"].sum()) if "Preventive_Savings_Potential_INR" in critical.columns else 0

    by_cat = critical.groupby("Asset_Category").agg(
        Count=("Asset_ID", "count"),
        Total_Savings=("Preventive_Savings_Potential_INR", "sum"),
    ).reset_index().sort_values("Total_Savings", ascending=False) if "Asset_Category" in critical.columns else pd.DataFrame()

    for _, row in by_cat.head(5).iterrows():
        gain = int(row["Total_Savings"])
        recs.append({
            "Recommendation_ID": f"REC-MAINT-{len(recs)+1:03d}",
            "Category": "🔧 Preventive Maintenance",
            "Asset_Category": row["Asset_Category"],
            "Country": "All Countries",
            "Action": f"Schedule immediate preventive service for {int(row['Count'])} {row['Asset_Category']}s",
            "Detail": (
                f"Critical failure risk assets: {int(row['Count'])} | "
                f"Potential breakdown cost avoided: {cr(gain)}"
            ),
            "Expected_Annual_Gain_INR": gain,
            "Expected_Annual_Gain_Label": cr(gain),
            "Priority": "Critical",
            "Action_Owner": "Maintenance Manager",
            "Implementation_Timeline": "Immediate",
        })
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 6: ASSET RETIREMENT
# ═══════════════════════════════════════════════════════════════════════════════
def rule_asset_retirement(retire_df):
    recs = []
    if retire_df.empty:
        return recs

    by_cat = retire_df.groupby("Asset_Category").agg(
        Count=("Asset_ID", "count"),
        Total_Savings=("Annual_Savings_If_Replaced_INR", "sum"),
        Total_Replace_Cost=("Replacement_Cost_INR", "sum"),
    ).reset_index().sort_values("Total_Savings", ascending=False)

    for _, row in by_cat.head(5).iterrows():
        gain = int(row["Total_Savings"])
        recs.append({
            "Recommendation_ID": f"REC-RETIRE-{len(recs)+1:03d}",
            "Category": "🗑️ Asset Retirement",
            "Asset_Category": row["Asset_Category"],
            "Country": "All Countries",
            "Action": f"Retire {int(row['Count'])} aging {row['Asset_Category']}s",
            "Detail": (
                f"Assets recommended for retirement: {int(row['Count'])} | "
                f"Annual maintenance savings: {cr(gain)} | "
                f"Replacement investment: {cr(int(row['Total_Replace_Cost']))}"
            ),
            "Expected_Annual_Gain_INR": gain,
            "Expected_Annual_Gain_Label": cr(gain),
            "Priority": "Medium",
            "Action_Owner": "Fleet Manager / CFO",
            "Implementation_Timeline": "60 Days",
        })
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# RULE 7: COUNTRY EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════
def rule_country_expansion(country_df):
    recs = []
    if country_df.empty or "Growth_Potential" not in country_df.columns:
        return recs

    high_opp = country_df[country_df["Growth_Potential"] == "High Opportunity"].sort_values(
        "Total_Revenue_Loss_INR", ascending=False
    )

    for _, row in high_opp.head(3).iterrows():
        gain = int(row.get("Total_Revenue_Loss_INR", 0) * 0.35)
        if gain > 100000:
            recs.append({
                "Recommendation_ID": f"REC-EXPAND-{len(recs)+1:03d}",
                "Category": "🌍 Country Expansion",
                "Asset_Category": "All Categories",
                "Country": row["Country"],
                "Action": f"Increase fleet deployment in {row['Country']}",
                "Detail": (
                    f"Current utilization: {row.get('Avg_Utilization_Pct', 0):.1f}% | "
                    f"Revenue loss from idle: {cr(int(row.get('Total_Revenue_Loss_INR', 0)))} | "
                    f"Growth potential: High"
                ),
                "Expected_Annual_Gain_INR": gain,
                "Expected_Annual_Gain_Label": cr(gain),
                "Priority": "Medium",
                "Action_Owner": "CEO / Business Development",
                "Implementation_Timeline": "120 Days",
            })
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLIDATE & RANK
# ═══════════════════════════════════════════════════════════════════════════════
def consolidate_recommendations(all_recs):
    df = pd.DataFrame(all_recs)
    if df.empty:
        return df

    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    df["Priority_Rank"] = df["Priority"].map(priority_order).fillna(3)
    df = df.sort_values(["Priority_Rank", "Expected_Annual_Gain_INR"], ascending=[True, False])
    df["Rank"] = range(1, len(df) + 1)
    df = df.drop(columns=["Priority_Rank"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  RENTAL PROFIT OPTIMIZATION — AI RECOMMENDATION ENGINE")
    print("=" * 70)

    txn = load("rental_transactions_processed")
    assets = load("asset_master_processed")
    util = load("asset_utilization_processed")
    customers = load("customers_processed")
    fleet_df = load("fleet_expansion")
    retire_df = load("asset_retirement_candidates")
    country_df = load("country_performance")
    churn_df = load("churn_predictions")
    maint_scores = load("predictive_maintenance_scores")

    print("\n[GENERATING RECOMMENDATIONS]")
    all_recs = []
    all_recs += rule_pricing_optimization(txn, assets)
    print(f"  ✓ Pricing recommendations: {len([r for r in all_recs if 'PRICE' in r['Recommendation_ID']])}")

    reloc_recs = rule_fleet_relocation(util, assets)
    all_recs += reloc_recs
    print(f"  ✓ Fleet relocation recommendations: {len(reloc_recs)}")

    fleet_recs = rule_fleet_expansion(fleet_df)
    all_recs += fleet_recs
    print(f"  ✓ Fleet expansion recommendations: {len(fleet_recs)}")

    churn_recs = rule_customer_retention(churn_df, customers)
    all_recs += churn_recs
    print(f"  ✓ Customer retention recommendations: {len(churn_recs)}")

    maint_recs = rule_preventive_maintenance(maint_scores)
    all_recs += maint_recs
    print(f"  ✓ Preventive maintenance recommendations: {len(maint_recs)}")

    retire_recs = rule_asset_retirement(retire_df)
    all_recs += retire_recs
    print(f"  ✓ Asset retirement recommendations: {len(retire_recs)}")

    expand_recs = rule_country_expansion(country_df)
    all_recs += expand_recs
    print(f"  ✓ Country expansion recommendations: {len(expand_recs)}")

    # Consolidate
    recs_df = consolidate_recommendations(all_recs)
    path = os.path.join(PROC_DIR, "recommendations.csv")
    recs_df.to_csv(path, index=False)
    print(f"\n  ✓ Total recommendations generated: {len(recs_df)}")

    total_gain = int(recs_df["Expected_Annual_Gain_INR"].sum()) if len(recs_df) > 0 else 0
    print(f"\n  {'='*60}")
    print(f"  TOTAL ANNUAL PROFIT OPPORTUNITY: {cr(total_gain)}")
    print(f"  {'='*60}")

    # Print top 10
    print("\n  TOP 10 EXECUTIVE RECOMMENDATIONS:")
    print(f"  {'─'*70}")
    for _, row in recs_df.head(10).iterrows():
        print(f"\n  #{int(row['Rank'])} [{row['Priority']}] {row['Category']}")
        print(f"     Action: {row['Action']}")
        print(f"     Impact: {row['Expected_Annual_Gain_Label']}")
        print(f"     Owner:  {row['Action_Owner']} | Timeline: {row['Implementation_Timeline']}")

    print("\n" + "=" * 70)
    print("  ✅ RECOMMENDATION ENGINE COMPLETE")
    print("  Next: python scripts/06_powerbi_exports.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
