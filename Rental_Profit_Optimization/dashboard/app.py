"""
===============================================================================
RENTAL BUSINESS PROFIT OPTIMIZATION PLATFORM
Streamlit Executive Dashboard — app.py
===============================================================================
8-Page Executive Dashboard:
  1. CEO Profit Command Center
  2. Revenue Analytics
  3. Asset Analytics
  4. Maintenance Analytics
  5. Customer Analytics
  6. Profit Analytics
  7. AI Recommendations
  8. Cross-Country Analysis
===============================================================================
"""

import os
import sys

# ── Auto-bootstrap: generate data if not already present ─────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
SCRIPTS_DIR = os.path.join(BASE_DIR, "..", "scripts")
_sentinel = os.path.join(PROC_DIR, "rental_transactions_processed.csv")

if not os.path.exists(_sentinel):
    import streamlit as st
    import importlib.util

    def _run(name):
        path = os.path.join(SCRIPTS_DIR, name)
        spec = importlib.util.spec_from_file_location("m", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    with st.spinner("⚙️ First-time setup: generating data & training models (~3 min)..."):
        for s in ["01_generate_data.py", "02_preprocess_data.py",
                  "03_business_analytics.py", "04_ml_models.py",
                  "05_recommendation_engine.py", "06_powerbi_exports.py"]:
            _run(s)
    st.success("✅ Setup complete! Reloading dashboard...")
    st.rerun()
import warnings
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rental Profit Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "..", "reports")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0a0e1a; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
  }
  [data-testid="stSidebar"] .stMarkdown h2 {
    color: #63b3ed; font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase;
  }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(135deg, rgba(15,25,50,0.95) 0%, rgba(10,18,40,0.95) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 20px 22px;
    margin: 6px 0;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .kpi-card:hover {
    border-color: rgba(99,179,237,0.5);
    box-shadow: 0 8px 32px rgba(99,179,237,0.15);
    transform: translateY(-2px);
  }
  .kpi-label {
    color: #718096; font-size: 0.75rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;
  }
  .kpi-value {
    color: #f0f4f8; font-size: 1.6rem; font-weight: 800; line-height: 1;
  }
  .kpi-value-green { color: #48bb78; }
  .kpi-value-blue  { color: #63b3ed; }
  .kpi-value-gold  { color: #f6c90e; }
  .kpi-value-red   { color: #fc8181; }
  .kpi-value-purple{ color: #b794f4; }
  .kpi-sub {
    color: #718096; font-size: 0.72rem; margin-top: 6px;
  }
  .kpi-delta-up   { color: #48bb78; font-size: 0.78rem; font-weight: 600; }
  .kpi-delta-down { color: #fc8181; font-size: 0.78rem; font-weight: 600; }

  /* Section Headers */
  .section-header {
    background: linear-gradient(135deg, rgba(99,179,237,0.08), rgba(72,187,120,0.08));
    border-left: 4px solid #63b3ed;
    border-radius: 0 12px 12px 0;
    padding: 12px 20px;
    margin: 20px 0 16px 0;
  }
  .section-header h3 {
    color: #e2e8f0; font-size: 1.1rem; font-weight: 700; margin: 0;
  }

  /* Page Header */
  .page-hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2d5a 50%, #0f2040 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 30px 36px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .page-hero h1 {
    color: #f0f4f8; font-size: 1.8rem; font-weight: 800; margin: 0 0 6px 0;
    background: linear-gradient(135deg, #63b3ed, #48bb78);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .page-hero p { color: #718096; font-size: 0.9rem; margin: 0; }

  /* Recommendation Cards */
  .rec-critical {
    background: linear-gradient(135deg, rgba(252,129,74,0.1), rgba(252,129,74,0.05));
    border: 1px solid rgba(252,129,74,0.4);
    border-radius: 12px; padding: 16px 20px; margin: 8px 0;
  }
  .rec-high {
    background: linear-gradient(135deg, rgba(99,179,237,0.1), rgba(99,179,237,0.05));
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 12px; padding: 16px 20px; margin: 8px 0;
  }
  .rec-medium {
    background: linear-gradient(135deg, rgba(72,187,120,0.08), rgba(72,187,120,0.04));
    border: 1px solid rgba(72,187,120,0.25);
    border-radius: 12px; padding: 16px 20px; margin: 8px 0;
  }
  .rec-title { color: #e2e8f0; font-weight: 700; font-size: 0.95rem; }
  .rec-detail { color: #718096; font-size: 0.82rem; margin-top: 4px; }
  .rec-impact { color: #48bb78; font-weight: 700; font-size: 1.05rem; }
  .rec-badge-critical {
    background: #fc8181; color: #fff; border-radius: 20px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 700;
  }
  .rec-badge-high {
    background: #63b3ed; color: #fff; border-radius: 20px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 700;
  }
  .rec-badge-medium {
    background: #48bb78; color: #fff; border-radius: 20px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 700;
  }

  /* Streamlit overrides */
  div[data-testid="metric-container"] {
    background: rgba(15,25,50,0.8);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px; padding: 16px;
  }
  .stDataFrame { border-radius: 12px; overflow: hidden; }
  h1, h2, h3 { color: #e2e8f0 !important; }
  .stSelectbox label, .stMultiSelect label { color: #718096 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_data(name, nrows=None):
    path = os.path.join(PROC_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_report(name):
    path = os.path.join(REPORT_DIR, f"{name}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ── Plotly Theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#a0aec0", size=12),
    title_font=dict(color="#e2e8f0", size=15, family="Inter"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a0aec0")),
    xaxis=dict(gridcolor="rgba(99,179,237,0.08)", color="#a0aec0", showline=False),
    yaxis=dict(gridcolor="rgba(99,179,237,0.08)", color="#a0aec0", showline=False),
    colorway=["#63b3ed", "#48bb78", "#f6c90e", "#fc8181", "#b794f4",
               "#fbd38d", "#4fd1c5", "#feb2b2", "#9ae6b4", "#90cdf4"],
    margin=dict(t=50, l=10, r=10, b=10),
)

COLOR_SEQ = ["#63b3ed", "#48bb78", "#f6c90e", "#fc8181", "#b794f4",
             "#fbd38d", "#4fd1c5", "#feb2b2"]


def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_cr(val):
    if pd.isna(val):
        return "N/A"
    val = float(val)
    if abs(val) >= 1e7:
        return f"₹{val/1e7:.2f} Cr"
    elif abs(val) >= 1e5:
        return f"₹{val/1e5:.2f} L"
    else:
        return f"₹{val:,.0f}"


def kpi_card(label, value, color="kpi-value-blue", sub="", delta=""):
    delta_html = ""
    if delta:
        cls = "kpi-delta-up" if str(delta).startswith("+") else "kpi-delta-down"
        delta_html = f'<div class="{cls}">{delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="{color}">{value}</div>
      {delta_html}
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title, icon="📊"):
    st.markdown(f"""
    <div class="section-header">
      <h3>{icon} {title}</h3>
    </div>
    """, unsafe_allow_html=True)


def page_hero(title, subtitle):
    st.markdown(f"""
    <div class="page-hero">
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ RENTAL PROFIT")
    st.markdown("**AI Command Center**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏆 CEO Command Center",
            "📈 Revenue Analytics",
            "🏗️ Asset Analytics",
            "🔧 Maintenance Analytics",
            "👥 Customer Analytics",
            "💰 Profit Analytics",
            "🤖 AI Recommendations",
            "🌍 Cross-Country Analysis",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("## Filters")

    # Global filters loaded lazily
    txn_sample = load_data("rental_transactions_processed", nrows=50000)

    country_opts = ["All Countries"]
    cat_opts = ["All Categories"]

    if not txn_sample.empty:
        if "Country" in txn_sample.columns:
            country_opts += sorted(txn_sample["Country"].dropna().unique().tolist())
        if "Asset_Category" in txn_sample.columns:
            cat_opts += sorted(txn_sample["Asset_Category"].dropna().unique().tolist())

    selected_country = st.selectbox("🌍 Country", country_opts)
    selected_cat = st.selectbox("🏗️ Asset Category", cat_opts)

    year_opts = ["All Years"]
    if not txn_sample.empty and "Year" in txn_sample.columns:
        year_opts += sorted(txn_sample["Year"].dropna().unique().astype(int).astype(str).tolist())
    selected_year = st.selectbox("📅 Year", year_opts)

    st.markdown("---")
    st.markdown("""
    <div style="color:#4a5568; font-size:0.72rem; text-align:center;">
      India 🇮🇳 | UAE 🇦🇪 | Saudi 🇸🇦<br>Qatar 🇶🇦 | Oman 🇴🇲 | Kuwait 🇰🇼<br><br>
      <b style="color:#63b3ed;">Built with AI & Analytics</b>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER UTILITY
# ═══════════════════════════════════════════════════════════════════════════════
def apply_filters(df):
    if df.empty:
        return df
    if selected_country != "All Countries" and "Country" in df.columns:
        df = df[df["Country"] == selected_country]
    if selected_cat != "All Categories" and "Asset_Category" in df.columns:
        df = df[df["Asset_Category"] == selected_cat]
    if selected_year != "All Years" and "Year" in df.columns:
        df = df[df["Year"].astype(str) == selected_year]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: CEO PROFIT COMMAND CENTER
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏆 CEO Command Center":
    page_hero(
        "CEO Profit Command Center",
        "Real-time executive intelligence across India & GCC | Industrial & Infrastructure Rentals"
    )

    # Load data
    txn = apply_filters(load_data("rental_transactions_processed", nrows=200000))
    util = apply_filters(load_data("asset_utilization_processed", nrows=100000))
    maint = apply_filters(load_data("maintenance_records_processed", nrows=100000))
    customers = load_data("customers_processed", nrows=50000)
    impact = load_data("financial_impact_summary")
    forecast = load_data("revenue_forecast_2025")
    recs = load_data("recommendations")

    # ── KPI Row 1: Core Business Metrics ────────────────────────────────────
    section_header("Current Business Performance", "📊")
    c1, c2, c3, c4, c5 = st.columns(5)

    total_rev = int(txn["Gross_Revenue_INR"].sum()) if not txn.empty else 0
    total_profit = int(txn["Net_Profit_INR"].sum()) if not txn.empty else 0
    avg_margin = round(txn["Rental_Margin_Pct"].mean(), 1) if not txn.empty else 0
    avg_util = round(util["Utilization_Rate_Pct"].mean(), 1) if not util.empty else 0
    total_rentals = len(txn) if not txn.empty else 0

    with c1:
        kpi_card("Total Revenue", fmt_cr(total_rev), "kpi-value-blue", "3-Year Portfolio")
    with c2:
        kpi_card("Net Profit", fmt_cr(total_profit), "kpi-value-green", "After All Costs")
    with c3:
        kpi_card("Profit Margin", f"{avg_margin:.1f}%", "kpi-value-gold", "Avg Rental Margin")
    with c4:
        kpi_card("Asset Utilization", f"{avg_util:.1f}%", "kpi-value-purple", "Fleet Average")
    with c5:
        kpi_card("Total Rentals", f"{total_rentals:,}", "kpi-value-blue", "Active Contracts")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Row 2: Opportunity Metrics ──────────────────────────────────────
    section_header("Annual Profit Opportunity", "💡")
    c1, c2, c3, c4, c5 = st.columns(5)

    total_opportunity = int(recs["Expected_Annual_Gain_INR"].sum()) if not recs.empty else 0
    pricing_opp = int(impact["Revenue_Increase_From_Pricing_INR"].iloc[0]) if not impact.empty and "Revenue_Increase_From_Pricing_INR" in impact.columns else 0
    fleet_opp = int(impact["Revenue_From_Fleet_Expansion_INR"].iloc[0]) if not impact.empty and "Revenue_From_Fleet_Expansion_INR" in impact.columns else 0
    maint_savings = int(impact["Savings_From_Downtime_Reduction_INR"].iloc[0]) if not impact.empty and "Savings_From_Downtime_Reduction_INR" in impact.columns else 0
    churn_opp = int(impact["Revenue_Retention_From_Churn_Prevention_INR"].iloc[0]) if not impact.empty and "Revenue_Retention_From_Churn_Prevention_INR" in impact.columns else 0

    with c1:
        kpi_card("Total Profit Opportunity", fmt_cr(total_opportunity), "kpi-value-gold",
                 "Annual AI-Identified Gains", "🎯 Action Required")
    with c2:
        kpi_card("Pricing Opportunity", fmt_cr(pricing_opp), "kpi-value-green",
                 "Underpriced Asset Revenue")
    with c3:
        kpi_card("Fleet Expansion Gain", fmt_cr(fleet_opp), "kpi-value-blue",
                 "New Asset ROI")
    with c4:
        kpi_card("Maintenance Savings", fmt_cr(maint_savings), "kpi-value-purple",
                 "Preventive Maintenance")
    with c5:
        kpi_card("Customer Retention", fmt_cr(churn_opp), "kpi-value-green",
                 "Revenue Saved from Churn")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Row 3: Risk Metrics ──────────────────────────────────────────────
    section_header("Risk & Loss Indicators", "⚠️")
    c1, c2, c3, c4, c5 = st.columns(5)

    idle_loss = int(util["Revenue_Loss_From_Idle_INR"].sum()) if not util.empty else 0
    total_maint_cost = int(maint["Maintenance_Cost_INR"].sum()) if not maint.empty else 0
    breakdowns = int(maint["Failure_Flag"].sum()) if not maint.empty and "Failure_Flag" in maint.columns else 0
    churn_customers = int((customers["Churn_Probability"] > 0.70).sum()) if not customers.empty else 0
    total_assets_raw = load_data("asset_master_processed", nrows=1000)
    idle_assets = int((total_assets_raw["Asset_Status"] == "Idle").sum()) if not total_assets_raw.empty else 0

    with c1:
        kpi_card("Idle Asset Loss", fmt_cr(idle_loss), "kpi-value-red", "Revenue Leakage")
    with c2:
        kpi_card("Maintenance Cost", fmt_cr(total_maint_cost), "kpi-value-red", "Total Spend")
    with c3:
        kpi_card("Equipment Breakdowns", f"{breakdowns:,}", "kpi-value-red", "Historical Events")
    with c4:
        kpi_card("High Churn Risk", f"{churn_customers:,}", "kpi-value-red", "Customers at Risk")
    with c5:
        kpi_card("Idle Assets", f"{idle_assets:,}", "kpi-value-red", "Zero Revenue")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue Forecast ─────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        section_header("2025 Revenue Forecast", "📈")
        if not forecast.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast["Month"], y=forecast["Upper_Bound_INR"],
                fill=None, mode="lines", line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast["Month"], y=forecast["Lower_Bound_INR"],
                fill="tonexty", mode="lines", line=dict(width=0),
                fillcolor="rgba(99,179,237,0.12)", name="Forecast Range"
            ))
            fig.add_trace(go.Scatter(
                x=forecast["Month"], y=forecast["Forecasted_Revenue_INR"],
                mode="lines+markers", name="Forecasted Revenue",
                line=dict(color="#63b3ed", width=3),
                marker=dict(size=8, color="#63b3ed"),
                text=[fmt_cr(v) for v in forecast["Forecasted_Revenue_INR"]],
                textposition="top center"
            ))
            fig.update_layout(title="Monthly Revenue Forecast — 2025", **PLOTLY_LAYOUT)
            fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Profit Mix", "🍩")
        if not txn.empty and "Asset_Category" in txn.columns:
            cat_profit = txn.groupby("Asset_Category")["Net_Profit_INR"].sum().reset_index()
            cat_profit = cat_profit[cat_profit["Net_Profit_INR"] > 0]
            fig2 = px.pie(
                cat_profit, values="Net_Profit_INR", names="Asset_Category",
                color_discrete_sequence=COLOR_SEQ, hole=0.55
            )
            fig2.update_traces(textinfo="percent+label", textfont_size=11)
            fig2.update_layout(title="Profit by Equipment", **PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Top Recommendations Preview ──────────────────────────────────────────
    section_header("Top 5 Executive Actions", "🎯")
    if not recs.empty:
        for _, row in recs.head(5).iterrows():
            priority = str(row.get("Priority", "Medium"))
            css_class = f"rec-{priority.lower()}"
            badge_class = f"rec-badge-{priority.lower()}"
            st.markdown(f"""
            <div class="{css_class}">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div class="rec-title">#{int(row.get('Rank',0))} {row.get('Action','')}</div>
                  <div class="rec-detail">{row.get('Detail','')}</div>
                  <div class="rec-detail" style="margin-top:4px;">
                    👤 {row.get('Action_Owner','')} &nbsp;|&nbsp; ⏱ {row.get('Implementation_Timeline','')}
                  </div>
                </div>
                <div style="text-align:right; min-width:130px;">
                  <div class="rec-impact">{row.get('Expected_Annual_Gain_Label','')}</div>
                  <span class="{badge_class}">{priority}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: REVENUE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Revenue Analytics":
    page_hero("Revenue Analytics", "Detailed revenue breakdown by time, region, category, and customer")

    txn = apply_filters(load_data("rental_transactions_processed", nrows=200000))
    if txn.empty:
        st.warning("⚠️ No transaction data found. Run script 01 and 02 first.")
        st.stop()

    txn["Rental_Start_Date"] = pd.to_datetime(txn["Rental_Start_Date"], errors="coerce")
    txn["Month"] = txn["Rental_Start_Date"].dt.to_period("M").astype(str)
    txn["Year"] = txn["Rental_Start_Date"].dt.year

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Revenue", fmt_cr(txn["Gross_Revenue_INR"].sum()), "kpi-value-blue")
    with c2:
        kpi_card("Total Profit", fmt_cr(txn["Net_Profit_INR"].sum()), "kpi-value-green")
    with c3:
        kpi_card("Avg Daily Rate", fmt_cr(txn["Effective_Daily_Rate_INR"].mean()), "kpi-value-gold")
    with c4:
        kpi_card("Avg Discount", f"{txn['Discount_Pct'].mean():.1f}%", "kpi-value-red", "Revenue Leakage")

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue Trend
    section_header("Revenue & Profit Trend", "📉")
    monthly = txn.groupby("Month").agg(
        Revenue=("Gross_Revenue_INR", "sum"),
        Profit=("Net_Profit_INR", "sum"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["Month"], y=monthly["Revenue"],
                         name="Revenue", marker_color="#63b3ed", opacity=0.8))
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Profit"],
                              name="Profit", line=dict(color="#48bb78", width=3),
                              mode="lines+markers", yaxis="y2"))
    fig.update_layout(
        title="Monthly Revenue vs Profit", yaxis2=dict(overlaying="y", side="right",
        showgrid=False, color="#a0aec0"), **PLOTLY_LAYOUT
    )
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section_header("Revenue by Country", "🌍")
        country_rev = txn.groupby("Country")["Gross_Revenue_INR"].sum().reset_index().sort_values("Gross_Revenue_INR", ascending=True)
        fig = px.bar(country_rev, x="Gross_Revenue_INR", y="Country",
                     orientation="h", color="Gross_Revenue_INR",
                     color_continuous_scale=["#0d1b3e", "#63b3ed"],
                     text=[fmt_cr(v) for v in country_rev["Gross_Revenue_INR"]])
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Revenue by Country", **PLOTLY_LAYOUT)
        fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Revenue by Asset Category", "🏗️")
        cat_rev = txn.groupby("Asset_Category")["Gross_Revenue_INR"].sum().reset_index().sort_values("Gross_Revenue_INR", ascending=False)
        fig = px.bar(cat_rev, x="Asset_Category", y="Gross_Revenue_INR",
                     color="Asset_Category", color_discrete_sequence=COLOR_SEQ,
                     text=[fmt_cr(v) for v in cat_rev["Gross_Revenue_INR"]])
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(title="Revenue by Equipment Type", **PLOTLY_LAYOUT, showlegend=False)
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section_header("Revenue by Project Type", "🏭")
        proj_rev = txn.groupby("Project_Type")["Gross_Revenue_INR"].sum().reset_index().sort_values("Gross_Revenue_INR", ascending=False).head(10)
        fig = px.bar(proj_rev, x="Gross_Revenue_INR", y="Project_Type",
                     orientation="h", color="Gross_Revenue_INR",
                     color_continuous_scale=["#0d1b3e", "#48bb78"],
                     text=[fmt_cr(v) for v in proj_rev["Gross_Revenue_INR"]])
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Top Project Types by Revenue", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        section_header("Revenue by Rental Mode", "📋")
        mode_rev = txn.groupby("Rental_Mode").agg(
            Revenue=("Gross_Revenue_INR", "sum"),
            Count=("Transaction_ID", "count"),
            Avg_Days=("Rental_Days", "mean"),
        ).reset_index()
        fig = px.pie(mode_rev, values="Revenue", names="Rental_Mode",
                     color_discrete_sequence=COLOR_SEQ, hole=0.5)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(title="Revenue Split by Rental Mode", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ASSET ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Asset Analytics":
    page_hero("Asset Analytics", "Fleet performance, utilization rates, idle analysis, and replacement insights")

    assets = apply_filters(load_data("asset_master_processed"))
    util = apply_filters(load_data("asset_utilization_processed", nrows=100000))
    idle_df = load_data("idle_assets")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Total Fleet", f"{len(assets):,}", "kpi-value-blue", "Active Assets")
    with c2:
        active = int((assets["Asset_Status"] == "Active").sum()) if not assets.empty else 0
        kpi_card("Active Assets", f"{active:,}", "kpi-value-green")
    with c3:
        idle = int((assets["Asset_Status"] == "Idle").sum()) if not assets.empty else 0
        kpi_card("Idle Assets", f"{idle:,}", "kpi-value-red", "Zero Revenue")
    with c4:
        maintenance_ct = int((assets["Asset_Status"] == "Under Maintenance").sum()) if not assets.empty else 0
        kpi_card("Under Maintenance", f"{maintenance_ct:,}", "kpi-value-gold")
    with c5:
        avg_util = round(util["Utilization_Rate_Pct"].mean(), 1) if not util.empty else 0
        kpi_card("Fleet Utilization", f"{avg_util:.1f}%", "kpi-value-purple", "Industry Avg: 68%")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_header("Utilization by Asset Category", "📊")
        if not util.empty:
            cat_util = util.groupby("Asset_Category").agg(
                Avg_Util=("Utilization_Rate_Pct", "mean"),
                Industry_Avg=("Industry_Avg_Util_Pct", "mean"),
            ).reset_index().sort_values("Avg_Util", ascending=False)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cat_util["Asset_Category"], y=cat_util["Avg_Util"],
                name="Current Utilization", marker_color="#63b3ed"
            ))
            fig.add_trace(go.Scatter(
                x=cat_util["Asset_Category"], y=cat_util["Industry_Avg"],
                name="Industry Average", mode="lines+markers",
                line=dict(color="#f6c90e", dash="dash", width=2),
                marker=dict(size=8)
            ))
            fig.update_layout(title="Utilization vs Industry Average", **PLOTLY_LAYOUT)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Fleet Status Distribution", "🔵")
        if not assets.empty:
            status_ct = assets["Asset_Status"].value_counts().reset_index()
            status_ct.columns = ["Status", "Count"]
            colors_map = {"Active": "#48bb78", "Idle": "#fc8181",
                          "Under Maintenance": "#f6c90e", "Retired": "#718096"}
            colors = [colors_map.get(s, "#63b3ed") for s in status_ct["Status"]]
            fig = px.pie(status_ct, values="Count", names="Status",
                         color_discrete_sequence=colors, hole=0.55)
            fig.update_traces(textinfo="percent+value")
            fig.update_layout(title="Asset Status Breakdown", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    section_header("Idle Asset Revenue Loss by Category & Country", "🚨")
    if not util.empty:
        idle_loss = util.groupby(["Asset_Category", "Country"]).agg(
            Total_Loss=("Revenue_Loss_From_Idle_INR", "sum"),
            Avg_Util=("Utilization_Rate_Pct", "mean"),
        ).reset_index()
        fig = px.treemap(
            idle_loss, path=["Country", "Asset_Category"],
            values="Total_Loss", color="Avg_Util",
            color_continuous_scale=["#fc8181", "#f6c90e", "#48bb78"],
            title="Idle Revenue Loss — Size = Revenue Lost | Color = Utilization Rate"
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section_header("Asset Age Distribution", "📅")
        if not assets.empty:
            fig = px.histogram(assets, x="Asset_Age_Years", nbins=30,
                               color_discrete_sequence=["#63b3ed"],
                               title="Fleet Age Profile")
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_xaxes(title="Asset Age (Years)")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        section_header("Daily Rate by Category", "💵")
        if not assets.empty:
            fig = px.box(assets, x="Asset_Category", y="Daily_Rental_Rate_INR",
                          color="Asset_Category", color_discrete_sequence=COLOR_SEQ,
                          title="Daily Rental Rate Distribution")
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    # Replacement Candidates
    retire_df = load_data("asset_retirement_candidates")
    if not retire_df.empty:
        section_header(f"Asset Replacement Candidates ({len(retire_df):,} Assets)", "🗑️")
        display_cols = ["Asset_ID", "Asset_Category", "Country", "Asset_Age_Years",
                        "Total_Breakdowns", "Annual_Savings_If_Replaced_INR"]
        display_cols = [c for c in display_cols if c in retire_df.columns]
        top_retire = retire_df[display_cols].head(20)
        if "Annual_Savings_If_Replaced_INR" in top_retire.columns:
            top_retire["Annual Savings"] = top_retire["Annual_Savings_If_Replaced_INR"].apply(fmt_cr)
        st.dataframe(top_retire.rename(columns={
            "Asset_ID": "Asset ID", "Asset_Category": "Category",
            "Asset_Age_Years": "Age (Yrs)", "Total_Breakdowns": "Breakdowns"
        }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MAINTENANCE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Maintenance Analytics":
    page_hero("Maintenance Analytics", "Downtime, costs, breakdown patterns, and predictive failure risk")

    maint = apply_filters(load_data("maintenance_records_processed", nrows=100000))
    maint_scores = load_data("predictive_maintenance_scores")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Maintenance Cost", fmt_cr(maint["Maintenance_Cost_INR"].sum()) if not maint.empty else "N/A", "kpi-value-red")
    with c2:
        total_downtime = round(maint["Downtime_Hours"].sum() / 24, 0) if not maint.empty else 0
        kpi_card("Total Downtime", f"{int(total_downtime):,} Days", "kpi-value-red")
    with c3:
        breakdowns = int(maint["Failure_Flag"].sum()) if not maint.empty and "Failure_Flag" in maint.columns else 0
        kpi_card("Total Breakdowns", f"{breakdowns:,}", "kpi-value-red")
    with c4:
        rev_lost = int(maint["Revenue_Lost_From_Downtime_INR"].sum()) if not maint.empty and "Revenue_Lost_From_Downtime_INR" in maint.columns else 0
        kpi_card("Revenue Lost (Downtime)", fmt_cr(rev_lost), "kpi-value-red")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_header("Maintenance Cost by Type", "🔩")
        if not maint.empty:
            cost_type = maint.groupby("Maintenance_Type").agg(
                Total_Cost=("Maintenance_Cost_INR", "sum"),
                Count=("Maintenance_ID", "count"),
            ).reset_index()
            fig = px.bar(cost_type, x="Maintenance_Type", y="Total_Cost",
                          color="Maintenance_Type", color_discrete_sequence=COLOR_SEQ,
                          text=[fmt_cr(v) for v in cost_type["Total_Cost"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Cost by Maintenance Type", **PLOTLY_LAYOUT, showlegend=False)
            fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Downtime by Asset Category", "⏰")
        if not maint.empty:
            down_cat = maint.groupby("Asset_Category")["Downtime_Hours"].sum().reset_index().sort_values("Downtime_Hours", ascending=True)
            fig = px.bar(down_cat, x="Downtime_Hours", y="Asset_Category",
                          orientation="h", color="Downtime_Hours",
                          color_continuous_scale=["#1a2d5a", "#fc8181"],
                          text=down_cat["Downtime_Hours"].apply(lambda x: f"{x/24:.0f}d"))
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Total Downtime by Category", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    section_header("Maintenance Cost vs Asset Age", "📈")
    if not maint.empty and "Asset_Age_At_Event" in maint.columns:
        maint["Age_Band"] = pd.cut(
            maint["Asset_Age_At_Event"],
            bins=[0, 3, 6, 10, 20],
            labels=["0-3 Years", "3-6 Years", "6-10 Years", "10+ Years"]
        )
        age_cost = maint.groupby("Age_Band", observed=True).agg(
            Avg_Cost=("Maintenance_Cost_INR", "mean"),
            Total_Cost=("Maintenance_Cost_INR", "sum"),
            Avg_Downtime=("Downtime_Hours", "mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=age_cost["Age_Band"].astype(str), y=age_cost["Total_Cost"],
                              name="Total Cost", marker_color="#fc8181", yaxis="y"))
        fig.add_trace(go.Scatter(x=age_cost["Age_Band"].astype(str), y=age_cost["Avg_Downtime"],
                                  name="Avg Downtime (hrs)", line=dict(color="#f6c90e", width=3),
                                  mode="lines+markers", yaxis="y2"))
        fig.update_layout(title="Maintenance Cost Increases with Asset Age",
                           yaxis2=dict(overlaying="y", side="right", showgrid=False, color="#a0aec0"),
                           **PLOTLY_LAYOUT)
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    # High risk assets
    section_header("High-Risk Assets (Predicted Failure >75%)", "🚨")
    if not maint_scores.empty and "Failure_Risk_Tier" in maint_scores.columns:
        critical = maint_scores[maint_scores["Failure_Risk_Tier"].isin(["Critical", "High Risk"])].copy()
        if len(critical) > 0:
            if selected_country != "All Countries" and "Country" in critical.columns:
                critical = critical[critical["Country"] == selected_country]
            if selected_cat != "All Categories" and "Asset_Category" in critical.columns:
                critical = critical[critical["Asset_Category"] == selected_cat]

            display_cols = ["Asset_ID", "Asset_Category", "Country", "Asset_Age_Years",
                             "Failure_Risk_Tier", "Failure_Probability_Predicted",
                             "Preventive_Savings_Potential_INR"]
            display_cols = [c for c in display_cols if c in critical.columns]
            top_risk = critical[display_cols].sort_values(
                "Failure_Probability_Predicted", ascending=False
            ).head(25)
            if "Failure_Probability_Predicted" in top_risk.columns:
                top_risk["Failure Risk %"] = (top_risk["Failure_Probability_Predicted"] * 100).round(1)
            if "Preventive_Savings_Potential_INR" in top_risk.columns:
                top_risk["Savings If Prevented"] = top_risk["Preventive_Savings_Potential_INR"].apply(fmt_cr)
            st.dataframe(top_risk, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: CUSTOMER ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Analytics":
    page_hero("Customer Analytics", "Segmentation, churn risk, lifetime value, and top accounts")

    customers = load_data("customers_processed", nrows=50000)
    cust_rev = load_data("top_customers")
    churn = load_data("churn_predictions")

    if selected_country != "All Countries" and "Country" in customers.columns:
        customers = customers[customers["Country"] == selected_country]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Total Customers", f"{len(customers):,}", "kpi-value-blue")
    with c2:
        platinum = int((customers["Customer_Segment"] == "Platinum").sum()) if not customers.empty else 0
        kpi_card("Platinum Accounts", f"{platinum:,}", "kpi-value-gold")
    with c3:
        high_churn = int((customers["Churn_Probability"] > 0.70).sum()) if not customers.empty else 0
        kpi_card("High Churn Risk", f"{high_churn:,}", "kpi-value-red")
    with c4:
        at_risk_rev = int(customers["Revenue_At_Risk_INR"].sum()) if not customers.empty and "Revenue_At_Risk_INR" in customers.columns else 0
        kpi_card("Revenue at Risk", fmt_cr(at_risk_rev), "kpi-value-red")
    with c5:
        avg_ltv = int(customers["Lifetime_Value_INR"].mean()) if not customers.empty else 0
        kpi_card("Avg Customer LTV", fmt_cr(avg_ltv), "kpi-value-green")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        section_header("Customer Segment Distribution", "🏆")
        if not customers.empty:
            seg = customers["Customer_Segment"].value_counts().reset_index()
            seg.columns = ["Segment", "Count"]
            colors = {"Platinum": "#f6c90e", "Gold": "#63b3ed",
                       "Silver": "#a0aec0", "Bronze": "#fc8181"}
            color_list = [colors.get(s, "#63b3ed") for s in seg["Segment"]]
            fig = px.pie(seg, values="Count", names="Segment",
                          color_discrete_sequence=color_list, hole=0.55)
            fig.update_traces(textinfo="percent+value")
            fig.update_layout(title="Customer Segments", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Churn Risk Distribution", "⚠️")
        if not customers.empty and "Churn_Risk_Tier" in customers.columns:
            churn_dist = customers["Churn_Risk_Tier"].value_counts().reset_index()
            churn_dist.columns = ["Risk Tier", "Count"]
            risk_colors = {"Low Risk": "#48bb78", "Medium Risk": "#f6c90e",
                            "High Risk": "#fc8181", "Critical Risk": "#9b2335"}
            fig = px.bar(churn_dist, x="Risk Tier", y="Count",
                          color="Risk Tier", color_discrete_map=risk_colors,
                          text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Churn Risk Tiers", **PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section_header("LTV by Customer Category", "💼")
        if not customers.empty:
            cat_ltv = customers.groupby("Customer_Category").agg(
                Avg_LTV=("Lifetime_Value_INR", "mean"),
                Total_LTV=("Lifetime_Value_INR", "sum"),
                Count=("Customer_ID", "count"),
            ).reset_index().sort_values("Total_LTV", ascending=True)
            fig = px.bar(cat_ltv, x="Total_LTV", y="Customer_Category",
                          orientation="h", color="Total_LTV",
                          color_continuous_scale=["#0d1b3e", "#48bb78"],
                          text=[fmt_cr(v) for v in cat_ltv["Total_LTV"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Total LTV by Customer Category", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        section_header("Revenue by Country", "🌍")
        if not customers.empty:
            country_ltv = customers.groupby("Country")["Lifetime_Value_INR"].sum().reset_index()
            fig = px.bar(country_ltv.sort_values("Lifetime_Value_INR", ascending=False),
                          x="Country", y="Lifetime_Value_INR",
                          color="Country", color_discrete_sequence=COLOR_SEQ,
                          text=[fmt_cr(v) for v in country_ltv.sort_values("Lifetime_Value_INR", ascending=False)["Lifetime_Value_INR"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Customer LTV by Country", **PLOTLY_LAYOUT, showlegend=False)
            fig.update_yaxes(tickprefix="₹")
            st.plotly_chart(fig, use_container_width=True)

    section_header("Top 20 Customers by Revenue", "⭐")
    if not cust_rev.empty:
        top20 = cust_rev[["Customer_ID", "Customer_Name", "Customer_Category",
                            "Customer_Segment", "Country", "Total_Revenue_INR",
                            "Total_Profit_INR", "Churn_Probability"]].head(20)
        top20["Revenue"] = top20["Total_Revenue_INR"].apply(fmt_cr)
        top20["Profit"] = top20["Total_Profit_INR"].apply(fmt_cr)
        top20["Churn Risk"] = (top20["Churn_Probability"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            top20[["Customer_Name", "Customer_Category", "Country", "Revenue", "Profit", "Churn Risk"]],
            use_container_width=True, hide_index=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: PROFIT ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Profit Analytics":
    page_hero("Profit Analytics", "Asset-level, location-level, and contract-level profitability deep-dive")

    txn = apply_filters(load_data("rental_transactions_processed", nrows=200000))
    fin = apply_filters(load_data("financial_performance_processed"))
    pricing = load_data("pricing_optimization")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Net Profit", fmt_cr(txn["Net_Profit_INR"].sum()) if not txn.empty else "N/A", "kpi-value-green")
    with c2:
        kpi_card("Avg Profit Margin", f"{txn['Rental_Margin_Pct'].mean():.1f}%" if not txn.empty else "N/A", "kpi-value-gold")
    with c3:
        under = int(txn["Underpriced_Flag"].sum()) if not txn.empty and "Underpriced_Flag" in txn.columns else 0
        kpi_card("Underpriced Contracts", f"{under:,}", "kpi-value-red", "Below Market Rate")
    with c4:
        pricing_opp = int(txn["Pricing_Opportunity_INR"].sum()) if not txn.empty and "Pricing_Opportunity_INR" in txn.columns else 0
        kpi_card("Pricing Opportunity", fmt_cr(pricing_opp), "kpi-value-gold")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        section_header("Profit Margin by Asset Category", "📊")
        if not txn.empty:
            cat_margin = txn.groupby("Asset_Category").agg(
                Avg_Margin=("Rental_Margin_Pct", "mean"),
                Total_Profit=("Net_Profit_INR", "sum"),
            ).reset_index().sort_values("Avg_Margin", ascending=False)
            fig = px.bar(cat_margin, x="Asset_Category", y="Avg_Margin",
                          color="Avg_Margin", color_continuous_scale=["#fc8181", "#f6c90e", "#48bb78"],
                          text=cat_margin["Avg_Margin"].apply(lambda x: f"{x:.1f}%"))
            fig.add_hline(y=30, line_dash="dash", line_color="#63b3ed",
                           annotation_text="Target Margin: 30%")
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Average Profit Margin by Category", **PLOTLY_LAYOUT)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Profit vs Revenue Scatter", "🔵")
        if not txn.empty:
            sample = txn.sample(min(5000, len(txn)), random_state=42)
            fig = px.scatter(
                sample, x="Gross_Revenue_INR", y="Net_Profit_INR",
                color="Asset_Category", size="Rental_Days",
                color_discrete_sequence=COLOR_SEQ,
                hover_data=["Country", "Project_Type", "Rental_Days"],
                opacity=0.6
            )
            fig.update_layout(title="Revenue vs Profit by Contract", **PLOTLY_LAYOUT)
            fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
            fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

    section_header("Pricing Gap Analysis — Current vs Competitor Rate", "💡")
    if not pricing.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current Rate", x=pricing["Asset_Category"],
                              y=pricing["Avg_Current_Rate_INR"], marker_color="#fc8181"))
        fig.add_trace(go.Bar(name="Competitor Rate", x=pricing["Asset_Category"],
                              y=pricing["Avg_Competitor_Rate_INR"], marker_color="#48bb78"))
        fig.add_trace(go.Bar(name="Recommended Rate", x=pricing["Asset_Category"],
                              y=pricing["Recommended_Rate_INR"], marker_color="#63b3ed"))
        fig.update_layout(title="Pricing Gap: Where We Can Increase Revenue",
                           barmode="group", **PLOTLY_LAYOUT)
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section_header("Profit by Customer Segment", "🏷️")
        if not txn.empty and "Customer_Segment" in txn.columns:
            seg_profit = txn.groupby("Customer_Segment").agg(
                Total_Profit=("Net_Profit_INR", "sum"),
                Avg_Margin=("Rental_Margin_Pct", "mean"),
            ).reset_index()
            fig = px.bar(seg_profit, x="Customer_Segment", y="Total_Profit",
                          color="Customer_Segment", color_discrete_sequence=COLOR_SEQ,
                          text=[fmt_cr(v) for v in seg_profit["Total_Profit"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Profit by Customer Segment", **PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        section_header("Margin Band Distribution", "🎯")
        if not txn.empty and "Margin_Band" in txn.columns:
            margin_dist = txn["Margin_Band"].value_counts().reset_index()
            margin_dist.columns = ["Margin Band", "Count"]
            colors = {"Loss": "#fc8181", "Low (<15%)": "#fbd38d",
                       "Medium (15-30%)": "#f6c90e", "Good (30-50%)": "#48bb78",
                       "Excellent (>50%)": "#4fd1c5"}
            fig = px.pie(margin_dist, values="Count", names="Margin Band",
                          color="Margin Band", color_discrete_map=colors, hole=0.5)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(title="Contract Margin Profile", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7: AI RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Recommendations":
    page_hero("AI Recommendation Engine", "Ranked profit improvement actions for management decision-making")

    recs = load_data("recommendations")

    if recs.empty:
        st.warning("⚠️ No recommendations found. Run script 05 first.")
        st.stop()

    # Summary KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Recommendations", f"{len(recs):,}", "kpi-value-blue")
    with c2:
        critical_ct = int((recs["Priority"] == "Critical").sum())
        kpi_card("Critical Actions", f"{critical_ct}", "kpi-value-red", "Act Immediately")
    with c3:
        high_ct = int((recs["Priority"] == "High").sum())
        kpi_card("High Priority", f"{high_ct}", "kpi-value-gold", "This Month")
    with c4:
        total_gain = int(recs["Expected_Annual_Gain_INR"].sum())
        kpi_card("Total Annual Opportunity", fmt_cr(total_gain), "kpi-value-green", "AI Identified")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter by category
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        priority_filter = st.multiselect("Filter by Priority", ["Critical", "High", "Medium"],
                                          default=["Critical", "High", "Medium"])
    with col_filter2:
        cat_filter = st.multiselect("Filter by Category",
                                     recs["Category"].unique().tolist(),
                                     default=recs["Category"].unique().tolist())

    filtered_recs = recs[
        (recs["Priority"].isin(priority_filter)) &
        (recs["Category"].isin(cat_filter))
    ]

    # Opportunity by category chart
    section_header("Annual Opportunity by Action Category", "📊")
    opp_by_cat = filtered_recs.groupby("Category")["Expected_Annual_Gain_INR"].sum().reset_index()
    opp_by_cat = opp_by_cat.sort_values("Expected_Annual_Gain_INR", ascending=True)
    fig = px.bar(opp_by_cat, x="Expected_Annual_Gain_INR", y="Category",
                  orientation="h", color="Expected_Annual_Gain_INR",
                  color_continuous_scale=["#1a2d5a", "#48bb78"],
                  text=[fmt_cr(v) for v in opp_by_cat["Expected_Annual_Gain_INR"]])
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Annual Revenue/Profit Opportunity by Action Category", **PLOTLY_LAYOUT)
    fig.update_xaxes(tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation Cards
    section_header(f"All Recommendations ({len(filtered_recs)} Actions)", "📋")
    for _, row in filtered_recs.iterrows():
        priority = str(row.get("Priority", "Medium"))
        css_class = f"rec-{priority.lower()}"
        badge_class = f"rec-badge-{priority.lower()}"
        cat_icon = row.get("Category", "")[:2]
        st.markdown(f"""
        <div class="{css_class}">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="flex:1;">
              <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span class="{badge_class}">{priority}</span>
                <span style="color:#63b3ed; font-size:0.78rem; font-weight:600;">{row.get('Category','')}</span>
              </div>
              <div class="rec-title">#{int(row.get('Rank',0))} {row.get('Action','')}</div>
              <div class="rec-detail" style="margin-top:6px;">{row.get('Detail','')}</div>
              <div style="margin-top:8px; display:flex; gap:16px; flex-wrap:wrap;">
                <span style="color:#718096; font-size:0.78rem;">👤 <b style="color:#a0aec0">{row.get('Action_Owner','')}</b></span>
                <span style="color:#718096; font-size:0.78rem;">⏱ <b style="color:#a0aec0">{row.get('Implementation_Timeline','')}</b></span>
                <span style="color:#718096; font-size:0.78rem;">🌍 <b style="color:#a0aec0">{row.get('Country','')}</b></span>
              </div>
            </div>
            <div style="text-align:right; min-width:150px; padding-left:16px;">
              <div style="color:#718096; font-size:0.72rem; margin-bottom:4px;">ANNUAL GAIN</div>
              <div class="rec-impact">{row.get('Expected_Annual_Gain_Label','')}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8: CROSS-COUNTRY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Cross-Country Analysis":
    page_hero("Cross-Country Performance Analysis", "Comparing India vs GCC: Revenue, Profit, Utilization & Growth Potential")

    country_df = load_data("country_performance")
    txn = load_data("rental_transactions_processed", nrows=200000)
    fleet_df = load_data("fleet_expansion")
    proj_df = load_data("project_opportunity")

    if not country_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Countries Active", f"{len(country_df)}", "kpi-value-blue", "India + GCC")
        with c2:
            best_country = country_df.loc[country_df["Total_Revenue_INR"].idxmax(), "Country"] if "Total_Revenue_INR" in country_df.columns else "N/A"
            kpi_card("Top Revenue Country", best_country, "kpi-value-gold")
        with c3:
            best_util = country_df.loc[country_df["Avg_Utilization_Pct"].idxmax(), "Country"] if "Avg_Utilization_Pct" in country_df.columns else "N/A"
            kpi_card("Best Utilization", best_util, "kpi-value-green")
        with c4:
            growth_count = int((country_df["Growth_Potential"] == "High Opportunity").sum()) if "Growth_Potential" in country_df.columns else 0
            kpi_card("High Opportunity Markets", f"{growth_count}", "kpi-value-purple")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            section_header("Revenue by Country", "💰")
            if "Total_Revenue_INR" in country_df.columns:
                fig = px.bar(country_df.sort_values("Total_Revenue_INR", ascending=False),
                              x="Country", y="Total_Revenue_INR",
                              color="Country", color_discrete_sequence=COLOR_SEQ,
                              text=[fmt_cr(v) for v in country_df.sort_values("Total_Revenue_INR", ascending=False)["Total_Revenue_INR"]])
                fig.update_traces(textposition="outside")
                fig.update_layout(title="Total Revenue — All Countries", **PLOTLY_LAYOUT, showlegend=False)
                fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_header("Utilization vs Profit Margin", "📊")
            if "Avg_Utilization_Pct" in country_df.columns and "Avg_Margin_Pct" in country_df.columns:
                fig = px.scatter(country_df, x="Avg_Utilization_Pct", y="Avg_Margin_Pct",
                                  size="Total_Revenue_INR", color="Country",
                                  color_discrete_sequence=COLOR_SEQ,
                                  text="Country", hover_data=["Total_Revenue_INR"])
                fig.update_traces(textposition="top center")
                fig.update_layout(title="Utilization vs Margin Bubble Chart", **PLOTLY_LAYOUT)
                fig.update_xaxes(ticksuffix="%", title="Fleet Utilization")
                fig.update_yaxes(ticksuffix="%", title="Profit Margin")
                st.plotly_chart(fig, use_container_width=True)

        # Country Scorecard
        section_header("Country Performance Scorecard", "🏆")
        if not country_df.empty:
            display = country_df.copy()
            if "Total_Revenue_INR" in display.columns:
                display["Revenue"] = display["Total_Revenue_INR"].apply(fmt_cr)
            if "Total_Profit_INR" in display.columns:
                display["Profit"] = display["Total_Profit_INR"].apply(fmt_cr)
            if "Avg_Margin_Pct" in display.columns:
                display["Margin"] = display["Avg_Margin_Pct"].apply(lambda x: f"{x:.1f}%")
            if "Avg_Utilization_Pct" in display.columns:
                display["Utilization"] = display["Avg_Utilization_Pct"].apply(lambda x: f"{x:.1f}%")
            if "Total_Downtime_Hours" in display.columns:
                display["Downtime (Days)"] = (display["Total_Downtime_Hours"] / 24).apply(lambda x: f"{int(x):,}")
            if "Total_Maint_Cost_INR" in display.columns:
                display["Maint Cost"] = display["Total_Maint_Cost_INR"].apply(fmt_cr)
            if "Growth_Potential" in display.columns:
                display["Growth Potential"] = display["Growth_Potential"]

            show_cols = ["Country", "Revenue", "Profit", "Margin", "Utilization",
                          "Downtime (Days)", "Maint Cost", "Growth Potential"]
            show_cols = [c for c in show_cols if c in display.columns]
            st.dataframe(display[show_cols], use_container_width=True, hide_index=True)

    # Project Sector Analysis
    if not proj_df.empty:
        section_header("Project Sector ROI Analysis", "🏭")
        col3, col4 = st.columns(2)
        with col3:
            fig = px.bar(proj_df.head(10).sort_values("Total_Revenue_INR", ascending=True),
                          x="Total_Revenue_INR", y="Project_Type",
                          orientation="h", color="Avg_Margin_Pct",
                          color_continuous_scale=["#fc8181", "#f6c90e", "#48bb78"],
                          text=[fmt_cr(v) for v in proj_df.head(10).sort_values("Total_Revenue_INR", ascending=True)["Total_Revenue_INR"]])
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Top Sectors by Revenue", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.scatter(proj_df, x="Avg_Rental_Days", y="Avg_Margin_Pct",
                              size="Total_Revenue_INR", color="Project_Type",
                              color_discrete_sequence=COLOR_SEQ,
                              text="Project_Type", hover_data=["Total_Revenue_INR"])
            fig.update_traces(textposition="top center")
            fig.update_layout(title="Sector ROI: Rental Duration vs Margin", **PLOTLY_LAYOUT)
            fig.update_xaxes(title="Avg Rental Duration (Days)")
            fig.update_yaxes(ticksuffix="%", title="Profit Margin")
            st.plotly_chart(fig, use_container_width=True)

    # Fleet Expansion
    if not fleet_df.empty:
        section_header("Fleet Expansion Opportunities", "📈")
        if "Expected_Annual_Revenue_INR" in fleet_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fleet_df["Asset_Category"],
                y=fleet_df["Total_Purchase_Cost_INR"] if "Total_Purchase_Cost_INR" in fleet_df.columns else [],
                name="Investment Required", marker_color="#fc8181"
            ))
            fig.add_trace(go.Bar(
                x=fleet_df["Asset_Category"],
                y=fleet_df["Expected_Annual_Revenue_INR"],
                name="Annual Revenue Expected", marker_color="#48bb78"
            ))
            fig.update_layout(title="Fleet Expansion: Investment vs Expected Revenue",
                               barmode="group", **PLOTLY_LAYOUT)
            fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
