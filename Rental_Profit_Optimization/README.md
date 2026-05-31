# Rental Business Profit Optimization Platform

## Overview
An enterprise-grade AI/Analytics platform for industrial rental businesses operating across **India and GCC countries** (Saudi Arabia, UAE, Qatar, Oman, Kuwait).

Designed for executives to answer: **"How can we increase annual profit by ₹10+ Crore using data-driven decisions?"**

---

## Equipment Coverage
- Manlift (Boom Lift, Scissor Lift, Articulated Boom)
- Genset (Diesel & Gas)
- Transformer Rentals
- Tower Light Rentals
- Air Compressor Rentals
- Welding Machine Rentals
- Mobile Crane Rentals
- Forklift & Telehandler Rentals
- Earthmoving Equipment

---

## Project Structure
```
Rental_Profit_Optimization/
├── data/
│   ├── raw/              ← 6 enterprise datasets (570K+ records)
│   └── processed/        ← Cleaned + feature-engineered data
├── models/               ← Trained ML models (.pkl)
├── scripts/
│   ├── 01_generate_data.py
│   ├── 02_preprocess_data.py
│   ├── 03_business_analytics.py
│   ├── 04_ml_models.py
│   ├── 05_recommendation_engine.py
│   └── 06_powerbi_exports.py
├── dashboard/
│   └── app.py            ← 8-page Streamlit dashboard
├── reports/              ← Data quality + analytics reports
├── powerbi_exports/      ← 11 Power BI ready CSVs
├── database/
│   └── rental_db.sqlite
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
cd Rental_Profit_Optimization
pip install -r requirements.txt
```

### 2. Run All Scripts in Order
```bash
python scripts/01_generate_data.py       # Generate 570K+ records (~5 min)
python scripts/02_preprocess_data.py     # Clean & engineer features (~2 min)
python scripts/03_business_analytics.py  # Run 8 business modules (~1 min)
python scripts/04_ml_models.py           # Train 5 ML models (~10 min)
python scripts/05_recommendation_engine.py  # Generate AI recommendations
python scripts/06_powerbi_exports.py     # Export Power BI CSVs
```

### 3. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## Dashboard Pages
1. **CEO Profit Command Center** — Executive KPIs & total profit opportunity
2. **Revenue Analytics** — Trends, region, category, country breakdown
3. **Asset Analytics** — Utilization, idle analysis, fleet performance
4. **Maintenance Analytics** — Downtime, cost, high-risk assets
5. **Customer Analytics** — Segments, churn risk, LTV, top accounts
6. **Profit Analytics** — Profit by asset/location/country
7. **AI Recommendations** — Ranked actions with ₹ impact
8. **Cross-Country Analysis** — India vs GCC performance comparison

---

## ML Models
| Model | Algorithm | Business Output |
|---|---|---|
| Predictive Maintenance | XGBoost Classifier | Failure Probability, Downtime Risk |
| Revenue Forecasting | XGBoost Regressor | 12-Month Revenue Forecast |
| Customer Churn | XGBoost Classifier | Churn Score, Revenue at Risk |
| Asset Utilization | XGBoost Regressor | Idle Risk, Utilization Forecast |
| Profit Prediction | XGBoost Regressor | Contract Profitability |

---

## Financial Impact Areas
- ₹ Revenue Increase via Pricing Optimization
- ₹ Savings via Predictive Maintenance
- ₹ Gains via Asset Utilization Improvement
- ₹ Revenue Retention via Churn Prevention
- ₹ Returns via Fleet Expansion
- ₹ Savings via Asset Retirement

---

## Countries Covered
🇮🇳 India | 🇸🇦 Saudi Arabia | 🇦🇪 UAE | 🇶🇦 Qatar | 🇴🇲 Oman | 🇰🇼 Kuwait
