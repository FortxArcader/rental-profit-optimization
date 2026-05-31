"""
===============================================================================
RENTAL BUSINESS PROFIT OPTIMIZATION PLATFORM
Script 01: Enterprise-Scale Data Generation
===============================================================================
Generates 6 datasets covering 570,000+ records for industrial rental businesses
operating across India and GCC countries.

Datasets:
  - asset_master.csv          : 20,000 assets
  - rental_transactions.csv   : 200,000+ rental records
  - maintenance_records.csv   : 100,000 maintenance events
  - customers.csv             : 50,000 customers
  - financial_performance.csv : 100,000+ financial records
  - asset_utilization.csv     : 100,000+ utilization records

Business Focus: Generators, Manlifts, Cranes, Compressors, Tower Lights,
                Transformers, Forklifts, Earthmoving Equipment
Countries: India, Saudi Arabia, UAE, Qatar, Oman, Kuwait
===============================================================================
"""

import os
import sys
import random
import sqlite3
import warnings
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
fake = Faker("en_IN")
Faker.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DB_PATH = os.path.join(BASE_DIR, "database", "rental_db.sqlite")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ── Business Constants ───────────────────────────────────────────────────────
COUNTRIES = {
    "India": {
        "cities": ["Delhi NCR", "Mumbai", "Bangalore", "Chennai", "Hyderabad",
                   "Pune", "Kolkata", "Ahmedabad", "Rajasthan", "Gujarat",
                   "Odisha", "Jharkhand", "Assam", "UP", "MP"],
        "currency": "INR", "fx_rate": 1.0
    },
    "Saudi Arabia": {
        "cities": ["Riyadh", "Jeddah", "Dammam", "Jubail", "Yanbu", "Khobar", "Abha"],
        "currency": "SAR", "fx_rate": 22.5
    },
    "UAE": {
        "cities": ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah"],
        "currency": "AED", "fx_rate": 23.0
    },
    "Qatar": {
        "cities": ["Doha", "Al Wakrah", "Al Khor", "Mesaieed", "Ras Laffan"],
        "currency": "QAR", "fx_rate": 23.3
    },
    "Oman": {
        "cities": ["Muscat", "Sohar", "Salalah", "Sur", "Nizwa"],
        "currency": "OMR", "fx_rate": 221.0
    },
    "Kuwait": {
        "cities": ["Kuwait City", "Ahmadi", "Salmiya", "Hawalli", "Farwaniya"],
        "currency": "KWD", "fx_rate": 276.0
    }
}

ASSET_CATEGORIES = {
    "Genset": {
        "subcategories": ["Diesel Genset", "Gas Genset", "Silent Genset", "Open Genset"],
        "capacities": ["20 KVA", "62.5 KVA", "82.5 KVA", "125 KVA", "160 KVA",
                       "250 KVA", "500 KVA", "750 KVA", "1000 KVA", "1500 KVA"],
        "daily_rate_range": (4500, 85000),
        "purchase_cost_range": (350000, 8500000),
        "fuel_types": ["Diesel", "Gas", "Diesel"],
        "manufacturers": ["Cummins", "Caterpillar", "Perkins", "Kirloskar", "Mahindra",
                          "Ashok Leyland", "Volvo", "MTU", "FG Wilson"],
        "service_interval_days": 90,
        "avg_lifespan_years": 12
    },
    "Manlift": {
        "subcategories": ["Scissor Lift", "Boom Lift", "Articulated Boom Lift",
                          "Telescopic Boom Lift", "Spider Lift"],
        "capacities": ["6M", "8M", "10M", "12M", "15M", "18M", "20M", "24M", "30M", "40M"],
        "daily_rate_range": (6000, 75000),
        "purchase_cost_range": (800000, 12000000),
        "fuel_types": ["Electric", "Diesel", "Hybrid"],
        "manufacturers": ["JLG", "Genie", "Haulotte", "Skyjack", "Manitou", "Snorkel", "Dingli"],
        "service_interval_days": 60,
        "avg_lifespan_years": 10
    },
    "Crane": {
        "subcategories": ["Mobile Crane", "Tower Crane", "Crawler Crane",
                          "Pick & Carry Crane", "All-Terrain Crane"],
        "capacities": ["5T", "10T", "15T", "20T", "30T", "50T", "75T", "100T", "150T", "200T"],
        "daily_rate_range": (18000, 350000),
        "purchase_cost_range": (2500000, 80000000),
        "fuel_types": ["Diesel"],
        "manufacturers": ["Liebherr", "Tadano", "Manitowoc", "Kobelco", "XCMG",
                          "Sany", "Zoomlion", "Grove"],
        "service_interval_days": 45,
        "avg_lifespan_years": 15
    },
    "Forklift": {
        "subcategories": ["Electric Forklift", "Diesel Forklift", "Gas Forklift",
                          "Reach Truck", "Pallet Jack"],
        "capacities": ["1.5T", "2T", "3T", "5T", "7T", "10T", "15T"],
        "daily_rate_range": (3500, 25000),
        "purchase_cost_range": (500000, 5000000),
        "fuel_types": ["Electric", "Diesel", "Gas"],
        "manufacturers": ["Toyota", "Komatsu", "Hyster", "Yale", "Crown", "Mitsubishi",
                          "Linde", "Jungheinrich"],
        "service_interval_days": 60,
        "avg_lifespan_years": 10
    },
    "Telehandler": {
        "subcategories": ["Rotating Telehandler", "Fixed Telehandler", "Heavy Duty Telehandler"],
        "capacities": ["2.5T/6M", "3T/7M", "4T/14M", "6T/10M", "7T/17M"],
        "daily_rate_range": (8000, 45000),
        "purchase_cost_range": (2500000, 9000000),
        "fuel_types": ["Diesel"],
        "manufacturers": ["Manitou", "JCB", "Merlo", "Haulotte", "Magni", "Liebherr"],
        "service_interval_days": 60,
        "avg_lifespan_years": 10
    },
    "Tower Light": {
        "subcategories": ["LED Tower Light", "Metal Halide Tower Light",
                          "Solar Tower Light", "Hybrid Tower Light"],
        "capacities": ["4x1000W", "4x1500W", "6x1000W", "4x LED 200W"],
        "daily_rate_range": (1500, 8000),
        "purchase_cost_range": (150000, 800000),
        "fuel_types": ["Diesel", "Solar", "Hybrid"],
        "manufacturers": ["Generac", "Terex", "Doosan", "Atlas Copco", "Wacker Neuson"],
        "service_interval_days": 120,
        "avg_lifespan_years": 8
    },
    "Compressor": {
        "subcategories": ["Portable Air Compressor", "Stationary Compressor",
                          "Screw Compressor", "Reciprocating Compressor"],
        "capacities": ["185 CFM", "375 CFM", "600 CFM", "900 CFM", "1200 CFM"],
        "daily_rate_range": (3500, 35000),
        "purchase_cost_range": (300000, 5000000),
        "fuel_types": ["Diesel", "Electric"],
        "manufacturers": ["Atlas Copco", "Ingersoll Rand", "Kaeser", "Doosan",
                          "Chicago Pneumatic", "CompAir"],
        "service_interval_days": 90,
        "avg_lifespan_years": 12
    },
    "Transformer": {
        "subcategories": ["Distribution Transformer", "Power Transformer",
                          "Step-Up Transformer", "Step-Down Transformer"],
        "capacities": ["100 KVA", "250 KVA", "500 KVA", "1000 KVA",
                       "2000 KVA", "5000 KVA"],
        "daily_rate_range": (5000, 95000),
        "purchase_cost_range": (800000, 25000000),
        "fuel_types": ["Electric"],
        "manufacturers": ["ABB", "Siemens", "Schneider Electric", "BHEL",
                          "Crompton Greaves", "Toshiba"],
        "service_interval_days": 180,
        "avg_lifespan_years": 20
    },
    "Welding Machine": {
        "subcategories": ["MIG Welder", "TIG Welder", "Arc Welder",
                          "Plasma Cutter", "Submerged Arc Welder"],
        "capacities": ["200A", "300A", "400A", "500A", "600A"],
        "daily_rate_range": (800, 6500),
        "purchase_cost_range": (50000, 800000),
        "fuel_types": ["Electric", "Diesel"],
        "manufacturers": ["Lincoln Electric", "Miller", "ESAB", "Fronius",
                          "Kemppi", "Hobart"],
        "service_interval_days": 90,
        "avg_lifespan_years": 8
    },
    "Earthmoving": {
        "subcategories": ["Excavator", "Bulldozer", "Motor Grader",
                          "Wheel Loader", "Backhoe Loader", "Compactor"],
        "capacities": ["0.3 CBM", "0.6 CBM", "1.0 CBM", "1.5 CBM",
                       "2.0 CBM", "3.0 CBM"],
        "daily_rate_range": (12000, 120000),
        "purchase_cost_range": (3000000, 40000000),
        "fuel_types": ["Diesel"],
        "manufacturers": ["Caterpillar", "Komatsu", "Hitachi", "Volvo",
                          "JCB", "Doosan", "Liebherr", "Hyundai"],
        "service_interval_days": 30,
        "avg_lifespan_years": 12
    }
}

CUSTOMER_CATEGORIES = [
    "Government", "EPC Contractor", "Oil & Gas", "Power Company",
    "Infrastructure Developer", "Telecom Operator", "Real Estate Developer",
    "Industrial Manufacturer", "Mining Company", "Shipyard"
]

PROJECT_TYPES = [
    "Oil & Gas", "Power Plant", "Metro Project", "Airport Project",
    "Data Center", "Construction", "Telecom", "Industrial Shutdown",
    "Refinery", "Petrochemical", "Road & Highway", "Port & Harbor",
    "Solar Farm", "Wind Farm", "Smart City", "Desalination Plant"
]

RENTAL_MODES = ["Daily", "Weekly", "Monthly"]
OWNERSHIP_TYPES = ["Owned", "Leased", "Hire Purchase"]
PURCHASE_COUNTRIES = ["India", "Japan", "Germany", "USA", "China", "South Korea", "UK"]

# ── Helper Functions ──────────────────────────────────────────────────────────
def weighted_country():
    """India gets ~60% share, GCC splits the rest."""
    countries = list(COUNTRIES.keys())
    weights = [0.60, 0.14, 0.12, 0.06, 0.05, 0.03]
    return random.choices(countries, weights=weights, k=1)[0]


def weighted_category():
    """Gensets and Manlifts dominate fleet."""
    cats = list(ASSET_CATEGORIES.keys())
    weights = [0.22, 0.20, 0.08, 0.10, 0.07, 0.10, 0.08, 0.06, 0.05, 0.04]
    return random.choices(cats, weights=weights, k=1)[0]


def add_noise(value, pct=0.15):
    """Add ±pct% random noise."""
    return value * (1 + random.uniform(-pct, pct))


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 1 — ASSET MASTER  (20,000 assets)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_asset_master(n=20000):
    print(f"  Generating {n:,} assets...")
    records = []
    for i in range(1, n + 1):
        cat_name = weighted_category()
        cat = ASSET_CATEGORIES[cat_name]
        country = weighted_country()
        city = random.choice(COUNTRIES[country]["cities"])
        sub = random.choice(cat["subcategories"])
        capacity = random.choice(cat["capacities"])
        fuel = random.choice(cat["fuel_types"])
        manufacturer = random.choice(cat["manufacturers"])
        ownership = random.choices(OWNERSHIP_TYPES, weights=[0.70, 0.20, 0.10])[0]
        purchase_country = random.choice(PURCHASE_COUNTRIES)

        age_years = random.triangular(0.5, 15, 4)
        purchase_cost = int(add_noise(random.uniform(*cat["purchase_cost_range"])))
        # Depreciation: 15% reducing balance per year
        book_value = int(purchase_cost * ((1 - 0.15) ** age_years))

        # Status weighted by age: older → more likely idle/maintenance
        if age_years < 3:
            status_weights = [0.80, 0.12, 0.05, 0.03]
        elif age_years < 8:
            status_weights = [0.68, 0.15, 0.10, 0.07]
        else:
            status_weights = [0.50, 0.18, 0.18, 0.14]
        status = random.choices(
            ["Active", "Under Maintenance", "Idle", "Retired"],
            weights=status_weights
        )[0]

        daily_rate = int(add_noise(random.uniform(*cat["daily_rate_range"])))
        service_interval = cat["service_interval_days"] + random.randint(-10, 10)
        last_service_date = datetime.today() - timedelta(days=random.randint(0, service_interval))
        next_service_date = last_service_date + timedelta(days=service_interval)

        records.append({
            "Asset_ID": f"AST{i:06d}",
            "Asset_Category": cat_name,
            "Asset_Subcategory": sub,
            "Rental_Capacity": capacity,
            "Manufacturer": manufacturer,
            "Fuel_Type": fuel,
            "Ownership_Type": ownership,
            "Purchase_Country": purchase_country,
            "Country": country,
            "City": city,
            "Branch": city,
            "Asset_Age_Years": round(age_years, 1),
            "Purchase_Cost_INR": purchase_cost,
            "Book_Value_INR": book_value,
            "Daily_Rental_Rate_INR": daily_rate,
            "Asset_Status": status,
            "Service_Interval_Days": service_interval,
            "Last_Service_Date": last_service_date.strftime("%Y-%m-%d"),
            "Next_Service_Date": next_service_date.strftime("%Y-%m-%d"),
            "Commission_Date": (datetime.today() - timedelta(days=int(age_years * 365))).strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(records)
    path = os.path.join(RAW_DIR, "asset_master.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved asset_master.csv  [{len(df):,} rows]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 2 — CUSTOMERS  (50,000 customers)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_customers(n=50000):
    print(f"  Generating {n:,} customers...")
    records = []
    segments = {
        "Platinum": {"min_ltv": 5000000, "max_ltv": 50000000, "weight": 0.05},
        "Gold":     {"min_ltv": 1000000, "max_ltv": 5000000,  "weight": 0.15},
        "Silver":   {"min_ltv": 250000,  "max_ltv": 1000000,  "weight": 0.35},
        "Bronze":   {"min_ltv": 50000,   "max_ltv": 250000,   "weight": 0.45},
    }

    for i in range(1, n + 1):
        country = weighted_country()
        city = random.choice(COUNTRIES[country]["cities"])
        cat = random.choice(CUSTOMER_CATEGORIES)
        seg_name = random.choices(
            list(segments.keys()),
            weights=[v["weight"] for v in segments.values()]
        )[0]
        seg = segments[seg_name]
        ltv = int(add_noise(random.uniform(seg["min_ltv"], seg["max_ltv"])))
        total_rentals = int(ltv / random.uniform(25000, 200000))
        total_rentals = max(1, total_rentals)
        churn_prob = np.clip(
            random.gauss(0.25, 0.20) + (0.02 * random.randint(0, 5)),
            0.01, 0.99
        )
        days_since_last = random.randint(0, 730)
        credit_limit = int(ltv * random.uniform(0.3, 1.5))

        company_name = fake.company()

        records.append({
            "Customer_ID": f"CUS{i:06d}",
            "Customer_Name": company_name,
            "Customer_Category": cat,
            "Customer_Segment": seg_name,
            "Country": country,
            "City": city,
            "Total_Rentals": total_rentals,
            "Lifetime_Value_INR": ltv,
            "Days_Since_Last_Rental": days_since_last,
            "Churn_Probability": round(churn_prob, 4),
            "Credit_Limit_INR": credit_limit,
            "Payment_Terms_Days": random.choice([15, 30, 45, 60, 90]),
            "Account_Manager": fake.name(),
            "Registration_Date": (datetime.today() - timedelta(days=random.randint(30, 3650))).strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(records)
    path = os.path.join(RAW_DIR, "customers.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved customers.csv  [{len(df):,} rows]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 3 — RENTAL TRANSACTIONS  (200,000+ records)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_transactions(assets_df, customers_df, n=200000):
    print(f"  Generating {n:,} rental transactions...")
    asset_ids = assets_df["Asset_ID"].tolist()
    asset_rates = assets_df.set_index("Asset_ID")["Daily_Rental_Rate_INR"].to_dict()
    asset_cats = assets_df.set_index("Asset_ID")["Asset_Category"].to_dict()
    asset_countries = assets_df.set_index("Asset_ID")["Country"].to_dict()
    customer_ids = customers_df["Customer_ID"].tolist()
    customer_cats = customers_df.set_index("Customer_ID")["Customer_Category"].to_dict()
    customer_segs = customers_df.set_index("Customer_ID")["Customer_Segment"].to_dict()

    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    records = []
    for i in range(1, n + 1):
        asset_id = random.choice(asset_ids)
        customer_id = random.choice(customer_ids)

        rental_start = start_date + timedelta(days=random.randint(0, date_range_days))
        rental_mode = random.choices(RENTAL_MODES, weights=[0.30, 0.25, 0.45])[0]

        if rental_mode == "Daily":
            rental_days = random.randint(1, 14)
        elif rental_mode == "Weekly":
            rental_days = random.randint(7, 60)
        else:
            rental_days = random.randint(30, 365)

        rental_end = rental_start + timedelta(days=rental_days)
        base_rate = asset_rates.get(asset_id, 5000)

        # Competitor rate is within ±20% of our rate
        competitor_rate = int(base_rate * random.uniform(0.80, 1.20))

        # Discounts are more common for longer rentals and platinum customers
        seg = customer_segs.get(customer_id, "Bronze")
        if rental_mode == "Monthly" and seg == "Platinum":
            discount_pct = random.uniform(10, 25)
        elif rental_mode == "Monthly":
            discount_pct = random.uniform(5, 15)
        elif rental_mode == "Weekly":
            discount_pct = random.uniform(0, 10)
        else:
            discount_pct = random.uniform(0, 5)

        effective_rate = base_rate * (1 - discount_pct / 100)
        gross_revenue = effective_rate * rental_days
        # Costs: mobilization, fuel, operator (if applicable)
        mobilization_cost = base_rate * random.uniform(0.5, 2.0)
        operating_cost_pct = random.uniform(0.25, 0.45)
        operating_cost = gross_revenue * operating_cost_pct
        total_cost = mobilization_cost + operating_cost
        net_profit = gross_revenue - total_cost
        margin_pct = (net_profit / gross_revenue * 100) if gross_revenue > 0 else 0

        project_type = random.choice(PROJECT_TYPES)
        country = asset_countries.get(asset_id, "India")

        payment_status = random.choices(
            ["Paid", "Pending", "Overdue", "Partial"],
            weights=[0.65, 0.18, 0.10, 0.07]
        )[0]

        records.append({
            "Transaction_ID": f"TXN{i:07d}",
            "Asset_ID": asset_id,
            "Customer_ID": customer_id,
            "Asset_Category": asset_cats.get(asset_id, ""),
            "Customer_Category": customer_cats.get(customer_id, ""),
            "Customer_Segment": seg,
            "Country": country,
            "Project_Type": project_type,
            "Rental_Mode": rental_mode,
            "Rental_Start_Date": rental_start.strftime("%Y-%m-%d"),
            "Rental_End_Date": rental_end.strftime("%Y-%m-%d"),
            "Rental_Days": rental_days,
            "Base_Daily_Rate_INR": int(base_rate),
            "Competitor_Rate_INR": competitor_rate,
            "Discount_Pct": round(discount_pct, 2),
            "Effective_Daily_Rate_INR": int(effective_rate),
            "Gross_Revenue_INR": int(gross_revenue),
            "Operating_Cost_INR": int(operating_cost),
            "Mobilization_Cost_INR": int(mobilization_cost),
            "Total_Cost_INR": int(total_cost),
            "Net_Profit_INR": int(net_profit),
            "Rental_Margin_Pct": round(margin_pct, 2),
            "Payment_Status": payment_status,
            "Invoice_Date": rental_end.strftime("%Y-%m-%d"),
        })

        if i % 50000 == 0:
            print(f"    ... {i:,} transactions done")

    df = pd.DataFrame(records)
    path = os.path.join(RAW_DIR, "rental_transactions.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved rental_transactions.csv  [{len(df):,} rows]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 4 — MAINTENANCE RECORDS  (100,000 records)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_maintenance(assets_df, n=100000):
    print(f"  Generating {n:,} maintenance records...")
    asset_ids = assets_df["Asset_ID"].tolist()
    asset_ages = assets_df.set_index("Asset_ID")["Asset_Age_Years"].to_dict()
    asset_cats = assets_df.set_index("Asset_ID")["Asset_Category"].to_dict()
    asset_countries = assets_df.set_index("Asset_ID")["Country"].to_dict()

    maint_types = {
        "Preventive": {"cost_mult": 0.6, "downtime_hrs": (4, 24), "weight": 0.45},
        "Corrective": {"cost_mult": 1.8, "downtime_hrs": (24, 168), "weight": 0.30},
        "Breakdown":  {"cost_mult": 3.5, "downtime_hrs": (48, 720), "weight": 0.15},
        "Inspection": {"cost_mult": 0.3, "downtime_hrs": (2, 8), "weight": 0.10},
    }

    base_costs = {
        "Genset": 35000, "Manlift": 45000, "Crane": 120000,
        "Forklift": 25000, "Telehandler": 55000, "Tower Light": 12000,
        "Compressor": 30000, "Transformer": 60000,
        "Welding Machine": 10000, "Earthmoving": 80000
    }

    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    records = []
    for i in range(1, n + 1):
        asset_id = random.choice(asset_ids)
        cat = asset_cats.get(asset_id, "Genset")
        age = asset_ages.get(asset_id, 5)
        country = asset_countries.get(asset_id, "India")

        maint_type = random.choices(
            list(maint_types.keys()),
            weights=[v["weight"] for v in maint_types.values()]
        )[0]
        mt = maint_types[maint_type]

        base_cost = base_costs.get(cat, 30000)
        # Older assets cost more to maintain
        age_factor = 1 + (age * 0.08)
        cost = int(add_noise(base_cost * mt["cost_mult"] * age_factor))
        downtime_hrs = round(random.uniform(*mt["downtime_hrs"]), 1)

        maint_date = start_date + timedelta(days=random.randint(0, date_range_days))
        failure_flag = 1 if maint_type == "Breakdown" else 0

        # Failure probability based on age
        failure_prob = np.clip(0.05 + (age * 0.04) + random.gauss(0, 0.05), 0.01, 0.99)

        records.append({
            "Maintenance_ID": f"MNT{i:07d}",
            "Asset_ID": asset_id,
            "Asset_Category": cat,
            "Country": country,
            "Maintenance_Type": maint_type,
            "Maintenance_Date": maint_date.strftime("%Y-%m-%d"),
            "Maintenance_Cost_INR": cost,
            "Downtime_Hours": downtime_hrs,
            "Failure_Flag": failure_flag,
            "Failure_Probability": round(failure_prob, 4),
            "Asset_Age_At_Event": round(age, 1),
            "Technician_Name": fake.name(),
            "Spare_Parts_Cost_INR": int(cost * random.uniform(0.2, 0.6)),
            "Labour_Cost_INR": int(cost * random.uniform(0.2, 0.4)),
            "Resolution_Days": round(downtime_hrs / 24, 1),
        })

        if i % 25000 == 0:
            print(f"    ... {i:,} maintenance records done")

    df = pd.DataFrame(records)
    path = os.path.join(RAW_DIR, "maintenance_records.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved maintenance_records.csv  [{len(df):,} rows]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 5 — FINANCIAL PERFORMANCE  (100,000+ records)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_financial(assets_df, n=100000):
    print(f"  Generating {n:,} financial performance records...")
    months = pd.date_range("2022-01", "2024-12", freq="MS")
    asset_sample = assets_df.sample(min(n // len(months) + 1, len(assets_df)), random_state=42)

    records = []
    for _, asset in asset_sample.iterrows():
        for month in months:
            base_rev = asset["Daily_Rental_Rate_INR"] * random.uniform(12, 28)
            seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * month.month / 12)
            revenue = int(base_rev * seasonal * add_noise(1.0, 0.20))
            cost = int(revenue * random.uniform(0.40, 0.65))
            depreciation = int(asset["Purchase_Cost_INR"] * 0.15 / 12)
            gross_profit = revenue - cost
            net_profit = gross_profit - depreciation
            margin = round((net_profit / revenue * 100) if revenue > 0 else 0, 2)

            records.append({
                "Record_ID": f"FIN{len(records)+1:07d}",
                "Asset_ID": asset["Asset_ID"],
                "Asset_Category": asset["Asset_Category"],
                "Country": asset["Country"],
                "Branch": asset["Branch"],
                "Month": month.strftime("%Y-%m"),
                "Year": month.year,
                "Quarter": f"Q{((month.month - 1) // 3) + 1}",
                "Gross_Revenue_INR": revenue,
                "Operating_Cost_INR": cost,
                "Depreciation_INR": depreciation,
                "Gross_Profit_INR": gross_profit,
                "Net_Profit_INR": net_profit,
                "Profit_Margin_Pct": margin,
            })

            if len(records) >= n:
                break
        if len(records) >= n:
            break

    df = pd.DataFrame(records[:n])
    path = os.path.join(RAW_DIR, "financial_performance.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved financial_performance.csv  [{len(df):,} rows]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 6 — ASSET UTILIZATION  (100,000+ records)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_utilization(assets_df, n=100000):
    print(f"  Generating {n:,} utilization records...")
    months = pd.date_range("2022-01", "2024-12", freq="MS")
    asset_sample = assets_df.sample(min(n // len(months) + 1, len(assets_df)), random_state=99)

    industry_avg_util = {
        "Genset": 0.72, "Manlift": 0.68, "Crane": 0.62,
        "Forklift": 0.65, "Telehandler": 0.63, "Tower Light": 0.70,
        "Compressor": 0.67, "Transformer": 0.74, "Welding Machine": 0.58,
        "Earthmoving": 0.64
    }

    records = []
    for _, asset in asset_sample.iterrows():
        base_util = industry_avg_util.get(asset["Asset_Category"], 0.65)
        # Older assets have lower utilization
        age_penalty = min(asset["Asset_Age_Years"] * 0.015, 0.20)
        base_util = max(base_util - age_penalty, 0.20)

        for month in months:
            seasonal_boost = 0.08 * np.sin(2 * np.pi * (month.month - 3) / 12)
            util_rate = np.clip(
                base_util + seasonal_boost + random.gauss(0, 0.07),
                0.05, 1.0
            )
            days_in_month = (month + pd.DateOffset(months=1) - month).days
            days_rented = int(util_rate * days_in_month)
            idle_days = days_in_month - days_rented
            revenue_loss = int(idle_days * asset["Daily_Rental_Rate_INR"])

            records.append({
                "Utilization_ID": f"UTL{len(records)+1:07d}",
                "Asset_ID": asset["Asset_ID"],
                "Asset_Category": asset["Asset_Category"],
                "Country": asset["Country"],
                "Branch": asset["Branch"],
                "Month": month.strftime("%Y-%m"),
                "Year": month.year,
                "Quarter": f"Q{((month.month - 1) // 3) + 1}",
                "Days_In_Month": days_in_month,
                "Days_Rented": days_rented,
                "Idle_Days": idle_days,
                "Utilization_Rate_Pct": round(util_rate * 100, 2),
                "Industry_Avg_Util_Pct": round(base_util * 100, 2),
                "Revenue_Loss_From_Idle_INR": revenue_loss,
                "Daily_Rate_INR": int(asset["Daily_Rental_Rate_INR"]),
            })

            if len(records) >= n:
                break
        if len(records) >= n:
            break

    df = pd.DataFrame(records[:n])
    path = os.path.join(RAW_DIR, "asset_utilization.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved asset_utilization.csv  [{len(df):,} rows]")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SQLITE LOADER
# ═══════════════════════════════════════════════════════════════════════════════
def load_to_sqlite(datasets: dict):
    print("\n  Loading datasets into SQLite...")
    conn = sqlite3.connect(DB_PATH)
    for name, df in datasets.items():
        df.to_sql(name, conn, if_exists="replace", index=False)
        print(f"    ✓ Table '{name}' loaded  [{len(df):,} rows]")
    conn.close()
    print(f"  ✓ SQLite database saved: {DB_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  RENTAL PROFIT OPTIMIZATION — DATA GENERATION")
    print("  Generating enterprise-scale datasets (570,000+ records)")
    print("=" * 70)

    print("\n[1/6] Asset Master Dataset")
    assets = generate_asset_master(20000)

    print("\n[2/6] Customer Dataset")
    customers = generate_customers(50000)

    print("\n[3/6] Rental Transactions Dataset")
    transactions = generate_transactions(assets, customers, 200000)

    print("\n[4/6] Maintenance Records Dataset")
    maintenance = generate_maintenance(assets, 100000)

    print("\n[5/6] Financial Performance Dataset")
    financial = generate_financial(assets, 100000)

    print("\n[6/6] Asset Utilization Dataset")
    utilization = generate_utilization(assets, 100000)

    total = len(assets) + len(customers) + len(transactions) + len(maintenance) + len(financial) + len(utilization)
    print(f"\n  📊 Total Records Generated: {total:,}")

    load_to_sqlite({
        "asset_master": assets,
        "customers": customers,
        "rental_transactions": transactions,
        "maintenance_records": maintenance,
        "financial_performance": financial,
        "asset_utilization": utilization,
    })

    print("\n" + "=" * 70)
    print("  ✅ DATA GENERATION COMPLETE")
    print("  Next: python scripts/02_preprocess_data.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
