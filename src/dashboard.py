import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- Setup paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "feature_engineered_data.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "XGBoost_model.pkl"

# --- Load data and model ---
if not DATA_PATH.exists():
    st.error(f"❌ Data file not found at: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# --- Prepare features ---
drop_cols = ["sale_price_gbp", "property_id", "listing_date", "transaction_type"]
X_raw = df.drop(columns=[col for col in drop_cols if col in df.columns])
X = pd.get_dummies(X_raw, drop_first=True).fillna(0)

# --- Make predictions ---
df["predicted_price"] = model.predict(X).round(2)

# --- Extract district from one-hot columns ---
district_cols = [col for col in X.columns if col.startswith("location_district_")]

def get_district(row):
    for col in district_cols:
        if row.get(col, 0) == 1:
            return col.replace("location_district_", "")
    return "Unknown"

df["location_district"] = X.apply(get_district, axis=1)

# --- Streamlit Layout ---
st.set_page_config(page_title="UK Housing Dashboard", page_icon="🏡")
st.title("🏡 Smarter UK Housing Insights")

st.markdown("""
This dashboard helps buyers, investors, and agents understand the housing market  
by connecting key **drivers of value** (size, location, quality) with actionable insights.
""")

# Budget Explorer
st.header(" What Can You Afford?")
budget = st.slider("Select your budget (£)", 100_000, 2_000_000, 500_000, step=50_000)
affordable = df[df["predicted_price"] <= budget]
st.success(f"{len(affordable)} properties match your budget.")

st.dataframe(
    affordable[[
        "location_city",
        "location_district",
        "property_type",
        "predicted_price",
        "square_footage",
        "bedrooms",
        "bathrooms",
        "property_age"
    ]]
)

#  Space-for-Money Analysis
st.header(" Space vs. Price")
st.markdown("Which districts give you the **most square footage per pound**?")
df["space_value"] = df["square_footage"] / df["predicted_price"]
space_rank = df.groupby("location_district")["space_value"].mean().sort_values(ascending=False)
space_rank_df = space_rank.head(10).reset_index().rename(columns={"space_value": "Avg SqFt per £"})
st.bar_chart(space_rank_df.set_index("location_district"))

# Undervalued Districts
st.header(" Potentially Undervalued Areas")
df["residual"] = df["predicted_price"] - df["sale_price_gbp"]
undervalued = df[df["residual"] > 0].groupby("location_district")["residual"].mean().sort_values(ascending=False)
undervalued_df = undervalued.head(10).reset_index().rename(columns={"residual": "Avg Residual (£)"})
st.bar_chart(undervalued_df.set_index("location_district"))

# Scenario Analysis
st.header(" What If the Market Shifts?")
adjustment = st.slider("Simulate Market Change (%)", -20, 20, 0)
df["adjusted_price"] = (df["predicted_price"] * (1 + adjustment / 100)).round(2)
st.line_chart(df[["predicted_price", "adjusted_price"]].head(100))

# Risk Detection
st.header(" Properties with Potential Risk")
risky = df[(df["days_on_market"] > 120) | (df["market_trend_index"] < 0.3)]
st.dataframe(
    risky[[
        "location_city",
        "location_district",
        "sale_price_gbp",
        "days_on_market",
        "market_trend_index"
    ]]
)

# Footer
st.markdown("---")
st.caption("Created by Ifeoma Adigwe • Powered by Streamlit • © 2025")
