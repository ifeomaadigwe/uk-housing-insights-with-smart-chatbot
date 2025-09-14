# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, ARTIFACTS_DIR

# Load cleaned data from data/
data_path = DATA_DIR / "cleaned_uk_housing_market.csv"
df = pd.read_csv(data_path)

# Ensure artifacts folder exists
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Helper to save plots
def save_plot(fig, name):
    fig.savefig(ARTIFACTS_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)

# 1. Distribution of Sale Prices
fig = plt.figure()
sns.histplot(df["sale_price_gbp"], bins=50, kde=True)
plt.title("Distribution of Sale Prices")
save_plot(fig, "sale_price_distribution")

# 2. Average Price by Property Type
fig = plt.figure()
sns.barplot(x="property_type", y="sale_price_gbp", data=df)
plt.title("Average Sale Price by Property Type")
save_plot(fig, "price_by_property_type")

# 3. Price vs Square Footage
fig = plt.figure()
sns.scatterplot(x="square_footage", y="sale_price_gbp", hue="property_type", data=df)
plt.title("Sale Price vs Square Footage")
save_plot(fig, "price_vs_square_footage")

# 4. Price vs Bedrooms
fig = plt.figure()
sns.boxplot(x="bedrooms", y="sale_price_gbp", data=df)
plt.title("Sale Price by Number of Bedrooms")
save_plot(fig, "price_by_bedrooms")

# 5. Price vs Bathrooms
fig = plt.figure()
sns.boxplot(x="bathrooms", y="sale_price_gbp", data=df)
plt.title("Sale Price by Number of Bathrooms")
save_plot(fig, "price_by_bathrooms")

# 6. Price by Build Quality Rating
fig = plt.figure()
sns.barplot(x="build_quality_rating", y="sale_price_gbp", data=df)
plt.title("Average Price by Build Quality Rating")
save_plot(fig, "price_by_build_quality")

# 7. Days on Market Distribution
fig = plt.figure()
sns.histplot(df["days_on_market"], bins=40, kde=True)
plt.title("Distribution of Days on Market")
save_plot(fig, "days_on_market_distribution")

# 8. Price vs Amenities Score
fig = plt.figure()
sns.scatterplot(x="nearby_amenities_score", y="sale_price_gbp", data=df)
plt.title("Price vs Nearby Amenities Score")
save_plot(fig, "price_vs_amenities")

# 9. Monthly Revenue vs Sale Price
fig = plt.figure()
sns.scatterplot(x="sale_price_gbp", y="revenue_gbp_monthly", data=df)
plt.title("Monthly Revenue vs Sale Price")
save_plot(fig, "revenue_vs_price")

# 10. Price Trends Over Time
df["listing_date"] = pd.to_datetime(df["listing_date"])
df["listing_month"] = df["listing_date"].dt.to_period("M")
monthly_avg = df.groupby("listing_month")["sale_price_gbp"].mean().reset_index()
fig = plt.figure()
sns.lineplot(x="listing_month", y="sale_price_gbp", data=monthly_avg)
plt.title("Average Sale Price Over Time")
plt.xticks(rotation=45)
save_plot(fig, "price_trend_over_time")

print("✅ EDA visuals saved to artifacts folder.")
