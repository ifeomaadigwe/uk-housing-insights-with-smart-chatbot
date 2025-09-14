import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, ARTIFACTS_DIR, ENABLE_PROPERTY_AGE, ENABLE_LISTING_MONTH, ENABLE_DEMAND_INDEX

#  Load cleaned data
df = pd.read_csv(DATA_DIR / "cleaned_uk_housing_market.csv")

#  Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if ENABLE_PROPERTY_AGE and "year_built" in df.columns:
        df["property_age"] = 2025 - df["year_built"]

    if ENABLE_LISTING_MONTH and "listing_date" in df.columns:
        df["listing_date"] = pd.to_datetime(df["listing_date"])
        df["listing_month"] = df["listing_date"].dt.month

    if ENABLE_DEMAND_INDEX and "days_on_market" in df.columns:
        df["demand_index"] = 1 / (df["days_on_market"] + 1)

    return df

#  Apply feature engineering
df_fe = engineer_features(df)

#  Save feature-engineered data to data/
fe_path = DATA_DIR / "feature_engineered_data.csv"
df_fe.to_csv(fe_path, index=False)

#  Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

#  Helper to save plots
def save_plot(fig, name):
    fig.savefig(ARTIFACTS_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)

#  1. Histogram of Property Age
if "property_age" in df_fe.columns:
    fig = plt.figure()
    sns.histplot(df_fe["property_age"], bins=30, kde=True)
    plt.title("Distribution of Property Age")
    save_plot(fig, "property_age_histogram")

#  2. Boxplot of Sale Price by Listing Month
if "listing_month" in df_fe.columns:
    fig = plt.figure()
    sns.boxplot(x="listing_month", y="sale_price_gbp", data=df_fe)
    plt.title("Sale Price by Listing Month")
    save_plot(fig, "price_by_listing_month")

#  3. Scatterplot of Demand Index vs Sale Price
if "demand_index" in df_fe.columns:
    fig = plt.figure()
    sns.scatterplot(x="demand_index", y="sale_price_gbp", data=df_fe)
    plt.title("Demand Index vs Sale Price")
    save_plot(fig, "demand_vs_price")

#  4. Correlation Heatmap
fig = plt.figure()
corr = df_fe.select_dtypes(include="number").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
save_plot(fig, "correlation_heatmap")

print(f" Feature engineering complete. Data saved to: {fe_path}")
print(f" Visuals saved to: {ARTIFACTS_DIR}")
