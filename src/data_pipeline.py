import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path for clean imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, ARTIFACTS_DIR

# Load raw dataset
raw_path = DATA_DIR / "uk_housing_market.csv"
df = pd.read_csv(raw_path)

# Basic cleaning function
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    if "sale_price_gbp" in df.columns:
        df["sale_price_gbp"].fillna(df["sale_price_gbp"].median(), inplace=True)
    if "days_on_market" in df.columns:
        df["days_on_market"].fillna(df["days_on_market"].median(), inplace=True)

    # Drop rows missing critical info
    df.dropna(subset=["location_city", "listing_date"], inplace=True)

    return df

# Apply cleaning
cleaned_df = clean_data(df)

# Save cleaned dataset to data/
cleaned_path = DATA_DIR / "cleaned_uk_housing_market.csv"
cleaned_df.to_csv(cleaned_path, index=False)

print(f" Cleaned data saved to: {cleaned_path}")
