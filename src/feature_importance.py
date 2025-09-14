import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_DIR, ARTIFACTS_DIR

# Load feature-engineered data
df = pd.read_csv(DATA_DIR / "feature_engineered_data.csv")
drop_cols = ["sale_price_gbp", "property_id", "listing_date", "transaction_type"]
X_raw = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Encode and clean
X_encoded = pd.get_dummies(X_raw, drop_first=True)
X = X_encoded.fillna(X_encoded.median())

# Load best model
model = joblib.load(ARTIFACTS_DIR / "XGBoost_model.pkl")

# Extract feature importances
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot top 15 features
top_n = 15
fig = plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(top_n), palette="viridis")
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
fig.savefig(ARTIFACTS_DIR / "xgboost_feature_importance.png")
plt.close(fig)

print("✅ Feature importance analysis complete. Visual saved to artifacts.")
