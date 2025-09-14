import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.config import DATA_DIR, ARTIFACTS_DIR, RANDOM_STATE, TEST_SIZE, TARGET_MAE, TARGET_RMSE, TARGET_R2

#  Load feature-engineered data
df = pd.read_csv(DATA_DIR / "feature_engineered_data.csv")

#  Select features and target
drop_cols = ["sale_price_gbp", "property_id", "listing_date", "transaction_type"]
X_raw = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df["sale_price_gbp"]

#  Encode categorical features and handle missing values
X_encoded = pd.get_dummies(X_raw, drop_first=True)
X = X_encoded.fillna(X_encoded.median())
y = y.fillna(y.median())

#  Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

#  Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(random_state=RANDOM_STATE)
}

#  Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    #  Save model
    model_path = ARTIFACTS_DIR / f"{name}_model.pkl"
    joblib.dump(model, model_path)

#  Print results
for name, metrics in results.items():
    print(f"\n {name} Performance:")
    print(f"MAE: £{metrics['MAE']:.2f} {'✅' if metrics['MAE'] <= TARGET_MAE else '❌'}")
    print(f"RMSE: £{metrics['RMSE']:.2f} {'✅' if metrics['RMSE'] <= TARGET_RMSE else '❌'}")
    print(f"R2: {metrics['R2']:.2f} {'✅' if metrics['R2'] >= TARGET_R2 else '❌'}")
    print(f"Model saved to: {ARTIFACTS_DIR / f'{name}_model.pkl'}")
