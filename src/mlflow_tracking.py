import os
import sys
from pathlib import Path
import dagshub
import mlflow
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, ARTIFACTS_DIR, TEST_SIZE, RANDOM_STATE

# STEP 1: Set team name and experiment name
TEAM_NAME = "TEAM_ONE"
EXPERIMENT_NAME = "Hackathon-UK-Housing"

# STEP 2: Set DagsHub credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "smksean"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9ef9a5775d83b50bedf6708b96bac89c454b8563"

# STEP 3: Connect to DagsHub MLflow server
dagshub.init(repo_owner="smksean", repo_name="hackathon-logging", mlflow=True)

# STEP 4: Load data and prepare features
df = pd.read_csv(DATA_DIR / "feature_engineered_data.csv")
drop_cols = ["sale_price_gbp", "property_id", "listing_date", "transaction_type"]
X_raw = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df["sale_price_gbp"].fillna(df["sale_price_gbp"].median())

X_encoded = pd.get_dummies(X_raw, drop_first=True)
X = X_encoded.fillna(X_encoded.median())

# STEP 5: Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# STEP 6: Load trained model
model_path = ARTIFACTS_DIR / "XGBoost_model.pkl"
model = joblib.load(model_path)

# STEP 7: Predict and compute metrics on test set
y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print("✅ Metrics calculated on test set")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.4f}")

# STEP 8: Log results to MLflow
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=TEAM_NAME):
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.sklearn.log_model(model, "model")

print(f"🎉 Run logged for {TEAM_NAME}! Check the leaderboard on DagsHub.")
print("🔗 View your experiment at: https://dagshub.com/smksean/hackathon-logging/mlflow")
print("🔗 View the leaderboard at: https://dagshub.com/smksean/hackathon-logging/leaderboard")