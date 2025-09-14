from pathlib import Path

# Paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure artifacts folder exists
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Modeling configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Benchmark targets
TARGET_MAE = 178_419.22
TARGET_RMSE = 230_304.13
TARGET_R2 = 0.80

# Feature engineering flags
ENABLE_PROPERTY_AGE = True
ENABLE_LISTING_MONTH = True
ENABLE_DEMAND_INDEX = True
