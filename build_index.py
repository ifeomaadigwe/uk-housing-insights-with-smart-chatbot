# build_index.py

import os
import pickle
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# ==============================
# 1. --- Import project config 
# ==============================

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, ARTIFACTS_DIR

# --- Configuration ---
EMBED_MODEL = "intfloat/e5-small-v2"
SAVE_INDEX = Path("index.faiss")
SAVE_DOCS = Path("docs.pkl")

# --- Load CSV housing dataset ---
print(" Loading feature-engineered dataset...")
df = pd.read_csv(DATA_DIR / "feature_engineered_data.csv")

# ==============================
# 2. Convert Rows → Housing Descriptions
# ==============================
print("📝 Converting housing rows to text descriptions...")

rows_as_text = []

for i, row in df.iterrows():
    city = row.get("location_city", "Unknown city")
    district = row.get("location_district", "Unknown district")
    price = row.get("sale_price_gbp", None)
    bedrooms = row.get("bedrooms", None)
    bathrooms = row.get("bathrooms", None)
    sqft = row.get("square_footage", None)
    property_type = row.get("property_type", "property")
    property_age = row.get("property_age", None)

    # Build a descriptive sentence
    desc = f"Property in {city}, {district}. It is a {property_type}"

    if bedrooms and not pd.isna(bedrooms):
        desc += f" with {int(bedrooms)} bedrooms"
    if bathrooms and not pd.isna(bathrooms):
        desc += f" and {int(bathrooms)} bathrooms"
    if sqft and not pd.isna(sqft):
        desc += f", covering about {int(sqft)} square feet"
    if property_age and not pd.isna(property_age):
        desc += f". It is approximately {int(property_age)} years old"

    if price and not pd.isna(price):
        desc += f". The listed sale price is £{int(price):,}"
    
    rows_as_text.append({"id": i, "text": desc})

print(f"✅ Converted {len(rows_as_text)} housing rows into text descriptions")

# ==============================
# 3. Create Embeddings
# ==============================
print("🔄 Generating embeddings...")
embed_model = SentenceTransformer("intfloat/e5-small-v2")
embeddings = embed_model.encode(
    [r["text"] for r in rows_as_text],
    normalize_embeddings=True,
    show_progress_bar=True
)
embeddings = np.array(embeddings).astype("float32")

# ==============================
# 4. Create FAISS Index
# ==============================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS + docs
faiss.write_index(index, "index.faiss")
with open("docs.pkl", "wb") as f:
    pickle.dump(rows_as_text, f)

print(" FAISS index and docs.pkl created successfully!")
print("Files saved: index.faiss, docs.pkl")
