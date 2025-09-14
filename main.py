from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://openrouter.ai/api/v1")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)

# Initialize FastAPI app
app = FastAPI(
    title="UK Housing Market Prediction API",
    description="Predict property prices, detect high-risk listings, and chat with a housing assistant.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data
DATA_DIR = Path(__file__).resolve().parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

df = pd.read_csv(DATA_DIR / "feature_engineered_data.csv")
model = joblib.load(ARTIFACTS_DIR / "XGBoost_model.pkl")

drop_cols = ["sale_price_gbp", "property_id", "listing_date", "transaction_type"]
X_raw = df.drop(columns=[col for col in drop_cols if col in df.columns])
X = pd.get_dummies(X_raw, drop_first=True).fillna(0)

# Load RAG components
index = faiss.read_index("index.faiss")
with open("docs.pkl", "rb") as f:
    doc_chunks = pickle.load(f)
embed_model = SentenceTransformer("intfloat/e5-small-v2")

# ------------------------------
# Root Endpoint
# ------------------------------
@app.get("/", tags=["Welcome"])
def read_root():
    return {"message": "Welcome to the UK Housing Market API. Use /docs to explore endpoints."}

# ------------------------------
# Price Prediction Endpoint
# ------------------------------
@app.get("/predict-price", tags=["Prediction"])
def predict_price(
    bedrooms: int = Query(...),
    bathrooms: int = Query(...),
    square_footage: int = Query(...),
    property_age: int = Query(...),
    build_quality: float = Query(...),
    nearby_amenities_score: float = Query(...),
    market_trend_index: float = Query(...),
    listing_month: int = Query(...),
    listing_year: int = Query(...),
    location_city: str = Query(...),
    property_type: str = Query(...),
    location_district: str = Query(...)
):
    input_data = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "square_footage": square_footage,
        "property_age": property_age,
        "build_quality": build_quality,
        "nearby_amenities_score": nearby_amenities_score,
        "market_trend_index": market_trend_index,
        "listing_month": listing_month,
        "listing_year": listing_year,
        "location_city": location_city,
        "property_type": property_type,
        f"location_district_{location_district}": 1
    }])

    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(input_data)[0]
    return {"predicted_price": f"£{prediction:,.2f}"}

# ------------------------------
# High-Risk Properties Endpoint
# ------------------------------
@app.get("/high-risk-properties", tags=["Risk Analysis"])
def get_high_risk_properties():
    def risk_level(row):
        if row["market_trend_index"] < 0.3 or row["days_on_market"] > 120:
            return "🔴 High Risk"
        elif 0.3 <= row["market_trend_index"] <= 0.6:
            return "🟡 Moderate Risk"
        elif row["market_trend_index"] > 0.6 and row["days_on_market"] < 60:
            return "🟢 Low Risk"
        return "🟡 Moderate Risk"

    df["risk_level"] = df.apply(risk_level, axis=1)
    risky = df[df["risk_level"].str.contains("Risk")]

    return JSONResponse(content=risky[[
        "location_city", "sale_price_gbp", "days_on_market",
        "market_trend_index", "risk_level"
    ]].to_dict(orient="records"))

# ------------------------------
# Chatbot Endpoint
# ------------------------------
@app.get("/chat-query", tags=["Chatbot"])
def chat_query(question: str = Query(...)):
    query_embedding = embed_model.encode(question, normalize_embeddings=True)
    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, 5)
    retrieved_chunks = [doc_chunks[i]["text"] for i in indices[0]]
    context = "\n---\n".join(retrieved_chunks)[:3000]

    prompt = f"""You are a helpful assistant for UK housing market insights.
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )

    return {"response": response.choices[0].message.content}
