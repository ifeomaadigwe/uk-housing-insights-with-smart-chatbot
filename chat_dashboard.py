# chat_dashboard.py

import os
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from fpdf import FPDF
import datetime
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt

# ==============================
# Load environment variables
# ==============================
if "DEEPSEEK_API_KEY" in st.secrets:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
    API_BASE = st.secrets.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
else:
    load_dotenv()
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

if not API_KEY:
    raise ValueError("❌ API key not found. Please set it in .env (local) or Streamlit Secrets (cloud).")

# ✅ Initialize DeepSeek client
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

# ==============================
# Load Data & Model
# ==============================
DATA_DIR = Path(__file__).resolve().parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

df = pd.read_csv(DATA_DIR / "feature_engineered_data.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

model = joblib.load(ARTIFACTS_DIR / "XGBoost_model.pkl")

drop_cols = ["sale_price_gbp", "property_id", "listing_date", "transaction_type"]
X_raw = df.drop(columns=[col for col in drop_cols if col in df.columns])
X = pd.get_dummies(X_raw, drop_first=True).fillna(0)
df["predicted_price"] = model.predict(X).round(2)

district_cols = [col for col in X.columns if col.startswith("location_district_")]

def get_district(row):
    for col in district_cols:
        if row[col] == 1:
            return col.replace("location_district_", "")
    return "Unknown"

df["location_district"] = X.apply(get_district, axis=1)

# ==============================
# RAG Chatbot Setup
# ==============================
INDEX_PATH = Path("index.faiss")
DOCS_PATH = Path("docs.pkl")

index = faiss.read_index(str(INDEX_PATH))
with open(DOCS_PATH, "rb") as f:
    doc_chunks = pickle.load(f)

embed_model = SentenceTransformer("intfloat/e5-small-v2")

def rag_chat(query, top_k=5):
    try:
        query_embedding = embed_model.encode(query, normalize_embeddings=True)
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = index.search(query_embedding, top_k)
        retrieved_chunks = [doc_chunks[i]["text"] for i in indices[0]]

        context = "\n---\n".join(retrieved_chunks)[:3000]
        prompt = f"""You are a helpful assistant for UK housing market insights.
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ API call failed: {e}"

# ==============================
# PDF Export Helper
# ==============================
def export_chat_to_pdf(chat_history, filename="chat_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "UK Housing Market Chat Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)

    for role, message in chat_history:
        prefix = "User: " if role == "user" else "Assistant: "
        pdf.multi_cell(0, 10, prefix + message)
        pdf.ln(2)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("chat/housing_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{timestamp}_{filename}"
    pdf.output(str(path))
    return path

# ==============================
# Streamlit Dashboard
# ==============================
st.set_page_config(page_title="UK Housing Market Dashboard", layout="wide")
st.title("🏠 UK Housing Market Insights")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Smart Chatbot", "🔮 Price Prediction"])

# ------------------------------
# TAB 1: Market Dashboard
# ------------------------------
with tab1:
    st.subheader("Budget-Based Property Suggestions")
    budget = st.slider("Select your budget (£)", 100000, 2000000, 500000, step=50000)
    affordable = df[df["predicted_price"] <= budget]

    st.subheader("Filter by District")
    selected_district = st.selectbox("Filter by district", df["location_district"].unique())
    filtered_df = affordable[affordable["location_district"] == selected_district]

    st.dataframe(
        filtered_df[[
            "location_city", "property_type", "predicted_price",
            "square_footage", "bedrooms", "bathrooms", "property_age"
        ]].reset_index(drop=True)
    )

    st.subheader("Top Undervalued Districts")
    df["residual"] = df["predicted_price"] - df["sale_price_gbp"]
    undervalued = (
        df[df["residual"] > 0]
        .groupby("location_district")["residual"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(undervalued.head(10))

    st.subheader("Market Scenario Simulation")
    adjustment = st.slider("Simulate Market Change (%)", -20, 20, 0)
    df["adjusted_price"] = (df["predicted_price"] * (1 + adjustment / 100)).round(2)
    st.line_chart(df[["sale_price_gbp", "adjusted_price"]].head(100).reset_index(drop=True))

    st.subheader("⚠️ High-Risk Properties ")
    st.markdown("""
    - 🔴 **High Risk**: `market_trend_index < 0.3` or `days_on_market > 120`  
    - 🟡 **Moderate Risk**: `market_trend_index between 0.3 and 0.6`  
    - 🟢 **Low Risk**: `market_trend_index > 0.6` and `days_on_market < 60`  
    """)

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
    st.dataframe(
        risky[[
            "location_city", "sale_price_gbp", "days_on_market",
            "market_trend_index", "risk_level"
        ]].reset_index(drop=True)
    )

# ------------------------------
# TAB 2: Chatbot
# ------------------------------
with tab2:
    st.subheader("💬 Ask the UK Housing Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Type your question here:")

    if st.button("Send") and user_input:
        with st.spinner("Thinking..."):
            reply = rag_chat(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", reply))

    if st.session_state.chat_history:
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Assistant:** {message}")

        if st.button("Save this conversation as a Housing Report"):
            pdf_path = export_chat_to_pdf(st.session_state.chat_history)
            st.success(f"📄 Report saved to {pdf_path}")

# ------------------------------
# TAB 3: Customer Housing Prediction
# ------------------------------
with tab3:
    st.subheader(" Predict Property Price")

    district = st.selectbox("Select a district:", df["location_district"].unique())
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
    sqft = st.number_input("Square Footage", 200, 10000, 1500)
    property_age = st.number_input("Property Age (years)", 0, 200, 10)
    build_quality = st.slider("Build Quality Rating", 1.0, 10.0, 7.0)
    amenities = st.slider("Nearby Amenities Score", 0.0, 10.0, 5.0)
    market_trend = st.slider("Market Trend Index", 0.0, 1.0, 1.0)
    listing_month = st.slider("Listing Month", 1, 12, 6)
    listing_year = st.slider("Listing Year", 2000, 2030, 2025)
    location_city = st.text_input("Location City", "London")
    property_type = st.text_input("Property Type", "Detached")

    if st.button("Predict Price"):
        input_data = pd.DataFrame([{
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "square_footage": sqft,
            "property_age": property_age,
            "build_quality": build_quality,
            "nearby_amenities_score": amenities,
            "market_trend_index": market_trend,
            "listing_month": listing_month,
            "listing_year": listing_year,
            "location_city": location_city,
            "property_type": property_type,
            "location_district_" + district: 1
        }])

        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        prediction = model.predict(input_data)[0]
        st.success(f"🏠 Predicted Property Price: £{prediction:,.2f}")

# Footer
st.markdown("---")
st.caption("Created by Ifeoma Adigwe • Powered by Streamlit & DeepSeek @2025")
