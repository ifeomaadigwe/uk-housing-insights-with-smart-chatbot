# rag_chatbot.py

import os
import torch
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ==============================
# Load environment variables
# ==============================
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("❌ DeepSeek API key not found. Please set it in your .env file.")

# Initialize OpenAI client (DeepSeek compatible)
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ==============================
# Device check (safe)
# ==============================
if torch.cuda.is_available():
    try:
        torch.zeros(1).to("cuda")  # sanity check
        device = "cuda"
    except Exception as e:
        print(f"⚠️ CUDA detected but unusable: {e}. Falling back to CPU.")
        device = "cpu"
else:
    device = "cpu"

print(f"📌 Using device: {device}")

# ==============================
# Load FAISS Index + Embeddings
# ==============================
index = faiss.read_index("index.faiss")
with open("docs.pkl", "rb") as f:
    doc_chunks = pickle.load(f)

# FIX: Load model without device first, then move to device
embed_model = SentenceTransformer("intfloat/e5-small-v2")
embed_model = embed_model.to(device)

# ==============================
# RAG Chat Function
# ==============================
def rag_chat(query, top_k=5):
    query_embedding = embed_model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [doc_chunks[i]["text"] for i in indices[0]]

    context = "\n---\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant for UK housing market insights.
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content  

# ==============================
# CLI Chat Loop
# ==============================
if __name__ == "__main__":
    print("💬 DeepSeek RAG Chatbot Ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = rag_chat(user_input)
        print("DeepSeek:", reply.strip())