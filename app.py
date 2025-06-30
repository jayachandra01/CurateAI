import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="CurateAI: Wikipedia Link Recommender", layout="centered")
st.title("📚 CurateAI: Wikipedia Link Recommender")

# Load model and dataset
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    data_path = "wiki_documents.json"
    with open(data_path, "r", encoding="utf-8") as f:
        wiki_data = json.load(f)
    embeddings = np.array([item["embedding"] for item in wiki_data])
    knn = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn.fit(embeddings)
    return model, wiki_data, knn, embeddings

model, wiki_data, knn, embeddings = load_model_and_data()

# Input section
user_input = st.text_area("Paste any topic or paragraph here:", height=100)
num_suggestions = st.slider("How many suggestions would you like?", 1, 5, 3)
submit = st.button("🔍 Recommend")

# Recommend logic
if submit and user_input.strip():
    query_embedding = model.encode([user_input])
    distances, indices = knn.kneighbors(query_embedding, n_neighbors=num_suggestions)
    st.success("Here are your recommendations:")
    for i, idx in enumerate(indices[0], 1):
        item = wiki_data[idx]
        title = item.get("title", "Untitled")
        url = item.get("url", "#")
        st.markdown(f"{i}. [{title}]({url})")
