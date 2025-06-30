
import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import wikipediaapi

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("wiki_documents.pkl", "rb") as f:
    titles = pickle.load(f)

with open("wiki_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

index = faiss.read_index("wiki_faiss.index")

wiki = wikipediaapi.Wikipedia("en")

# Search logic
def get_recommendations(query, k=5):
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding).astype("float32"), k)
    results = []
    for i in I[0]:
        title = titles[i]
        page = wiki.page(title)
        summary = page.summary[0:300] + "..." if page.exists() else "No summary available."
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        results.append((title, summary, url))
    return results

# Streamlit UI
st.set_page_config(page_title="CurateAI", page_icon="🧠")
st.title("📚 CurateAI – Topic-based Reading Recommender")
st.write("Enter a concept or article theme, and get the most relevant Wikipedia articles.")

query = st.text_input("Enter your topic or question:")
top_k = st.slider("How many results?", 1, 10, 5)

if st.button("Get Recommendations") and query:
    with st.spinner("Finding relevant articles..."):
        recs = get_recommendations(query, k=top_k)
        for i, (title, summary, url) in enumerate(recs, 1):
            st.markdown(f"### {i}. [{title}]({url})")
            st.markdown(f"{summary}\n")
