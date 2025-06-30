import streamlit as st
import wikipedia
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Reading List Recommender", layout="centered")
st.title("📚 Reading List Recommender")
st.caption("Paste any article URL to get semantically relevant reading recommendations.")

@st.cache_data
def extract_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs)
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return None

@st.cache_resource
def load_knowledge_base():
    with open("wiki_documents.pkl", "rb") as f:
        documents = pickle.load(f)
    with open("wiki_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    index = faiss.read_index("wiki_faiss.index")
    return documents, embeddings, index

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_recommendations(article_text, model, index, documents, k=5):
    embedding = model.encode([article_text])
    D, I = index.search(np.array(embedding), k)
    return [documents[i] for i in I[0]]

url = st.text_input("Enter an article URL:")
top_k = st.slider("Number of recommendations", 3, 10, 5)

if st.button("🔍 Get Recommendations") and url:
    article_text = extract_article_text(url)
    if not article_text or len(article_text.strip()) < 200:
        st.warning("Could not extract enough article content.")
        st.stop()

    st.success("Article loaded! Getting recommendations...")

    documents, embeddings, index = load_knowledge_base()
    model = load_embedding_model()
    results = get_recommendations(article_text, model, index, documents, k=top_k)

    st.subheader("📌 Recommended Readings:")
    for i, doc in enumerate(results):
        st.markdown(f"**{i+1}. {doc['title']}**")
        st.caption(doc['text'])