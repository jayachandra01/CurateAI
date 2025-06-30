import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from wikipediaapi import Wikipedia
import requests
from bs4 import BeautifulSoup
import logging

# --- Load Resources ---
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("wiki_faiss.index")

with open("wiki_documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open("wiki_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

wiki = Wikipedia("en", headers={'User-Agent': 'CurateAI/1.0 (contact@example.com)'})
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- Helper Functions ---
def fetch_wikipedia_text(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, "html.parser")
        paragraphs = soup.select("p")
        return " ".join(p.get_text() for p in paragraphs if p.get_text().strip())
    except Exception as e:
        st.error(f"Failed to fetch article: {e}")
        return ""

def get_recommendations(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = [documents[i] for i in indices[0]]
    return results

def title_to_url(title):
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

# --- UI ---
st.set_page_config(page_title="CurateAI – Reading List Recommender", page_icon="📚")
st.title("📚 CurateAI – Reading List Recommender")
st.caption("Paste any article URL or text to get smart reading suggestions from Wikipedia.")

input_type = st.radio("Choose input method:", ["Paste article text", "Paste article URL"])

user_input = ""
if input_type == "Paste article text":
    user_input = st.text_area("Paste your article text here:")
else:
    url = st.text_area("Paste a Wikipedia article URL here:")
    if url:
        user_input = fetch_wikipedia_text(url)

top_k = st.slider("How many suggestions would you like?", 1, 10, 5)
if st.button("🔍 Recommend") and user_input:
    results = get_recommendations(user_input, k=top_k)
    st.success("Here are your recommendations:")
    for idx, title in enumerate(results, 1):
        st.markdown(f"{idx}. [{title}]({title_to_url(title)})")