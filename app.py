
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import wikipediaapi
import json
from utils import get_wiki_articles, get_recommendations

# Load the FAISS index and metadata
index = faiss.read_index("wiki_faiss.index")
with open("wiki_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="CurateAI – Reading List Recommender")

st.title("📚 CurateAI – Reading List Recommender")
st.markdown("Paste any article URL or text to get smart reading suggestions from Wikipedia.")

input_mode = st.radio("Choose input method:", ["Paste article text", "Paste article URL"])

if input_mode == "Paste article text":
    user_input = st.text_area("Paste your article text here:")
else:
    user_input = st.text_area("Paste a Wikipedia article URL here:")

top_k = st.slider("How many suggestions would you like?", 1, 10, 3)

if st.button("🔍 Recommend"):
    if len(user_input.strip()) < 20:
        st.warning("Please enter a valid article with enough content.")
    else:
        try:
            results = get_recommendations(user_input, model, index, documents, k=top_k)
            st.success("Here are your recommendations:")
            for idx, title in enumerate(results, 1):
                st.markdown(f"{idx}. [{title['title']}]({title['url']})")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
