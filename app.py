import streamlit as st
import wikipedia
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="CurateAI: Smart Reading Recommender", layout="centered")
st.title("📚 CurateAI – Reading List Recommender")
st.caption("Paste any article URL or text to get smart reading suggestions from Wikipedia.")

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
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def title_to_wiki_url(title):
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

def get_recommendations(article_text, model, index, documents, k=5, min_score=0.35):
    embedding = model.encode([article_text])
    D, I = index.search(np.array(embedding), k)
    results = []
    for i in range(len(I[0])):
        score = D[0][i]
        if score > min_score:
            title = documents[I[0][i]]
            url = title_to_wiki_url(title)
            results.append((title, url, round(float(score), 4)))
    return results

option = st.radio("Choose input method:", ("Paste article text", "Paste article URL"))

user_input = ""
if option == "Paste article text":
    user_input = st.text_area("Paste your article text here:")
elif option == "Paste article URL":
    url = st.text_input("Paste URL here:")
    if url:
        user_input = extract_article_text(url)

top_k = st.slider("How many suggestions would you like?", 3, 10, 5)

if st.button("🔍 Recommend"):
    if not user_input or len(user_input.strip()) < 200:
        st.warning("Please enter a valid article with enough content.")
        st.stop()

    documents, embeddings, index = load_knowledge_base()
    model = load_embedding_model()
    results = get_recommendations(user_input, model, index, documents, k=top_k)

    if results:
        st.subheader("🔗 Recommended Wikipedia Readings:")
        for title, link, score in results:
            st.markdown(f"- [{title}]({link}) – Similarity: `{score}`")
    else:
        st.info("No relevant matches found. Try a more detailed article.")