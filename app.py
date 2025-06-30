import streamlit as st
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load resources
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("wiki_nn_model.pkl")
with open("wiki_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

def recommend_articles(text, k=5):
    query_vec = vectorizer.transform([text])
    distances, indices = model.kneighbors(query_vec, n_neighbors=k)
    results = [documents[i] for i in indices[0]]
    return results

# UI
st.set_page_config(page_title="CurateAI", layout="centered")
st.title("📚 CurateAI: Wikipedia Link Recommender")

user_input = st.text_area("Paste any topic or paragraph here:")
num_suggestions = st.slider("How many suggestions would you like?", 1, 5, 3)

if st.button("🔍 Recommend"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        st.success("Here are your recommendations:")
        for i, item in enumerate(recommend_articles(user_input, k=num_suggestions), 1):
            st.markdown(f"{i}. [{item['title']}]({item['url']})")