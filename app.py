
import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

# Load model and index
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with open("wiki_documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open("wiki_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

index = faiss.read_index("wiki_faiss.index")

# Functions
def title_to_wiki_url(title):
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

def get_recommendations(query, model, index, documents, k=5):
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding).astype("float32"), k)
    titles = [documents[i]['title'] for i in I[0]]
    urls = [title_to_wiki_url(t) for t in titles]
    return urls

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.text for para in paragraphs])
        return content.strip()
    except Exception as e:
        return None

# Streamlit UI
st.set_page_config(page_title="CurateAI – Reading List Recommender", page_icon="📚")
st.title("📚 CurateAI – Reading List Recommender")
st.write("Paste any article URL or text to get smart reading suggestions from Wikipedia.")

input_mode = st.radio("Choose input method:", ["Paste article text", "Paste article URL"])

user_input = ""
if input_mode == "Paste article text":
    user_input = st.text_area("Paste your article text here:")
else:
    url_input = st.text_input("Paste the article URL here:")
    if url_input:
        user_input = get_text_from_url(url_input)

top_k = st.slider("How many suggestions would you like?", 1, 10, 3)

if st.button("🔍 Recommend"):
    if user_input and len(user_input.split()) > 20:
        with st.spinner("Generating recommendations..."):
            results = get_recommendations(user_input, model, index, documents, k=top_k)
            st.success("Here are your recommendations:")
            for i, url in enumerate(results, 1):
                st.markdown(f"{i}. [{url.split('/')[-1].replace('_', ' ')}]({url})")
    else:
        st.warning("Please enter a valid article with enough content.")
