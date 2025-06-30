
import json
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

def get_wiki_articles(titles):
    results = []
    for title in titles:
        results.append({
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        })
    return results

def embed_text(text, model: SentenceTransformer):
    embedding = model.encode([text])[0]
    return np.array(embedding).astype("float32")

def get_recommendations(query, model, index, documents, k=5):
    embedding = embed_text(query, model)
    D, I = index.search(np.array([embedding]), k)
    results = []
    for idx in I[0]:
        if idx < len(documents):
            results.append(documents[idx])
    return results
