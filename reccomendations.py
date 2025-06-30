import faiss
import json
from sentence_transformers import SentenceTransformer, util

# Load FAISS index and documents
index = faiss.read_index("wiki_faiss.index")

with open("wiki_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

titles = list(documents.keys())
embeddings = [doc["embedding"] for doc in documents.values()]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_recommendations(query, k=5):
    try:
        if isinstance(query, str):
            query_embedding = model.encode(query)
        else:
            raise ValueError("Query must be a string")

        scores, indices = index.search([query_embedding], k)
        results = [(titles[i], float(scores[0][idx]))
                   for idx, i in enumerate(indices[0]) if i < len(titles)]
        return results
    except Exception as e:
        print("Error in get_recommendations:", e)
        return []
