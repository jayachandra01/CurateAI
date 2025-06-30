# CurateAI 📚✨

An intelligent **Reading List Recommender** that takes any article as input and returns a curated list of related topics using state-of-the-art **NLP techniques** and **Wikipedia embeddings**.

### 🚀 Live Demo
Try it on Streamlit 👉 [https://curateai.streamlit.app](https://curateai.streamlit.app)

---

## 🔍 What it Does

Given a block of text (like a news article, blog post, or paragraph), this app:
1. **Understands the context** using a Sentence-BERT model.
2. **Compares it** against ~500 Wikipedia topics.
3. **Returns the top 5–10 most relevant topics** you should read next.

Use cases include:
- Academic research
- Personalized learning
- Reading plan suggestions
- Knowledge graph enrichment

---

## 🧠 How it Works

1. **Sentence Embeddings**  
   Uses `all-MiniLM-L6-v2` via HuggingFace Transformers to embed the input and Wikipedia corpus.

2. **Vector Similarity Search**  
   Uses FAISS (Facebook AI Similarity Search) for fast nearest-neighbor lookup.

3. **Wikipedia as Knowledge Base**  
   Uses a curated set of 200+ high-quality Wikipedia articles as a topic base.

---

## 📦 Tech Stack

| Component            | Tool / Library              |
|----------------------|-----------------------------|
| Embedding Model      | `sentence-transformers`     |
| Vector Search        | `FAISS`                     |
| Data Source          | `Wikipedia` (scraped)       |
| UI                   | `Streamlit`                 |
| Language             | `Python 3.11`               |

---

