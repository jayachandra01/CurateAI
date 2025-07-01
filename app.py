import streamlit as st
from transformers import pipeline
import wikipedia
import urllib.parse
import torch

# Inject custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stTextArea textarea {
            font-size: 16px;
            line-height: 1.5;
        }
        .stSlider > div[data-baseweb="slider"] {
            margin-top: -10px;
        }
        .recommend-box {
            padding: 1rem;
            background-color: #e8f5e9;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .title {
            font-size: 2.4rem;
            font-weight: bold;
            color: #333333;
        }
        .subheader {
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
            color: #555;
        }
        .wiki-link {
            font-size: 1.1rem;
            padding: 5px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="CurateAI: Wikipedia Link Recommender", page_icon="📚")

# Title and instructions
st.markdown('<div class="title">📚 CurateAI: Wikipedia Link Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Paste any topic, paragraph, or sentence below. CurateAI will recommend the most relevant Wikipedia articles.</div>', unsafe_allow_html=True)

# User input
input_text = st.text_area(" ", placeholder="Paste your topic here...", height=160)

# Slider for number of suggestions
num_suggestions = st.slider("🔢 How many suggestions would you like?", 1, 10, 5)

# Load Hugging Face text2text generation pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base", device=0 if torch.cuda.is_available() else -1)

# Wikipedia link fetcher
def get_wikipedia_links(keywords, limit):
    found = []
    seen_titles = set()

    for kw in keywords:
        try:
            search_results = wikipedia.search(kw, results=5)
            for title in search_results:
                if title in seen_titles:
                    continue
                try:
                    summary = wikipedia.summary(title, sentences=1, auto_suggest=False)
                    url_title = urllib.parse.quote(title.replace(" ", "_"))
                    url = f"https://en.wikipedia.org/wiki/{url_title}"
                    found.append((title, url))
                    seen_titles.add(title)
                    break
                except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                    continue
        except Exception:
            continue
        if len(found) >= limit:
            break

    if len(found) < limit:
        for kw in keywords:
            if len(found) >= limit:
                break
            try:
                search_results = wikipedia.search(kw, results=5)
                for title in search_results:
                    if title in seen_titles:
                        continue
                    url_title = urllib.parse.quote(title.replace(" ", "_"))
                    url = f"https://en.wikipedia.org/wiki/{url_title}"
                    found.append((title, url))
                    seen_titles.add(title)
            except:
                continue

    return found[:limit]

# Recommend button
if st.button("🔍 Recommend"):
    if not input_text.strip():
        st.warning("⚠️ Please enter a topic or paragraph.")
    else:
        with st.spinner("🧠 Thinking... Generating Wikipedia recommendations..."):
            try:
                prompt = f"Extract important topics from: {input_text}"
                response = generator(prompt, max_new_tokens=32, num_return_sequences=1)[0]['generated_text']
                keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
                recommendations = get_wikipedia_links(keywords, num_suggestions)

                if recommendations:
                    st.markdown('<div class="recommend-box">', unsafe_allow_html=True)
                    st.success("✅ Here are your recommendations:")
                    for idx, (title, link) in enumerate(recommendations, 1):
                        st.markdown(f'<div class="wiki-link">{idx}. <a href="{link}" target="_blank">{title}</a></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No valid Wikipedia links found. Try refining the input.")

            except Exception as e:
                st.error(f"❌ Something went wrong: `{str(e)}`")

