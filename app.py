import streamlit as st
from transformers import pipeline
import wikipedia
import urllib.parse

# Set page config
st.set_page_config(page_title="CurateAI: Wikipedia Link Recommender", page_icon="📚")

# Title and instructions
st.title("📚 CurateAI: Wikipedia Link Recommender")
st.write("Paste any topic or paragraph here:")

# User input
input_text = st.text_area(" ", height=150)

# Slider for number of suggestions
num_suggestions = st.slider("How many suggestions would you like?", 1, 10, 5)

# Generate pipeline from Hugging Face (Flan-T5 base)
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Button to recommend
if st.button("🔍 Recommend"):
    if not input_text.strip():
        st.warning("Please enter a topic or paragraph.")
    else:
        with st.spinner("Generating recommendations..."):
            try:
                # Generate keywords from input using text2text generation
                prompt = f"Extract important topics from: {input_text}"
                response = generator(prompt, max_new_tokens=32, num_return_sequences=1)[0]['generated_text']
                keywords = [kw.strip() for kw in response.split(',') if kw.strip()]

                # Wikipedia link fetcher
                def get_wikipedia_links(keywords, limit):
                    found = []
                    seen_titles = set()
                    for kw in keywords:
                        try:
                            # Search Wikipedia
                            search_results = wikipedia.search(kw, results=5)
                            for title in search_results:
                                if title in seen_titles:
                                    continue
                                try:
                                    # Resolve title and verify page exists
                                    summary = wikipedia.summary(title, sentences=1, auto_suggest=False)
                                    url_title = urllib.parse.quote(title.replace(" ", "_"))
                                    url = f"https://en.wikipedia.org/wiki/{url_title}"
                                    found.append((title, url))
                                    seen_titles.add(title)
                                    break  # move to next keyword
                                except wikipedia.exceptions.DisambiguationError:
                                    continue
                                except wikipedia.exceptions.PageError:
                                    continue
                        except Exception:
                            continue
                        if len(found) >= limit:
                            break
                    return found

                # Get valid Wikipedia links
                recommendations = get_wikipedia_links(keywords, num_suggestions)

                # Show results
                st.success("Here are your recommendations:")
                for idx, (title, link) in enumerate(recommendations, 1):
                    st.markdown(f"{idx}. [{title}]({link})")

                if not recommendations:
                    st.warning("No valid Wikipedia links found. Try refining the input.")

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")


