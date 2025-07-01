
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="CurateAI: Wikipedia Link Recommender", layout="centered")
st.title("📚 CurateAI: Wikipedia Link Recommender")

user_input = st.text_area("Paste any topic or paragraph here:")
num_links = st.slider("How many suggestions would you like?", min_value=1, max_value=10, value=5)

# Load a Hugging Face model for text generation
generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)


if st.button("🔍 Recommend") and user_input:
    with st.spinner("Generating recommendations..."):
        prompt = f"Suggest {num_links} specific and relevant Wikipedia article titles (only titles, no explanation) for this passage:\n{user_input}\n\nReturn them as a numbered list."

        try:
            response = generator(prompt, max_length=256)[0]['generated_text']
            suggestions = response.choices[0].message.content.strip().split("\n")
st.success("Here are your recommendations:")
for i, suggestion in enumerate(suggestions, start=1):
    title = suggestion.strip()
    if title:
        # Remove numbering from model output if any (e.g., "1. Mahatma Gandhi")
        if "." in title[:4]:
            title = title.split(".", 1)[1].strip()

        wiki_link = title.replace(" ", "_")
        st.markdown(f"{i}. [{title}](https://en.wikipedia.org/wiki/{wiki_link})")

        except Exception as e:
            st.error(f"Error: {e}")
