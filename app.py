import streamlit as st
from transformers import pipeline
import urllib.parse

# Load the T5 model for text-to-text generation
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Streamlit app UI setup
st.set_page_config(page_title="CurateAI: Wikipedia Link Recommender", layout="centered")
st.title("📚 CurateAI: Wikipedia Link Recommender")

# User input
user_input = st.text_area("Paste any topic or paragraph here:")
num_links = st.slider("How many suggestions would you like?", min_value=1, max_value=10, value=5)

# Handle recommendation button
if st.button("🔍 Recommend") and user_input:
    with st.spinner("Generating recommendations..."):
        try:
            # Create prompt for the model
            prompt = f"Suggest {num_links} relevant Wikipedia article titles based on this content:\n{user_input}"

            # Generate response using the model
            output = generator(prompt, max_length=256, num_return_sequences=1)
            raw_response = output[0]['generated_text']

            # Format and display each suggestion as a clickable Wikipedia link
            suggestions = raw_response.strip().split("\n")
            st.success("Here are your recommendations:")
            for i, suggestion in enumerate(suggestions, start=1):
                if suggestion.strip():
                    title = suggestion.strip().replace("*", "").strip()
                    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                    wiki_link = f"https://en.wikipedia.org/wiki/{encoded_title}"
                    st.markdown(f"{i}. [{title}]({wiki_link})")

        except Exception as e:
            st.error(f"Error: {e}")

