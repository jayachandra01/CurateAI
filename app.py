
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="CurateAI: Wikipedia Link Recommender", layout="centered")
st.title("📚 CurateAI: Wikipedia Link Recommender")

user_input = st.text_area("Paste any topic or paragraph here:")
num_links = st.slider("How many suggestions would you like?", min_value=1, max_value=10, value=5)

# Load a Hugging Face model for text generation
generator = pipeline("text2text-generation", model="google/flan-t5-base")

if st.button("🔍 Recommend") and user_input:
    with st.spinner("Generating recommendations..."):
        prompt = f"List {num_links} relevant Wikipedia article titles based on the following content: {user_input}"
        try:
            response = generator(prompt, max_length=256)[0]['generated_text']
            suggestions = response.strip().split("\n")
            st.success("Here are your recommendations:")
            for i, suggestion in enumerate(suggestions, start=1):
                if suggestion.strip():
                    st.markdown(f"{i}. {suggestion}")
        except Exception as e:
            st.error(f"Error: {e}")
