import streamlit as st
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="CurateAI: Wikipedia Link Recommender", layout="centered")

st.title("📚 CurateAI: Wikipedia Link Recommender")

user_input = st.text_area("Paste any topic or paragraph here:")
num_links = st.slider("How many suggestions would you like?", min_value=1, max_value=10, value=5)

if st.button("🔍 Recommend") and user_input:
    with st.spinner("Generating recommendations..."):
        prompt = f"Suggest {num_links} specific and relevant Wikipedia article links based on this content: {user_input}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that recommends Wikipedia article links."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )

            suggestions = response.choices[0].message.content.strip().split("\n")
            st.success("Here are your recommendations:")
            for i, suggestion in enumerate(suggestions, start=1):
                if suggestion.strip():
                    st.markdown(f"{i}. {suggestion}")

        except Exception as e:
            st.error(f"Error: {e}")
