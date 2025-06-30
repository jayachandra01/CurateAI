import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
import os

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📚 CurateAI: Wikipedia Link Recommender")

input_text = st.text_area("Paste any topic or paragraph here:")

num_links = st.slider("How many suggestions would you like?", 1, 10, 5)

if st.button("🔍 Recommend"):
    if input_text:
        if input_text.startswith("http"):
            try:
                response = requests.get(input_text)
                soup = BeautifulSoup(response.text, "html.parser")
                text_content = soup.get_text()
            except Exception as e:
                st.error(f"Failed to fetch URL: {e}")
                st.stop()
        else:
            text_content = input_text

        prompt = f"Suggest {num_links} specific and relevant Wikipedia article links based on this content:

{text_content}

Respond only with clickable Wikipedia links in markdown list format."

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = completion.choices[0].message["content"]
            st.markdown("### Here are your recommendations:")
            st.markdown(answer)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
    else:
        st.warning("Please enter a topic or URL.")