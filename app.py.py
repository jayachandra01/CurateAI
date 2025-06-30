
import streamlit as st
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="CurateAI: Wikipedia Link Recommender", page_icon="📚")

st.title("📚 CurateAI: Wikipedia Link Recommender")

st.write("Paste any topic or paragraph here:")
input_text = st.text_area("", height=120)

st.write("How many suggestions would you like?")
num_suggestions = st.slider("", 1, 10, 5)

def get_gpt_recommendations(prompt_text, n=5):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that recommends relevant Wikipedia article titles based on a given topic or paragraph."},
        {"role": "user", "content": f"Topic: {prompt_text}\nGive me {n} related Wikipedia article titles."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5
    )

    reply = response['choices'][0]['message']['content']
    return [line.strip("- ").strip() for line in reply.strip().split("\n") if line]

def title_to_link(title):
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

if st.button("🔍 Recommend"):
    with st.spinner("Getting GPT recommendations..."):
        try:
            suggestions = get_gpt_recommendations(input_text, num_suggestions)
            links = [title_to_link(s) for s in suggestions]
            st.success("Here are your recommendations:")
            for i, (title, link) in enumerate(zip(suggestions, links), start=1):
                st.markdown(f"{i}. [{title}]({link})")
        except Exception as e:
            st.error(f"Error: {e}")
