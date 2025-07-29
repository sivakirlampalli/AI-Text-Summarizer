import streamlit as st
from transformers import pipeline

# Set up the Streamlit app UIc
st.set_page_config(page_title="AI Text Summarizer")
st.title("🧠 AI Text Summarizer")
st.write("Paste your long text below and get a short summary using HuggingFace's BART model.")

# Get user input
user_input = st.text_area("Enter your long text here:")

# Create the summarizer pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Summarize button
if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarizer(user_input, max_length=130, min_length=30, do_sample=False)
            st.success("Done!")
            st.subheader("📝 Summary:")
            st.write(summary[0]['summary_text'])
