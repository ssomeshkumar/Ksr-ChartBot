import streamlit as st
import fitz  # from PyMuPDF
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()  # Load environment variables from .env

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract Q&A pairs from PDF
def extract_qa(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    pattern = r"\n?\s*(\d{1,3})\.\s+(.+?)\n\s*➤\s+(.+?)(?=\n\s*\d{1,3}\.|$)"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    questions = []
    answers = []

    for _, question, answer in matches:
        questions.append(question.strip())
        answers.append(answer.strip())

    return questions, answers

# Build FAISS index
def build_index(questions):
    if not questions:
        return None, None
    embeddings = model.encode(questions)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Call OpenRouter API
def call_llm_api(context, question):
    prompt = f"Answer the user's question using only the college handbook info below:\n\n{context}\n\nUser question: {question}\nAnswer:"

    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = os.getenv("OPENROUTER_API_KEY")  # Read from .env

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for college admission and policy FAQ."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"⚠️ API Error: No valid 'choices' in response.\n\nResponse: {data}"

    except Exception as e:
        return f"⚠️ API exception: {e}"

# Streamlit UI
st.set_page_config(page_title="KSRCE FAQ Bot", layout="centered")
st.title("🎓 KSRCE FAQ Bot")
st.write("Ask anything about KSRCE admissions, policies, or placements.")

questions, answers = extract_qa("ksrfaq1.pdf")

if not questions:
    st.error("❌ No questions were extracted from the PDF. Check the format.")
    st.stop()

index, _ = build_index(questions)

user_query = st.text_input("📩 Type your question:")

if user_query:
    query_embedding = model.encode([user_query])
    D, I = index.search(np.array(query_embedding), k=1)
    matched_q = questions[I[0][0]]
    matched_a = answers[I[0][0]]
    context = f"Q: {matched_q}\nA: {matched_a}"

    answer = call_llm_api(context, user_query)

    st.markdown("### ✅ Answer")
    st.write(answer)
