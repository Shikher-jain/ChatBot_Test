import os
import io
import time
import json
import torch
import streamlit as st
# import SpeechRecognition as sr
import pyttsx3

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="NEURAL ENTITY",
    page_icon="🧠",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small").to(DEVICE)
    return tokenizer, model

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

tokenizer, model = load_llm()
embedder = load_embedder()

# ---------------- MEMORY ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    st.session_state.chunks = []

# ---------------- VOICE ----------------
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# def listen():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         audio = r.listen(source)
#     return r.recognize_google(audio)
#     # return "What is the time now?"

# ---------------- PDF → RAG ----------------
def ingest_pdf(file):
    reader = PdfReader(file)
    text = "\n".join(page.extract_text() for page in reader.pages)

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    st.session_state.vector_db = index
    st.session_state.chunks = chunks

def rag_query(query):
    if not st.session_state.vector_db:
        return ""

    q_emb = embedder.encode([query])
    D, I = st.session_state.vector_db.search(q_emb, 2)
    return "\n".join([st.session_state.chunks[i] for i in I[0]])

# ---------------- AGENT TOOLS ----------------
def tool_time():
    return time.ctime()

def tool_math(expr):
    try:
        return str(eval(expr))
    except:
        return "Invalid math"

TOOLS = {
    "time": tool_time,
    "math": tool_math
}

def agent_decide(prompt):
    if "time" in prompt:
        return TOOLS["time"]()
    if "calculate" in prompt:
        return TOOLS["math"](prompt.split("calculate")[-1])
    return None

# ---------------- LLM RESPONSE ----------------
def generate(prompt):
    context = rag_query(prompt)
    agent_result = agent_decide(prompt)

    final_prompt = f"""
Context:
{context}

Agent:
{agent_result}

User:
{prompt}

Answer:
"""

    inputs = tokenizer.encode(final_prompt + tokenizer.eos_token, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs,
        max_length=300,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- UI ----------------
st.markdown("""
<style>
body { background: black; color: white; }
.chat { background: rgba(255,255,255,0.05); padding:15px; border-radius:12px; }
.user { color:#00f5ff; }
.bot { color:#ff00ff; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧠 NEURAL ENTITY")
st.caption("PDF brain • Voice • Tools • Intelligence")

# ---------------- PDF UPLOAD ----------------
pdf = st.file_uploader("🧬 Upload PDF to implant knowledge", type=["pdf"])
if pdf:
    ingest_pdf(pdf)
    st.success("Knowledge implanted.")

# ---------------- INPUT ----------------
col1, col2 = st.columns([4,1])

with col1:
    prompt = st.text_input("Transmit Thought")


with col2:
    if st.button("🎤"):
        prompt = st.text_input("Transmit Thought")
        # prompt = listen()

# ---------------- CHAT ----------------
if prompt:
    st.session_state.chat.append(("You", prompt))
    response = generate(prompt)
    st.session_state.chat.append(("Entity", response))
    speak(response)

for role, msg in st.session_state.chat:
    cls = "user" if role == "You" else "bot"
    st.markdown(f"<div class='chat {cls}'><b>{role}:</b><br>{msg}</div>", unsafe_allow_html=True)
