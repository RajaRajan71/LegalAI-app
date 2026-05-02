import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Suppress HuggingFace warnings and memory bloat
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Set Streamlit UI configuration
st.set_page_config(page_title="Legal AI Assistant", layout="wide")
st.title("⚖️ Legal AI Assistant")
st.write("Analyze and query legal documents using multi-agent reasoning.")

# Load from .env locally
load_dotenv()

# Safely check environment variables or Hugging Face secrets
if "GROQ_API_KEY" in os.environ:
    groq_api_key = os.environ["GROQ_API_KEY"]
else:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("GROQ_API_KEY is not set. Please set it in your environment variables or Hugging Face secrets.")
        st.stop()

# Initialize Groq Client
client = Groq(api_key=groq_api_key)

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Load with low-memory settings
    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

def rewrite_query(query):
    prompt = f"""
Rewrite the following legal question to be more clear and detailed.

Question:
{query}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

def retrieve(query, db):
    return db.similarity_search(query, k=4)

def reason(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
You are a legal reasoning expert.
Analyze the question step-by-step using the context.

Context:
{context}

Question:
{query}

Explain reasoning clearly.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def generate_answer(query, reasoning):
    prompt = f"""
You are a legal assistant.
Using the reasoning below, give a final clear legal answer.

Reasoning:
{reasoning}

Question:
{query}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

def format_output(answer, docs):
    sources = list(set([
        doc.metadata.get("source", "Unknown")
        for doc in docs
    ]))
    return {
        "answer": answer,
        "sources": sources
    }

# Streamlit interface
query = st.text_input("Enter your legal question:", placeholder="e.g., What are the terms of the agreement regarding termination?")

if st.button("Ask", type="primary"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                # Initialize Database
                db = load_vector_db()
                
                # Run the multi-agent logic
                improved_query = rewrite_query(query)
                docs = retrieve(improved_query, db)
                reasoning = reason(improved_query, docs)
                answer = generate_answer(improved_query, reasoning)
                output = format_output(answer, docs)
                
                # Display Output
                st.subheader("📜 Answer")
                st.write(output["answer"])

                st.subheader("📚 Sources")
                sources = output.get("sources", [])
                if sources:
                    for src in sources:
                        st.caption(f"• {src}")
                else:
                    st.caption("No specific sources cited.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
