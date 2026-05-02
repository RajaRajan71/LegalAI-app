import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress HuggingFace warnings and memory bloat
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Set Streamlit UI configuration
st.set_page_config(page_title="Legal AI Assistant", layout="wide")

# App Header
st.title("⚖️ Legal AI Assistant")
st.write("Analyze and query Indian statutory laws, regulations, and landmark precedents using multi-agent reasoning.")

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
    os.makedirs("vectorstore", exist_ok=True)
    
    # If the database folder doesn't exist, create it on the fly
    if not os.path.exists("vectorstore/db_faiss/index.faiss"):
        st.info("Generating vector database for the first time... this might take a moment.")
        
        # Create data directory if not present
        os.makedirs("data", exist_ok=True)
        
        loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            # Fallback warning if no PDFs are present
            st.warning("No PDF files found in the 'data/' folder. Please ensure PDFs are uploaded.")
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        texts = splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        db = FAISS.from_documents(texts, embeddings)
        db.save_local("vectorstore/db_faiss")
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
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

# --- Advanced UI Elements ---

# Popover menu for settings
with st.popover("⚙️ App Settings & Documentation Info"):
    st.markdown("#### Database Documents Supported")
    st.caption("• Companies Act 2013 & Limited Liability Partnership (LLP) Act 2008")
    st.caption("• The Indian Contract Act, 1872 & Sale of Goods Act 1930")
    st.caption("• Consumer Protection Act 2019 & Precedents (Vodafone, etc.)")
    if st.button("Reload Vector Database"):
        with st.spinner("Rebuilding database..."):
            load_vector_db()

# Example Questions Layout
st.markdown("#### Example Questions You Can Ask:")
cols = st.columns(3)
with cols[0]:
    if st.button("Free Consent in Contract Act"):
        st.session_state["preset_query"] = "What constitutes Free Consent and what are the effects of coercion under the Indian Contract Act?"
with cols[1]:
    if st.button("Vodafone International Holdings"):
        st.caption("How did the Supreme Court interpret offshore acquisitions in the Vodafone case?")
with cols[2]:
    if st.button("Consumer Protection Act"):
        st.caption("What penalties can be imposed for misleading advertisements?")

# Grab a preset query if a button was clicked
preset_val = st.session_state.get("preset_query", "")
query = st.text_input("Enter your legal question:", value=preset_val, placeholder="e.g., What are the terms of the agreement regarding termination?")

if st.button("Ask", type="primary"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing and Reasoning... 🤔"):
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
