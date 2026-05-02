import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Suppress symlink and dataset warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def create_vector_db():
    print("📄 Loading PDFs...")

    # Load PDFs from the 'data' directory
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    print(f"✅ Loaded {len(documents)} pages")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    print(f"✅ Split into {len(texts)} chunks")
    print("🔄 Loading embeddings...")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("⚡ Creating FAISS DB...")

    # Create FAISS database
    db = FAISS.from_documents(texts, embeddings)
    
    # Save the vector database locally
    os.makedirs("vectorstore", exist_ok=True)
    db.save_local("vectorstore/db_faiss")

    print("✅ DB created successfully in vectorstore/db_faiss")

if __name__ == "__main__":
    create_vector_db()