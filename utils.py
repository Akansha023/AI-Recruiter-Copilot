"""
Core AI utility functions for the Recruiter Copilot.
Handles PDF extraction, embedding, vector search, and LLM calls.
"""

import os
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
import uuid

# ─── Configure Gemini ─────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")   # free & fast

# ─── Embedding Model ───────────────────────────────────────────────────────────
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ─── ChromaDB (in-memory, resets per session) ─────────────────────────────────
chroma_client = chromadb.Client()


def get_or_create_collection(name: str = "resume_store"):
    """Get existing collection or create new one."""
    try:
        return chroma_client.get_collection(name)
    except Exception:
        return chroma_client.create_collection(name)


# ─── PDF Extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_file) -> str:
    """Extract raw text from an uploaded PDF file object."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()


# ─── Text Chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


# ─── Embedding & Storage ───────────────────────────────────────────────────────
def store_resume_embeddings(chunks: list[str], collection_name: str = "resume_store") -> str:
    """
    Embed and store chunks in ChromaDB.
    Returns collection name used.
    """
    # Use unique collection per analysis to avoid ID conflicts
    unique_name = f"{collection_name}_{uuid.uuid4().hex[:8]}"
    collection = chroma_client.create_collection(unique_name)

    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )
    return unique_name


def retrieve_relevant_context(query: str, collection_name: str, n_results: int = 4) -> str:
    """
    Retrieve the most relevant chunks from stored embeddings.
    Returns concatenated context string.
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        query_embedding = embed_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count())
        )
        return "\n\n".join(results['documents'][0])
    except Exception as e:
        return f"[Context retrieval error: {e}]"


# ─── LLM Call ─────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    """Send prompt to Gemini and return text response."""
    if not GOOGLE_API_KEY:
        return "⚠️ No API key found. Please set GOOGLE_API_KEY environment variable."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"


# ─── Full Pipeline ─────────────────────────────────────────────────────────────
def process_resume(pdf_file) -> tuple[str, str]:
    """
    Full pipeline: PDF → text → chunks → embeddings.
    Returns (raw_text, collection_name)
    """
    raw_text = extract_text_from_pdf(pdf_file)
    if not raw_text:
        raise ValueError("Could not extract text from PDF. Try a different file.")

    chunks = chunk_text(raw_text)
    collection_name = store_resume_embeddings(chunks)
    return raw_text, collection_name
