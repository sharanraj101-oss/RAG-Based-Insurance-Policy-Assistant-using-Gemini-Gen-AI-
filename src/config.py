import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "kb"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vectorstore"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# LLM & Embedding Settings
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking Configuration
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Retrieval Configuration
TOP_K_DENSE = 6
TOP_K_SPARSE = 6
TOP_K_FINAL = 3
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

# System Prompts
SYSTEM_PROMPT = """You are an AI Insurance Policy Assistant specialized in explaining policy terms, coverage details, claim procedures, premiums, and IRDAI regulations.

Strict Directives:
1. Answer the user's question using ONLY the provided policy context and tool outputs.
2. If the context does not contain enough information, clearly state: "I could not find relevant details in the current policy documentation." Do NOT invent or fabricate insurance coverage or terms.
3. Be professional, empathetic, clear, and structured (use bullet points and bold headers).
4. Always cite the specific policy clause or source document when answering.
5. If the query asks for live claim status or premium calculations, guide the user to use the interactive tools or provide clear instructions.
"""
