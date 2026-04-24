"""
Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent


class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    API_KEY = GEMINI_API_KEY or GROQ_API_KEY
    
    # Paths
    DATA_PATH = PROJECT_ROOT / "data"
    CHROMA_PATH = PROJECT_ROOT / "vectorstore" / "chroma_db"
    
    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Models
    GROQ_MODEL = "llama-3.3-70b-versatile"
    MODEL_NAME = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    TOP_K = 5