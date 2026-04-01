"""
Centralised configuration — reads from .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class Config:
    # ── LLM (Cohere) ─────────────────────────────────────────────────
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    cohere_model: str = os.getenv("COHERE_MODEL", "command-r-plus-08-2024")

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ── Vector Store ──────────────────────────────────────────────────
    chroma_db_path: Path = BASE_DIR / os.getenv("CHROMA_DB_PATH", "chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION_NAME", "ddr_inspection_docs")

    # ── Paths ─────────────────────────────────────────────────────────
    input_dir: Path = BASE_DIR / os.getenv("INPUT_DIR", "inputs")
    output_dir: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
    images_dir: Path = BASE_DIR / os.getenv("IMAGES_DIR", "outputs/images")

    # ── Default PDF filenames ──────────────────────────────────────────────────
    # These match the actual files in inputs/. Override via CLI --inspection / --thermal.
    default_inspection_pdf: str = os.getenv("INSPECTION_PDF", "Sample_Report.pdf")
    default_thermal_pdf: str = os.getenv("THERMAL_PDF", "Thermal_Images.pdf")

    # ── Chunking ──────────────────────────────────────────────────────
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ── Report Metadata ───────────────────────────────────────────────
    report_title: str = os.getenv("REPORT_TITLE", "Detailed Diagnosis Report")
    company_name: str = os.getenv("COMPANY_NAME", "UrbanRoof Private Limited")
    company_website: str = os.getenv("COMPANY_WEBSITE", "www.urbanroof.in")

    # ── RAG Retrieval ─────────────────────────────────────────────────
    retrieval_k: int = 8  # top-k chunks per query

    def validate(self):
        if not self.cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY is not set. "
                "Please add your Cohere API key to .env  "
                "(get one free at https://dashboard.cohere.com/api-keys)"
            )

    def __init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.validate()
