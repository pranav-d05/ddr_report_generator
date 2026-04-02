"""
Centralised configuration — reads from .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

BASE_DIR = Path(__file__).parent.parent


class Config:
    # ── LLM (OpenRouter) ─────────────────────────────────────────────
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ── Vector Store ──────────────────────────────────────────────────
    chroma_db_path: Path = BASE_DIR / os.getenv("CHROMA_DB_PATH", "chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION_NAME", "ddr_inspection_docs")

    # ── Paths ─────────────────────────────────────────────────────────
    input_dir: Path  = BASE_DIR / os.getenv("INPUT_DIR",  "inputs")
    output_dir: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
    images_dir: Path = BASE_DIR / os.getenv("IMAGES_DIR", "outputs/images")

    # ── Default PDF filenames ──────────────────────────────────────────────────
    default_inspection_pdf: str = os.getenv("INSPECTION_PDF", "Sample_Report.pdf")
    default_thermal_pdf: str    = os.getenv("THERMAL_PDF",    "Thermal_Images.pdf")

    # ── Chunking ──────────────────────────────────────────────────────
    # Larger chunks preserve the "Impacted Area N / Negative side / Positive side"
    # structure of the inspection report intact within a single chunk.
    chunk_size: int    = int(os.getenv("CHUNK_SIZE",    "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ── Report Metadata ───────────────────────────────────────────────
    report_title: str    = os.getenv("REPORT_TITLE",    "Detailed Diagnosis Report")
    company_name: str    = os.getenv("COMPANY_NAME",    "UrbanRoof Private Limited")
    company_website: str = os.getenv("COMPANY_WEBSITE", "www.urbanroof.in")

    # ── RAG Retrieval ─────────────────────────────────────────────────
    retrieval_k: int = 8   # top-k chunks per query

    # ── LangSmith Observability ───────────────────────────────────────
    langsmith_tracing:  bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower().strip() == "true"
    langsmith_api_key:  str  = os.getenv("LANGCHAIN_API_KEY",    "").strip()
    langsmith_project:  str  = os.getenv("LANGCHAIN_PROJECT",    "ddr-report-generator").strip()
    langsmith_endpoint: str  = os.getenv("LANGCHAIN_ENDPOINT",   "https://api.smith.langchain.com").strip()

    def validate(self):
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Please add your OpenRouter API key to .env. "
            )

    def setup_tracing(self) -> None:
        """
        Bootstrap LangSmith tracing.

        When LANGCHAIN_TRACING_V2=true this method ensures all required env vars
        are in place (LangChain reads them at import time AND at call time), then
        logs a clear status message.
        """
        from src.utils.logger import get_logger
        _log = get_logger(__name__)

        if not self.langsmith_tracing:
            _log.info("LangSmith tracing ✓ disabled (set LANGCHAIN_TRACING_V2=true to enable)")
            return

        if not self.langsmith_api_key or self.langsmith_api_key == "your-langsmith-api-key-here":
            _log.warning(
                "LangSmith tracing is enabled but LANGCHAIN_API_KEY is not set. "
                "Tracing will be skipped. Add your key to .env: https://smith.langchain.com"
            )
            return

        # Ensure the env vars are set for langchain/langsmith internals
        os.environ["LANGCHAIN_TRACING_V2"]  = "true"
        os.environ["LANGCHAIN_API_KEY"]      = self.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"]      = self.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"]     = self.langsmith_endpoint

        _log.info(
            f"LangSmith tracing ✓ ENABLED — project='{self.langsmith_project}' "
            f"endpoint='{self.langsmith_endpoint}'"
        )
        _log.info(f"  → View traces at: https://smith.langchain.com/projects")

    def __init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.validate()
