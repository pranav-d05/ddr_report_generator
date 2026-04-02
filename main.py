"""
DDR Report Generator - Main Entry Point
Run with: uv run python main.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger
from src.config import Config
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.vectorstore.store import VectorStore
from src.graph.pipeline import DDRPipeline
from src.report.builder import PDFReportBuilder

try:
    from langsmith import trace as ls_trace
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Detailed Diagnosis Report (DDR) from inspection PDFs"
    )
    parser.add_argument("--inspection", type=str, default=None,
                        help="Path to the inspection report PDF")
    parser.add_argument("--thermal",    type=str, default=None,
                        help="Path to the thermal report PDF")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output PDF path")
    parser.add_argument("--reingest",   action="store_true",
                        help="Force re-ingestion: clears ChromaDB and re-processes PDFs")
    parser.add_argument("--clean-images", action="store_true",
                        help="Delete all previously extracted images before re-extracting")
    return parser.parse_args()


def _clean_all_images(images_dir: Path) -> None:
    """Remove all extracted PNG and JSON files from the images directory."""
    removed = 0
    for pattern in ["*.png", "*.json"]:
        for f in images_dir.glob(pattern):
            try:
                f.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Could not remove {f.name}: {e}")
    if removed:
        logger.info(f"Cleaned up {removed} image/metadata files from {images_dir}")


def main():
    args   = parse_args()
    config = Config()

    logger.info("=" * 60)
    logger.info("DDR Report Generator Starting")
    logger.info("=" * 60)

    # ── 0. Bootstrap LangSmith tracing ─────────────────────────────
    config.setup_tracing()

    # ── 1. Resolve input PDFs ──────────────────────────────────────
    inspection_pdf = (
        Path(args.inspection) if args.inspection
        else config.input_dir / config.default_inspection_pdf
    )
    thermal_pdf = (
        Path(args.thermal) if args.thermal
        else config.input_dir / config.default_thermal_pdf
    )

    missing = []
    if not inspection_pdf.exists():
        missing.append(str(inspection_pdf))
    if not thermal_pdf.exists():
        missing.append(str(thermal_pdf))

    if missing:
        logger.error(f"Missing input PDFs: {missing}")
        sys.exit(1)

    logger.info(f"Inspection PDF : {inspection_pdf}")
    logger.info(f"Thermal PDF    : {thermal_pdf}")

    # ── 2. Clean images if requested ──────────────────────────────
    if args.clean_images or args.reingest:
        logger.info("Cleaning extracted images…")
        _clean_all_images(config.images_dir)

    # ── 3. Initialise Vector Store ─────────────────────────────────
    vector_store = VectorStore(config)

    if args.reingest:
        logger.info("--reingest flag: clearing ChromaDB collection")
        vector_store.clear()

    # ── 4. Ingest PDFs (skip if already indexed) ───────────────────
    if vector_store.is_empty() or args.reingest:
        logger.info("Ingesting PDFs into vector store…")
        parser  = PDFParser(config)
        chunker = DocumentChunker(config)

        inspection_docs = parser.parse(inspection_pdf, doc_type="inspection")
        thermal_docs    = parser.parse(thermal_pdf,    doc_type="thermal")

        all_docs = inspection_docs + thermal_docs
        logger.info(
            f"Extracted {len(inspection_docs)} pages from inspection, "
            f"{len(thermal_docs)} from thermal"
        )

        chunks = chunker.chunk(all_docs)
        logger.info(f"Created {len(chunks)} text chunks")

        vector_store.add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB")
    else:
        logger.info("ChromaDB already populated — skipping ingestion (use --reingest to force)")

    # ── 5. Extract images from PDFs ────────────────────────────────
    logger.info("Extracting images from PDFs…")
    parser    = PDFParser(config)
    image_map = parser.extract_images(inspection_pdf, thermal_pdf)

    total_imgs = sum(len(v) for v in image_map.values())
    logger.info(
        f"Total images: {total_imgs} "
        f"({len(image_map.get('inspection', []))} inspection, "
        f"{len(image_map.get('thermal', []))} thermal)"
    )

    # Log section distribution
    from collections import Counter
    for dtype, imgs in image_map.items():
        dist = Counter(i.get("section_hint", "?") for i in imgs)
        logger.info(f"  [{dtype}] section distribution: {dict(dist)}")

    # ── 6. Verify OpenRouter API key ───────────────────────────────
    logger.info("Verifying OpenRouter API key…")
    try:
        import httpx
        resp = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={
                "Authorization": f"Bearer {config.openrouter_api_key}",
                "HTTP-Referer": "https://urbanroof.in",
                "X-Title": "DDR Report Generator",
            },
            timeout=10,
        )
        if resp.status_code == 401 or resp.status_code == 403:
            logger.error(f"OpenRouter API key is INVALID (Code: {resp.status_code}). Response: {resp.text}")
            logger.error("Update OPENROUTER_API_KEY in .env")
            sys.exit(1)
        elif resp.status_code != 200:
            logger.warning(f"OpenRouter check returned {resp.status_code}: {resp.text}")
        logger.info("OpenRouter API key valid ✓")
    except Exception as e:
        logger.warning(f"Could not pre-check OpenRouter API key: {e} — proceeding anyway")

    # ── 7. Run LangGraph DDR pipeline ─────────────────────────────
    logger.info("Running LangGraph DDR generation pipeline…")
    pipeline = DDRPipeline(config, vector_store)

    # Wrap the full pipeline in a LangSmith trace when tracing is enabled
    if _LANGSMITH_AVAILABLE and config.langsmith_tracing:
        with ls_trace(
            name="ddr-report-pipeline",
            run_type="chain",
            project_name=config.langsmith_project,
            metadata={
                "inspection_pdf":    inspection_pdf.name,
                "thermal_pdf":       thermal_pdf.name,
                "openrouter_model":  config.openrouter_model,
                "total_images":      total_imgs,
                "inspection_images": len(image_map.get("inspection", [])),
                "thermal_images":    len(image_map.get("thermal",    [])),
            },
            tags=["ddr", "urbanroof", config.openrouter_model],
        ):
            report_state = pipeline.run(image_map=image_map)
    else:
        report_state = pipeline.run(image_map=image_map)

    logger.info("Pipeline complete — all sections generated")


    # ── 8. Build PDF report ────────────────────────────────────────
    logger.info("Building final PDF report…")
    builder     = PDFReportBuilder(config)
    output_path = builder.build(report_state, output_path=args.output)

    logger.info("=" * 60)
    logger.info(f"DDR Report generated successfully!")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
