"""
DDR Report Generator - Main Entry Point
Run with: uv run python main.py
"""

import argparse
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger
from src.config import Config
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.vectorstore.store import VectorStore
from src.graph.pipeline import DDRPipeline
from src.report.builder import PDFReportBuilder

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Detailed Diagnosis Report (DDR) from inspection PDFs"
    )
    parser.add_argument(
        "--inspection",
        type=str,
        default=None,
        help="Path to the inspection report PDF (default: inputs/Sample_Report.pdf)",
    )
    parser.add_argument(
        "--thermal",
        type=str,
        default=None,
        help="Path to the thermal report PDF (default: inputs/Thermal_Images.pdf)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: outputs/DDR_Report_<timestamp>.pdf)",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Force re-ingestion: clears ChromaDB and re-processes PDFs",
    )
    parser.add_argument(
        "--clean-images",
        action="store_true",
        help="Delete all previously extracted images before re-extracting",
    )
    return parser.parse_args()


def _purge_old_images(images_dir: Path) -> None:
    """
    Remove legacy per-page image files (inspection_pageXXX_imgYY.png,
    thermal_pageXXX_imgYY.png) that were produced by the old extractor.
    The new xref-based extractor produces *_xrefNNNNN.png filenames instead.
    """
    removed = 0
    for pattern in ["inspection_page*.png", "thermal_page*.png"]:
        for f in images_dir.glob(pattern):
            try:
                f.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Could not remove old image {f.name}: {e}")
    if removed:
        logger.info(f"Cleaned up {removed} legacy per-page image files")


def main():
    args = parse_args()
    config = Config()

    logger.info("=" * 60)
    logger.info("DDR Report Generator Starting")
    logger.info("=" * 60)

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
        logger.error(
            "Place your PDFs in the inputs/ folder or use "
            "--inspection / --thermal to specify paths."
        )
        sys.exit(1)

    logger.info(f"Inspection PDF : {inspection_pdf}")
    logger.info(f"Thermal PDF    : {thermal_pdf}")

    # ── 2. Purge legacy images if requested ───────────────────────
    if args.clean_images or args.reingest:
        logger.info("Purging legacy per-page image files…")
        _purge_old_images(config.images_dir)

    # ── 3. Initialise Vector Store ─────────────────────────────────
    vector_store = VectorStore(config)

    if args.reingest:
        logger.info("--reingest flag set: clearing existing ChromaDB collection")
        vector_store.clear()

    # ── 4. Ingest PDFs (skip if already indexed) ───────────────────
    if vector_store.is_empty() or args.reingest:
        logger.info("Ingesting PDFs into vector store…")

        parser = PDFParser(config)
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
    parser = PDFParser(config)
    image_map = parser.extract_images(inspection_pdf, thermal_pdf)
    total_imgs = sum(len(v) for v in image_map.values())
    logger.info(f"Extracted {total_imgs} unique images total "
                f"({len(image_map.get('inspection', []))} inspection, "
                f"{len(image_map.get('thermal', []))} thermal)")

    # ── 6. Verify Cohere API key ───────────────────────────────────
    logger.info("Verifying Cohere API key…")
    try:
        import httpx
        resp = httpx.get(
            "https://api.cohere.com/v1/check-api-key",
            headers={"Authorization": f"Bearer {config.cohere_api_key}"},
            timeout=10,
        )
        if resp.status_code == 401 or (
            resp.status_code == 200 and not resp.json().get("valid", True)
        ):
            logger.error(
                "Cohere API key is INVALID.\n"
                "  Update COHERE_API_KEY in .env:\n"
                "  https://dashboard.cohere.com/api-keys"
            )
            sys.exit(1)
        logger.info("Cohere API key valid ✓")
    except Exception as e:
        logger.warning(f"Could not pre-check Cohere API key: {e} — proceeding anyway")

    # ── 7. Run LangGraph DDR pipeline ─────────────────────────────
    logger.info("Running LangGraph DDR generation pipeline…")
    pipeline = DDRPipeline(config, vector_store)
    report_state = pipeline.run(image_map=image_map)
    logger.info("Pipeline complete — all sections generated")

    # ── 8. Build PDF report ────────────────────────────────────────
    logger.info("Building final PDF report…")
    builder = PDFReportBuilder(config)
    output_path = builder.build(report_state, output_path=args.output)

    logger.info("=" * 60)
    logger.info(f"DDR Report generated successfully!")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
