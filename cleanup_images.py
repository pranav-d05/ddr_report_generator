"""
cleanup_images.py — Remove legacy per-page duplicate images from outputs/images/

The old pdf_parser extracted images per-page (inspection_pageXXX_imgYY.png,
thermal_pageXXX_imgYY.png) creating thousands of duplicates.
The new extractor uses xref-based deduplication (*_xrefNNNNN.png).

Run this once:
    uv run python cleanup_images.py

No API key required — this script only touches the local filesystem.
"""

import sys
from pathlib import Path

# Resolve project root so this works from any working directory
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    # Derive images_dir directly from .env or fall back to default — no Config() needed
    from dotenv import load_dotenv
    import os
    load_dotenv(PROJECT_ROOT / ".env")

    images_dir = Path(
        os.getenv("IMAGES_DIR", str(PROJECT_ROOT / "outputs" / "images"))
    ).resolve()

    if not images_dir.exists():
        logger.info(f"Images directory does not exist: {images_dir}")
        return

    logger.info(f"Scanning: {images_dir}")

    patterns = ["inspection_page*.png", "thermal_page*.png"]
    total_removed = 0
    total_size = 0

    for pattern in patterns:
        files = list(images_dir.glob(pattern))
        for f in files:
            try:
                size = f.stat().st_size
                f.unlink()
                total_removed += 1
                total_size += size
            except Exception as e:
                logger.warning(f"Could not remove {f.name}: {e}")

    size_mb = total_size / (1024 * 1024)
    if total_removed:
        logger.info(f"Removed {total_removed} legacy image files ({size_mb:.1f} MB freed)")
    else:
        logger.info("No legacy per-page images found — nothing to remove")

    # Report what remains
    remaining_xref = list(images_dir.glob("*_xref*.png"))
    logger.info(f"Remaining xref-based images: {len(remaining_xref)}")


if __name__ == "__main__":
    main()
