"""
cleanup_images.py — Remove legacy per-page duplicate images from outputs/images/

The old pdf_parser extracted images per-page (inspection_pageXXX_imgYY.png,
thermal_pageXXX_imgYY.png) creating thousands of duplicates.
The new extractor uses xref-based deduplication (*_xrefNNNNN.png).

Run this once:
    uv run python cleanup_images.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    config = Config()
    images_dir = config.images_dir

    if not images_dir.exists():
        logger.info(f"Images directory does not exist: {images_dir}")
        return

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
    logger.info(f"Removed {total_removed} legacy image files ({size_mb:.1f} MB freed)")

    # Report what remains
    remaining_xref = list(images_dir.glob("*_xref*.png"))
    logger.info(f"Remaining xref-based images: {len(remaining_xref)}")


if __name__ == "__main__":
    main()
