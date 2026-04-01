"""
Diagnostic script — run this to verify image extraction and section assignment
BEFORE running the full DDR pipeline.

Usage:
    uv run python diagnose.py
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.ingestion.pdf_parser import PDFParser


def main():
    config = Config()

    inspection_pdf = config.input_dir / config.default_inspection_pdf
    thermal_pdf    = config.input_dir / config.default_thermal_pdf

    print(f"\n{'='*60}")
    print("DDR Image Extraction Diagnostic")
    print(f"{'='*60}\n")

    # Clean images for fresh extraction
    print("Cleaning old extracted images...")
    removed = 0
    for f in config.images_dir.glob("*.png"):
        f.unlink()
        removed += 1
    for f in config.images_dir.glob("*.json"):
        f.unlink()
        removed += 1
    print(f"  Removed {removed} files\n")

    parser = PDFParser(config)
    image_map = parser.extract_images(inspection_pdf, thermal_pdf)

    for doc_type, images in image_map.items():
        print(f"\n{'─'*50}")
        print(f"  {doc_type.upper()} IMAGES: {len(images)} total")
        print(f"{'─'*50}")

        dist = Counter(i["section_hint"] for i in images)
        print(f"  Section distribution: {dict(dist)}\n")

        # Show overlay vs real photo counts for thermal
        if doc_type == "thermal":
            overlays = sum(1 for i in images if i.get("is_thermal_overlay"))
            reals    = len(images) - overlays
            print(f"  Real photos: {reals}, Thermal overlays: {overlays}\n")

        for img in images:
            overlay_flag = " [THERMAL OVERLAY]" if img.get("is_thermal_overlay") else " [REAL PHOTO]"
            print(
                f"  xref={img['xref']:05d} | seq={img.get('seq_idx','?'):3} | "
                f"page={img['page']:2} | hint={img['section_hint']:<15} | "
                f"{img['width']}x{img['height']}{overlay_flag if doc_type=='thermal' else ''}"
            )

    # Verify per-area image assignment
    print(f"\n{'='*60}")
    print("Area → Image Assignment Preview")
    print(f"{'='*60}\n")

    from src.graph.nodes import DDR_AREAS, AREA_TO_SECTION_KEY
    from src.graph.nodes import _assign_images_for_area

    for area in DDR_AREAS:
        section_key = AREA_TO_SECTION_KEY.get(area, "general")
        img_dict    = _assign_images_for_area(image_map, section_key)
        n_v = len(img_dict.get("visual",  []))
        n_t = len(img_dict.get("thermal", []))
        status = "✓" if (n_v + n_t) > 0 else "✗ NO IMAGES"
        print(f"  {status}  {area:<35} → {n_v} visual, {n_t} thermal")
        for p in img_dict.get("visual",  []):
            print(f"        [V] {Path(p).name}")
        for p in img_dict.get("thermal", []):
            print(f"        [T] {Path(p).name}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
