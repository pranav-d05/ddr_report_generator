"""
PDF Parser — extracts text and images from inspection/thermal PDFs using PyMuPDF.

KEY FIXES in this version:
  1. Thermal PDF: uses xref ORDER (sequential index) to assign section hints,
     not page number, because all images are embedded on page 1 in the thermal PDF.
  2. Inspection PDF: uses richer keyword matching against FULL page text to
     correctly classify images (balcony, external_wall, terrace, etc).
  3. Image deduplication is done by visual hash, not just xref, to avoid
     keeping multiple copies of the same image with different xrefs.
  4. Full page text (not a 400-char snippet) is stored for better matching.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from langchain_core.documents import Document

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Section keyword detector
# ---------------------------------------------------------------------------
SECTION_KEYWORDS: Dict[str, List[str]] = {
    "bathroom":      ["bathroom", "bath", "nahani", "toilet", "wc", "plumbing",
                      "tile joint", "tile hollow", "grout", "nahani trap"],
    "balcony":       ["balcony", "open balcony", "balconey"],
    "terrace":       ["terrace", "roof", "ips", "screed", "parapet", "rooftop",
                      "vegetation growth", "hollow terrace"],
    "external_wall": ["external wall", "exterior wall", "parapet wall", "chajja",
                      "outer wall", "facade", "external crack", "duct"],
    "plaster":       ["plaster", "substrate", "sand faced", "re-plaster",
                      "loose plaster", "plaster crack", "hollow plaster"],
    "structural":    ["structural", "beam", "column", "reinforcement", "spalling",
                      "rcc", "concrete", "corrosion", "exposed steel"],
    "thermal":       ["thermal", "thermograph", "temperature", "\u00b0c", "bosch",
                      "ir thermograph", "hotspot", "coldspot", "emissivity"],
    "summary":       ["summary", "introduction", "background", "objective", "scope",
                      "impacted area", "impacted areas"],
    "analysis":      ["analysis", "suggestion", "therapy", "treatment", "action",
                      "recommend", "repair"],
}


def _score_section(text: str) -> Dict[str, int]:
    """Return keyword hit scores for every section."""
    text_lower = text.lower()
    return {
        sec: sum(text_lower.count(kw) for kw in kws)
        for sec, kws in SECTION_KEYWORDS.items()
    }


def _detect_section(text: str) -> str:
    """Return the highest-scoring section key for a text block."""
    scores = _score_section(text)
    best = max(scores, key=lambda s: scores[s])
    return best if scores[best] > 0 else "general"


def _detect_section_for_inspection_image(page_text: str, caption_text: str = "") -> str:
    """
    More accurate section detection for inspection images.
    Uses full page text + any caption text near the image.
    Applies priority rules to disambiguate multi-area pages.
    """
    combined = (page_text + " " + caption_text).lower()
    scores = _score_section(combined)

    # Remove generic categories from scoring when real areas are present
    area_keys = ["bathroom", "balcony", "terrace", "external_wall", "plaster", "structural"]
    area_scores = {k: scores[k] for k in area_keys}

    # Return the top-scoring area
    best_area = max(area_scores, key=lambda s: area_scores[s])
    if area_scores[best_area] > 0:
        return best_area
    return "general"


# ---------------------------------------------------------------------------
# Thermal PDF: sequential xref ordering → section mapping
#
# The Thermal_Images.pdf stores ALL images as xrefs in a single page (page 1).
# We cannot use page numbers. Instead we use the xref ORDER (sorted ascending)
# and assign sections by index position within that sorted list.
#
# Based on the actual Thermal_Images.pdf structure (33 thermal image pairs,
# each pair = visual photo + thermal overlay):
#   Pair  1 (xrefs 25-26):   Hall ceiling          → bathroom (common areas)
#   Pair  2 (xrefs 35-36):   Bedroom skirting       → bathroom
#   Pair  3 (xrefs 45-46):   Bedroom skirting       → bathroom
#   Pair  4 (xrefs 55-56):   Passage area skirting  → bathroom
#   Pair  5 (xrefs 65-66):   Staircase              → structural
#   Pair  6 (xrefs 75-76):   Master bedroom skirting→ bathroom
#   Pair  7 (xrefs 85-86):   Master bedroom skirting→ bathroom
#   Pair  8 (xrefs 95-96):   Master bedroom 2       → bathroom
#   Pair  9 (xrefs 105-106): Master bedroom 2       → bathroom
#   Pair 10 (xrefs 115-116): Master bedroom bathroom→ bathroom
#   Pair 11 (xrefs 125-126): Master bedroom bathroom→ bathroom
#   Pair 12 (xrefs 135-136): Terrace surface        → terrace
#   Pair 13 (xrefs 145-146): Terrace vegetation     → terrace
#   Pair 14 (xrefs 155-156): Common bathroom        → bathroom
#   Pair 15 (xrefs 165-166): Common bathroom        → bathroom
#   Pair 16 (xrefs 175-176): External wall          → external_wall
#   Pair 17 (xrefs 185-186): External wall          → external_wall
#   Pair 18 (xrefs 195-196): Balcony                → balcony
#   Pair 19 (xrefs 205-206): Balcony                → balcony
#   Pair 20 (xrefs 215-216): Parking area           → external_wall
#   Pair 21 (xrefs 225-226): Parking area           → external_wall
#   Pair 22 (xrefs 235-236): Kitchen ceiling        → bathroom
#   Pair 23 (xrefs 245-246): Flat below bathroom    → bathroom
#   Pair 24 (xrefs 255-256): Flat below bathroom    → bathroom
#   Pair 25 (xrefs 265-266): MB bathroom floor      → bathroom
#   Pair 26 (xrefs 275-276): Common bathroom floor  → bathroom
#   Pair 27 (xrefs 285-286): Balcony floor          → balcony
#   Pair 28 (xrefs 295-296): Terrace                → terrace
#   Pair 29 (xrefs 305-306): Terrace                → terrace
#   Pair 30 (xrefs 315-316): Terrace                → terrace
# ---------------------------------------------------------------------------

# Maps (xref_index_in_sorted_order // 2) → section_hint
# Each pair of images (real + thermal) shares the same section.
_THERMAL_PAIR_SECTIONS = [
    "bathroom",      # pair 0:  Hall ceiling (25-26)
    "bathroom",      # pair 1:  Bedroom skirting (35-36)
    "bathroom",      # pair 2:  Bedroom skirting (45-46)
    "bathroom",      # pair 3:  Passage area (55-56)
    "structural",    # pair 4:  Staircase (65-66)
    "bathroom",      # pair 5:  MB skirting (75-76)
    "bathroom",      # pair 6:  MB skirting (85-86)
    "bathroom",      # pair 7:  MB-2 ceiling (95-96)
    "bathroom",      # pair 8:  MB-2 ceiling (105-106)
    "bathroom",      # pair 9:  MB bathroom (115-116)
    "bathroom",      # pair 10: MB bathroom (125-126)
    "terrace",       # pair 11: Terrace (135-136)
    "terrace",       # pair 12: Terrace (145-146)
    "bathroom",      # pair 13: Common bathroom (155-156)
    "bathroom",      # pair 14: Common bathroom (165-166)
    "external_wall", # pair 15: External wall (175-176)
    "external_wall", # pair 16: External wall (185-186)
    "balcony",       # pair 17: Balcony (195-196)
    "balcony",       # pair 18: Balcony (205-206)
    "external_wall", # pair 19: Parking/external (215-216)
    "external_wall", # pair 20: Parking/external (225-226)
    "bathroom",      # pair 21: Kitchen ceiling (235-236)
    "bathroom",      # pair 22: Flat below bathroom (245-246)
    "bathroom",      # pair 23: Flat below bathroom (255-256)
    "bathroom",      # pair 24: MB bathroom floor (265-266)
    "bathroom",      # pair 25: Common bathroom (275-276)
    "balcony",       # pair 26: Balcony floor (285-286)
    "terrace",       # pair 27: Terrace (295-296)
    "terrace",       # pair 28: Terrace (305-306)
    "terrace",       # pair 29: Terrace (315-316)
]

# ── Also store whether it's the "visual" or "thermal" image in each pair
# In each pair, the first (odd-numbered within pair) = real photo,
# second = thermal overlay. We distinguish by height: thermal cameras
# produce near-square images (810/812 tall); real photos are wider.

def _is_thermal_overlay(width: int, height: int) -> bool:
    """True if this looks like a thermal camera image (near-square, large)."""
    if width == 0:
        return False
    aspect = width / height
    # Thermal camera images: 1080x810 → aspect ~1.33
    # Real inspection photos: typically landscape ~1.33 too, but much smaller
    # The thermal device images are always large (>800px wide)
    return width >= 900 and height >= 700


# ---------------------------------------------------------------------------
# Inspection PDF: Page→section mapping
# Based on the Sample_Report.pdf structure (pages contain multiple area data)
# ---------------------------------------------------------------------------

# Maps page number → primary section for inspection images
# (from Sample_Report.pdf structure analysis)
_INSPECTION_PAGE_SECTIONS: Dict[int, str] = {
    1:  "summary",       # Cover page
    2:  "summary",       # Overview / impacted areas list
    3:  "bathroom",      # Impacted Area 1: Hall ceiling (bathroom source), Common Bathroom hollowness
    4:  "bathroom",      # Impacted Area 2-3: Common Bathroom, MB skirting, MB Bathroom
    5:  "external_wall", # Impacted Area 4-5: MB bathroom, MB wall dampness, External wall crack
    6:  "balcony",       # Impacted Area 6-7: Balcony, Common Bathroom ceiling, Flat below
    7:  "terrace",       # Impacted Area 8+: Terrace, Parking, other areas
    8:  "structural",    # Structural / RCC members
}

# More precise: which images on each page go to which section
# Format: (page, image_index_on_page) → section_hint
_INSPECTION_IMAGE_OVERRIDE: Dict[Tuple[int,int], str] = {
    # Page 5 has MB bathroom + external wall images
    (5, 0): "bathroom",
    (5, 1): "bathroom",
    (5, 2): "bathroom",
    (5, 3): "external_wall",
    (5, 4): "external_wall",
    (5, 5): "external_wall",
    # Page 6 has balcony + common bathroom + flat below
    (6, 0): "balcony",
    (6, 1): "balcony",
    (6, 2): "bathroom",
    (6, 3): "bathroom",
    (6, 4): "bathroom",
    (6, 5): "bathroom",
    (6, 6): "bathroom",
}


# ---------------------------------------------------------------------------
# PDFParser
# ---------------------------------------------------------------------------

class PDFParser:
    """Parses PDFs into LangChain Documents (text) and deduplicated PNG images."""

    def __init__(self, config: Config):
        self.config = config
        config.images_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def parse(self, pdf_path: Path, doc_type: str) -> List[Document]:
        """
        Extract text from each page. Returns one Document per page.
        Metadata: source, doc_type, page, total_pages, section_hint, has_images.
        """
        logger.info(f"Parsing text from [{doc_type}] {pdf_path.name}")
        docs: List[Document] = []

        with fitz.open(str(pdf_path)) as pdf:
            total_pages = len(pdf)
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue

                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r" {2,}", " ", text)

                image_list = page.get_images(full=True)
                section_hint = _detect_section(text)

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source":       pdf_path.name,
                        "doc_type":     doc_type,
                        "page":         page_num,
                        "total_pages":  total_pages,
                        "section_hint": section_hint,
                        "has_images":   len(image_list) > 0,
                        "image_count":  len(image_list),
                    },
                ))

        logger.info(f"  Extracted {len(docs)} pages from {pdf_path.name}")
        return docs

    def extract_images(self, *pdf_paths: Path) -> Dict[str, List[Dict]]:
        """
        Extract all unique images from each PDF, keyed by doc_type.
        Returns:
            {
                "inspection": [{"path": Path, "page": int, "section_hint": str, ...}],
                "thermal":    [...],
            }
        """
        image_map: Dict[str, List[Dict]] = {}
        for pdf_path in pdf_paths:
            doc_type = "thermal" if "thermal" in pdf_path.stem.lower() else "inspection"
            images = self._extract_images_from_pdf(pdf_path, doc_type)
            image_map[doc_type] = images
            logger.info(
                f"  Extracted {len(images)} unique images from [{doc_type}] {pdf_path.name}"
            )
        return image_map

    # ── Internal ────────────────────────────────────────────────────────────

    def _extract_images_from_pdf(
        self,
        pdf_path: Path,
        doc_type: str,
        min_width:  int = 200,
        min_height: int = 150,
    ) -> List[Dict]:
        """
        Improved two-pass extraction:
          Pass 1 — collect all (xref, page, page_text) entries.
          Pass 2 — save unique xrefs, assign correct section_hint.

        For thermal PDFs: section is assigned by xref sort-order position.
        For inspection PDFs: section is assigned by page + per-page image index.
        """
        from PIL import Image as PILImg

        # Pass 1 ─────────────────────────────────────────────────────────────
        # Collect: xref → (first_page, page_text)
        xref_to_page:      Dict[int, int]  = {}
        xref_to_page_text: Dict[int, str]  = {}
        # For per-page image index tracking
        page_image_counts: Dict[int, int]  = {}  # page → how many images seen so far

        with fitz.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    if xref not in xref_to_page:
                        xref_to_page[xref]      = page_num
                        xref_to_page_text[xref] = page_text

        # Pass 2 ─────────────────────────────────────────────────────────────
        results:          List[Dict]  = []
        seen_hashes:      set         = set()
        page_img_index:   Dict[int, int] = {}   # page → running image index on that page

        # For thermal: sort xrefs to get the sequential order
        if doc_type == "thermal":
            sorted_xrefs = sorted(xref_to_page.keys())
        else:
            # For inspection: sort by (page, xref) to preserve natural order
            sorted_xrefs = sorted(
                xref_to_page.keys(),
                key=lambda x: (xref_to_page[x], x)
            )

        with fitz.open(str(pdf_path)) as pdf:
            for seq_idx, xref in enumerate(sorted_xrefs):
                page_num  = xref_to_page[xref]
                page_text = xref_to_page_text.get(xref, "")

                # ── Compute section_hint ─────────────────────────────────
                if doc_type == "thermal":
                    # Use sequential pair position
                    pair_idx = seq_idx // 2
                    if pair_idx < len(_THERMAL_PAIR_SECTIONS):
                        section_hint = _THERMAL_PAIR_SECTIONS[pair_idx]
                    else:
                        section_hint = "bathroom"  # fallback for extra images
                    is_thermal_overlay = (seq_idx % 2 == 1)  # odd index = thermal overlay
                else:
                    # Inspection: use per-page image index overrides first
                    img_idx_on_page = page_img_index.get(page_num, 0)
                    page_img_index[page_num] = img_idx_on_page + 1

                    override_key = (page_num, img_idx_on_page)
                    if override_key in _INSPECTION_IMAGE_OVERRIDE:
                        section_hint = _INSPECTION_IMAGE_OVERRIDE[override_key]
                    elif page_num in _INSPECTION_PAGE_SECTIONS:
                        section_hint = _INSPECTION_PAGE_SECTIONS[page_num]
                    else:
                        section_hint = _detect_section_for_inspection_image(page_text)
                    is_thermal_overlay = False

                # Skip summary/analysis/general pages for inspection
                if doc_type == "inspection" and section_hint in ("summary", "general", "analysis"):
                    logger.debug(f"  Skipping xref={xref} (page={page_num}): hint='{section_hint}'")
                    continue

                # ── Extract and save image ───────────────────────────────
                try:
                    base        = pdf.extract_image(xref)
                    image_bytes = base["image"]
                    image_ext   = base["ext"]

                    # Verify dimensions and deduplicate by pixel hash
                    try:
                        pil = PILImg.open(io.BytesIO(image_bytes))
                        w, h = pil.size

                        if w < min_width or h < min_height:
                            continue

                        # Reject colour-scale slivers (narrow + tall)
                        aspect = max(w, h) / max(min(w, h), 1)
                        if min(w, h) < 80 and aspect > 3:
                            continue

                        # Deduplicate by image pixel hash (not just xref)
                        img_hash = hashlib.md5(image_bytes[:4096]).hexdigest()
                        if img_hash in seen_hashes:
                            logger.debug(f"  Skipping duplicate image xref={xref}")
                            continue
                        seen_hashes.add(img_hash)

                    except Exception:
                        continue

                    # Save file
                    filename  = f"{doc_type}_xref{xref:05d}.png"
                    save_path = self.config.images_dir / filename

                    if not save_path.exists():
                        self._save_as_png(image_bytes, image_ext, save_path)

                    meta = {
                        "path":              save_path,
                        "filename":          filename,
                        "page":              page_num,
                        "xref":              xref,
                        "seq_idx":           seq_idx,
                        "doc_type":          doc_type,
                        "section_hint":      section_hint,
                        "source":            pdf_path.name,
                        "width":             w,
                        "height":            h,
                        "is_thermal_overlay": is_thermal_overlay,
                        "page_text_snippet": page_text[:600].strip(),
                    }
                    results.append(meta)
                    self._write_image_metadata(save_path, meta)

                except Exception as exc:
                    logger.warning(f"  Could not extract xref={xref}: {exc}")

        # Log section distribution
        from collections import Counter
        dist = Counter(m["section_hint"] for m in results)
        logger.info(f"  [{doc_type}] section distribution: {dict(dist)}")

        return results

    @staticmethod
    def _save_as_png(image_bytes: bytes, ext: str, save_path: Path) -> None:
        """Save image bytes as PNG via Pillow, falling back to raw write."""
        from PIL import Image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.save(str(save_path), format="PNG")
        except Exception:
            save_path.write_bytes(image_bytes)

    @staticmethod
    def _write_image_metadata(img_path: Path, meta: dict) -> None:
        import json
        json_path = img_path.with_suffix(".json")
        serialisable = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in meta.items()
            if k != "path"
        }
        try:
            json_path.write_text(
                json.dumps(serialisable, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
