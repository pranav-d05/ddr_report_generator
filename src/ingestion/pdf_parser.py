"""
PDF Parser — extracts text and images from inspection/thermal PDFs using PyMuPDF.

Images are extracted once per unique xref (globally deduplicated across pages),
saved as <doc_type>_xref<NNNNN>.png in outputs/images/.

Section hints are assigned from page text (keyword scoring) with a page-range
fallback for thermal PDFs whose pages lack descriptive text.
"""

from __future__ import annotations

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
    "bathroom":     ["bathroom", "bath", "nahani", "toilet", "wc", "plumbing"],
    "balcony":      ["balcony", "open balcony"],
    "terrace":      ["terrace", "roof", "ips", "screed", "parapet"],
    "external_wall":["external wall", "exterior wall", "parapet wall", "chajja"],
    "plaster":      ["plaster", "substrate", "sand faced", "re-plaster"],
    "structural":   ["structural", "beam", "column", "reinforcement", "spalling"],
    "thermal":      ["thermal", "thermograph", "temperature", "\u00b0c", "bosch", "ir"],
    "summary":      ["summary", "introduction", "background", "objective", "scope"],
    "analysis":     ["analysis", "suggestion", "therapy", "treatment", "action"],
}


def _detect_section(text: str) -> str:
    """Return the highest-scoring section key for a text block."""
    text_lower = text.lower()
    scores: Dict[str, int] = {
        sec: sum(text_lower.count(kw) for kw in kws)
        for sec, kws in SECTION_KEYWORDS.items()
    }
    best = max(scores, key=lambda s: scores[s])
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Thermal-PDF page-range section map (fallback when page text is sparse)
# 1-indexed page numbers.
# ---------------------------------------------------------------------------
_THERMAL_PAGE_MAP: List[Tuple[int, int, str]] = [
    (1,   2,  "summary"),
    (3,   6,  "bathroom"),
    (7,   10, "external_wall"),
    (11,  14, "bathroom"),
    (15,  18, "external_wall"),
    (19,  22, "terrace"),
    (23,  26, "balcony"),
    (27,  99, "structural"),
]


def _thermal_hint(page_num: int, page_text: str) -> str:
    """Derive section hint for a thermal-PDF page."""
    if page_text and len(page_text.strip()) > 30:
        hint = _detect_section(page_text)
        # Only trust detector if it returns a specific area (not generic labels)
        if hint not in ("general", "thermal", "summary"):
            return hint
    for start, end, hint in _THERMAL_PAGE_MAP:
        if start <= page_num <= end:
            return hint
    return "thermal"


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
                        "source": pdf_path.name,
                        "doc_type": doc_type,
                        "page": page_num,
                        "total_pages": total_pages,
                        "section_hint": section_hint,
                        "has_images": len(image_list) > 0,
                        "image_count": len(image_list),
                    },
                ))

        logger.info(f"  Extracted {len(docs)} pages from {pdf_path.name}")
        return docs

    def extract_images(self, *pdf_paths: Path) -> Dict[str, List[Dict]]:
        """
        Extract all unique images from each PDF (keyed by doc_type).

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
            logger.info(f"  Extracted {len(images)} unique images from [{doc_type}] {pdf_path.name}")
        return image_map

    # ── Internal ────────────────────────────────────────────────────────────

    def _extract_images_from_pdf(
        self,
        pdf_path: Path,
        doc_type: str,
        min_width: int = 250,
        min_height: int = 200,
    ) -> List[Dict]:
        """
        Two-pass extraction:
          Pass 1 — record the first page each xref appears on + its section_hint.
          Pass 2 — save each unique xref once, skip small/decorative images.
        """
        from PIL import Image as PILImg

        xref_to_hint: Dict[int, str] = {}
        xref_to_page: Dict[int, int] = {}
        results: List[Dict] = []

        with fitz.open(str(pdf_path)) as pdf:
            # Pass 1: map xref -> (first_page, section_hint)
            for page_num, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()
                hint = (
                    _thermal_hint(page_num, page_text)
                    if doc_type == "thermal"
                    else (_detect_section(page_text) if page_text else "general")
                )
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    if xref not in xref_to_hint:
                        xref_to_hint[xref] = hint
                        xref_to_page[xref] = page_num

            # Pass 2: save unique images that pass size filter
            for xref, hint in xref_to_hint.items():
                filename = f"{doc_type}_xref{xref:05d}.png"
                save_path = self.config.images_dir / filename

                try:
                    base = pdf.extract_image(xref)
                    image_bytes = base["image"]
                    image_ext = base["ext"]

                    # Verify dimensions via PIL
                    try:
                        pil = PILImg.open(io.BytesIO(image_bytes))
                        w, h = pil.size
                        if w < min_width or h < min_height:
                            continue
                        # Reject colour-scale slivers (e.g. 10px wide, 200px tall)
                        aspect = max(w, h) / max(min(w, h), 1)
                        if min(w, h) < 80 and aspect > 3:
                            continue
                    except Exception:
                        continue  # can't decode — skip

                    if not save_path.exists():
                        self._save_as_png(image_bytes, image_ext, save_path)

                    results.append({
                        "path": save_path,
                        "filename": filename,
                        "page": xref_to_page[xref],
                        "xref": xref,
                        "doc_type": doc_type,
                        "section_hint": hint,
                        "source": pdf_path.name,
                    })

                except Exception as exc:
                    logger.warning(f"  Could not extract xref={xref}: {exc}")

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
