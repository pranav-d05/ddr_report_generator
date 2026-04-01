"""
ReportLab PDF Builder — assembles the final DDR PDF from the pipeline state.

Sections generated:
  Cover Page (canvas-drawn, no flowable height issues)
  Table of Contents
  1. Property Issue Summary
  2. Area-wise Observations (with images)
  3. Probable Root Causes
  4. Severity Assessment (table)
  5. Recommended Actions
  6. Additional Notes
  7. Missing / Unclear Information
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.config import Config
from src.graph.state import DDRState
from src.report.styles import (
    A4, MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM,
    PAGE_WIDTH, PAGE_HEIGHT, CONTENT_WIDTH,
    NAVY, ORANGE, LIGHT_GREY, MID_GREY, DARK_GREY, WHITE,
    STYLE_COVER_TITLE, STYLE_COVER_SUBTITLE,
    STYLE_SECTION_HEADING, STYLE_SUBSECTION_HEADING,
    STYLE_BODY, STYLE_BULLET, STYLE_FOOTER, STYLE_TOC_ENTRY,
    STYLE_LABEL, SEVERITY_TABLE_STYLE,
    severity_color,
)
from src.utils.helpers import timestamp_str, human_timestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)

# DDR section order
DDR_AREAS = [
    "Bathroom / Internal Wet Areas",
    "Balcony",
    "Terrace / Roof",
    "External Wall",
    "Plaster / Substrate",
    "Structural Elements",
]


# ── Cover page drawn directly on canvas (avoids HRFlowable-in-Table None height bug) ──

def _draw_cover(canvas, doc):
    """OnPage callback: draws the full-bleed navy cover page using raw canvas calls."""
    canvas.saveState()

    # Background fill
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)

    # Company name (top area)
    canvas.setFillColor(colors.HexColor("#AABBCC"))
    canvas.setFont("Helvetica", 11)
    canvas.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT * 0.68,
                             doc.config.company_name.upper())

    # Report title (large)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 30)
    canvas.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT * 0.57,
                             doc.config.report_title)

    # Orange accent bar
    bar_w = 8 * cm
    bar_h = 3
    canvas.setFillColor(ORANGE)
    canvas.rect(
        (PAGE_WIDTH - bar_w) / 2,
        PAGE_HEIGHT * 0.52,
        bar_w, bar_h, fill=1, stroke=0,
    )

    # Generated date
    canvas.setFillColor(colors.HexColor("#CCDDEE"))
    canvas.setFont("Helvetica", 10)
    canvas.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT * 0.47,
                             f"Generated: {human_timestamp()}")

    # Website
    canvas.setFillColor(colors.HexColor("#AABBCC"))
    canvas.setFont("Helvetica", 10)
    canvas.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT * 0.43,
                             doc.config.company_website)

    # Bottom navy strip label
    canvas.setFillColor(ORANGE)
    canvas.rect(0, 0, PAGE_WIDTH, 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(PAGE_WIDTH / 2, 0.4 * cm,
                             "CONFIDENTIAL — FOR CLIENT USE ONLY")

    canvas.restoreState()


def _draw_content_page(canvas, doc):
    """OnPage callback for all content pages: draws footer only."""
    page_num = canvas.getPageNumber()

    canvas.saveState()
    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_LEFT, MARGIN_BOTTOM - 4 * mm,
                PAGE_WIDTH - MARGIN_RIGHT, MARGIN_BOTTOM - 4 * mm)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(MARGIN_LEFT, MARGIN_BOTTOM - 9 * mm,
                      f"{doc.config.company_name} — {doc.config.report_title}")
    canvas.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, MARGIN_BOTTOM - 9 * mm,
                           f"Page {page_num}")
    canvas.restoreState()


class PDFReportBuilder:
    """Builds the final DDR PDF from a fully populated DDRState."""

    def __init__(self, config: Config):
        self.config = config

    def build(self, state: DDRState, output_path: Optional[str] = None) -> Path:
        if output_path:
            out = Path(output_path)
        else:
            ts = timestamp_str()
            out = self.config.output_dir / f"DDR_Report_{ts}.pdf"

        out.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Building PDF report → {out}")

        # ── Document setup ─────────────────────────────────────────────────
        doc = BaseDocTemplate(
            str(out),
            pagesize=A4,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )
        # Attach config so page callbacks can access it
        doc.config = self.config

        cover_frame = Frame(0, 0, PAGE_WIDTH, PAGE_HEIGHT,
                            leftPadding=0, rightPadding=0,
                            topPadding=0, bottomPadding=0, id="cover")

        content_frame = Frame(
            MARGIN_LEFT, MARGIN_BOTTOM,
            CONTENT_WIDTH, PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            id="content",
        )

        doc.addPageTemplates([
            PageTemplate(id="Cover",   frames=[cover_frame],   onPage=_draw_cover),
            PageTemplate(id="Content", frames=[content_frame], onPage=_draw_content_page),
        ])

        # ── Assemble story ─────────────────────────────────────────────────
        story: list = []

        # Cover: just a blank spacer — the canvas callback draws everything
        story.append(Spacer(1, PAGE_HEIGHT))
        story.append(NextPageTemplate("Content"))
        story.append(PageBreak())

        story += self._table_of_contents()
        story.append(PageBreak())
        story += self._section_property_summary(state)
        story += self._section_area_observations(state)
        story += self._section_root_causes(state)
        story += self._section_severity(state)
        story += self._section_recommended_actions(state)
        story += self._section_additional_notes(state)
        story += self._section_missing_info(state)

        doc.build(story)
        logger.info(f"PDF report written: {out}")
        return out

    # ── Table of Contents ──────────────────────────────────────────────────

    def _table_of_contents(self) -> list:
        elements: list = []
        elements.append(Paragraph("Table of Contents", STYLE_SECTION_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1, color=NAVY))
        elements.append(Spacer(1, 6))
        toc_items = [
            "1.  Property Issue Summary",
            "2.  Area-wise Observations",
            "3.  Probable Root Causes",
            "4.  Severity Assessment",
            "5.  Recommended Actions",
            "6.  Additional Notes",
            "7.  Missing / Unclear Information",
        ]
        for item in toc_items:
            elements.append(Paragraph(item, STYLE_TOC_ENTRY))
        return elements

    # ── Section helpers ────────────────────────────────────────────────────

    def _section_header(self, number: int, title: str) -> list:
        return [
            Paragraph(f"{number}.  {title}", STYLE_SECTION_HEADING),
            HRFlowable(width=CONTENT_WIDTH, thickness=1, color=ORANGE, spaceAfter=8),
        ]

    def _body_text(self, text: str) -> list:
        """Convert raw LLM text into Paragraph flowables, handling bullets."""
        elements: list = []
        if not text or "Not Available" in text[:30]:
            elements.append(Paragraph(text or "Not Available", STYLE_BODY))
            return elements

        for line in text.splitlines():
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 4))
            elif line.startswith(("- ", "• ", "* ")):
                elements.append(Paragraph(f"• {line[2:]}", STYLE_BULLET))
            elif line and line[0].isdigit() and len(line) > 2 and line[1] in ".):" :
                elements.append(Paragraph(line, STYLE_BULLET))
            else:
                elements.append(Paragraph(line, STYLE_BODY))
        return elements

    def _area_block(self, area: str, text: str, images: Dict[str, List]) -> list:
        """
        Render a single area sub-section with optional visual and thermal images.
        `images` is expected as {"visual": [Path, ...], "thermal": [Path, ...]}.
        Paths may be Path objects or strings — both are handled.
        """
        elements: list = []
        elements.append(Paragraph(area, STYLE_SUBSECTION_HEADING))
        elements += self._body_text(text)

        for img_type, label in [("visual", "Visual Evidence"), ("thermal", "Thermal Analysis")]:
            raw_paths = images.get(img_type, [])
            if not raw_paths:
                continue

            # Coerce everything to Path and filter for existence
            valid_paths: List[Path] = []
            for p in raw_paths[:2]:
                p = Path(p)
                if p.exists() and p.stat().st_size > 0:
                    valid_paths.append(p)

            if not valid_paths:
                continue

            elements.append(Spacer(1, 6))
            elements.append(Paragraph(f"<i>{label}</i>", STYLE_BODY))

            img_elements: list = []
            for img_path in valid_paths:
                try:
                    max_w = (CONTENT_WIDTH / 2) - 0.5 * cm
                    max_h = 7 * cm
                    dw, dh = self._scale_image(img_path, max_w, max_h)
                    img_el = Image(str(img_path), width=dw, height=dh)
                    img_elements.append(img_el)
                except Exception as exc:
                    logger.warning(f"Could not embed {img_type} image {img_path.name}: {exc}")

            if img_elements:
                # Pad to 2 columns so Table colWidths is always valid
                while len(img_elements) < 2:
                    img_elements.append(Spacer(1, 1))

                img_table = Table(
                    [img_elements],
                    colWidths=[CONTENT_WIDTH / 2] * 2,
                )
                img_table.setStyle(TableStyle([
                    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING",    (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                elements.append(img_table)

        elements.append(Spacer(1, 8))
        return elements

    @staticmethod
    def _scale_image(img_path: Path, max_w: float, max_h: float):
        """Return (draw_width, draw_height) scaled to fit max_w × max_h."""
        from PIL import Image as PILImage
        try:
            with PILImage.open(str(img_path)) as pil_img:
                orig_w, orig_h = pil_img.size
        except Exception:
            return max_w, max_h
        ratio = min(max_w / orig_w, max_h / orig_h)
        return orig_w * ratio, orig_h * ratio

    # ── Section renderers ──────────────────────────────────────────────────

    def _section_property_summary(self, state: DDRState) -> list:
        elements = self._section_header(1, "Property Issue Summary")
        elements += self._body_text(state.get("property_summary", "Not Available"))
        elements.append(PageBreak())
        return elements

    def _section_area_observations(self, state: DDRState) -> list:
        elements = self._section_header(2, "Area-wise Observations")
        observations = state.get("area_observations", {})
        area_images   = state.get("area_images",      {})
        for area in DDR_AREAS:
            obs  = observations.get(area, "Not Available — no observations recorded.")
            img_dict = area_images.get(area, {})
            elements += self._area_block(area, obs, img_dict)
        elements.append(PageBreak())
        return elements

    def _section_root_causes(self, state: DDRState) -> list:
        elements = self._section_header(3, "Probable Root Causes")
        root_causes = state.get("root_causes", {})
        for area in DDR_AREAS:
            elements.append(Paragraph(area, STYLE_SUBSECTION_HEADING))
            elements += self._body_text(root_causes.get(area, "Not Available"))
            elements.append(Spacer(1, 6))
        elements.append(PageBreak())
        return elements

    def _section_severity(self, state: DDRState) -> list:
        elements = self._section_header(4, "Severity Assessment")
        severity = state.get("severity", {})

        table_data = [
            [
                Paragraph("Area", STYLE_LABEL),
                Paragraph("Severity", STYLE_LABEL),
                Paragraph("Reasoning", STYLE_LABEL),
            ]
        ]
        for area in DDR_AREAS:
            raw = severity.get(area, "Not Available")
            sev_label = "N/A"
            reasoning = raw
            for line in raw.splitlines():
                if line.lower().startswith("severity:"):
                    sev_label = line.split(":", 1)[1].strip()
                elif line.lower().startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()

            table_data.append([
                Paragraph(area, STYLE_BODY),
                Paragraph(f"<b>{sev_label}</b>", STYLE_BODY),
                Paragraph(reasoning, STYLE_BODY),
            ])

        col_widths = [CONTENT_WIDTH * 0.28, CONTENT_WIDTH * 0.14, CONTENT_WIDTH * 0.58]
        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0), NAVY),
            ("TEXTCOLOR",     (0, 0), (-1,  0), WHITE),
            ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1,  0), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GREY, WHITE]),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("GRID",          (0, 0), (-1, -1), 0.5, MID_GREY),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ]))
        elements.append(tbl)
        elements.append(PageBreak())
        return elements

    def _section_recommended_actions(self, state: DDRState) -> list:
        elements = self._section_header(5, "Recommended Actions")
        actions = state.get("recommended_actions", {})
        for area in DDR_AREAS:
            elements.append(Paragraph(area, STYLE_SUBSECTION_HEADING))
            elements += self._body_text(actions.get(area, "Not Available"))
            elements.append(Spacer(1, 6))
        elements.append(PageBreak())
        return elements

    def _section_additional_notes(self, state: DDRState) -> list:
        elements = self._section_header(6, "Additional Notes")
        elements += self._body_text(state.get("additional_notes", "Not Available"))
        elements.append(PageBreak())
        return elements

    def _section_missing_info(self, state: DDRState) -> list:
        elements = self._section_header(7, "Missing / Unclear Information")
        elements += self._body_text(state.get("missing_info", "Not Available"))
        return elements
