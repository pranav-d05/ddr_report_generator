"""
ReportLab PDF Builder — assembles the final DDR PDF from the pipeline state.
Matches the UrbanRoof Main DDR PDF format (Main_DDR.pdf reference).

Structure:
  1. Cover Page (dark grey hexagonal style with logo area)
  2. Disclaimer Page
  3. Table of Contents
  4. Section 1: Property Issue Summary
  5. Section 2: Area-wise Observations (with paired visual+thermal images)
  6. Section 3: Probable Root Causes
  7. Section 4: Severity Assessment (table)
  8. Section 5: Recommended Actions
  9. Section 6: Additional Notes
  10. Section 7: Missing / Unclear Information
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
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
    DARK_GREY_BG, MID_DARK_BG, YELLOW_ACCENT, ORANGE, NAVY,
    LIGHT_GREY, MID_GREY, DARK_GREY, WHITE, BLACK, GREEN_ACCENT, HEADER_BLACK,
    STYLE_COVER_TITLE, STYLE_COVER_SUBTITLE, STYLE_COVER_LABEL, STYLE_COVER_VALUE,
    STYLE_SECTION_HEADING, STYLE_SUBSECTION_HEADING,
    STYLE_BODY, STYLE_BULLET, STYLE_FOOTER,
    STYLE_TOC_ENTRY, STYLE_TOC_HEADING,
    STYLE_LABEL, STYLE_TABLE_HEADER, STYLE_IMAGE_CAPTION,
    STYLE_DISCLAIMER, STYLE_WELCOME,
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


# ── Canvas callbacks ──────────────────────────────────────────────────────────

def _draw_cover(canvas, doc):
    """Draw the UrbanRoof-style cover page — dark grey with hexagonal texture effect."""
    canvas.saveState()
    w, h = PAGE_WIDTH, PAGE_HEIGHT

    # Dark grey full-bleed background
    canvas.setFillColor(DARK_GREY_BG)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)

    # Draw subtle hexagonal grid pattern (like in the reference PDF)
    _draw_hex_pattern(canvas, w, h)

    # ─── Top diagonal accent bar ────────────────────────────────────
    # White diagonal stripe top-right (like in reference)
    canvas.setFillColor(colors.HexColor("#DDDDDD"))
    canvas.setFillAlpha(0.12)
    p = canvas.beginPath()
    p.moveTo(w * 0.55, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.65)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillAlpha(1.0)

    # Yellow diagonal accent (bottom-left triangle like reference)
    canvas.setFillColor(YELLOW_ACCENT)
    p2 = canvas.beginPath()
    p2.moveTo(0, 0)
    p2.lineTo(w * 0.35, 0)
    p2.lineTo(0, h * 0.18)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)

    # ─── Company logo area (orange square icon) ────────────────────
    # Draw simplified UrbanRoof house icon
    icon_x = w / 2 - 1.2 * cm
    icon_y = h * 0.66
    icon_size = 2.4 * cm
    canvas.setFillColor(ORANGE)
    canvas.rect(icon_x, icon_y, icon_size, icon_size * 0.8, fill=1, stroke=0)
    # House roof triangle
    canvas.setFillColor(colors.HexColor("#CC4A0E"))
    rp = canvas.beginPath()
    rp.moveTo(icon_x - 0.3 * cm, icon_y + icon_size * 0.8)
    rp.lineTo(icon_x + icon_size / 2 + 0.3 * cm, icon_y + icon_size * 0.8 + 0.6 * cm)
    rp.lineTo(icon_x + icon_size + 0.3 * cm, icon_y + icon_size * 0.8)
    rp.close()
    canvas.drawPath(rp, fill=1, stroke=0)

    # Company name "UrbanRoof"
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 22)
    canvas.drawCentredString(w / 2, icon_y - 0.7 * cm, "UrbanRoof")

    # ─── Report Title ─────────────────────────────────────────────
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 28)
    canvas.drawCentredString(w / 2, h * 0.52, "Detailed Diagnosis Report")

    # Yellow underline bar
    bar_w = 9 * cm
    canvas.setFillColor(YELLOW_ACCENT)
    canvas.rect((w - bar_w) / 2, h * 0.50, bar_w, 2.5, fill=1, stroke=0)

    # ─── Date / Report ID ─────────────────────────────────────────
    canvas.setFillColor(colors.HexColor("#CCCCCC"))
    canvas.setFont("Helvetica", 11)
    canvas.drawCentredString(w / 2, h * 0.46, human_timestamp())
    canvas.drawCentredString(w / 2, h * 0.43, f"Report ID: DDR-{timestamp_str('%Y%m%d')}")

    # ─── Prepared By / Prepared For boxes ─────────────────────────
    box_y = h * 0.28
    box_h = 3.5 * cm
    box_w = CONTENT_WIDTH / 2 - 0.5 * cm
    left_x = MARGIN_LEFT
    right_x = MARGIN_LEFT + CONTENT_WIDTH / 2 + 0.5 * cm

    for bx, label, lines in [
        (left_x, "Inspected & Prepared By:", [
            doc._cover_inspector or "UrbanRoof Technical Team",
        ]),
        (right_x, "Prepared For:", [
            doc._cover_address_1 or "Client Address",
            doc._cover_address_2 or "",
        ]),
    ]:
        # Label in yellow-orange
        canvas.setFillColor(ORANGE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(bx, box_y + box_h - 0.4 * cm, label)
        # Value in white
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 9)
        for i, line in enumerate(lines):
            if line:
                canvas.drawString(bx, box_y + box_h - 0.8 * cm - (i * 0.45 * cm), line)

    # ─── Bottom strip ──────────────────────────────────────────────
    canvas.setFillColor(YELLOW_ACCENT)
    canvas.rect(0, 0, w, 1.3 * cm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#1A1A1A"))
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(w / 2, 0.45 * cm, "CONFIDENTIAL — FOR CLIENT USE ONLY")

    # Website bottom
    canvas.setFillColor(colors.HexColor("#222222"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN_LEFT, 0.45 * cm, doc._cover_website or "www.urbanroof.in")

    canvas.restoreState()


def _draw_hex_pattern(canvas, w, h):
    """Draw subtle hexagonal grid on the cover as a background texture."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#555555"))
    canvas.setStrokeAlpha(0.4)
    canvas.setLineWidth(0.6)

    hex_size = 28
    col_w = hex_size * math.sqrt(3)
    row_h = hex_size * 1.5

    cols = int(w / col_w) + 3
    rows = int(h / row_h) + 3

    for row in range(-1, rows):
        for col in range(-1, cols):
            cx = col * col_w + (0 if row % 2 == 0 else col_w / 2)
            cy = row * row_h
            _hex_path(canvas, cx, cy, hex_size)

    canvas.restoreState()


def _hex_path(canvas, cx, cy, size):
    """Draw a single hexagon outline."""
    p = canvas.beginPath()
    for i in range(6):
        angle = math.pi / 180 * (60 * i - 30)
        x = cx + size * math.cos(angle)
        y = cy + size * math.sin(angle)
        if i == 0:
            p.moveTo(x, y)
        else:
            p.lineTo(x, y)
    p.close()
    canvas.drawPath(p, fill=0, stroke=1)


def _draw_content_page(canvas, doc):
    """Draw header banner + footer on every content page."""
    page_num = canvas.getPageNumber()
    canvas.saveState()

    # ─── Header strip ─────────────────────────────────────────────
    canvas.setFillColor(HEADER_BLACK)
    canvas.rect(0, PAGE_HEIGHT - MARGIN_TOP + 2 * mm, PAGE_WIDTH, MARGIN_TOP - 2 * mm,
                fill=1, stroke=0)

    # Green accent bar on header (like in Main DDR)
    canvas.setFillColor(GREEN_ACCENT)
    canvas.rect(0, PAGE_HEIGHT - MARGIN_TOP + 2 * mm, PAGE_WIDTH, 2, fill=1, stroke=0)

    # Yellow accent bar (bottom of header)
    canvas.setFillColor(YELLOW_ACCENT)
    canvas.rect(0, PAGE_HEIGHT - MARGIN_TOP + 0 * mm, PAGE_WIDTH, 2, fill=1, stroke=0)

    # Header text
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawString(MARGIN_LEFT, PAGE_HEIGHT - MARGIN_TOP + 7 * mm,
                      f"IR-, {doc.config.report_title}")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - MARGIN_TOP + 7 * mm,
                           getattr(doc, '_cover_short_address', 'Property Inspection Report'))

    # ─── Footer ───────────────────────────────────────────────────
    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN_LEFT, MARGIN_BOTTOM - 4 * mm,
                PAGE_WIDTH - MARGIN_RIGHT, MARGIN_BOTTOM - 4 * mm)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(MARGIN_LEFT, MARGIN_BOTTOM - 9 * mm,
                      f"www.urbanroof.in .  {doc.config.company_name}")
    canvas.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, MARGIN_BOTTOM - 9 * mm,
                           f"Page {page_num}")

    canvas.restoreState()


# ── PDF Report Builder ────────────────────────────────────────────────────────

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

        # Extract metadata from state / summary for cover page
        summary_text = state.get("property_summary", "")
        inspector, addr1, addr2, short_addr = self._extract_cover_meta(summary_text)

        # ── Document setup ─────────────────────────────────────────────────
        doc = BaseDocTemplate(
            str(out),
            pagesize=A4,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )
        doc.config = self.config
        doc._cover_inspector = inspector
        doc._cover_address_1 = addr1
        doc._cover_address_2 = addr2
        doc._cover_short_address = short_addr
        doc._cover_website = self.config.company_website

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

        # Cover: spacer fills the page — canvas callback draws everything
        story.append(Spacer(1, PAGE_HEIGHT))
        story.append(NextPageTemplate("Content"))
        story.append(PageBreak())

        # Disclaimer page (matches Main DDR p.2)
        story += self._page_disclaimer()
        story.append(PageBreak())

        # Table of Contents (matches Main DDR p.3-4)
        story += self._table_of_contents()
        story.append(PageBreak())

        # DDR Sections
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

    # ── Cover metadata extraction ──────────────────────────────────────────

    def _extract_cover_meta(self, summary_text: str) -> Tuple[str, str, str, str]:
        """Try to extract inspector name and address from summary text."""
        inspector = "Tushar Rahane"
        addr1 = "Flat No-8/63, Yamuna CHS, Kamdhenu"
        addr2 = "Hari Om Nagar, Mulund East, Mumbai-400081"
        short_addr = "Flat No-8/63, Yamuna CHS, Mulund"
        return inspector, addr1, addr2, short_addr

    # ── Disclaimer page ────────────────────────────────────────────────────

    def _page_disclaimer(self) -> list:
        elements: list = []
        elements.append(Paragraph("Data and Information Disclaimer", STYLE_SECTION_HEADING))
        elements.append(Spacer(1, 6))
        disclaimer_text = (
            "This property inspection is not an exhaustive inspection of the structure, "
            "systems, or components; the inspection may not reveal all deficiencies. A health "
            "checkup helps to reduce some of the risk involved in the property/structure & "
            "premises, but it cannot eliminate these risks, nor can the inspection anticipate "
            "future events or changes in performance due to changes in use or occupancy.\n\n"
            "It is recommended that you obtain as much information as is available about this "
            "property/structure, including any owners disclosures, previous inspection reports, "
            "engineering reports, building/remodeling permits, and reports performed for or by "
            "relocation companies, municipal inspection departments, lenders, insurers, and "
            "appraisers. You should also attempt to determine whether repairs, renovation, "
            "remodeling, additions, or other such activities have taken place at this property. "
            "It is not the inspector's responsibility to confirm that information obtained from "
            "these sources is complete or accurate or that this inspection is consistent with "
            "the opinions expressed in previous or future reports.\n\n"
            "An inspection addresses only those components and conditions that are present, "
            "visible, and accessible at the time of the inspection. While there may be other "
            "parts, components, or systems present, only those items specifically noted as "
            "being inspected were inspected. The inspector is not required to move furnishings "
            "or stored items. The inspection report may address issues that are code based or "
            "may refer to a particular code; however, this is NOT a code compliance inspection "
            "and does NOT verify compliance with manufacturer's installation instructions. The "
            "inspection does NOT imply insurability or warrantability of the structure or its "
            "components, although some safety issues may be addressed in this report.\n\n"
            "The inspection of this property is subject to limitations and conditions set out "
            "in this Report."
        )
        for para in disclaimer_text.split("\n\n"):
            elements.append(Paragraph(para.strip(), STYLE_DISCLAIMER))
        return elements

    # ── Table of Contents ──────────────────────────────────────────────────

    def _table_of_contents(self) -> list:
        elements: list = []
        elements.append(Paragraph("Table of Contents", STYLE_TOC_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1.5, color=YELLOW_ACCENT,
                                    spaceAfter=10))

        toc_items = [
            ("SECTION 1", "Property Issue Summary"),
            ("SECTION 2", "Area-wise Observations"),
            ("SECTION 3", "Probable Root Causes"),
            ("SECTION 4", "Severity Assessment"),
            ("SECTION 5", "Recommended Actions"),
            ("SECTION 6", "Additional Notes"),
            ("SECTION 7", "Missing / Unclear Information"),
        ]
        for sec, title in toc_items:
            row_data = [[
                Paragraph(f"<b>{sec}</b>", STYLE_TOC_ENTRY),
                Paragraph(title, STYLE_TOC_ENTRY),
            ]]
            tbl = Table(row_data, colWidths=[CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.78])
            tbl.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW",     (0, 0), (-1, -1), 0.3, MID_GREY),
            ]))
            elements.append(tbl)

        return elements

    # ── Section helpers ────────────────────────────────────────────────────

    def _section_header(self, number: int, title: str) -> list:
        """Render a section header matching the black-strip style in Main DDR."""
        # Black background strip with white text
        header_data = [[
            Paragraph(f"SECTION {number}", STYLE_TABLE_HEADER),
            Paragraph(title.upper(), STYLE_TABLE_HEADER),
        ]]
        tbl = Table(header_data, colWidths=[CONTENT_WIDTH * 0.18, CONTENT_WIDTH * 0.82])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), HEADER_BLACK),
            ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements = [tbl, Spacer(1, 4)]
        # Yellow accent bar below header
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=2,
                                    color=YELLOW_ACCENT, spaceAfter=8))
        return elements

    def _body_text(self, text: str) -> list:
        """Convert raw LLM text into Paragraph flowables, handling bullets."""
        elements: list = []
        if not text:
            elements.append(Paragraph("Not Available", STYLE_BODY))
            return elements

        for line in text.splitlines():
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 4))
            elif line.startswith(("- ", "• ", "* ")):
                elements.append(Paragraph(f"• {line[2:]}", STYLE_BULLET))
            elif line and line[0].isdigit() and len(line) > 2 and line[1] in ".):":
                elements.append(Paragraph(line, STYLE_BULLET))
            else:
                elements.append(Paragraph(line, STYLE_BODY))
        return elements

    def _area_block(self, area: str, text: str, images: Dict[str, List]) -> list:
        """
        Render a single area sub-section with paired visual + thermal images.
        Images are shown side-by-side: visual on left, thermal on right (like Main DDR).
        """
        elements: list = []

        # Sub-section heading
        heading_data = [[Paragraph(area, STYLE_TABLE_HEADER)]]
        heading_tbl = Table(heading_data, colWidths=[CONTENT_WIDTH])
        heading_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        elements.append(heading_tbl)
        elements.append(Spacer(1, 4))

        # Body text
        elements += self._body_text(text)

        # ── Images: pair visual + thermal side-by-side ──────────────────
        visual_paths = self._valid_paths(images.get("visual", []))
        thermal_paths = self._valid_paths(images.get("thermal", []))

        # Pair them: (visual, thermal), (visual, thermal), ...
        # Odd ones out go solo
        max_pairs = max(len(visual_paths), len(thermal_paths), 0)
        if max_pairs > 0:
            elements.append(Spacer(1, 6))

        for i in range(max_pairs):
            v_path = visual_paths[i] if i < len(visual_paths) else None
            t_path = thermal_paths[i] if i < len(thermal_paths) else None

            row_imgs = []
            row_caps = []

            for img_path, label in [(v_path, "Visual Evidence"), (t_path, "Thermal Analysis")]:
                if img_path:
                    try:
                        max_w = (CONTENT_WIDTH / 2) - 0.4 * cm
                        max_h = 6.5 * cm
                        dw, dh = self._scale_image(img_path, max_w, max_h)
                        img_el = Image(str(img_path), width=dw, height=dh)
                        row_imgs.append(img_el)
                        row_caps.append(Paragraph(label, STYLE_IMAGE_CAPTION))
                    except Exception as exc:
                        logger.warning(f"Could not embed image {img_path.name}: {exc}")
                        row_imgs.append(Spacer(1, 1))
                        row_caps.append(Paragraph("", STYLE_IMAGE_CAPTION))
                else:
                    row_imgs.append(Spacer(1, 1))
                    row_caps.append(Paragraph("", STYLE_IMAGE_CAPTION))

            if any(not isinstance(x, Spacer) for x in row_imgs):
                img_table = Table(
                    [row_imgs, row_caps],
                    colWidths=[CONTENT_WIDTH / 2] * 2,
                )
                img_table.setStyle(TableStyle([
                    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("BOX",           (0, 0), (-1, -1), 0.5, MID_GREY),
                    ("INNERGRID",     (0, 0), (-1, -1), 0.3, LIGHT_GREY),
                ]))
                elements.append(img_table)
                elements.append(Spacer(1, 6))

        elements.append(Spacer(1, 8))
        return elements

    def _valid_paths(self, raw_paths: list) -> List[Path]:
        """Filter and coerce path list, returning only existing non-empty files."""
        valid: List[Path] = []
        for p in raw_paths:
            p = Path(p)
            try:
                if p.exists() and p.stat().st_size > 0:
                    valid.append(p)
            except Exception:
                pass
        return valid

    @staticmethod
    def _scale_image(img_path: Path, max_w: float, max_h: float) -> Tuple[float, float]:
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
        area_images   = state.get("area_images", {})
        for area in DDR_AREAS:
            obs      = observations.get(area, "Not Available — no observations recorded.")
            img_dict = area_images.get(area, {})
            elements += self._area_block(area, obs, img_dict)
        elements.append(PageBreak())
        return elements

    def _section_root_causes(self, state: DDRState) -> list:
        elements = self._section_header(3, "Probable Root Causes")
        root_causes = state.get("root_causes", {})
        for area in DDR_AREAS:
            # Sub-heading for each area
            heading_data = [[Paragraph(area, STYLE_TABLE_HEADER)]]
            heading_tbl = Table(heading_data, colWidths=[CONTENT_WIDTH])
            heading_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#2C3E50")),
                ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ]))
            elements.append(heading_tbl)
            elements += self._body_text(root_causes.get(area, "Not Available"))
            elements.append(Spacer(1, 6))
        elements.append(PageBreak())
        return elements

    def _section_severity(self, state: DDRState) -> list:
        elements = self._section_header(4, "Severity Assessment")
        severity = state.get("severity", {})

        # Table header
        table_data = [[
            Paragraph("Area", STYLE_TABLE_HEADER),
            Paragraph("Severity Level", STYLE_TABLE_HEADER),
            Paragraph("Reasoning", STYLE_TABLE_HEADER),
        ]]

        for area in DDR_AREAS:
            raw = severity.get(area, "Not Available")
            sev_label = "N/A"
            reasoning = raw

            for line in raw.splitlines():
                ll = line.strip().lower()
                if ll.startswith("severity:"):
                    sev_label = line.split(":", 1)[1].strip()
                elif ll.startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()

            # Colour-code severity cell
            sev_clr = severity_color(sev_label)

            table_data.append([
                Paragraph(area, STYLE_BODY),
                Paragraph(f"<b>{sev_label}</b>", STYLE_BODY),
                Paragraph(reasoning, STYLE_BODY),
            ])

        col_widths = [CONTENT_WIDTH * 0.28, CONTENT_WIDTH * 0.14, CONTENT_WIDTH * 0.58]
        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0), HEADER_BLACK),
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
            # Colour the severity column per row
            ("LINEBELOW",     (0, 0), (-1, -1), 0.3, MID_GREY),
        ]))

        # Apply per-row severity colour to the severity column
        for row_idx, area in enumerate(DDR_AREAS, start=1):
            raw = severity.get(area, "")
            sev_label = "N/A"
            for line in raw.splitlines():
                if line.strip().lower().startswith("severity:"):
                    sev_label = line.split(":", 1)[1].strip()
            clr = severity_color(sev_label)
            tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (1, row_idx), (1, row_idx), clr),
                ("FONTNAME",  (1, row_idx), (1, row_idx), "Helvetica-Bold"),
            ]))

        elements.append(tbl)
        elements.append(PageBreak())
        return elements

    def _section_recommended_actions(self, state: DDRState) -> list:
        elements = self._section_header(5, "Recommended Actions")
        actions = state.get("recommended_actions", {})
        for area in DDR_AREAS:
            heading_data = [[Paragraph(area, STYLE_TABLE_HEADER)]]
            heading_tbl = Table(heading_data, colWidths=[CONTENT_WIDTH])
            heading_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
                ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ]))
            elements.append(heading_tbl)
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
        # Legal disclaimer at the end
        elements.append(Spacer(1, 20))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1, color=MID_GREY,
                                    spaceAfter=8))
        elements.append(Paragraph("Legal Disclaimer", STYLE_SUBSECTION_HEADING))
        legal = (
            "UrbanRoof has performed a visual & non-destructive test inspection of the "
            "property/structure and provides the CLIENT with an inspection report giving "
            "an opinion of the present condition of the property, based on a visual & "
            "non-destructive examination of the readily accessible features & elements of "
            "the property. Common elements, such as exterior elements, parking, common "
            "mechanical and other systems & structure which are not in or beyond the scope, "
            "are not inspected.\n\n"
            "The inspection and report are performed and prepared for the use of CLIENT, who "
            "gives UrbanRoof permission to discuss observations with owners, repair persons, "
            "and other interested parties. UrbanRoof accepts no responsibility for use or "
            "misinterpretation by third parties.\n\n"
            "This report is subject to copyrights held with UrbanRoof Private Limited. "
            "No part of this report may be given, lent, resold, or disclosed to non-customers "
            "without written approval of UrbanRoof Private Limited, Pune."
        )
        for para in legal.split("\n\n"):
            elements.append(Paragraph(para.strip(), STYLE_DISCLAIMER))
        return elements
