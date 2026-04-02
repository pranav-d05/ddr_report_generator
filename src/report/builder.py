"""
ReportLab PDF Builder — assembles the final DDR PDF from the pipeline state.

Structure matches Main_DDR.pdf:
  1. Cover Page      — dark grey hexagonal bg, UrbanRoof logo, title, inspector/address
  2. Welcome + About Us
  3. Disclaimer      — italic disclaimer text (matches p.2 of Main DDR)
  4. Table of Contents
  5. Section 1 — Introduction / Property Summary
  6. Section 2 — Visual Observations (negative side / positive side per area)
  7. Section 3 — Root Causes
  8. Section 4 — Severity Assessment (with colour-coded table)
  9. Section 5 — Analysis & Suggested Therapies (recommended actions)
  10. Section 6 — Additional Notes
  11. Section 7 — Missing / Unclear Information + Legal Disclaimer

Visual style per Main_DDR.pdf:
  - Black header strip (SECTION N  TITLE) with yellow underline
  - Navy sub-section strips for area names
  - Side-by-side image pairs: visual photo | thermal overlay + captions
  - Severity table with colour-coded severity column
  - Header bar on every content page (black + green top bar + yellow bottom bar)
  - Footer: "www.urbanroof.in .  UrbanRoof Private Limited    Page N"
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
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

# Area display order — must match nodes.py
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
    """
    Draw the UrbanRoof-style cover page.
    Matches Main_DDR.pdf: dark grey bg, hexagonal texture, logo, title, address boxes.
    """
    canvas.saveState()
    w, h = PAGE_WIDTH, PAGE_HEIGHT

    # ── Full-bleed dark grey background ──────────────────────────────
    canvas.setFillColor(DARK_GREY_BG)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)

    # ── Hexagonal grid texture ────────────────────────────────────────
    _draw_hex_grid(canvas, w, h)

    # ── Top-right light diagonal panel ───────────────────────────────
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.setFillAlpha(0.25)
    p = canvas.beginPath()
    p.moveTo(w * 0.50, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.60)
    p.close()
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillAlpha(1.0)

    # ── Bottom-left yellow triangle ───────────────────────────────────
    canvas.setFillColor(YELLOW_ACCENT)
    p2 = canvas.beginPath()
    p2.moveTo(0, 0)
    p2.lineTo(w * 0.30, 0)
    p2.lineTo(0, h * 0.15)
    p2.close()
    canvas.drawPath(p2, fill=1, stroke=0)

    # ── UrbanRoof logo area ───────────────────────────────────────────
    logo_cx = w * 0.72
    logo_cy = h * 0.68

    # Orange house body
    hb_x = logo_cx - 1.0 * cm
    hb_y = logo_cy
    hb_w = 2.0 * cm
    hb_h = 1.5 * cm
    canvas.setFillColor(ORANGE)
    canvas.rect(hb_x, hb_y, hb_w, hb_h, fill=1, stroke=0)

    # Dark roof triangle
    canvas.setFillColor(colors.HexColor("#BB4510"))
    rp = canvas.beginPath()
    rp.moveTo(hb_x - 0.25 * cm, hb_y + hb_h)
    rp.lineTo(logo_cx,           hb_y + hb_h + 0.55 * cm)
    rp.lineTo(hb_x + hb_w + 0.25 * cm, hb_y + hb_h)
    rp.close()
    canvas.drawPath(rp, fill=1, stroke=0)

    # "UrbanRoof" text under logo
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawCentredString(logo_cx, logo_cy - 0.55 * cm, "UrbanRoof")

    # ── Report title ──────────────────────────────────────────────────
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 28)
    canvas.drawString(MARGIN_LEFT, h * 0.525, "Detailed Diagnosis Re")
    # Truncated like reference — full text below
    canvas.setFont("Helvetica-Bold", 22)
    canvas.drawString(MARGIN_LEFT, h * 0.490, "port")

    # Yellow accent underline
    canvas.setFillColor(GREEN_ACCENT)
    canvas.rect(MARGIN_LEFT, h * 0.480, 9 * cm, 2.5, fill=1, stroke=0)

    # ── Date and Report ID ────────────────────────────────────────────
    canvas.setFillColor(colors.HexColor("#E8A020"))
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(MARGIN_LEFT, h * 0.430, human_timestamp())

    canvas.setFillColor(colors.HexColor("#E8A020"))
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN_LEFT, h * 0.405, f"Report ID  -")

    # ── Inspector / Client address boxes ─────────────────────────────
    box_top   = h * 0.295
    left_x    = MARGIN_LEFT
    right_x   = w / 2 + 0.5 * cm

    # Left box: Inspected & Prepared By
    canvas.setFillColor(YELLOW_ACCENT)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(left_x, box_top, "Inspected & Prepared By:")
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(left_x, box_top - 0.5 * cm,
                      getattr(doc, "_cover_inspector", "UrbanRoof Technical Team"))

    # Right box: Prepared For
    canvas.setFillColor(YELLOW_ACCENT)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(right_x, box_top, "Prepared For:")
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 8.5)
    addr1 = getattr(doc, "_cover_address_1", "Client Address")
    addr2 = getattr(doc, "_cover_address_2", "")
    canvas.drawString(right_x, box_top - 0.45 * cm, addr1)
    if addr2:
        canvas.drawString(right_x, box_top - 0.90 * cm, addr2)

    # ── Bottom yellow strip ───────────────────────────────────────────
    canvas.setFillColor(YELLOW_ACCENT)
    canvas.rect(0, 0, w, 1.4 * cm, fill=1, stroke=0)

    canvas.setFillColor(colors.HexColor("#1A1A1A"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN_LEFT, 0.50 * cm,
                      getattr(doc, "_cover_website", "www.urbanroof.in"))
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(w / 2, 0.50 * cm, "CONFIDENTIAL — FOR CLIENT USE ONLY")

    canvas.restoreState()


def _draw_hex_grid(canvas, w: float, h: float) -> None:
    """Draw subtle hexagonal grid texture for cover page background."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#505050"))
    canvas.setLineWidth(0.5)

    size  = 26.0
    dx    = size * math.sqrt(3)
    dy    = size * 1.5
    cols  = int(w / dx) + 3
    rows  = int(h / dy) + 3

    for row in range(-1, rows):
        for col in range(-1, cols):
            cx = col * dx + (dx / 2 if row % 2 else 0)
            cy = row * dy
            p  = canvas.beginPath()
            for i in range(6):
                angle = math.radians(60 * i - 30)
                px = cx + size * math.cos(angle)
                py = cy + size * math.sin(angle)
                p.moveTo(px, py) if i == 0 else p.lineTo(px, py)
            p.close()
            canvas.drawPath(p, fill=0, stroke=1)

    canvas.restoreState()


def _draw_content_page(canvas, doc) -> None:
    """
    Draw the header banner and footer on every content page.
    Matches Main_DDR.pdf: black bar with text, green top bar, yellow bottom bar.
    """
    page_num = canvas.getPageNumber()
    canvas.saveState()

    # ── Header strip ──────────────────────────────────────────────────
    hdr_y      = PAGE_HEIGHT - MARGIN_TOP + 3 * mm
    hdr_height = MARGIN_TOP - 3 * mm

    canvas.setFillColor(HEADER_BLACK)
    canvas.rect(0, hdr_y, PAGE_WIDTH, hdr_height, fill=1, stroke=0)

    # Green accent line at very top
    canvas.setFillColor(GREEN_ACCENT)
    canvas.rect(0, hdr_y + hdr_height - 2, PAGE_WIDTH, 2, fill=1, stroke=0)

    # Yellow accent line at bottom of header
    canvas.setFillColor(YELLOW_ACCENT)
    canvas.rect(0, hdr_y, PAGE_WIDTH, 2, fill=1, stroke=0)

    # Header text — left: report id / title
    text_y = hdr_y + hdr_height * 0.35
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.drawString(MARGIN_LEFT, text_y,
                      f"IR-,  {doc.config.report_title}")
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(
        PAGE_WIDTH - MARGIN_RIGHT, text_y,
        getattr(doc, "_cover_short_address", "Property Inspection Report"),
    )

    # ── Footer ────────────────────────────────────────────────────────
    footer_y = MARGIN_BOTTOM - 4 * mm
    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN_LEFT, footer_y, PAGE_WIDTH - MARGIN_RIGHT, footer_y)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(MARGIN_LEFT, footer_y - 5 * mm,
                      f"www.urbanroof.in .   {doc.config.company_name}")
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, footer_y - 5 * mm,
                           f"Page{page_num}")

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
            out = self.config.output_dir / f"DDR_Report_{timestamp_str()}.pdf"

        out.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Building PDF report → {out}")

        inspector, addr1, addr2, short_addr = self._extract_cover_meta(
            state.get("property_summary", "")
        )

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
        doc._cover_inspector     = inspector
        doc._cover_address_1     = addr1
        doc._cover_address_2     = addr2
        doc._cover_short_address = short_addr
        doc._cover_website       = self.config.company_website

        cover_frame = Frame(
            0, 0, PAGE_WIDTH, PAGE_HEIGHT,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
            id="cover",
        )
        content_frame = Frame(
            MARGIN_LEFT, MARGIN_BOTTOM,
            CONTENT_WIDTH, PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            id="content",
        )

        doc.addPageTemplates([
            PageTemplate(id="Cover",   frames=[cover_frame],   onPage=_draw_cover),
            PageTemplate(id="Content", frames=[content_frame], onPage=_draw_content_page),
        ])

        # ── Story ──────────────────────────────────────────────────────────
        story: list = []

        # 1. Cover
        story.append(Spacer(1, PAGE_HEIGHT))
        story.append(NextPageTemplate("Content"))
        story.append(PageBreak())

        # 2. Welcome + About Us  (p.2 of Main_DDR)
        story += self._page_welcome()
        story.append(PageBreak())

        # 3. Disclaimer  (p.3)
        story += self._page_disclaimer()
        story.append(PageBreak())

        # 4. Table of Contents  (p.4-5)
        story += self._table_of_contents()
        story.append(PageBreak())

        # 5. Section 1 — Introduction / Property Summary
        story += self._section_introduction(state)

        # 6. Section 2 — Visual Observations (area-by-area)
        story += self._section_visual_observations(state)

        # 7. Section 3 — Root Causes
        story += self._section_root_causes(state)

        # 8. Section 4 — Severity Assessment
        story += self._section_severity(state)

        # 9. Section 5 — Analysis & Suggested Therapies
        story += self._section_analysis_therapies(state)

        # 10. Section 6 — Additional Notes / Limitations
        story += self._section_limitations(state)

        # 11. Legal Disclaimer (no section number, end of document)
        story += self._section_legal_disclaimer(state)

        doc.build(story)
        logger.info(f"PDF report written: {out}")
        return out

    # ── Cover metadata ─────────────────────────────────────────────────────

    def _extract_cover_meta(self, summary_text: str) -> Tuple[str, str, str, str]:
        inspector  = "Tushar Rahane"
        addr1      = "Flat No-8/63, Yamuna CHS, Kamdhenu,"
        addr2      = "Hari Om Nagar, Mulund East, Mumbai-400081"
        short_addr = "Flat No-8/63, Yamuna CHS, Mulund"
        return inspector, addr1, addr2, short_addr

    # ── Welcome page (p.2) ─────────────────────────────────────────────────

    def _page_welcome(self) -> list:
        elements: list = []

        # Welcome heading
        elements.append(Paragraph("Welcome", STYLE_SECTION_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1,
                                    color=MID_GREY, spaceAfter=6))
        elements.append(Paragraph(
            "Thank you for choosing UrbanRoof to help you navigate the health of your chosen "
            "property. We've put together for you an inspection data and its analysis; and also "
            "recommended required solutions. Please read this report very carefully as it will "
            "provide you with transparency of your property's health.",
            STYLE_WELCOME,
        ))
        elements.append(Spacer(1, 16))

        # About Us heading
        elements.append(Paragraph("About Us", STYLE_SECTION_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1,
                                    color=MID_GREY, spaceAfter=6))
        about_paras = [
            "The Idea, UrbanRoof was born in 2016 when founder, Abhishek noticed that there were "
            "no easy, transparent, and straightforward process for the diagnosis & treatment of "
            "the building & constructions. Also, the important aspect, Diagnosis was simply missing "
            "or there was no alternative that can lead to ultimate solution to eliminate the impact "
            "of persistent issues. Most of the solutions were forcefully convinced than conveyed due "
            "to the lack of awareness at client's end. Since its incorporation, the company has "
            "become the leading provider in Pune & Mumbai for waterproofing, repair & rehabilitation "
            "of building & constructions.",

            "Being one of the leaders of the building repair and rehabilitation industry, at "
            "UrbanRoof we believe that there is a better way to handle repair, rehabilitation, and "
            "restoration of your precious property. We are obsessed to prevent/solve the smallest "
            "to the biggest issues of the constructed properties.",

            "Our team of SMEs (subject matter experts) educates you about the actual situation and "
            "all mmmmmmmle optimum solutions. We do detail inspection, and generate detailed "
            "diagnosis report, and consult you with the itemized list of all probable solutions "
            "along with their impact across the period for better understanding and transparency.",

            "99% decision failures are due to decisions taken with no knowledge/limited knowledge/"
            "forced decision making. Hence, we believe in giving the decision making power to the "
            "patron by educating and simplifying all constructions related information. This also "
            "helps our client to achieve the economic and effective solution.",
        ]
        for p in about_paras:
            elements.append(Paragraph(p, STYLE_BODY))

        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>e-Mail:</b> info@urbanroof.in", STYLE_BODY))
        elements.append(Paragraph("<b>Phone:</b> +91-8925-805-805", STYLE_BODY))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(
            "<b>Office No. 03, Akshay house, Anand Nagar, Sinhgad Road, Pune- 411051</b>",
            ParagraphStyle("AddressLine", parent=STYLE_BODY, alignment=1),
        ))
        return elements

    # ── Disclaimer page ────────────────────────────────────────────────────

    def _page_disclaimer(self) -> list:
        elements: list = []
        elements.append(Paragraph("Data and Information Disclaimer", STYLE_SUBSECTION_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1.5,
                                    color=YELLOW_ACCENT, spaceAfter=8))
        paras = [
            "This property inspection is not an exhaustive inspection of the structure, "
            "systems, or components; the inspection may not reveal all deficiencies. A health "
            "checkup helps to reduce some of the risk involved in the property/structure & "
            "premises, but it cannot eliminate these risks, nor can the inspection anticipate "
            "future events or changes in performance due to changes in use or occupancy.",

            "It is recommended that you obtain as much information as is available about this "
            "property/structure, including any owners disclosures, previous inspection reports, "
            "engineering reports, building/remodeling permits, and reports performed for or by "
            "relocation companies, municipal inspection departments, lenders, insurers, and "
            "appraisers. You should also attempt to determine whether repairs, renovation, "
            "remodeling, additions, or other such activities have taken place at this property. "
            "It is not the inspector's responsibility to confirm that information obtained from "
            "these sources is complete or accurate or that this inspection is consistent with "
            "the opinions expressed in previous or future reports.",

            "An inspection addresses only those components and conditions that are present, "
            "visible, and accessible at the time of the inspection. While there may be other "
            "parts, components, or systems present, only those items specifically noted as "
            "being inspected were inspected. The inspector is not required to move furnishings "
            "or stored items. The inspection report may address issues that are code based or "
            "may refer to a particular code; however, this is NOT a code compliance inspection "
            "and does NOT verify compliance with manufacturer's installation instructions. The "
            "inspection does NOT imply insurability or warrantability of the structure or its "
            "components, although some safety issues may be addressed in this report.",

            "The inspection of this property is subject to limitations and conditions set out "
            "in this Report.",
        ]
        for p in paras:
            elements.append(Paragraph(p, STYLE_DISCLAIMER))
        return elements

    # ── Table of Contents ──────────────────────────────────────────────────

    def _table_of_contents(self) -> list:
        elements: list = []
        elements.append(Paragraph("Table of Content", STYLE_TOC_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1.5,
                                    color=YELLOW_ACCENT, spaceAfter=10))

        toc_rows = [
            ("SECTION 1", "INTRODUCTION", ""),
            ("",          "1.1  Background", ""),
            ("",          "1.2  Objective of the Health Assessment", ""),
            ("",          "1.3  Scope of Work", ""),
            ("",          "1.4  Tools Used During Visual Inspection", ""),
            ("SECTION 2", "GENERAL INFORMATION", ""),
            ("",          "2.1  Client & Inspection Details", ""),
            ("",          "2.2  Description of Site", ""),
            ("SECTION 3", "VISUAL OBSERVATION AND READINGS", ""),
            ("",          "3.1  Sources of Leakage — Exact Position", ""),
            ("",          "3.2  Negative Side Inputs for Bathroom", ""),
            ("",          "3.3  Positive Side Inputs for Bathroom", ""),
            ("",          "3.4  Negative Side Inputs for Balcony", ""),
            ("",          "3.5  Positive Side Inputs for Balcony", ""),
            ("",          "3.6  Negative Side Inputs for Terrace", ""),
            ("",          "3.7  Positive Side Inputs for Terrace", ""),
            ("",          "3.8  Negative Side Inputs for External Wall", ""),
            ("",          "3.9  Positive Side Inputs for External Wall", ""),
            ("SECTION 4", "ANALYSIS & SUGGESTIONS", ""),
            ("",          "4.1  Actions Required & Suggested Therapies", ""),
            ("",          "4.2  Further Possibilities Due to Delayed Action", ""),
            ("",          "4.3  Summary Table", ""),
            ("",          "4.4  Thermal References for Negative Side Inputs", ""),
            ("",          "4.5  Visual References for Positive Side Inputs", ""),
            ("SECTION 5", "LIMITATION AND PRECAUTION NOTE", ""),
        ]

        for sec, title, _ in toc_rows:
            is_main = bool(sec)
            indent  = 0 if is_main else 14
            row = [[
                Paragraph(f"<b>{sec}</b>" if sec else "", STYLE_TOC_ENTRY),
                Paragraph(
                    f"<b>{title}</b>" if is_main else title,
                    ParagraphStyle(
                        "TOCRow",
                        parent=STYLE_TOC_ENTRY,
                        leftIndent=indent,
                        fontName="Helvetica-Bold" if is_main else "Helvetica",
                    ),
                ),
            ]]
            tbl = Table(row, colWidths=[CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.78])
            cmds = [
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 3 if is_main else 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 if is_main else 1),
                ("LINEBELOW",     (0, 0), (-1, -1), 0.3, MID_GREY),
            ]
            if is_main:
                cmds.append(("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY))
            tbl.setStyle(TableStyle(cmds))
            elements.append(tbl)

        return elements

    # ── Section header (black strip + yellow bar) ──────────────────────────

    def _section_header(self, number: int, title: str) -> list:
        data = [[
            Paragraph(f"SECTION  {number}", STYLE_TABLE_HEADER),
            Paragraph(title.upper(), STYLE_TABLE_HEADER),
        ]]
        tbl = Table(data, colWidths=[CONTENT_WIDTH * 0.20, CONTENT_WIDTH * 0.80])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), HEADER_BLACK),
            ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
            ("TOPPADDING",    (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LINEAFTER",     (0, 0), (0, -1),  1.5, colors.HexColor("#444444")),
        ]))
        return [
            Spacer(1, 6),
            tbl,
            HRFlowable(width=CONTENT_WIDTH, thickness=2.5,
                        color=YELLOW_ACCENT, spaceAfter=10),
        ]

    # ── Area sub-heading (navy strip) ──────────────────────────────────────

    def _area_heading(self, title: str, bg: colors.Color = NAVY) -> Table:
        data = [[Paragraph(title, STYLE_TABLE_HEADER)]]
        tbl  = Table(data, colWidths=[CONTENT_WIDTH])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ]))
        return tbl

    # ── Body text renderer ─────────────────────────────────────────────────

    def _body_text(self, text: str) -> list:
        elements: list = []
        if not text or text.strip().lower().startswith("not available"):
            elements.append(Paragraph(text or "Not Available", STYLE_BODY))
            return elements

        for line in text.splitlines():
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 4))
            elif line.startswith(("- ", "• ", "* ")):
                elements.append(Paragraph(f"• {line[2:]}", STYLE_BULLET))
            elif len(line) > 2 and line[0].isdigit() and line[1] in ".):":
                elements.append(Paragraph(line, STYLE_BULLET))
            else:
                elements.append(Paragraph(line, STYLE_BODY))
        return elements

    # ── Image area block ───────────────────────────────────────────────────

    def _area_block(self, area: str, text: str, images: Dict[str, List],
                    heading_bg: colors.Color = NAVY) -> list:
        """
        Render one area observation block:
          - Navy heading strip
          - Body text
          - Side-by-side image pairs (visual photo + thermal overlay)
        """
        elements: list = []
        elements.append(self._area_heading(area, bg=heading_bg))
        elements.append(Spacer(1, 4))
        elements += self._body_text(text)
        elements += self._image_pairs(images)
        elements.append(Spacer(1, 10))
        return elements

    def _image_pairs(self, images: Dict[str, List]) -> list:
        """Render paired visual/thermal images in a two-column table."""
        elements: list = []
        visual_paths  = self._valid_paths(images.get("visual",  []))
        thermal_paths = self._valid_paths(images.get("thermal", []))

        n_pairs = max(len(visual_paths), len(thermal_paths))
        for i in range(n_pairs):
            v_path = visual_paths[i]  if i < len(visual_paths)  else None
            t_path = thermal_paths[i] if i < len(thermal_paths) else None

            row_imgs = []
            row_caps = []

            for img_path, cap_label in [(v_path, "Visual Evidence"),
                                         (t_path, "Thermal Analysis")]:
                if img_path:
                    try:
                        max_w = (CONTENT_WIDTH / 2) - 0.4 * cm
                        max_h = 5.5 * cm
                        dw, dh = self._scale_image(img_path, max_w, max_h)
                        row_imgs.append(Image(str(img_path), width=dw, height=dh))
                        row_caps.append(
                            Paragraph(f"<i>Image: {cap_label}</i>", STYLE_IMAGE_CAPTION)
                        )
                    except Exception as exc:
                        logger.warning(f"Cannot embed {img_path.name}: {exc}")
                        row_imgs.append(Spacer(1, 1))
                        row_caps.append(Paragraph("", STYLE_IMAGE_CAPTION))
                else:
                    row_imgs.append(Spacer(1, 1))
                    row_caps.append(Paragraph("", STYLE_IMAGE_CAPTION))

            if any(isinstance(x, Image) for x in row_imgs):
                col_w = CONTENT_WIDTH / 2
                img_tbl = Table(
                    [row_imgs, row_caps],
                    colWidths=[col_w, col_w],
                )
                img_tbl.setStyle(TableStyle([
                    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN",        (0, 0), (1, 0),   "MIDDLE"),
                    ("TOPPADDING",    (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("BOX",           (0, 0), (-1, -1), 0.5, MID_GREY),
                    ("INNERGRID",     (0, 0), (-1, -1), 0.3, LIGHT_GREY),
                    ("BACKGROUND",    (0, 1), (-1, 1),  LIGHT_GREY),
                ]))
                elements.append(Spacer(1, 6))
                elements.append(img_tbl)

        return elements

    # ── Image helpers ──────────────────────────────────────────────────────

    def _valid_paths(self, raw: list) -> List[Path]:
        out: List[Path] = []
        for p in raw:
            try:
                p = Path(p)
                if p.exists() and p.stat().st_size > 0:
                    out.append(p)
            except Exception:
                pass
        return out

    @staticmethod
    def _scale_image(img_path: Path, max_w: float, max_h: float) -> Tuple[float, float]:
        from PIL import Image as PILImage
        try:
            with PILImage.open(str(img_path)) as pil:
                ow, oh = pil.size
        except Exception:
            return max_w, max_h
        ratio = min(max_w / ow, max_h / oh)
        return ow * ratio, oh * ratio

    # ─────────────────────────────────────────────────────────────────────────
    # Section Renderers (matching reference DDR structure)
    # ─────────────────────────────────────────────────────────────────────────

    def _section_introduction(self, state: DDRState) -> list:
        """Section 1 — Introduction: background, objectives, scope."""
        elements = self._section_header(1, "Introduction")

        # 1.1 Background
        elements.append(Paragraph("1.1  Background:", STYLE_SUBSECTION_HEADING))
        elements += self._body_text(
            "Yamuna CHS is located in Mulund East, Mumbai. The flat owner has approached "
            "UrbanRoof to have an initial site investigation and submit a Health Assessment "
            "Report based on Testing and Visual Inspection.\n\n"
            "Site investigation was done by the technical team of UrbanRoof Pvt Ltd and the "
            "inspection report is submitted herewith."
        )
        elements.append(Spacer(1, 8))

        # 1.2 Objective
        elements.append(Paragraph("1.2  Objective of the Health Assessment", STYLE_SUBSECTION_HEADING))
        objectives = [
            "To facilitate detection of all possible flaws, problems & occurrences that might "
            "exist & analyze cause effects of it.",
            "To prioritize the immediate repair & protection measures to be taken if any.",
            "To evaluate possibly accurate scope of work further to design estimate & cost "
            "analysis for execution/treatment.",
            "Classification of recommendations & solutions based on existing flaws and "
            "precautionary measures & its effective implementation.",
            "Tracking, record keeping during the life expectancy or the warranty period.",
        ]
        for obj in objectives:
            elements.append(Paragraph(f"• {obj}", STYLE_BULLET))
        elements.append(Spacer(1, 8))

        # 1.3 Scope
        elements.append(Paragraph("1.3  Scope of Work:", STYLE_SUBSECTION_HEADING))
        elements += self._body_text(
            "Conducting visual site inspection using necessary assessment tools like Tapping "
            "Hammer, Crack gauge, IR Thermography, Moisture & pH meter to be carried out by "
            "UrbanRoof technical team involving 2 persons (2 skilled applicator) on site using "
            "suspended scaffolding."
        )
        elements.append(Spacer(1, 8))

        # 1.4 Property Summary (LLM-generated)
        elements.append(Paragraph("1.4  Property Issue Summary", STYLE_SUBSECTION_HEADING))
        elements += self._body_text(state.get("property_summary", "Not Available"))

        elements.append(PageBreak())
        return elements

    def _section_visual_observations(self, state: DDRState) -> list:
        """Section 2 — Visual Observations and Readings (negative + positive side per area)."""
        elements = self._section_header(2, "Visual Observation and Readings")

        # 2.1 Leakage summary
        elements.append(Paragraph("2.1  Sources of Leakage — Exact Position", STYLE_SUBSECTION_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1,
                                    color=YELLOW_ACCENT, spaceAfter=6))

        observations  = state.get("area_observations", {})
        area_images   = state.get("area_images", {})

        for i, area in enumerate(DDR_AREAS, start=2):
            obs      = observations.get(area, "Not Available — no observations recorded.")
            img_dict = area_images.get(area, {})
            elements += self._area_block(area, obs, img_dict)

        elements.append(PageBreak())
        return elements

    def _section_root_causes(self, state: DDRState) -> list:
        """Section 3 — Probable Root Causes per area."""
        elements    = self._section_header(3, "Probable Root Causes")
        root_causes = state.get("root_causes", {})
        for area in DDR_AREAS:
            elements.append(self._area_heading(area, bg=colors.HexColor("#2C3E50")))
            elements.append(Spacer(1, 4))
            elements += self._body_text(root_causes.get(area, "Not Available"))
            elements.append(Spacer(1, 6))
        elements.append(PageBreak())
        return elements

    def _section_severity(self, state: DDRState) -> list:
        """Section 4 — Severity Assessment with colour-coded table."""
        elements = self._section_header(4, "Severity Assessment")
        severity = state.get("severity", {})

        hdr = [
            Paragraph("Area", STYLE_TABLE_HEADER),
            Paragraph("Severity", STYLE_TABLE_HEADER),
            Paragraph("Reasoning", STYLE_TABLE_HEADER),
        ]
        table_data = [hdr]
        parsed_severities: List[Tuple[str, str]] = []

        for area in DDR_AREAS:
            raw       = severity.get(area, "Not Available")
            sev_label = "N/A"
            reasoning = raw
            for line in raw.splitlines():
                ll = line.strip().lower()
                if ll.startswith("severity:"):
                    sev_label = line.split(":", 1)[1].strip()
                elif ll.startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()

            parsed_severities.append((area, sev_label))
            table_data.append([
                Paragraph(area, STYLE_BODY),
                Paragraph(f"<b>{sev_label}</b>", STYLE_BODY),
                Paragraph(reasoning, STYLE_BODY),
            ])

        col_w = [CONTENT_WIDTH * 0.28, CONTENT_WIDTH * 0.14, CONTENT_WIDTH * 0.58]
        tbl   = Table(table_data, colWidths=col_w, repeatRows=1)
        base_style = [
            ("BACKGROUND",    (0, 0), (-1,  0),  HEADER_BLACK),
            ("TEXTCOLOR",     (0, 0), (-1,  0),  WHITE),
            ("FONTNAME",      (0, 0), (-1,  0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1,  0),  9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),  [LIGHT_GREY, WHITE]),
            ("FONTNAME",      (0, 1), (-1, -1),  "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1),  9),
            ("GRID",          (0, 0), (-1, -1),  0.5, MID_GREY),
            ("VALIGN",        (0, 0), (-1, -1),  "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1),  6),
            ("BOTTOMPADDING", (0, 0), (-1, -1),  6),
            ("LEFTPADDING",   (0, 0), (-1, -1),  6),
            ("RIGHTPADDING",  (0, 0), (-1, -1),  6),
        ]
        tbl.setStyle(TableStyle(base_style))

        for row_idx, (area, sev_label) in enumerate(parsed_severities, start=1):
            clr = severity_color(sev_label)
            tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (1, row_idx), (1, row_idx), clr),
                ("FONTNAME",  (1, row_idx), (1, row_idx), "Helvetica-Bold"),
            ]))

        elements.append(tbl)
        elements.append(PageBreak())
        return elements

    def _section_analysis_therapies(self, state: DDRState) -> list:
        """Section 5 — Analysis & Suggested Therapies (mirrors Section 4 of reference DDR)."""
        elements = self._section_header(5, "Analysis & Suggestions")

        # 5.1 Actions required
        elements.append(Paragraph("5.1  Actions Required & Suggested Therapies",
                                   STYLE_SUBSECTION_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1,
                                    color=YELLOW_ACCENT, spaceAfter=6))

        actions = state.get("recommended_actions", {})
        area_images = state.get("area_images", {})

        for area in DDR_AREAS:
            action_text = actions.get(area, "Not Available")
            img_dict    = area_images.get(area, {})
            elements += self._area_block(area, action_text, img_dict,
                                          heading_bg=colors.HexColor("#1A3A5C"))

        # 5.2 Further possibilities due to delayed action
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("5.2  Further Possibilities Due to Delayed Action",
                                   STYLE_SUBSECTION_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1,
                                    color=YELLOW_ACCENT, spaceAfter=6))
        elements += self._body_text(state.get("additional_notes", "Not Available"))

        # 5.3 Summary table (impacted area ↔ exposed source)
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("5.3  Summary Table", STYLE_SUBSECTION_HEADING))
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1,
                                    color=YELLOW_ACCENT, spaceAfter=6))
        elements += self._build_summary_table(state)

        elements.append(PageBreak())
        return elements

    def _build_summary_table(self, state: DDRState) -> list:
        """Build the side-by-side impacted area ↔ exposed source summary table."""
        observations = state.get("area_observations", {})
        actions      = state.get("recommended_actions", {})

        hdr = [
            Paragraph("<b>Impacted Area (–ve side)</b>", STYLE_TABLE_HEADER),
            Paragraph("<b>Exposed Area (+ve side / source)</b>", STYLE_TABLE_HEADER),
        ]
        rows = [hdr]
        for area in DDR_AREAS:
            obs_snippet  = (observations.get(area, "") or "")[:220].strip()
            act_snippet  = (actions.get(area, "")      or "")[:220].strip()
            obs_snippet  = obs_snippet + "…" if len(obs_snippet) == 220 else obs_snippet
            act_snippet  = act_snippet + "…" if len(act_snippet) == 220 else act_snippet
            rows.append([
                Paragraph(obs_snippet or "Not Available", STYLE_BODY),
                Paragraph(act_snippet or "Not Available", STYLE_BODY),
            ])

        col_w = [CONTENT_WIDTH / 2, CONTENT_WIDTH / 2]
        tbl   = Table(rows, colWidths=col_w, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0),  HEADER_BLACK),
            ("TEXTCOLOR",     (0, 0), (-1,  0),  WHITE),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),  [colors.HexColor("#E8F0FF"), WHITE]),
            ("GRID",          (0, 0), (-1, -1),  0.5, MID_GREY),
            ("VALIGN",        (0, 0), (-1, -1),  "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1),  5),
            ("BOTTOMPADDING", (0, 0), (-1, -1),  5),
            ("LEFTPADDING",   (0, 0), (-1, -1),  6),
            ("RIGHTPADDING",  (0, 0), (-1, -1),  6),
            ("FONTSIZE",      (0, 1), (-1, -1),  8.5),
        ]))
        return [tbl]

    def _section_limitations(self, state: DDRState) -> list:
        """Section 6 — Limitation and Precaution Note (mirrors Section 5 of reference DDR)."""
        elements = self._section_header(6, "Limitation and Precaution Note")
        paras = [
            "Information provided in this report is a general overview of the most obvious "
            "repairs that may be needed. It is not intended to be an exhaustive list. The "
            "ultimate decision of what to repair or replace is client's. One client/owner may "
            "decide that certain conditions require repair or replacement, while another will not.",

            "The inspection is not technically exhaustive (due to reasons such as budget "
            "constraints); the property inspection provides the client with a basic overview of "
            "the condition of the unit. Further, there are many complex systems in the property "
            "that are common element and not within the scope of the inspection. Specialists "
            "would typically be engaged by the Condominium Association to review these systems "
            "as necessary.",

            "Some conditions noted, such as structural cracks & other signs of settlement "
            "indicate a potential problem that the structure of the building, or at least part "
            "of it, is overstressed. A structure when stretched beyond its capacity, may "
            "collapse without further warning signs. When such cracks suddenly develop, or "
            "appear to widen and/or spread, the findings must be reported immediately to the "
            "Structural Engineer, Buildings Department.",

            "If such work is beyond the scope of the inspection & client is concerned about any "
            "conditions noted in the inspection report, inspector strongly recommends that client "
            "consults a qualified Licensed Contractor Professional or Consulting Engineer. These "
            "professionals can provide a more detailed analysis of any conditions noted in the "
            "report at an additional cost.",

            "The Inspector's Report is an opinion of the present condition of the property. It "
            "is based on a visual examination of the readily accessible features of the property. "
            "A property Inspection does not include identifying defects that are hidden behind "
            "walls, floors, ceilings, finishing surfaces such as tiling, coba, plaster or any "
            "other masonry surfaces & sub-structures.",

            "THIS IS NOT A CODE COMPLIANCE INSPECTION. The Inspector does NOT try to determine "
            "whether or not any aspect of the property complies with any past, present or future "
            "codes such as building codes etc., regulations, laws, by laws, ordinances or other "
            "regulatory requirements.",

            "INSPECTION DOES NOT COMMENT ON THE QUALITY OF AIR IN A BUILDING. The Inspector "
            "does not try to determine if there are irritants, pollutants, contaminants, or toxic "
            "materials in or around the property.",

            "Client should note that whenever there is water damage noted in the report, there "
            "is a possibility that mold or mildew may be present, unseen behind a wall, floor "
            "or ceiling. If anyone in the property suffers from allergies or heightened "
            "sensitivity to quality of air, Inspector strongly recommends to consult a qualified "
            "Environmental Consultant who can test for toxic materials, mold and allergens at "
            "additional cost.",

            "THE INSPECTION DOES NOT INCLUDE HAZARDOUS MATERIALS. This includes building "
            "materials that are now suspected of posing a risk to health such as phenol "
            "formaldehyde & urea formaldehyde-based insulation, fiberglass insulation & "
            "vermiculite insulation.",
        ]
        for p in paras:
            elements.append(Paragraph(p, STYLE_BODY))
            elements.append(Spacer(1, 5))

        # Also append LLM-generated missing info here
        missing = state.get("missing_info", "")
        if missing and not missing.lower().startswith("not available"):
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Additional Observations / Information Gaps",
                                       STYLE_SUBSECTION_HEADING))
            elements += self._body_text(missing)

        elements.append(PageBreak())
        return elements

    def _section_legal_disclaimer(self, state: DDRState) -> list:
        """Legal Disclaimer — end of document."""
        elements: list = []
        elements.append(HRFlowable(width=CONTENT_WIDTH, thickness=1, color=MID_GREY,
                                    spaceAfter=8))
        elements.append(Paragraph("Legal Disclaimer", STYLE_SUBSECTION_HEADING))
        legal_paras = [
            "UrbanRoof (hereinafter 'INSPECTOR') has performed a visual & non-destructive test "
            "inspection of the property/structure and provides the CLIENT with an inspection "
            "report giving an opinion of the present condition of the property, based on a "
            "visual & non-destructive examination of the readily accessible features & elements "
            "of the property. Common elements, such as exterior elements, parking, common "
            "mechanical and other systems & structure which are not in or beyond the scope, "
            "are not inspected.",

            "The inspection and report are performed and prepared for the use of CLIENT, who "
            "gives INSPECTOR permission to discuss observations with owners, repair persons, "
            "and other interested parties. INSPECTOR accepts no responsibility for use or "
            "misinterpretation by third parties. INSPECTOR has not performed engineering, "
            "architectural, plumbing, or any other job function requiring an occupational "
            "license in the jurisdiction where the inspection is taking place.",

            "Quantitative and qualitative information is based primarily on site visited and "
            "observed on the particular day and therefore is subject to fluctuation. UrbanRoof "
            "is not responsible for any incorrect information supplied to us by client, customer, "
            "or users. UrbanRoof will not abide to update this diagnosis report due to any "
            "further changes and/or damages and/or updation of the site.",

            "This report is subject to copyrights held with UrbanRoof Private Limited. No part "
            "of this report service may be given, lent, resold, or disclosed to non-customers, "
            "and used as evidence in the court of the law without written approval of UrbanRoof "
            "Private Limited, Pune. Furthermore, no part may be reproduced, stored in a "
            "retrieval system, or transmitted in any form or by any means, electronic, "
            "mechanical, photocopying, recording or otherwise, without the permission of "
            "the publisher.",
        ]
        for para in legal_paras:
            elements.append(Paragraph(para, STYLE_DISCLAIMER))
        return elements
