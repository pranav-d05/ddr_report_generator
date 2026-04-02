"""
ReportLab Style Constants for the DDR PDF Report.
Precisely matches the UrbanRoof DDR format from Main_DDR.pdf.
"""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import TableStyle

# ── Page dimensions ───────────────────────────────────────────────────────────

PAGE_WIDTH, PAGE_HEIGHT = A4          # 595.27 × 841.89 pt
MARGIN_LEFT   = 2.0 * cm
MARGIN_RIGHT  = 2.0 * cm
MARGIN_TOP    = 2.8 * cm             # room for header strip
MARGIN_BOTTOM = 2.5 * cm

CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT

# ── Colour palette (matches Main_DDR.pdf exactly) ────────────────────────────

DARK_GREY_BG   = colors.HexColor("#3A3A3A")   # Cover dark grey background
MID_DARK_BG    = colors.HexColor("#2E2E2E")   # Slightly darker cover panel
YELLOW_ACCENT  = colors.HexColor("#F2C811")   # UrbanRoof yellow/gold accent line
ORANGE         = colors.HexColor("#E8621A")   # UrbanRoof orange (logo, labels)
NAVY           = colors.HexColor("#1A2B4A")   # Sub-section heading navy bg
LIGHT_GREY     = colors.HexColor("#F4F4F4")   # Table alternating row tint
MID_GREY       = colors.HexColor("#CCCCCC")   # Borders / dividers
DARK_GREY      = colors.HexColor("#3D3D3D")   # Body text
WHITE          = colors.white
BLACK          = colors.black
GREEN_ACCENT   = colors.HexColor("#5CB85C")   # Top header green bar (like in DDR)
HEADER_BLACK   = colors.HexColor("#1C1C1C")   # Section header black strip bg

# Severity colours
SEVERITY_CLR: dict = {
    "critical": colors.HexColor("#C0392B"),
    "high":     colors.HexColor("#E67E22"),
    "medium":   colors.HexColor("#F39C12"),
    "low":      colors.HexColor("#27AE60"),
}

def severity_color(label: str) -> colors.Color:
    label_lower = label.lower()
    for key, clr in SEVERITY_CLR.items():
        if key in label_lower:
            return clr
    return DARK_GREY

# ── Paragraph Styles ─────────────────────────────────────────────────────────

# Cover — large white title
STYLE_COVER_TITLE = ParagraphStyle(
    "CoverTitle",
    fontName="Helvetica-Bold",
    fontSize=30,
    textColor=WHITE,
    alignment=TA_CENTER,
    spaceAfter=10,
    leading=38,
)

STYLE_COVER_SUBTITLE = ParagraphStyle(
    "CoverSubtitle",
    fontName="Helvetica",
    fontSize=13,
    textColor=colors.HexColor("#CCCCCC"),
    alignment=TA_CENTER,
    spaceAfter=6,
    leading=18,
)

STYLE_COVER_LABEL = ParagraphStyle(
    "CoverLabel",
    fontName="Helvetica-Bold",
    fontSize=9,
    textColor=YELLOW_ACCENT,
    alignment=TA_LEFT,
    spaceAfter=2,
    leading=13,
)

STYLE_COVER_VALUE = ParagraphStyle(
    "CoverValue",
    fontName="Helvetica",
    fontSize=9,
    textColor=WHITE,
    alignment=TA_LEFT,
    spaceAfter=4,
    leading=13,
)

# Section heading — used inside _section_header() tables (white text on black)
STYLE_SECTION_HEADING = ParagraphStyle(
    "SectionHeading",
    fontName="Helvetica-Bold",
    fontSize=12,
    textColor=WHITE,
    spaceBefore=0,
    spaceAfter=0,
    leading=16,
)

# Sub-section heading — navy bold, used for area blocks
STYLE_SUBSECTION_HEADING = ParagraphStyle(
    "SubsectionHeading",
    fontName="Helvetica-Bold",
    fontSize=11,
    textColor=NAVY,
    spaceBefore=10,
    spaceAfter=4,
    leading=14,
)

# Body text — justified, 9.5pt, dark grey
STYLE_BODY = ParagraphStyle(
    "Body",
    fontName="Helvetica",
    fontSize=9.5,
    textColor=DARK_GREY,
    alignment=TA_JUSTIFY,
    spaceBefore=3,
    spaceAfter=4,
    leading=14,
)

# Bullet item
STYLE_BULLET = ParagraphStyle(
    "Bullet",
    fontName="Helvetica",
    fontSize=9.5,
    textColor=DARK_GREY,
    leftIndent=14,
    firstLineIndent=0,
    spaceBefore=2,
    spaceAfter=2,
    leading=13,
)

# Footer
STYLE_FOOTER = ParagraphStyle(
    "Footer",
    fontName="Helvetica",
    fontSize=7.5,
    textColor=MID_GREY,
    alignment=TA_CENTER,
)

# Table of Contents entry
STYLE_TOC_ENTRY = ParagraphStyle(
    "TOCEntry",
    fontName="Helvetica",
    fontSize=10,
    textColor=DARK_GREY,
    spaceBefore=4,
    spaceAfter=4,
    leading=15,
)

STYLE_TOC_HEADING = ParagraphStyle(
    "TOCHeading",
    fontName="Helvetica-Bold",
    fontSize=16,
    textColor=NAVY,
    alignment=TA_CENTER,
    spaceBefore=12,
    spaceAfter=14,
    leading=22,
)

STYLE_LABEL = ParagraphStyle(
    "Label",
    fontName="Helvetica-Bold",
    fontSize=9,
    textColor=ORANGE,
    spaceBefore=2,
    spaceAfter=1,
)

# Used inside table header cells (white text on dark background)
STYLE_TABLE_HEADER = ParagraphStyle(
    "TableHeader",
    fontName="Helvetica-Bold",
    fontSize=9,
    textColor=WHITE,
    spaceBefore=0,
    spaceAfter=0,
    leading=12,
)

# Image captions
STYLE_IMAGE_CAPTION = ParagraphStyle(
    "ImageCaption",
    fontName="Helvetica-Oblique",
    fontSize=8,
    textColor=colors.HexColor("#888888"),
    alignment=TA_CENTER,
    spaceBefore=2,
    spaceAfter=6,
)

# Disclaimer / legal italic text
STYLE_DISCLAIMER = ParagraphStyle(
    "Disclaimer",
    fontName="Helvetica-Oblique",
    fontSize=9,
    textColor=DARK_GREY,
    alignment=TA_JUSTIFY,
    spaceBefore=5,
    spaceAfter=7,
    leading=13,
)

STYLE_WELCOME = ParagraphStyle(
    "Welcome",
    fontName="Helvetica",
    fontSize=9.5,
    textColor=DARK_GREY,
    alignment=TA_JUSTIFY,
    spaceBefore=4,
    spaceAfter=6,
    leading=14,
)

# ── Table style (Severity table) ─────────────────────────────────────────────

SEVERITY_TABLE_STYLE = TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  HEADER_BLACK),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, 0),  9),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GREY, WHITE]),
    ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",      (0, 1), (-1, -1), 9),
    ("GRID",          (0, 0), (-1, -1), 0.5, MID_GREY),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
])
