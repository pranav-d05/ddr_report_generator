"""
ReportLab Style Constants for the DDR PDF Report.
Matches the UrbanRoof DDR format from the Main_DDR.pdf sample.
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
MARGIN_TOP    = 2.5 * cm
MARGIN_BOTTOM = 2.5 * cm

CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT

# ── Colour palette (UrbanRoof brand — matches Main DDR) ───────────────────────

DARK_GREY_BG   = colors.HexColor("#3D3D3D")   # Cover dark grey background
MID_DARK_BG    = colors.HexColor("#2E2E2E")   # Slightly darker panels
YELLOW_ACCENT  = colors.HexColor("#F5C518")   # UrbanRoof yellow/gold accent
ORANGE         = colors.HexColor("#E8621A")   # UrbanRoof orange
NAVY           = colors.HexColor("#1A2B4A")   # Section heading navy
LIGHT_GREY     = colors.HexColor("#F5F5F5")   # Table alternating row
MID_GREY       = colors.HexColor("#CCCCCC")   # Borders / dividers
DARK_GREY      = colors.HexColor("#444444")   # Body text
WHITE          = colors.white
BLACK          = colors.black
GREEN_ACCENT   = colors.HexColor("#5CB85C")   # "Good" / green header bar
HEADER_BLACK   = colors.HexColor("#1C1C1C")   # Section header black strip

# Severity colours
SEVERITY_CLR = {
    "critical": colors.HexColor("#C0392B"),
    "high":     colors.HexColor("#E67E22"),
    "medium":   colors.HexColor("#F1C40F"),
    "low":      colors.HexColor("#27AE60"),
}

def severity_color(label: str) -> colors.Color:
    label_lower = label.lower()
    for key, clr in SEVERITY_CLR.items():
        if key in label_lower:
            return clr
    return DARK_GREY

# ── Paragraph Styles ─────────────────────────────────────────────────────────

_BASE = getSampleStyleSheet()

# Cover title - large white bold
STYLE_COVER_TITLE = ParagraphStyle(
    "CoverTitle",
    fontName="Helvetica-Bold",
    fontSize=30,
    textColor=WHITE,
    alignment=TA_CENTER,
    spaceAfter=10,
    leading=36,
)

STYLE_COVER_SUBTITLE = ParagraphStyle(
    "CoverSubtitle",
    fontName="Helvetica",
    fontSize=13,
    textColor=colors.HexColor("#DDDDDD"),
    alignment=TA_CENTER,
    spaceAfter=6,
    leading=18,
)

STYLE_COVER_LABEL = ParagraphStyle(
    "CoverLabel",
    fontName="Helvetica-Bold",
    fontSize=10,
    textColor=YELLOW_ACCENT,
    alignment=TA_LEFT,
    spaceAfter=2,
    leading=14,
)

STYLE_COVER_VALUE = ParagraphStyle(
    "CoverValue",
    fontName="Helvetica",
    fontSize=10,
    textColor=WHITE,
    alignment=TA_LEFT,
    spaceAfter=4,
    leading=14,
)

# Section heading — black strip style (like in DDR)
STYLE_SECTION_HEADING = ParagraphStyle(
    "SectionHeading",
    fontName="Helvetica-Bold",
    fontSize=13,
    textColor=WHITE,
    spaceBefore=12,
    spaceAfter=8,
    leading=18,
    backColor=HEADER_BLACK,
    borderPad=6,
    leftIndent=0,
)

# Sub-section heading (area names, H2-level) — navy bold
STYLE_SUBSECTION_HEADING = ParagraphStyle(
    "SubsectionHeading",
    fontName="Helvetica-Bold",
    fontSize=11,
    textColor=NAVY,
    spaceBefore=10,
    spaceAfter=4,
    leading=14,
)

# Body text
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
    spaceBefore=2,
    spaceAfter=2,
    leading=13,
    bulletIndent=4,
)

# Footer text
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
    textColor=NAVY,
    spaceBefore=4,
    spaceAfter=4,
    leading=15,
)

STYLE_TOC_HEADING = ParagraphStyle(
    "TOCHeading",
    fontName="Helvetica-Bold",
    fontSize=14,
    textColor=NAVY,
    alignment=TA_CENTER,
    spaceBefore=10,
    spaceAfter=10,
    leading=18,
)

STYLE_LABEL = ParagraphStyle(
    "Label",
    fontName="Helvetica-Bold",
    fontSize=9,
    textColor=ORANGE,
    spaceBefore=2,
    spaceAfter=1,
)

STYLE_TABLE_HEADER = ParagraphStyle(
    "TableHeader",
    fontName="Helvetica-Bold",
    fontSize=9,
    textColor=WHITE,
    spaceBefore=2,
    spaceAfter=1,
)

STYLE_IMAGE_CAPTION = ParagraphStyle(
    "ImageCaption",
    fontName="Helvetica-Oblique",
    fontSize=8,
    textColor=MID_GREY,
    alignment=TA_CENTER,
    spaceBefore=2,
    spaceAfter=4,
)

STYLE_DISCLAIMER = ParagraphStyle(
    "Disclaimer",
    fontName="Helvetica-Oblique",
    fontSize=9,
    textColor=DARK_GREY,
    alignment=TA_JUSTIFY,
    spaceBefore=4,
    spaceAfter=6,
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

# ── Table styles ─────────────────────────────────────────────────────────────

SEVERITY_TABLE_STYLE = TableStyle([
    ("BACKGROUND",   (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
    ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",     (0, 0), (-1, 0), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, WHITE]),
    ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",     (0, 1), (-1, -1), 9),
    ("GRID",         (0, 0), (-1, -1), 0.5, MID_GREY),
    ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",   (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
])
