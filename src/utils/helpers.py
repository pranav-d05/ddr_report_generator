"""
Shared utility / helper functions used across the DDR pipeline.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Timestamp ──────────────────────────────────────────────────────────────────

def timestamp_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return the current datetime as a formatted string, e.g. '20240401_183000'."""
    return datetime.now().strftime(fmt)


def human_timestamp(fmt: str = "%d %B %Y, %H:%M") -> str:
    """Human-readable timestamp for report headers, e.g. '01 April 2024, 18:30'."""
    return datetime.now().strftime(fmt)


# ── Filenames ──────────────────────────────────────────────────────────────────

def safe_filename(text: str, max_len: int = 64) -> str:
    """
    Convert an arbitrary string into a filesystem-safe filename stem.
    Replaces non-alphanumeric characters with underscores and trims length.
    """
    slug = re.sub(r"[^\w\-]", "_", text)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len]


# ── Text helpers ───────────────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 400, suffix: str = "…") -> str:
    """Truncate text to *max_chars*, appending *suffix* if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix


def clean_llm_output(text: str) -> str:
    """
    Strip common LLM artefacts: leading/trailing whitespace, triple-backtick
    fenced blocks (when the model wraps plain text in markdown code fences).
    """
    text = text.strip()
    # Remove markdown fences if the model wrapped output in them
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


# ── Section / severity helpers ─────────────────────────────────────────────────

SECTION_DISPLAY_NAMES: dict[str, str] = {
    "bathroom": "Bathroom / Internal Wet Areas",
    "balcony": "Balcony",
    "terrace": "Terrace / Roof",
    "external_wall": "External Wall",
    "plaster": "Plaster / Substrate",
    "structural": "Structural Elements",
    "thermal": "Thermal / IR Readings",
    "summary": "General Summary",
    "analysis": "Analysis",
    "general": "General",
}

SEVERITY_COLOURS: dict[str, str] = {
    "Critical": "#C0392B",
    "High":     "#E67E22",
    "Medium":   "#F1C40F",
    "Low":      "#27AE60",
}


def display_section(section_key: str) -> str:
    """Return a human-readable section name from an internal key."""
    return SECTION_DISPLAY_NAMES.get(section_key, section_key.replace("_", " ").title())


def severity_colour(severity_label: str) -> str:
    """Return a hex colour for a severity label (for PDF rendering)."""
    for key, colour in SEVERITY_COLOURS.items():
        if key.lower() in severity_label.lower():
            return colour
    return "#555555"


# ── Misc ───────────────────────────────────────────────────────────────────────

def flatten(nested: list[list[Any]]) -> list[Any]:
    """Flatten one level of nesting in a list."""
    return [item for sublist in nested for item in sublist]


def deduplicate_images(images: list[dict]) -> list[dict]:
    """
    Remove duplicate image records (same path) while preserving order.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for img in images:
        key = str(img.get("path", img.get("filename", "")))
        if key not in seen:
            seen.add(key)
            unique.append(img)
    return unique
