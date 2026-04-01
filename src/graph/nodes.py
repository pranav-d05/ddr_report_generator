"""
LangGraph Nodes — one function per DDR section.

Each node:
  1. Queries the vector store for relevant context.
  2. Calls the Cohere LLM.
  3. Writes its output into the shared DDRState.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Config
from src.graph.state import DDRState
from src.graph.prompts import (
    SYSTEM_PROMPT,
    PROPERTY_SUMMARY_PROMPT,
    AREA_OBSERVATIONS_PROMPT,
    ROOT_CAUSE_PROMPT,
    SEVERITY_PROMPT,
    RECOMMENDED_ACTIONS_PROMPT,
    ADDITIONAL_NOTES_PROMPT,
    MISSING_INFO_PROMPT,
)
from src.utils.logger import get_logger
from src.utils.helpers import clean_llm_output

logger = get_logger(__name__)

# ── DDR areas ────────────────────────────────────────────────────────────────

DDR_AREAS: List[str] = [
    "Bathroom / Internal Wet Areas",
    "Balcony",
    "Terrace / Roof",
    "External Wall",
    "Plaster / Substrate",
    "Structural Elements",
]

# Map display area name → section_hint keyword used in image metadata
AREA_TO_SECTION_KEY: Dict[str, str] = {
    "Bathroom / Internal Wet Areas": "bathroom",
    "Balcony":                        "balcony",
    "Terrace / Roof":                 "terrace",
    "External Wall":                  "external_wall",
    "Plaster / Substrate":            "plaster",
    "Structural Elements":            "structural",
}

# Fallback chain: if an area has no exact-match images, try these in order
AREA_FALLBACKS: Dict[str, List[str]] = {
    "bathroom":     ["plaster", "structural", "analysis"],
    "balcony":      ["external_wall", "plaster", "analysis"],
    "terrace":      ["structural", "analysis", "summary"],
    "external_wall":["plaster", "structural", "analysis"],
    "plaster":      ["external_wall", "structural", "analysis"],
    "structural":   ["external_wall", "plaster", "analysis"],
}


# ── LLM helpers ──────────────────────────────────────────────────────────────

def _build_llm(config: Config) -> ChatCohere:
    return ChatCohere(
        model=config.cohere_model,
        cohere_api_key=config.cohere_api_key,
        temperature=0.2,
        max_tokens=1024,
    )


def _call_llm(llm: ChatCohere, user_prompt: str, section_name: str = "") -> str:
    """
    Call the LLM with retry logic.
    Returns a string — never raises.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(1, 4):
        try:
            response = llm.invoke(messages)
            return clean_llm_output(response.content)
        except Exception as exc:
            err = str(exc)

            if "401" in err or "invalid_api_key" in err or "unauthorized" in err.lower():
                logger.error(
                    f"[{section_name}] Cohere auth failed — API key invalid/expired.\n"
                    "  Update COHERE_API_KEY in .env: https://dashboard.cohere.com/api-keys"
                )
                return (
                    "Not Available — Cohere authentication failed. "
                    "Update COHERE_API_KEY in .env and re-run."
                )

            if "429" in err or "rate" in err.lower():
                wait = 2 ** attempt
                logger.warning(f"[{section_name}] Rate limit, waiting {wait}s (attempt {attempt}/3)…")
                time.sleep(wait)
                continue

            if "timeout" in err.lower():
                wait = 5 * attempt
                logger.warning(f"[{section_name}] Timeout, waiting {wait}s (attempt {attempt}/3)…")
                time.sleep(wait)
                continue

            logger.error(f"[{section_name}] LLM error (attempt {attempt}/3): {exc}")
            if attempt == 3:
                return f"Not Available — LLM call failed after 3 attempts: {type(exc).__name__}"
            time.sleep(2)

    return "Not Available — all LLM retry attempts exhausted."


def _build_context(vector_store, query: str, k: int = 8, doc_type: str | None = None) -> str:
    """Retrieve relevant chunks and join into a context string."""
    filter_dict = {"doc_type": doc_type} if doc_type else None
    docs = vector_store.retrieve(query=query, k=k, filter=filter_dict)
    if not docs:
        return "No relevant context found in source documents."
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ── Image assignment ──────────────────────────────────────────────────────────

def _assign_images_for_area(
    image_map: Dict[str, List[Dict]],
    section_key: str,
    max_visual: int = 2,
    max_thermal: int = 2,
) -> Dict[str, List[Path]]:
    """
    Select up to max_visual inspection images and max_thermal thermal images
    for an area, using section_hint matching with fallback chains.
    """
    search_keys = [section_key] + AREA_FALLBACKS.get(section_key, [])

    def _collect(source_list: List[Dict], limit: int) -> List[Path]:
        seen: set = set()
        paths: List[Path] = []
        for key in search_keys:
            for img in source_list:
                p = img.get("path")
                if p and img.get("section_hint") == key and str(p) not in seen:
                    if Path(p).exists():
                        seen.add(str(p))
                        paths.append(Path(p))
                        if len(paths) >= limit:
                            return paths
        return paths

    visual_imgs  = _collect(image_map.get("inspection", []), max_visual)
    thermal_imgs = _collect(image_map.get("thermal",    []), max_thermal)

    return {"visual": visual_imgs, "thermal": thermal_imgs}


# ── Node 1 — Property Issue Summary ──────────────────────────────────────────

def node_property_summary(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 1] Generating Property Issue Summary…")
    context = _build_context(vector_store, "property inspection summary overview defects issues", k=10)
    llm = _build_llm(config)
    text = _call_llm(llm, PROPERTY_SUMMARY_PROMPT.format(context=context), "property_summary")
    logger.info("  ✓ Property summary generated")
    return {"property_summary": text}


# ── Node 2 — Area-wise Observations (+images) ─────────────────────────────────

def node_area_observations(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 2] Generating Area-wise Observations…")
    llm = _build_llm(config)
    observations: Dict[str, str] = {}
    area_images: Dict[str, Any] = {}
    image_map: Dict[str, List[Dict]] = state.get("image_map", {})

    for area in DDR_AREAS:
        section_key = AREA_TO_SECTION_KEY.get(area, "general")
        context = _build_context(vector_store, f"{area} defects observations leakage", k=8)
        text = _call_llm(llm, AREA_OBSERVATIONS_PROMPT.format(context=context, area=area), f"obs:{area}")
        observations[area] = text

        img_dict = _assign_images_for_area(image_map, section_key)
        area_images[area] = img_dict

        n_v = len(img_dict.get("visual", []))
        n_t = len(img_dict.get("thermal", []))
        logger.info(f"  ✓ {area} — {n_v} visual, {n_t} thermal image(s)")

    return {"area_observations": observations, "area_images": area_images}


# ── Node 3 — Probable Root Cause ──────────────────────────────────────────────

def node_root_causes(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 3] Generating Root Causes…")
    llm = _build_llm(config)
    root_causes: Dict[str, str] = {}
    for area in DDR_AREAS:
        context = _build_context(vector_store, f"{area} root cause reason origin failure", k=6)
        text = _call_llm(llm, ROOT_CAUSE_PROMPT.format(context=context, area=area), f"root:{area}")
        root_causes[area] = text
        logger.info(f"  ✓ {area}")
    return {"root_causes": root_causes}


# ── Node 4 — Severity Assessment ──────────────────────────────────────────────

def node_severity(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 4] Generating Severity Assessments…")
    llm = _build_llm(config)
    severity: Dict[str, str] = {}
    for area in DDR_AREAS:
        context = _build_context(vector_store, f"{area} severity damage extent structural risk", k=6)
        text = _call_llm(llm, SEVERITY_PROMPT.format(context=context, area=area), f"sev:{area}")
        severity[area] = text
        logger.info(f"  ✓ {area}")
    return {"severity": severity}


# ── Node 5 — Recommended Actions ──────────────────────────────────────────────

def node_recommended_actions(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 5] Generating Recommended Actions…")
    llm = _build_llm(config)
    actions: Dict[str, str] = {}
    for area in DDR_AREAS:
        context = _build_context(
            vector_store,
            f"{area} treatment repair remedy waterproofing recommendation",
            k=8,
        )
        text = _call_llm(llm, RECOMMENDED_ACTIONS_PROMPT.format(context=context, area=area), f"act:{area}")
        actions[area] = text
        logger.info(f"  ✓ {area}")
    return {"recommended_actions": actions}


# ── Node 6 — Additional Notes ─────────────────────────────────────────────────

def node_additional_notes(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 6] Generating Additional Notes…")
    context = _build_context(
        vector_store,
        "precautions limitations follow-up inspection warranty curing time",
        k=6,
    )
    llm = _build_llm(config)
    text = _call_llm(llm, ADDITIONAL_NOTES_PROMPT.format(context=context), "additional_notes")
    logger.info("  ✓ Additional notes generated")
    return {"additional_notes": text}


# ── Node 7 — Missing / Unclear Information ────────────────────────────────────

def node_missing_info(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 7] Identifying Missing / Unclear Information…")
    ctx_insp  = _build_context(vector_store, "missing unclear conflicting information", k=5, doc_type="inspection")
    ctx_therm = _build_context(vector_store, "missing unclear conflicting thermal readings", k=5, doc_type="thermal")
    combined  = f"[Inspection Context]\n{ctx_insp}\n\n[Thermal Context]\n{ctx_therm}"
    llm = _build_llm(config)
    text = _call_llm(llm, MISSING_INFO_PROMPT.format(context=combined), "missing_info")
    logger.info("  ✓ Missing info section generated")
    return {"missing_info": text}


# ── Node 8 — Compile ──────────────────────────────────────────────────────────

def node_compile(state: DDRState, config: Config, vector_store) -> Dict:
    """Validate state completeness. No LLM call."""
    logger.info("[Node 8] Compiling final report state…")
    required = [
        "property_summary", "area_observations", "root_causes",
        "severity", "recommended_actions", "additional_notes", "missing_info",
    ]
    missing = [s for s in required if not state.get(s)]
    if missing:
        logger.warning(f"  Sections not populated: {missing}")
    else:
        logger.info("  ✓ All sections populated")
    return {"errors": state.get("errors", [])}
