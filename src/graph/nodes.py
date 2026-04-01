"""
LangGraph Nodes — one function per DDR section.

Each node:
  1. Queries the vector store for relevant context.
  2. Calls the Cohere LLM with rich, specific prompts.
  3. Writes its output into the shared DDRState.

KEY FIXES:
  - Image assignment uses section_hint + is_thermal_overlay flag for proper pairing.
  - Context retrieval uses richer, area-specific queries.
  - LLM prompts pass the full retrieved context (not just snippets).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langsmith import traceable
except ImportError:
    # Graceful fallback: @traceable becomes a no-op decorator
    def traceable(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn
        return _wrap if args and callable(args[0]) else _wrap

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

# Per-area vector-store retrieval queries — specific to the inspection report structure
AREA_QUERIES: Dict[str, List[str]] = {
    "Bathroom / Internal Wet Areas": [
        "bathroom tile joint hollowness dampness plumbing nahani",
        "common bathroom master bedroom bathroom tile defects",
        "bathroom leakage ceiling dampness skirting",
    ],
    "Balcony": [
        "balcony tile joint hollowness dampness open balcony",
        "balcony external wall crack dampness",
    ],
    "Terrace / Roof": [
        "terrace roof screed crack hollow vegetation parapet",
        "terrace IPS surface waterproofing slope drainage",
    ],
    "External Wall": [
        "external wall crack hairline dampness chajja",
        "exterior wall paint spalling moisture ingress",
        "duct external wall crack master bedroom",
    ],
    "Plaster / Substrate": [
        "plaster loose hollow substrate sand faced re-plaster",
        "plaster crack separation bond coat",
    ],
    "Structural Elements": [
        "structural beam column RCC reinforcement spalling corrosion",
        "structural crack concrete exposed steel",
    ],
}


# ── LLM helpers ──────────────────────────────────────────────────────────────

def _build_llm(config: Config) -> ChatOpenRouter:
    return ChatOpenRouter(
        model=config.openrouter_model,
        api_key=config.openrouter_api_key,
        temperature=0.2,
        max_tokens=1024,
    )


def _call_llm(llm: ChatOpenRouter, user_prompt: str, section_name: str = "") -> str:
    """Call the LLM with retry logic. Returns a string — never raises."""
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
                    f"[{section_name}] OpenRouter auth failed — API key invalid/expired.\n"
                    "  Update OPENROUTER_API_KEY in .env: https://openrouter.ai/keys"
                )
                return (
                    "Not Available — OpenRouter authentication failed. "
                    "Update OPENROUTER_API_KEY in .env and re-run."
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


def _build_context(
    vector_store,
    queries: List[str],
    k: int = 6,
    doc_type: str | None = None,
) -> str:
    """
    Run multiple queries and merge the unique results into a single context string.
    This ensures broader coverage compared to a single query.
    """
    filter_dict = {"doc_type": doc_type} if doc_type else None
    seen_ids: set = set()
    all_docs = []

    for query in queries:
        docs = vector_store.retrieve(query=query, k=k, filter=filter_dict)
        for doc in docs:
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:80])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    if not all_docs:
        return "No relevant context found in source documents."

    return "\n\n---\n\n".join(d.page_content for d in all_docs)


# ── Image assignment ──────────────────────────────────────────────────────────

def _assign_images_for_area(
    image_map: Dict[str, List[Dict]],
    section_key: str,
    max_visual: int = 3,
    max_thermal: int = 3,
) -> Dict[str, List[Path]]:
    """
    Select inspection photos and thermal overlays for an area.

    Strategy:
    - For inspection images: match by section_hint, take the best ones
    - For thermal images: match by section_hint, separate real photos from thermal overlays
      using the is_thermal_overlay flag
    - Return {"visual": [...], "thermal": [...]} where:
        visual = inspection photos + real photos from thermal PDF
        thermal = thermal overlay images from thermal PDF
    """

    def _paths_for_section(
        source: List[Dict],
        section: str,
        limit: int,
        only_overlay: bool | None = None,
    ) -> List[Path]:
        """
        Filter source list by section_hint (and optionally overlay flag),
        return up to `limit` valid Paths.
        """
        results: List[Path] = []
        seen: set = set()
        for img in source:
            if img.get("section_hint") != section:
                continue
            if only_overlay is not None:
                if bool(img.get("is_thermal_overlay", False)) != only_overlay:
                    continue
            p = Path(img.get("path", ""))
            key = str(p)
            if key not in seen and p.exists() and p.stat().st_size > 0:
                seen.add(key)
                results.append(p)
                if len(results) >= limit:
                    break
        return results

    inspection_imgs = image_map.get("inspection", [])
    thermal_imgs    = image_map.get("thermal",    [])

    # Visual evidence = inspection photos for this section
    visual_paths = _paths_for_section(inspection_imgs, section_key, max_visual)

    # If no inspection photos found, supplement from thermal PDF real photos
    if not visual_paths:
        visual_paths = _paths_for_section(
            thermal_imgs, section_key, max_visual, only_overlay=False
        )

    # Thermal overlays = thermal camera images from thermal PDF
    thermal_paths = _paths_for_section(
        thermal_imgs, section_key, max_thermal, only_overlay=True
    )

    # If no thermal-specific overlays, try any thermal image for the section
    if not thermal_paths:
        thermal_paths = _paths_for_section(thermal_imgs, section_key, max_thermal)

    n_v = len(visual_paths)
    n_t = len(thermal_paths)
    logger.debug(f"  [{section_key}] → {n_v} visual, {n_t} thermal images")

    return {"visual": visual_paths, "thermal": thermal_paths}


# ── Node 1 — Property Issue Summary ──────────────────────────────────────────

@traceable(name="node_property_summary", run_type="chain", tags=["ddr-node"])
def node_property_summary(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 1] Generating Property Issue Summary…")
    context = _build_context(
        vector_store,
        queries=[
            "property inspection overview impacted areas rooms",
            "hall bedroom kitchen master bedroom parking area common bathroom",
            "dampness leakage seepage tile hollowness crack external wall",
            "summary of all defects observations",
        ],
        k=8,
    )
    llm  = _build_llm(config)
    text = _call_llm(llm, PROPERTY_SUMMARY_PROMPT.format(context=context), "property_summary")
    logger.info("  ✓ Property summary generated")
    return {"property_summary": text}


# ── Node 2 — Area-wise Observations (+images) ─────────────────────────────────

@traceable(name="node_area_observations", run_type="chain", tags=["ddr-node"])
def node_area_observations(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 2] Generating Area-wise Observations…")
    llm           = _build_llm(config)
    observations:  Dict[str, str]     = {}
    area_images:   Dict[str, Any]     = {}
    image_map:     Dict[str, List[Dict]] = state.get("image_map", {})

    for area in DDR_AREAS:
        section_key = AREA_TO_SECTION_KEY.get(area, "general")
        queries     = AREA_QUERIES.get(area, [f"{area} defects observations"])

        # Get context from both inspection and thermal docs
        insp_ctx  = _build_context(vector_store, queries, k=6, doc_type="inspection")
        therm_ctx = _build_context(vector_store, queries, k=4, doc_type="thermal")
        combined_ctx = f"[Inspection Observations]\n{insp_ctx}\n\n[Thermal Readings]\n{therm_ctx}"

        text = _call_llm(
            llm,
            AREA_OBSERVATIONS_PROMPT.format(context=combined_ctx, area=area),
            f"obs:{area}",
        )
        observations[area] = text

        img_dict = _assign_images_for_area(image_map, section_key)
        area_images[area] = img_dict

        n_v = len(img_dict.get("visual",  []))
        n_t = len(img_dict.get("thermal", []))
        logger.info(f"  ✓ {area} — {n_v} visual, {n_t} thermal image(s)")

    return {"area_observations": observations, "area_images": area_images}


# ── Node 3 — Probable Root Cause ──────────────────────────────────────────────

@traceable(name="node_root_causes", run_type="chain", tags=["ddr-node"])
def node_root_causes(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 3] Generating Root Causes…")
    llm         = _build_llm(config)
    root_causes: Dict[str, str] = {}

    for area in DDR_AREAS:
        queries = [
            q + " root cause reason origin failure" for q in AREA_QUERIES.get(area, [area])
        ]
        context = _build_context(vector_store, queries, k=6)
        text    = _call_llm(llm, ROOT_CAUSE_PROMPT.format(context=context, area=area), f"root:{area}")
        root_causes[area] = text
        logger.info(f"  ✓ {area}")

    return {"root_causes": root_causes}


# ── Node 4 — Severity Assessment ──────────────────────────────────────────────

@traceable(name="node_severity", run_type="chain", tags=["ddr-node"])
def node_severity(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 4] Generating Severity Assessments…")
    llm      = _build_llm(config)
    severity: Dict[str, str] = {}

    for area in DDR_AREAS:
        queries = [
            q + " severity damage extent structural risk" for q in AREA_QUERIES.get(area, [area])
        ]
        context = _build_context(vector_store, queries, k=6)
        text    = _call_llm(llm, SEVERITY_PROMPT.format(context=context, area=area), f"sev:{area}")
        severity[area] = text
        logger.info(f"  ✓ {area}")

    return {"severity": severity}


# ── Node 5 — Recommended Actions ──────────────────────────────────────────────

@traceable(name="node_recommended_actions", run_type="chain", tags=["ddr-node"])
def node_recommended_actions(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 5] Generating Recommended Actions…")
    llm     = _build_llm(config)
    actions: Dict[str, str] = {}

    for area in DDR_AREAS:
        queries = [
            q + " treatment repair remedy waterproofing recommendation"
            for q in AREA_QUERIES.get(area, [area])
        ]
        # Also include general therapy/treatment context
        queries.append("grouting treatment plaster work RCC treatment Dr Fixit URP")
        context = _build_context(vector_store, queries, k=8)
        text    = _call_llm(
            llm,
            RECOMMENDED_ACTIONS_PROMPT.format(context=context, area=area),
            f"act:{area}",
        )
        actions[area] = text
        logger.info(f"  ✓ {area}")

    return {"recommended_actions": actions}


# ── Node 6 — Additional Notes ─────────────────────────────────────────────────

@traceable(name="node_additional_notes", run_type="chain", tags=["ddr-node"])
def node_additional_notes(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 6] Generating Additional Notes…")
    context = _build_context(
        vector_store,
        queries=[
            "precautions limitations follow-up inspection warranty curing time",
            "limitation disclaimer scope of inspection",
            "structural engineer further investigation recommendation",
        ],
        k=6,
    )
    llm  = _build_llm(config)
    text = _call_llm(llm, ADDITIONAL_NOTES_PROMPT.format(context=context), "additional_notes")
    logger.info("  ✓ Additional notes generated")
    return {"additional_notes": text}


# ── Node 7 — Missing / Unclear Information ────────────────────────────────────

def node_missing_info(state: DDRState, config: Config, vector_store) -> Dict:
    logger.info("[Node 7] Identifying Missing / Unclear Information…")
    ctx_insp  = _build_context(
        vector_store,
        ["missing unclear conflicting information not sure"],
        k=5,
        doc_type="inspection",
    )
    ctx_therm = _build_context(
        vector_store,
        ["missing unclear conflicting thermal readings temperature"],
        k=5,
        doc_type="thermal",
    )
    combined = f"[Inspection Context]\n{ctx_insp}\n\n[Thermal Context]\n{ctx_therm}"
    llm  = _build_llm(config)
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

    # Log image assignment summary
    area_images = state.get("area_images", {})
    for area, imgs in area_images.items():
        n_v = len(imgs.get("visual", []))
        n_t = len(imgs.get("thermal", []))
        logger.info(f"  Images — {area}: {n_v} visual, {n_t} thermal")

    return {"errors": state.get("errors", [])}
