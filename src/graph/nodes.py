"""
LangGraph Nodes — one function per DDR section.

Each node:
  1. Queries the vector store for relevant context.
  2. Calls the OpenRouter LLM with rich, specific prompts.
  3. Writes its output into the shared DDRState.

KEY FIXES:
  - Image assignment uses section_hint + is_thermal_overlay flag for proper pairing.
  - Context retrieval uses richer, area-specific queries.
  - LLM prompts pass the full retrieved context (not just snippets).
  - Nodes 5/6/7 use prefetched_context — zero extra DB calls after Node 1.
"""

from __future__ import annotations

import time
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def _build_llm(config: Config, max_tokens: int = 600) -> ChatOpenRouter:
    """Build an LLM client. Pass right-sized max_tokens per section to reduce generation time."""
    return ChatOpenRouter(
        model=config.openrouter_model,
        api_key=config.openrouter_api_key,
        temperature=0.2,
        max_tokens=max_tokens,
    )


def _submit_with_context(pool: ThreadPoolExecutor, fn, *args):
    """
    Submit *fn* to the thread pool, propagating the current contextvars snapshot.
    This is what makes LangSmith trace IDs flow into worker threads — without it
    every thread spawned by ThreadPoolExecutor is an orphan in the trace tree.
    """
    ctx = copy_context()
    return pool.submit(ctx.run, fn, *args)


def _call_llm(llm: ChatOpenRouter, user_prompt: str, section_name: str = "") -> str:
    """Call the LLM with retry logic + elapsed-time logging. Returns a string — never raises."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    t0 = time.perf_counter()

    for attempt in range(1, 4):
        try:
            response = llm.invoke(messages)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(f"[{section_name}] LLM responded in {elapsed:.0f}ms")
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
    """

    def _paths_for_section(
        source: List[Dict],
        section: str,
        limit: int,
        only_overlay: bool | None = None,
    ) -> List[Path]:
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

    visual_paths = _paths_for_section(inspection_imgs, section_key, max_visual)

    if not visual_paths:
        visual_paths = _paths_for_section(
            thermal_imgs, section_key, max_visual, only_overlay=False
        )

    thermal_paths = _paths_for_section(
        thermal_imgs, section_key, max_thermal, only_overlay=True
    )

    if not thermal_paths:
        thermal_paths = _paths_for_section(thermal_imgs, section_key, max_thermal)

    n_v = len(visual_paths)
    n_t = len(thermal_paths)
    logger.debug(f"  [{section_key}] → {n_v} visual, {n_t} thermal images")

    return {"visual": visual_paths, "thermal": thermal_paths}


# ── Node 1 — Property Issue Summary ──────────────────────────────────────────

@traceable(name="node_property_summary", run_type="chain", tags=["ddr-node"])
def node_property_summary(state: DDRState, config: Config, vector_store) -> Dict:
    """
    Node 1: Generate property summary AND pre-fetch ALL context for every downstream node.

    Pre-fetching here (single-threaded, before Stage 1 fan-out) means:
    - All ChromaDB queries happen ONCE, on the main thread.
    - Stage 1-3 worker threads read from state["prefetched_context"] — zero DB calls.
    - Eliminates ChromaDB thread-safety concerns entirely.
    """
    t_node = time.perf_counter()
    logger.info("[Node 1] Generating Property Issue Summary + pre-fetching all context...")

    # ── 1a. Summary context (broad overview) ────────────────────────────────
    summary_ctx = _build_context(
        vector_store,
        queries=[
            "property inspection overview impacted areas rooms",
            "hall bedroom kitchen master bedroom parking area common bathroom",
            "dampness leakage seepage tile hollowness crack external wall",
            "summary of all defects observations",
        ],
        k=8,
    )

    # ── 1b. Pre-fetch per-area context for ALL downstream nodes ────────────────
    prefetched: Dict[str, Dict[str, str]] = {}

    for area in DDR_AREAS:
        queries = AREA_QUERIES.get(area, [area])
        insp  = _build_context(vector_store, queries, k=6, doc_type="inspection")
        therm = _build_context(vector_store, queries, k=4, doc_type="thermal")

        # Also fetch action-specific context for node 5
        action_queries = [
            q + " treatment repair remedy waterproofing recommendation"
            for q in queries
        ]
        action_queries.append("grouting treatment plaster work RCC treatment Dr Fixit URP")
        actions_ctx = _build_context(vector_store, action_queries, k=8)

        prefetched[area] = {
            "inspection": insp,
            "thermal":    therm,
            "actions":    actions_ctx,
        }

    # ── 1c. Global keys for nodes 6/7 ─────────────────────────────────────────
    prefetched["_global"] = {
        "all": _build_context(
            vector_store,
            ["dampness crack seepage defects all areas summary"],
            k=10,
        ),
        "notes": _build_context(
            vector_store,
            ["precautions limitations follow-up inspection warranty curing time",
             "limitation disclaimer scope structural engineer"],
            k=6,
        ),
        "missing_insp": _build_context(
            vector_store,
            ["missing unclear conflicting information not sure"],
            k=5, doc_type="inspection",
        ),
        "missing_therm": _build_context(
            vector_store,
            ["missing unclear conflicting thermal readings temperature"],
            k=5, doc_type="thermal",
        ),
    }

    logger.info(f"  Context pre-fetched for {len(DDR_AREAS)} areas + global keys")

    # ── 1d. Generate summary text ─────────────────────────────────────────────
    llm  = _build_llm(config, max_tokens=config.max_tokens_summary)
    text = _call_llm(llm, PROPERTY_SUMMARY_PROMPT.format(context=summary_ctx), "property_summary")

    elapsed_ms = (time.perf_counter() - t_node) * 1000
    logger.info(f"  [Node 1] done in {elapsed_ms:.0f}ms")
    return {
        "property_summary":   text,
        "prefetched_context": prefetched,
        "timings":            {"node1_ms": elapsed_ms},
    }


# ── Node 2 — Area-wise Observations (+images) ─────────────────────────────────

def _obs_for_area(
    area: str, config: Config, prefetched: Dict, image_map: Dict
) -> tuple:
    """Worker: build observation text + assign images for one area (no DB calls)."""
    t0          = time.perf_counter()
    section_key = AREA_TO_SECTION_KEY.get(area, "general")
    area_ctx    = prefetched.get(area, {})
    insp_ctx    = area_ctx.get("inspection", "No inspection context available.")
    therm_ctx   = area_ctx.get("thermal",    "No thermal context available.")
    combined    = f"[Inspection Observations]\n{insp_ctx}\n\n[Thermal Readings]\n{therm_ctx}"
    llm         = _build_llm(config, max_tokens=config.max_tokens_observation)
    text        = _call_llm(llm, AREA_OBSERVATIONS_PROMPT.format(context=combined, area=area), f"obs:{area}")
    img_dict    = _assign_images_for_area(image_map, section_key)
    elapsed     = (time.perf_counter() - t0) * 1000
    logger.info(f"  [obs] {area} done in {elapsed:.0f}ms")
    return area, text, img_dict


@traceable(name="node_area_observations", run_type="chain", tags=["ddr-node"])
def node_area_observations(state: DDRState, config: Config, vector_store) -> Dict:
    t_node      = time.perf_counter()
    logger.info("[Node 2] Area-wise Observations (parallel, context from cache)")
    image_map   = state.get("image_map", {})
    prefetched  = state.get("prefetched_context", {})
    observations: Dict[str, str] = {}
    area_images:  Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            _submit_with_context(pool, _obs_for_area, area, config, prefetched, image_map): area
            for area in DDR_AREAS
        }
        for fut in as_completed(futures):
            area, text, img_dict = fut.result()
            observations[area] = text
            area_images[area]   = img_dict

    elapsed_ms = (time.perf_counter() - t_node) * 1000
    logger.info(f"  [Node 2] all areas done in {elapsed_ms:.0f}ms")
    return {
        "area_observations": observations,
        "area_images":       area_images,
        "timings":           {**state.get("timings", {}), "node2_ms": elapsed_ms},
    }


# ── Node 3 — Probable Root Cause ──────────────────────────────────────────────

@traceable(name="node_root_causes", run_type="chain", tags=["ddr-node"])
def node_root_causes(state: DDRState, config: Config, vector_store) -> Dict:
    t_node     = time.perf_counter()
    logger.info("[Node 3] Root Causes (parallel, context from cache)")
    prefetched = state.get("prefetched_context", {})
    root_causes: Dict[str, str] = {}

    def _root_for_area(area: str) -> tuple:
        t0      = time.perf_counter()
        ctx     = prefetched.get(area, {})
        context = f"[Inspection]\n{ctx.get('inspection','')}\n\n[Thermal]\n{ctx.get('thermal','')}"
        llm     = _build_llm(config, max_tokens=config.max_tokens_root_cause)
        text    = _call_llm(llm, ROOT_CAUSE_PROMPT.format(context=context, area=area), f"root:{area}")
        logger.info(f"  [root] {area} done in {(time.perf_counter()-t0)*1000:.0f}ms")
        return area, text

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {_submit_with_context(pool, _root_for_area, area): area for area in DDR_AREAS}
        for fut in as_completed(futures):
            area, text = fut.result()
            root_causes[area] = text

    elapsed_ms = (time.perf_counter() - t_node) * 1000
    logger.info(f"  [Node 3] all areas done in {elapsed_ms:.0f}ms")
    return {
        "root_causes": root_causes,
        "timings":     {**state.get("timings", {}), "node3_ms": elapsed_ms},
    }


# ── Node 4 — Severity Assessment ──────────────────────────────────────────────

@traceable(name="node_severity", run_type="chain", tags=["ddr-node"])
def node_severity(state: DDRState, config: Config, vector_store) -> Dict:
    t_node     = time.perf_counter()
    logger.info("[Node 4] Severity Assessment (parallel, context from cache)")
    prefetched = state.get("prefetched_context", {})
    severity:  Dict[str, str] = {}

    def _sev_for_area(area: str) -> tuple:
        t0      = time.perf_counter()
        ctx     = prefetched.get(area, {})
        context = f"[Inspection]\n{ctx.get('inspection','')}\n\n[Thermal]\n{ctx.get('thermal','')}"
        llm     = _build_llm(config, max_tokens=config.max_tokens_severity)
        text    = _call_llm(llm, SEVERITY_PROMPT.format(context=context, area=area), f"sev:{area}")
        logger.info(f"  [sev] {area} done in {(time.perf_counter()-t0)*1000:.0f}ms")
        return area, text

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {_submit_with_context(pool, _sev_for_area, area): area for area in DDR_AREAS}
        for fut in as_completed(futures):
            area, text = fut.result()
            severity[area] = text

    elapsed_ms = (time.perf_counter() - t_node) * 1000
    logger.info(f"  [Node 4] all areas done in {elapsed_ms:.0f}ms")
    return {
        "severity": severity,
        "timings":  {**state.get("timings", {}), "node4_ms": elapsed_ms},
    }


# ── Node 5 — Recommended Actions (uses prefetched context) ────────────────────

@traceable(name="node_recommended_actions", run_type="chain", tags=["ddr-node"])
def node_recommended_actions(state: DDRState, config: Config, vector_store) -> Dict:
    """Parallel recommended actions — uses prefetched 'actions' context, zero DB calls."""
    t_node     = time.perf_counter()
    logger.info("[Node 5] Generating Recommended Actions (parallel, context from cache)")
    prefetched = state.get("prefetched_context", {})
    actions:    Dict[str, str] = {}

    def _act_for_area(area: str) -> tuple:
        t0      = time.perf_counter()
        ctx     = prefetched.get(area, {})
        # Use the pre-fetched action context (richer, includes treatment queries)
        context = ctx.get("actions", ctx.get("inspection", "No context available."))
        llm     = _build_llm(config, max_tokens=config.max_tokens_actions)
        text    = _call_llm(llm, RECOMMENDED_ACTIONS_PROMPT.format(context=context, area=area), f"act:{area}")
        logger.info(f"  [act] {area} done in {(time.perf_counter()-t0)*1000:.0f}ms")
        return area, text

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {_submit_with_context(pool, _act_for_area, area): area for area in DDR_AREAS}
        for fut in as_completed(futures):
            area, text = fut.result()
            actions[area] = text

    elapsed_ms = (time.perf_counter() - t_node) * 1000
    logger.info(f"  [Node 5] all areas done in {elapsed_ms:.0f}ms")
    return {
        "recommended_actions": actions,
        "timings": {**state.get("timings", {}), "node5_ms": elapsed_ms},
    }


# ── Node 6 — Additional Notes (uses prefetched context) ──────────────────────

@traceable(name="node_additional_notes", run_type="chain", tags=["ddr-node"])
def node_additional_notes(state: DDRState, config: Config, vector_store) -> Dict:
    """Additional notes — uses pre-fetched _global.notes context, zero DB calls."""
    t_node     = time.perf_counter()
    logger.info("[Node 6] Generating Additional Notes (context from cache)…")
    prefetched = state.get("prefetched_context", {})
    context    = prefetched.get("_global", {}).get("notes", "No context available.")
    llm        = _build_llm(config, max_tokens=config.max_tokens_notes)
    text       = _call_llm(llm, ADDITIONAL_NOTES_PROMPT.format(context=context), "additional_notes")
    elapsed_ms = (time.perf_counter() - t_node) * 1000
    logger.info(f"  ✓ Additional notes done in {elapsed_ms:.0f}ms")
    return {
        "additional_notes": text,
        "timings": {**state.get("timings", {}), "node6_ms": elapsed_ms},
    }


# ── Node 7 — Missing / Unclear Information (uses prefetched context) ──────────

def node_missing_info(state: DDRState, config: Config, vector_store) -> Dict:
    """Missing info — uses pre-fetched _global.missing_* context, zero DB calls."""
    t_node     = time.perf_counter()
    logger.info("[Node 7] Identifying Missing / Unclear Information (context from cache)…")
    prefetched = state.get("prefetched_context", {})
    g          = prefetched.get("_global", {})
    ctx_insp   = g.get("missing_insp",  "No inspection context.")
    ctx_therm  = g.get("missing_therm", "No thermal context.")
    combined   = f"[Inspection Context]\n{ctx_insp}\n\n[Thermal Context]\n{ctx_therm}"
    llm        = _build_llm(config, max_tokens=config.max_tokens_missing)
    text       = _call_llm(llm, MISSING_INFO_PROMPT.format(context=combined), "missing_info")
    elapsed_ms = (time.perf_counter() - t_node) * 1000
    logger.info(f"  ✓ Missing info done in {elapsed_ms:.0f}ms")
    return {
        "missing_info": text,
        "timings":      {**state.get("timings", {}), "node7_ms": elapsed_ms},
    }


# ── Node 8 — Compile ──────────────────────────────────────────────────────────

def node_compile(state: DDRState, config: Config, vector_store) -> Dict:
    """Validate state completeness and log timings. No LLM call."""
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

    # Log timing breakdown
    timings = state.get("timings", {})
    if timings:
        total = sum(timings.values())
        logger.info(f"  Timing breakdown: {timings}")
        logger.info(f"  Total tracked ms: {total:.0f}ms")

    # Log image assignment summary
    area_images = state.get("area_images", {})
    for area, imgs in area_images.items():
        n_v = len(imgs.get("visual", []))
        n_t = len(imgs.get("thermal", []))
        logger.info(f"  Images — {area}: {n_v} visual, {n_t} thermal")

    return {"errors": state.get("errors", [])}
