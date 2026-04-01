"""
LLM Prompt Templates for every DDR section node.

These prompts are tailored to the UrbanRoof inspection report format:
  - "Impacted Area N" with Negative/Positive side structure
  - Thermal readings with hotspot/coldspot temperatures
  - Specific defect types: dampness, tile hollowness, external cracks, etc.
"""

from __future__ import annotations

# ── System prompt (shared across all nodes) ────────────────────────────────────

SYSTEM_PROMPT = """You are an expert civil engineer and property diagnostics specialist
working for UrbanRoof Private Limited. Your task is to analyse raw inspection notes and
thermal imaging reports to produce a professional Detailed Diagnosis Report (DDR).

Guidelines:
- Write in clear, professional English appropriate for a property inspection report.
- Be SPECIFIC and FACTUAL — use exact locations, defect types, and measurements from context.
- The inspection reports use "Negative side" (observed damage/symptoms) and "Positive side"
  (identified source of the problem) terminology. Use both to write complete observations.
- Reference thermal readings (hotspot/coldspot temperatures, emissivity values) when available.
- If the provided context does not contain enough information for a section, write exactly:
  "Not Available — insufficient data in source documents."
- Do NOT fabricate data, temperatures, or measurements not present in the context.
- Format your output as clean prose. Use bullet points only when listing multiple distinct items.
- Keep language simple and client-friendly — avoid excessive technical jargon.
"""

# ── Node 1 — Property Issue Summary ───────────────────────────────────────────

PROPERTY_SUMMARY_PROMPT = """\
Based on the following excerpts from a property inspection report and thermal imaging data,
write a concise Property Issue Summary (200–300 words).

The summary should cover:
1. Property type, location, and general condition overview.
2. All impacted areas/rooms identified (e.g., Hall, Bedroom, Kitchen, Master Bedroom,
   Parking Area, Common Bathroom — use exactly the rooms mentioned in the context).
3. The primary defects observed: types of dampness, seepage, tile hollowness, cracks,
   plumbing issues — be specific about each area affected.
4. Overall urgency assessment (immediate / routine / monitoring).

Write this as a professional overview that a property owner can understand at a glance.

SOURCE CONTEXT (from inspection report and thermal data):
{context}

Property Issue Summary:"""

# ── Node 2 — Area-wise Observations ───────────────────────────────────────────

AREA_OBSERVATIONS_PROMPT = """\
Based on the inspection context below, write detailed observations for the area: {area}

Instructions:
- Extract ALL specific defects mentioned for this area.
- For each defect, state:
    a) Exact location (e.g., "skirting level of master bedroom", "tile joints of common bathroom")
    b) Type of defect (dampness, seepage, hollowness, crack, efflorescence, spalling, etc.)
    c) Whether it is a negative-side symptom (damage visible) or positive-side source (cause area)
    d) Any thermal reading corroboration (temperature values, hotspot/coldspot data)
- If multiple impacted areas within this zone are described, list each separately.
- For bathrooms: mention tile joint condition, nahani trap, plumbing, hollowness.
- For terrace: mention screed condition, slope, drainage, vegetation, IPS surface.
- For external walls: mention crack width, paint condition, moisture ingress.

If no information about {area} is found in the context, write exactly:
"Not Available — no observations recorded for this area."

SOURCE CONTEXT:
{context}

Observations for {area}:"""

# ── Node 3 — Probable Root Cause ──────────────────────────────────────────────

ROOT_CAUSE_PROMPT = """\
Based on the observations and context below, identify the probable root causes for defects
in the area: {area}

For each root cause:
1. State the specific cause clearly (e.g., "gaps in tile joints allowing moisture ingress",
   "failed waterproofing membrane on terrace screed", "hairline cracks in external plaster
   allowing water penetration").
2. Explain the mechanism briefly (1–2 sentences): how does this cause lead to the symptoms?
3. If the context mentions "positive side" source areas, use those as root causes.

Common root causes for this type of inspection:
- Tile joint gaps / grout failure → capillary action moisture rise
- IPS screed cracks → water channeling through slab
- External wall cracks → rain water ingress to interior
- Concealed plumbing leakage → continuous dampness
- Failed waterproofing → seepage and efflorescence
- Vegetation on terrace → root damage to waterproofing

If insufficient information is available, write:
"Not Available — root cause cannot be determined from available data."

SOURCE CONTEXT:
{context}

Probable Root Causes for {area}:"""

# ── Node 4 — Severity Assessment ──────────────────────────────────────────────

SEVERITY_PROMPT = """\
Based on the inspection context below, assess the severity of defects in the area: {area}

Severity levels:
- Critical : Structural risk, active water ingress into structural elements, immediate
             safety hazard, or risk of collapse/failure.
- High     : Significant defect worsening quickly; structural impact possible within
             1–2 years if not addressed; major moisture damage ongoing.
- Medium   : Moderate defect with functional and aesthetic impact; moisture causing
             paint/plaster damage but not yet structural; needs attention in 6–12 months.
- Low      : Minor cosmetic defect; no immediate structural risk; can be monitored.

Scoring criteria:
- Presence of live leakage or continuous seepage → High or Critical
- Tile hollowness + moisture dampness → Medium to High
- External wall cracks (>2mm) → High
- Hairline cracks only → Medium
- Terrace screed cracks + hollow sound → High
- Pure cosmetic paint spalling → Low to Medium

Format your response EXACTLY as:
Severity: <Critical / High / Medium / Low>
Reasoning: <1–2 sentence justification citing specific defects from context>

SOURCE CONTEXT:
{context}

Severity Assessment for {area}:"""

# ── Node 5 — Recommended Actions ──────────────────────────────────────────────

RECOMMENDED_ACTIONS_PROMPT = """\
Based on the defects and context below, provide specific recommended remediation actions
for the area: {area}

For each recommendation:
1. State the repair action clearly and specifically.
2. Specify materials/products where mentioned in context (e.g., Dr. Fixit URP, RTM Grout,
   Dr. Fixit Pidicrete URP, Dr. Fixit Lw+, Dr. Fixit HB).
3. Give the step-by-step sequence if multiple steps are required.
4. Mention curing times where relevant.

Standard treatments mentioned in UrbanRoof reports:
- Bathroom/Balcony grouting: V-cut joints → polymer modified mortar (Dr. Fixit URP) →
  RTM grout fill → 24-48 hour air cure
- Plaster repair: chip off → bonding coat (Dr. Fixit Pidicrete URP 1:1 with cement) →
  20-25mm sand-faced cement plaster (1:4 CM) → waterproofing compound (Dr. Fixit Lw+)
- RCC treatment: V-groove crack opening → heavy duty polymer mortar → Dr. Fixit HB for spalling
- Terrace waterproofing: remove existing screed → new IPS with waterproofing membrane
- External wall: crack filling → waterproof coating application

If insufficient information is available, write:
"Not Available — unable to determine remediation from available data."

SOURCE CONTEXT:
{context}

Recommended Actions for {area}:"""

# ── Node 6 — Additional Notes ─────────────────────────────────────────────────

ADDITIONAL_NOTES_PROMPT = """\
Based on the full inspection context below, write an Additional Notes section.

Cover these points (use what the context supports; omit what's not mentioned):
1. Inspection limitations: what was not inspected (concealed plumbing, behind walls, etc.)
2. Follow-up actions required: further tests (water ponding test, pull-off test, etc.)
3. Important precautions during repair (weather, curing conditions, safety)
4. Structural cracks warning: if any structural cracks were noted, state that a Registered
   Structural Engineer must be consulted immediately.
5. Seasonal consideration: if leakage is monsoon-specific, note that.
6. Warranty/guarantee recommendation for repair materials.

Keep to 150–250 words. Write in clear, professional language.

SOURCE CONTEXT:
{context}

Additional Notes:"""

# ── Node 7 — Missing / Unclear Information ────────────────────────────────────

MISSING_INFO_PROMPT = """\
Review the inspection context below and identify any information that is:
1. Mentioned but unclear or incomplete (e.g., "not sure", areas marked as uncertain).
2. Necessary for a complete diagnosis but absent from both source documents.
3. Conflicting between the inspection report and thermal report.
4. Areas where only partial observations are available.
5. Any measurements or readings that could not be confirmed.

List each item as a bullet point in this format:
- [Area/Topic]: Description of what is missing or unclear.

If all required information is present and consistent, write:
"No significant information gaps identified."

SOURCE CONTEXT (Inspection Report + Thermal Data):
{context}

Missing or Unclear Information:"""
