"""
LLM Prompt Templates for every DDR section node.

Each template uses Python str.format() placeholders: {context}
Some templates also accept {area} for area-specific sections.
"""

from __future__ import annotations

# ── System prompt (shared across all nodes) ────────────────────────────────────

SYSTEM_PROMPT = """You are an expert civil engineer and property diagnostics specialist
working for UrbanRoof Private Limited. Your task is to analyse raw inspection notes and
thermal imaging reports to produce a concise, professional Detailed Diagnosis Report (DDR).

Guidelines:
- Write in clear, technical English appropriate for a property inspection report.
- Be factual and specific — cite specific areas, materials, or observations where possible.
- If the provided context does not contain enough information for a section, write exactly:
  "Not Available — insufficient data in source documents."
- Do NOT fabricate data, temperatures, or measurements not present in the context.
- Format your output as clean prose (no markdown). Use bullet points only when listing
  multiple distinct items.
"""

# ── Node 1 — Property Issue Summary ───────────────────────────────────────────

PROPERTY_SUMMARY_PROMPT = """\
Based on the following excerpts from a property inspection report and thermal imaging report,
write a concise Property Issue Summary (150–250 words).

The summary should:
1. Identify the property type and general condition.
2. List the primary defects observed across all areas.
3. Mention the overall urgency level (e.g., immediate attention required / routine maintenance).

SOURCE CONTEXT:
{context}

Property Issue Summary:"""

# ── Node 2 — Area-wise Observations ───────────────────────────────────────────

AREA_OBSERVATIONS_PROMPT = """\
Based on the inspection excerpts below, write detailed observations for the area: {area}

Include:
- Specific defects visible (cracks, seepage, dampness, spalling, efflorescence, etc.)
- Location within the area (e.g., floor, wall junction, ceiling corner)
- Any measurements or temperatures mentioned in the context
- Whether thermal imaging corroborates the visual finding

If no information about this area is found, write exactly:
"Not Available — no observations recorded for this area."

SOURCE CONTEXT:
{context}

Observations for {area}:"""

# ── Node 3 — Probable Root Cause ──────────────────────────────────────────────

ROOT_CAUSE_PROMPT = """\
Based on the observations and context below, provide probable root causes for defects
in the area: {area}

For each root cause:
- State what the likely cause is (e.g., rising dampness, failed waterproofing membrane,
  inadequate slope, thermal bridging, construction joint failure).
- Briefly explain the mechanism (1–2 sentences).

If insufficient information is available, write exactly:
"Not Available — root cause cannot be determined from available data."

SOURCE CONTEXT:
{context}

Probable Root Causes for {area}:"""

# ── Node 4 — Severity Assessment ──────────────────────────────────────────────

SEVERITY_PROMPT = """\
Based on the inspection context below, provide a severity assessment for defects
in the area: {area}

Assign ONE severity level from: Critical / High / Medium / Low

Definitions:
- Critical : Structural risk, immediate safety hazard, or active water ingress causing
             damage to structural elements.
- High     : Significant defect that will worsen quickly without intervention; potential
             for structural impact within 1–2 years.
- Medium   : Moderate defect — aesthetic and functional issues; intervention needed within
             6–12 months.
- Low      : Minor cosmetic defect not requiring urgent action.

Format your response as:
Severity: <Level>
Reasoning: <1–2 sentence justification>

SOURCE CONTEXT:
{context}

Severity Assessment for {area}:"""

# ── Node 5 — Recommended Actions ──────────────────────────────────────────────

RECOMMENDED_ACTIONS_PROMPT = """\
Based on the defects and root causes described in the context below, provide specific
recommended remediation actions for the area: {area}

For each recommendation:
- State the treatment or repair action clearly.
- Specify the materials, methods, or products where known
  (e.g., crystalline waterproofing, polyurethane injection, IPS screed with membrane).
- Indicate the sequence if multiple steps are required.

If insufficient information is available, write exactly:
"Not Available — unable to determine remediation from available data."

SOURCE CONTEXT:
{context}

Recommended Actions for {area}:"""

# ── Node 6 — Additional Notes ─────────────────────────────────────────────────

ADDITIONAL_NOTES_PROMPT = """\
Based on the full inspection context below, write an Additional Notes section for the DDR.

This section should cover:
1. General precautions during repair works (safety, weather conditions, curing times).
2. Limitations of this inspection (e.g., concealed areas not inspected, equipment limits).
3. Follow-up inspections or tests recommended (e.g., water ponding test, pull-off test).
4. Any warranties or guarantees recommended for repair materials.

Keep the section to 100–200 words.

SOURCE CONTEXT:
{context}

Additional Notes:"""

# ── Node 7 — Missing / Unclear Information ────────────────────────────────────

MISSING_INFO_PROMPT = """\
Review the inspection context below and identify any information that is:
1. Mentioned but unclear or contradictory between the inspection report and thermal report.
2. Necessary for a complete diagnosis but absent from both source documents.
3. Areas where thermal readings conflict with visual observations.

List each item as a bullet point.
If all required information is present and consistent, write:
"No significant information gaps identified."

SOURCE CONTEXT:
{context}

Missing or Unclear Information:"""
