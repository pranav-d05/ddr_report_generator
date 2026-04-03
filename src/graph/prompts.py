"""
LLM Prompt Templates for every DDR section node.

PROMPT ENGINEERING TECHNIQUES USED:
  - Role + task scoping per node (not one generic system prompt)
  - Output anchoring with exact format constraints
  - Chain-of-thought for reasoning-heavy nodes (root cause, severity)
  - Negative prompting (explicit "do not" guards against hallucination)
  - Fallback instruction with exact string (prevents creative "Not Available" variants)
  - Context labelling (tagged sections so LLM knows what each block is)
  - Constraint-first ordering for tight token budgets (severity, missing info)
  - Generalisation guard on Node 5 (products from context, not hardcoded)
  - Absence-detection framing on Node 7 (gaps found by reasoning, not keyword search)
"""

from __future__ import annotations


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BASE SYSTEM PROMPT  (shared — kept deliberately lean)
#
# TECHNIQUE: Minimal base system prompt
# Reason: A bloated system prompt fights with every node's specific task.
# Keep the base to identity + universal rules only.
# Node-specific behaviour goes in the user prompt for that node.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """\
You are a senior civil engineer and property diagnostics specialist producing sections of a \
Detailed Diagnosis Report (DDR) for a professional inspection firm.

Universal rules — apply to every response:
- Use ONLY information present in the SOURCE CONTEXT provided. Do not invent facts, \
measurements, temperatures, or product names.
- "Negative side" means the location where damage is visible. \
"Positive side" means the source area causing that damage. Use both when available.
- If the context genuinely lacks enough information for a section, output the exact fallback \
string specified in the task instructions — do not paraphrase it.
- Write for a property owner, not a technical audience. Plain English, no unnecessary jargon.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 1 — Property Issue Summary
#
# TECHNIQUES:
#   1. Structured output scaffolding  — numbered sections tell the LLM exactly
#      what to produce, in what order, preventing free-form rambling.
#   2. Persona anchoring in user prompt — "a property owner reading this for
#      the first time" grounds the tone without relying solely on system prompt.
#   3. Word-count as a hard constraint — prevents both under-generation
#      (vague one-liners) and over-generation (verbose padding).
#   4. Context labelling — [INSPECTION DATA] / [THERMAL DATA] tags help the
#      LLM mentally separate the two source documents before synthesising.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROPERTY_SUMMARY_PROMPT = """\
TASK: Write the Property Issue Summary section of a DDR report.
TARGET READER: A property owner reading this for the first time — no technical background.
LENGTH: 200–300 words. Do not exceed 300 words.

Cover these four points in order:

1. PROPERTY OVERVIEW
   State the property type and general condition based on the context. \
Use room names exactly as they appear in the source (e.g., Hall, Master Bedroom, \
Common Bathroom, Parking Area — do not rename them).

2. IMPACTED AREAS
   List every room or zone where a defect was found. Be specific — "Master Bedroom \
skirting level" is better than just "Master Bedroom".

3. PRIMARY DEFECTS
   For each impacted area, name the defect type (dampness, seepage, tile hollowness, \
hairline crack, spalling, etc.). Reference thermal corroboration where the context \
provides temperature readings.

4. URGENCY ASSESSMENT
   Conclude with one sentence: overall urgency level — Immediate, Routine, or Monitoring \
— with a one-line reason.

SOURCE CONTEXT:
{context}

Property Issue Summary:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 2 — Area-wise Observations
#
# TECHNIQUES:
#   1. Task decomposition — break the observation into 4 sub-attributes
#      (location, defect type, negative/positive side, thermal) so the LLM
#      doesn't skip any dimension.
#   2. Area-specific sub-instructions — bathroom/terrace/wall checklists act
#      as few-shot hints, guiding coverage without restricting output.
#   3. Explicit fallback string — exact text to output when context is absent,
#      prevents the LLM inventing plausible-sounding observations.
#   4. Negative example guard — "Do not infer" stops hallucination when
#      context is thin but not completely empty.
#   5. Labelled context blocks — [Inspection Observations] / [Thermal Readings]
#      separation prevents the LLM from mixing up source types.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AREA_OBSERVATIONS_PROMPT = """\
TASK: Write the Area-wise Observations entry for: {area}

For every defect found in the context for this area, document all four attributes:
  a) LOCATION   — exact spot (e.g., "tile joints at skirting level", "ceiling above nahani trap")
  b) DEFECT TYPE — dampness / seepage / hollowness / crack / efflorescence / spalling / \
vegetation / other
  c) SIDE       — Negative side (damage symptom visible here) OR Positive side \
(moisture source originating here)
  d) THERMAL    — if thermal data shows a temperature reading or hotspot/coldspot at this \
location, include it; otherwise omit this attribute entirely

Area-specific checklist (include only what appears in the context):
  - Bathroom / Wet Areas : tile joint condition, nahani trap area, hollowness on tap, \
plumbing lines, ceiling/skirting dampness
  - Balcony              : tile joints, skirting level, external wall junction
  - Terrace / Roof       : IPS screed condition, hollow patches, slope towards drain, \
vegetation, parapet junction
  - External Wall        : crack width and orientation, paint condition, \
moisture staining, chajja/duct junction
  - Plaster / Substrate  : hollow sound areas, separation from substrate, \
crack pattern, bond failure
  - Structural Elements  : crack width, exposed reinforcement, spalling depth, \
corrosion staining

Do not infer or assume defects not described in the source context.

FALLBACK — if the context contains no observations for {area}, output this exact line:
Not Available — no observations recorded for this area.

SOURCE CONTEXT:
{context}

Observations for {area}:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 3 — Probable Root Cause
#
# TECHNIQUES:
#   1. Chain-of-thought (CoT) prompting — explicitly ask the LLM to reason
#      from symptom → mechanism → cause before stating the conclusion.
#      This reduces the chance of superficial or generic root causes.
#   2. Positive-side anchoring — instructs the LLM to prioritise the
#      document's own "Positive side" labels as ground-truth root causes.
#   3. Reference examples as few-shot hints — the example cause→mechanism
#      pairs teach the required level of specificity without over-constraining.
#   4. Generalisation guard — examples are framed as "patterns to reason
#      from" not as a fixed list to copy, so it works on any property.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT_CAUSE_PROMPT = """\
TASK: Identify the probable root causes for defects in: {area}

REASONING APPROACH — work through these steps before writing your answer:
  Step 1: Identify the symptoms described in the context (what damage is visible?).
  Step 2: Identify any "Positive side" source areas explicitly mentioned — these are \
the confirmed cause locations; treat them as primary root causes.
  Step 3: For remaining symptoms with no explicit source, reason about the most \
probable cause using construction knowledge.
  Step 4: Describe the cause-to-damage mechanism in 1–2 sentences.

FORMAT — for each root cause write:
  Cause: <specific cause statement>
  Mechanism: <how this cause produces the observed damage, 1–2 sentences>

REFERENCE PATTERNS (use these to calibrate specificity, not to copy blindly):
  - Open tile joints / grout failure → capillary moisture rises through joint gaps
  - IPS screed cracks → rainwater channels through slab to ceiling below
  - Hairline cracks in external plaster → wind-driven rain penetrates to inner wall
  - Concealed plumbing joint failure → slow continuous seepage with no surface crack
  - Failed waterproofing membrane → hydrostatic pressure pushes water through slab
  - Vegetation root growth → physical rupture of waterproofing layer

FALLBACK — if the context contains no defect information for {area}, output this exact line:
Not Available — root cause cannot be determined from available data.

SOURCE CONTEXT:
{context}

Probable Root Causes for {area}:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 4 — Severity Assessment
#
# TECHNIQUES:
#   1. Constraint-first ordering — the format requirement appears BEFORE
#      the criteria, not after. This is critical for tight token budgets
#      (120 tokens). The LLM outputs the answer first, not a preamble.
#   2. Output anchoring with a filled example — showing the exact format
#      with a placeholder example is far more reliable than describing it.
#   3. Hard stop instruction — "Output nothing else" eliminates the LLM
#      habit of restating the criteria or adding a closing remark.
#   4. Criteria as decision rules not prose — IF/THEN style scoring table
#      is faster to process and produces consistent label selection.
#   5. Trim verbose definitions — removed the long prose definitions since
#      they waste tokens. The scoring table already implies the definitions.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEVERITY_PROMPT = """\
TASK: Severity assessment for: {area}

OUTPUT FORMAT — respond in exactly these two lines, nothing before, nothing after:
Severity: <Critical | High | Medium | Low>
Reasoning: <one sentence citing the specific defect(s) from the context>

SCORING RULES — pick the highest level that applies:
  Critical → structural risk, active water into RCC/steel, collapse/failure risk
  High     → live leakage or continuous seepage; external wall cracks >2 mm; \
terrace screed cracked AND hollow
  Medium   → tile hollowness with dampness; hairline cracks; paint/plaster damage \
from moisture (no structural impact yet)
  Low      → cosmetic only; no moisture ingress; stable and not worsening

If the context contains no defect information for {area}, output:
Severity: Not Available
Reasoning: No defect data found in source documents for this area.

SOURCE CONTEXT:
{context}

Severity Assessment for {area}:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 5 — Recommended Actions
#
# TECHNIQUES:
#   1. Generalisation guard (key fix) — "use products named in the context"
#      replaces hardcoded Dr. Fixit names. The standard treatments are now
#      labelled as FALLBACK REFERENCE, only used when the document is silent
#      on products. This makes the prompt work on any inspection report.
#   2. Step-sequence instruction — numbered steps with curing time anchor
#      produce actionable outputs not vague advice.
#   3. Priority ordering — ask for most critical action first so truncation
#      (if max_tokens is hit) loses less-important steps, not the primary fix.
#   4. Negative prompting — "do not recommend actions for defects not present"
#      prevents the LLM from copying the fallback templates when the area
#      has no matching defects.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDED_ACTIONS_PROMPT = """\
TASK: Provide remediation actions for defects identified in: {area}

INSTRUCTIONS:
  1. List actions in order of priority — most critical defect first.
  2. For each action:
       - State what to do (specific repair verb: chip off / V-cut / apply / fill / re-lay)
       - Specify the material or product — use the name as it appears in the source \
context. If no product is named, use a generic description (e.g., "polymer-modified \
waterproofing compound").
       - Give the sequence of steps if more than one step is needed.
       - Include curing or drying time if the context mentions it.
  3. Do not recommend actions for defect types not present in the context for this area.

FALLBACK REFERENCE — use ONLY when the source context does not specify a repair method:
  Tile joint repair    : V-cut along joint → clean debris → fill with polymer-modified \
mortar → finish with matching grout → cure 24–48 hrs
  Plaster repair       : chip off loose plaster → apply bonding coat (cement slurry + \
bonding agent) → 20–25 mm sand-faced plaster (1:4 CM) → waterproofing coat → cure 7 days
  RCC crack repair     : open crack to V-groove → fill with structural polymer mortar → \
anti-corrosion coat on exposed steel → cure per product spec
  Terrace waterproofing: remove existing screed → apply waterproofing membrane → \
new IPS screed with slope towards drain
  External wall crack  : widen crack → fill with flexible sealant → waterproof paint coat

FALLBACK — if context contains no defect information for {area}, output this exact line:
Not Available — unable to determine remediation from available data.

SOURCE CONTEXT:
{context}

Recommended Actions for {area}:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 6 — Additional Notes
#
# TECHNIQUES:
#   1. Conditional inclusion instruction — "include only if context supports
#      it" prevents the LLM from padding with generic boilerplate when
#      the inspection doc is silent on that point.
#   2. Hard word count with both floor and ceiling — "100–200 words" stops
#      both the terse 2-line output and the 400-word essay.
#   3. Priority ordering — structural engineer warning first so it is never
#      lost to truncation.
#   4. Tone instruction at the end — placing it last acts as a final reminder
#      just before the LLM starts generating.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADDITIONAL_NOTES_PROMPT = """\
TASK: Write the Additional Notes section of the DDR.
LENGTH: 100–200 words. Include only points supported by the source context.

Cover in this priority order (skip any point the context does not support):

1. STRUCTURAL WARNING — if any structural crack, RCC damage, or exposed reinforcement \
was noted, include this sentence first:
   "A Registered Structural Engineer should be consulted immediately to assess the \
structural integrity before repair work begins."

2. INSPECTION SCOPE LIMITATIONS — what could not be assessed (e.g., areas behind walls, \
concealed plumbing runs, areas not accessible on the day of inspection).

3. RECOMMENDED FOLLOW-UP TESTS — tests that would confirm diagnosis \
(e.g., water ponding test for terrace, pull-off test for plaster adhesion, \
endoscopy for concealed pipe tracing). Include only tests relevant to defects found.

4. REPAIR PRECAUTIONS — weather conditions, curing environment, safety measures \
mentioned or implied by the defects found.

5. SEASONAL NOTE — if any leakage or defect is described as monsoon-related or \
weather-dependent, note that re-inspection after monsoon is advised.

Write in professional but accessible language. Do not add generic disclaimers not \
supported by the context.

SOURCE CONTEXT:
{context}

Additional Notes:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 7 — Missing / Unclear Information
#
# TECHNIQUES:
#   1. Absence-detection framing (key fix) — the old prompt asked the LLM
#      to look for words like "unclear" or "not sure" in the text. That fails
#      because inspection reports rarely use those words. The new approach
#      asks the LLM to reason about what a complete DDR REQUIRES and then
#      identify what is absent from the context. Gaps are found by logical
#      inference, not keyword search.
#   2. Required-fields checklist — gives the LLM an explicit definition of
#      "complete DDR" so it has a fixed target to compare against.
#   3. Conflict detection instruction — explicit instruction to flag when
#      inspection text and thermal data say different things.
#   4. Structured bullet format with field names — consistent output format
#      makes downstream parsing reliable.
#   5. Positive fallback (not "nothing found") — if everything is present,
#      a specific sentence confirms that rather than silence.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MISSING_INFO_PROMPT = """\
TASK: Identify information gaps in the source documents for this DDR.

APPROACH — do NOT search for the words "missing", "unclear", or "not sure" in the text.
Instead, reason by absence: compare what a complete DDR requires against what is \
actually present in the context below.

A COMPLETE DDR REQUIRES the following for each area with a reported defect:
  - Defect location (specific room and surface)
  - Defect type (dampness / crack / hollow / spalling / etc.)
  - Probable source (positive side) or at least a plausible cause
  - Severity indicator (size, extent, or degree of damage)
  - Thermal corroboration (if thermal camera was used in that area)

ALSO FLAG:
  - Conflicts: where the inspection report and thermal report describe the same area \
differently (e.g., inspection says "dry" but thermal shows hotspot).
  - Partial data: an area is mentioned but only one attribute is recorded \
(e.g., location noted but no defect type given).
  - Areas in the property that were likely inspected but have no entry at all.

OUTPUT FORMAT — one bullet per gap:
- [Area / Topic]: What is missing or conflicting and why it matters for the diagnosis.

FALLBACK — if the context provides all required information consistently, output:
No significant information gaps identified.

SOURCE CONTEXT (Inspection Report + Thermal Data):
{context}

Missing or Unclear Information:"""
