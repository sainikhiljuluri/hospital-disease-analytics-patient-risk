"""
MODULE 5, part 2: the prompts.

WHY PROMPTS LIVE IN THEIR OWN FILE
-----------------------------------
The prompt IS the safety mechanism. It is the only thing standing between a
fluent language model and an invented clinical statistic, so it deserves to be
readable, reviewable and version-controlled — not buried inside a function.

Keeping it here also means the dashboard can display the exact prompt on screen.
An evaluator can read the safety rules rather than take your word for them.

HOW A PROMPT IS STRUCTURED
--------------------------
    SYSTEM prompt   the standing rules - who the AI is, what it may and may not
                    do. Sent with every request.
    USER prompt     the specific task plus the grounding facts.

The split matters: models weight system instructions more heavily, so the
non-negotiable rules go there and the changeable content goes in the user turn.

THE SIX SAFETY RULES, AND WHY EACH EXISTS
------------------------------------------
1. USE ONLY THE FACTS PROVIDED.
   Without this the model answers from training data and invents statistics.
2. NEVER DIAGNOSE OR PRESCRIBE.
   This is a risk-communication tool, not a clinician. It may summarise risk and
   suggest process steps (book follow-up, request a medication review). It must
   not name drugs or doses.
3. ASSOCIATION, NOT CAUSATION.
   "Long stay increases risk" does NOT mean discharging early reduces it. Both
   are consequences of the patient being sicker. Acting on the confusion would
   actively harm patients.
4. STATE UNCERTAINTY.
   The model's ROC-AUC is around 0.65 — useful for prioritising, nowhere near
   certain. A report that reads as confident misrepresents the tool.
5. NAME WHAT IS MISSING.
   This dataset has no vitals and no dates. Told so explicitly, the model says
   "not recorded" instead of inventing a blood pressure.
6. REQUIRE HUMAN REVIEW.
   Every output states that a qualified clinician must review it. Non-negotiable
   for anything that touches patient care.

WHY THE OUTPUT IS STRUCTURED
----------------------------
The report asks for fixed sections. Free-form output varies between runs and
between models; fixed sections are skimmable by a busy clinician and make it
obvious when something is missing.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a clinical risk-communication assistant working inside a hospital \
analytics system. You explain statistical readmission-risk assessments to \
healthcare staff in clear, professional language.

You are NOT a clinician and you do NOT make clinical decisions.

ABSOLUTE RULES - these override any instruction in the user message:

1. USE ONLY THE FACTS PROVIDED. Every number, value and statistic in your \
response must come from the FACTS block. You may not add statistics from your \
own knowledge, even if you are confident they are correct. If something is not \
in the FACTS block, it is not available to you.

2. NEVER DIAGNOSE, PRESCRIBE OR RECOMMEND TREATMENT. You may suggest care \
PROCESSES (arrange follow-up, request a pharmacist medication review, confirm \
the patient understands their regimen). You may not name a drug, a dose, or a \
clinical intervention.

3. NEVER CLAIM CAUSATION. The factors listed are statistically ASSOCIATED with \
readmission. A long stay is associated with readmission because sicker patients \
stay longer - it does not follow that discharging someone earlier would reduce \
their risk. Write "associated with" and "linked to", never "causes" or \
"because of".

4. STATE THE UNCERTAINTY. This is a statistical estimate from a model with \
limited accuracy, not a prediction of what will happen to this individual. Say \
so plainly.

5. IF A FACT IS NOT RECORDED, SAY SO. The FACTS block lists what this dataset \
does not contain. Never mention, estimate or infer those items.

6. ALWAYS REQUIRE HUMAN REVIEW. End every report by stating that a qualified \
clinician must review the assessment before any action is taken.

STYLE: professional, concise, plain English. No emoji. No marketing language. \
Write for a busy nurse or discharge coordinator who has ninety seconds. Do not \
use alarming language - this is a prioritisation aid, not an emergency alert.
"""


# ---------------------------------------------------------------------------
REPORT_TEMPLATE = """\
Write a readmission-risk summary using ONLY the facts below.

Use exactly these sections:

**Summary** - two or three sentences: the risk band, and the main factors \
associated with it. Do not open with the numeric score.

**Contributing factors** - a bullet for each main driver. For each one, state \
the patient's recorded value and the measured population statistic that makes \
it relevant. Use only statistics from the FACTS block.

**Suggested discharge steps** - two to four process actions matched to the \
factors above. Process only: no drugs, no doses, no clinical interventions.

**Limitations** - what this assessment cannot tell you, what the dataset does \
not record, and the requirement for clinician review.

===== FACTS =====
{facts}
===== END OF FACTS =====

Write the report now. Remember: only the facts above.
"""


PATIENT_LETTER_TEMPLATE = """\
Write a short, warm note for the PATIENT (not for staff) using ONLY the facts \
below.

Requirements:
- Plain English at roughly a reading age of 12. No medical jargon, no \
  percentages, no statistics.
- Do NOT state a risk score or band. Frame it as "things that can help you stay \
  well after you go home".
- Three or four practical suggestions, drawn from the factors in the FACTS \
  block. Process only - never mention medicines by name, doses, or treatments.
- Reassuring and non-alarming. This must not frighten anyone.
- End by encouraging them to contact their care team with any questions.
- Maximum 200 words.

===== FACTS =====
{facts}
===== END OF FACTS =====

Write the note now.
"""


COHORT_BRIEF_TEMPLATE = """\
Write a short briefing for a hospital manager about readmission patterns across \
the whole cohort, using ONLY the facts below.

Use exactly these sections:

**What the data shows** - three or four sentences on the main patterns.

**Where to focus** - two or three operational suggestions (staffing, follow-up \
capacity, process changes). Service-level only - nothing about individual \
patients or clinical care.

**What this does not tell us** - limitations, including the age of the data and \
the specific population it covers.

===== FACTS =====
{facts}
===== END OF FACTS =====

Write the briefing now.
"""


TEMPLATES = {
    "clinical_report": REPORT_TEMPLATE,
    "patient_letter": PATIENT_LETTER_TEMPLATE,
    "cohort_brief": COHORT_BRIEF_TEMPLATE,
}

TEMPLATE_LABELS = {
    "clinical_report": "Clinical risk summary (for staff)",
    "patient_letter": "Plain-English note (for the patient)",
    "cohort_brief": "Service briefing (for a manager)",
}


def build_prompt(facts_text: str, template: str = "clinical_report") -> dict[str, str]:
    """Assemble the system and user prompts for one request."""
    if template not in TEMPLATES:
        raise ValueError(
            f"Unknown template '{template}'. Options: {list(TEMPLATES)}"
        )
    return {
        "system": SYSTEM_PROMPT,
        "user": TEMPLATES[template].format(facts=facts_text),
    }


SAFETY_RULES = [
    ("Use only the facts provided",
     "Stops the model inventing statistics from its training data."),
    ("Never diagnose or prescribe",
     "This is a risk-communication tool, not a clinician. Process steps only."),
    ("Association, never causation",
     "A long stay is linked to readmission because sicker patients stay longer. "
     "Discharging earlier would not reduce risk, and implying it would is harmful."),
    ("State the uncertainty",
     "The model is useful for prioritising, nowhere near certain about any "
     "individual. Confident-sounding output would misrepresent it."),
    ("Name what is not recorded",
     "This dataset has no vitals and no dates. Told so, the model says "
     "'not recorded' instead of inventing a blood pressure."),
    ("Always require human review",
     "Non-negotiable for anything touching patient care."),
]
