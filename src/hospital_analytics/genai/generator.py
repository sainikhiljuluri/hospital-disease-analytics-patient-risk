"""
MODULE 5, part 4: producing the report.

THE PIPELINE
------------
    patient row
      -> grounding.build_grounding()   assemble ONLY verified facts
      -> prompts.build_prompt()        wrap them in the safety rules
      -> providers.generate()          send to an LLM (or skip)
      -> validate_output()             check the rules were followed
      -> report

The last step is the one most projects leave out. A safety rule in a prompt is a
REQUEST, not a guarantee — language models do not always comply. Checking the
output afterwards is what turns a request into something you can rely on, and it
is the difference between "I told it not to" and "I verified it didn't".

WHEN THERE IS NO LANGUAGE MODEL
-------------------------------
`template_report()` builds the same structure from the same facts using plain
Python. It needs no API key, no internet and no downloads, and it is never
wrong, because there is nothing in it that could hallucinate.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from . import grounding as G
from . import prompts as P
from . import providers as V

# Words that would signal a broken safety rule. Not exhaustive — a heuristic
# check that catches the obvious failures and flags them for a human.
CAUSAL_WORDS = [
    r"\bcauses\b", r"\bcaused by\b", r"\bcausing\b", r"\bwill cause\b",
    r"\bleads to\b", r"\bresults in\b", r"\bdue to the\b",
]
PRESCRIPTION_WORDS = [
    r"\bmg\b", r"\bdose\b", r"\bdosage\b", r"\bprescribe\b", r"\bprescription\b",
    r"\bmetformin\b", r"\binsulin unit", r"\btitrate\b", r"\bmilligram",
]
CERTAINTY_WORDS = [
    r"\bwill definitely\b", r"\bcertain(ly)? to\b", r"\bguarantee",
    r"\bwill be readmitted\b", r"\bwill return to hospital\b",
]
# Facts this dataset does not contain. If they appear, something was invented.
FABRICATION_WORDS = [
    r"\bblood pressure\b", r"\bBMI\b", r"\bcholesterol\b", r"\bmmHg\b",
    r"\bsmoker\b", r"\bsmoking\b", r"\bbody mass\b",
]


# Lines that DECLARE something is unavailable are not fabrications - they are
# the opposite. A report saying "this dataset does not record blood pressure" is
# behaving correctly, and an earlier version of this checker failed it. Worth
# knowing: a validator that cries wolf gets switched off.
DISCLAIMER_MARKERS = (
    # declaring a field is unavailable
    "does not record", "not recorded", "no vital", "nothing above",
    "does not contain", "not available in this dataset", "were not measured",
    # DENYING causation - the opposite of a causal claim, and the second time
    # this checker flagged its own safety language. Any phrase that exists to
    # rule something out has to be exempt from the rule it names.
    "not causes", "not cause", "none is shown to cause", "does not establish",
    "not shown to cause", "associations, not", "association, not",
    "would not reduce", "does not follow", "not establish",
)


def _claim_text(text: str) -> str:
    """The text minus lines that merely declare what is unavailable."""
    kept = [
        line for line in text.lower().splitlines()
        if not any(marker in line for marker in DISCLAIMER_MARKERS)
    ]
    return "\n".join(kept)


# Which rules apply to which document. A patient letter is DELIBERATELY free of
# statistics, so requiring it to "state statistical uncertainty" would fail a
# correctly-written note. A cohort brief is about a service, not a person, so it
# needs limitations rather than individual clinician review.
#
# The four content prohibitions apply to everything. Only the positive
# requirements vary - a rule that fires on correct output teaches people to
# ignore the checker.
REQUIRED_BY_TEMPLATE = {
    "clinical_report": ["Requires clinician review", "States uncertainty"],
    "patient_letter": ["Refers to the care team"],
    "cohort_brief": ["States limitations"],
}


def validate_output(text: str, grounded: dict[str, Any],
                    template: str = "clinical_report") -> dict[str, Any]:
    """Check the generated text against the safety rules.

    Returns a pass/fail per rule plus the phrases that triggered any failure, so
    the dashboard can show exactly what went wrong rather than a bare verdict.
    """
    lowered = text.lower()
    claims = _claim_text(text)

    def hits(patterns: list[str], haystack: str) -> list[str]:
        found: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, haystack):
                found.append(match.group(0))
        return sorted(set(found))

    causal = hits(CAUSAL_WORDS, claims)
    prescriptive = hits(PRESCRIPTION_WORDS, claims)
    overconfident = hits(CERTAINTY_WORDS, claims)
    # Fabrication is checked ONLY outside disclaimer lines - see above.
    fabricated = hits(FABRICATION_WORDS, claims)

    has_review = any(
        phrase in lowered
        for phrase in ("clinician", "clinical review", "medical professional",
                       "healthcare professional", "qualified")
    )
    has_care_team = any(
        phrase in lowered
        for phrase in ("care team", "nurse", "doctor", "ask your")
    )
    has_uncertainty = any(
        phrase in lowered
        for phrase in ("estimate", "statistical", "not a prediction", "uncertain",
                       "associated", "likelihood", "not certain", "aid")
    )
    has_limitations = any(
        phrase in lowered
        for phrase in ("limitation", "does not tell", "historical", "not causes",
                       "association", "associated")
    )

    # Content prohibitions - apply to every document type.
    checks = [
        {"rule": "No causal claims", "passed": not causal, "found": causal},
        {"rule": "No prescribing", "passed": not prescriptive, "found": prescriptive},
        {"rule": "No overconfidence", "passed": not overconfident, "found": overconfident},
        {"rule": "No invented facts", "passed": not fabricated, "found": fabricated},
    ]

    # Positive requirements - vary by document type.
    available = {
        "Requires clinician review": has_review,
        "States uncertainty": has_uncertainty,
        "Refers to the care team": has_care_team,
        "States limitations": has_limitations,
    }
    for rule in REQUIRED_BY_TEMPLATE.get(template, REQUIRED_BY_TEMPLATE["clinical_report"]):
        checks.append({"rule": rule, "passed": available.get(rule, False), "found": []})

    failed = [c for c in checks if not c["passed"]]
    return {
        "checks": pd.DataFrame(checks),
        "passed": len(failed) == 0,
        "n_failed": len(failed),
        "failures": failed,
    }


# ---------------------------------------------------------------------------
def template_report(grounded: dict[str, Any], template: str = "clinical_report") -> str:
    """Build the report in Python, with no language model involved.

    Uses the identical facts the LLM would receive. The prose is stiffer; the
    content is the same, and it cannot hallucinate.
    """
    if template == "patient_letter":
        return _template_patient_letter(grounded)
    if template == "cohort_brief":
        return _template_cohort_brief(grounded)

    patient = grounded["patient"]
    prediction = grounded.get("prediction")
    drivers = grounded.get("drivers", [])
    population = {f["factor"]: f for f in grounded.get("population", [])}

    lines: list[str] = []
    lines.append(f"## Readmission risk summary — {grounded['pseudonym']}")
    lines.append("")

    # --- Summary ----------------------------------------------------------
    lines.append("**Summary**")
    lines.append("")
    if prediction:
        band = prediction["risk_band"]
        raising = [d for d in drivers if d["direction"] == "increases risk"][:3]
        named = ", ".join(d["feature"] for d in raising) if raising else "several recorded factors"
        lines.append(
            f"This admission falls in the **{band}** band for readmission within "
            f"30 days of discharge. The factors most associated with that "
            f"assessment are {named}. This is a statistical estimate produced by "
            f"a {prediction['model']} model (ROC-AUC {prediction['model_roc_auc']}), "
            f"not a prediction of what will happen to this individual."
        )
    else:
        lines.append(
            "No trained model is available, so this summary reports the recorded "
            "facts and the measured population statistics only. Train a model "
            "with `python scripts/04_train_model.py` for a risk band."
        )
    lines.append("")

    # --- Contributing factors --------------------------------------------
    lines.append("**Contributing factors**")
    lines.append("")
    if drivers:
        for driver in drivers[:6]:
            verb = "associated with higher risk" if driver["direction"] == "increases risk" \
                else "associated with lower risk"
            line = f"- **{driver['feature']}** (recorded value: {driver['value']}) — {verb}."
            # Look up the population statistic by an EXPLICIT map, not by fuzzy
            # word matching. An earlier version matched on shared words and
            # handed the same statistic to two different drivers, which is the
            # kind of error nobody notices because the output still reads well.
            match = population.get(FEATURE_TO_RISK_FACTOR.get(driver["feature"], ""))
            if match:
                line += (
                    f" In this cohort, {match['rate_with']}% of patients with this "
                    f"factor were readmitted against {match['rate_without']}% "
                    f"without it ({match['lift']}x)."
                )
            lines.append(line)
    else:
        for fact in grounded.get("population", [])[:5]:
            lines.append(f"- {fact['statement']}.")
    lines.append("")

    # --- Recorded facts ---------------------------------------------------
    lines.append("**Recorded for this admission**")
    lines.append("")
    for key, value in patient.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    # --- Suggested steps --------------------------------------------------
    lines.append("**Suggested discharge steps**")
    lines.append("")
    for step in _suggested_steps(drivers, patient, prediction):
        lines.append(f"- {step}")
    lines.append("")

    # --- Limitations ------------------------------------------------------
    lines.append("**Limitations**")
    lines.append("")
    lines.append(
        "- The factors above are **associated** with readmission; none is shown "
        "to cause it. A longer stay is linked to readmission because patients "
        "who are more unwell stay longer — discharging earlier would not reduce "
        "risk."
    )
    lines.append(
        f"- This dataset does not record: {', '.join(grounded['not_recorded'])}. "
        "Nothing above should be read as commenting on them."
    )
    lines.append(
        f"- The underlying data covers {grounded['cohort']['cohort']}. It is "
        "historical and may not reflect current practice."
    )
    if prediction:
        lines.append(
            f"- The model ranks risk reliably but its scores are not calibrated "
            f"probabilities. Read the band, not the number."
        )
    lines.append(
        "- **A qualified clinician must review this assessment before any action "
        "is taken.** It is a prioritisation aid, not a clinical decision."
    )
    lines.append("")
    lines.append("*Generated without a language model, from measured statistics only.*")
    return "\n".join(lines)


# Maps a model feature (as humanised in ml/predict.py) to the risk factor label
# used by metrics.risk_factor_summary(). Only genuine one-to-one pairs appear
# here. A driver with no entry simply gets no population statistic quoted - a
# missing number is honest, a wrong one is not.
FEATURE_TO_RISK_FACTOR = {
    "prior inpatient admissions": "Prior inpatient admission",
    "has at least one prior admission": "Prior inpatient admission",
    "prior emergency visits": "Prior emergency visit",
    "has at least one prior emergency visit": "Prior emergency visit",
    "prior outpatient visits": "Prior outpatient visit",
    "which recorded visit this is": "Repeat patient (not first visit)",
    "length of this stay": "Long stay (> 7 days)",
    "stay longer than 7 days": "Long stay (> 7 days)",
    "diagnoses recorded": "Complex case (9+ diagnoses)",
    "9 or more diagnoses": "Complex case (9+ diagnoses)",
    "medications during this stay": "Polypharmacy (> 20 meds)",
    "more than 20 medications": "Polypharmacy (> 20 meds)",
    "age": "Age 65 or over",
    "aged 65 or over": "Age 65 or over",
    "HbA1c was measured": "HbA1c tested during stay",
    "HbA1c above 7": "HbA1c high (> 7)",
    "serum glucose above 200": "Serum glucose high (> 200)",
    "diabetes regimen was changed": "Medication regimen changed",
    "on diabetes medication": "Diabetes medication prescribed",
    "admitted as an emergency": "Emergency admission",
}


def _suggested_steps(
    drivers: list[dict[str, Any]], patient: dict[str, Any],
    prediction: dict[str, Any] | None,
) -> list[str]:
    """Process suggestions matched to the factors present. No clinical advice."""
    steps: list[str] = []
    driver_text = " ".join(str(d["feature"]).lower() for d in drivers)

    if "prior" in driver_text or any("prior" in k for k in patient):
        steps.append(
            "Consider enrolment in the care-transitions programme, with a "
            "telephone follow-up within 48 hours of discharge."
        )
    if "medication" in driver_text or int(patient.get("medications during stay", 0) or 0) > 15:
        steps.append(
            "Request a pharmacist-led medication review before discharge to "
            "check for interactions and simplify the regimen where possible."
        )
    if "hba1c" in driver_text or patient.get("HbA1c") == "not measured during this stay":
        steps.append(
            "HbA1c was not measured during this admission. Consider whether a "
            "review of longer-term glycaemic control should be arranged."
        )
    if "stay" in driver_text or "diagnos" in driver_text:
        steps.append(
            "Confirm one named clinician owns the overall care plan, so no "
            "condition is managed in isolation after discharge."
        )
    if prediction and prediction.get("recommended_action"):
        steps.append(prediction["recommended_action"])

    steps.append(
        "Confirm the patient and any carer understand the discharge plan and "
        "know who to contact if their condition changes."
    )
    return steps[:5]


def _template_patient_letter(grounded: dict[str, Any]) -> str:
    patient = grounded["patient"]
    lines = [
        "## Staying well after you go home",
        "",
        "Thank you for letting us care for you during your stay. Here are a few "
        "things that can help you stay well now that you are going home.",
        "",
    ]
    tips = [
        "**Keep your follow-up appointment.** Talking to your care team soon "
        "after you go home is one of the most helpful things you can do.",
        "**Bring all your medicines to that appointment**, including anything "
        "you buy yourself, so the team can check them over together.",
    ]
    if int(patient.get("medications during stay", 0) or 0) > 15:
        tips.append(
            "**Ask for a medicines review.** You are taking several different "
            "medicines, and a pharmacist can go through them with you."
        )
    if patient.get("HbA1c") == "not measured during this stay":
        tips.append(
            "**Ask about a blood sugar check.** A longer-term blood sugar test "
            "was not done during this stay, and it may be worth arranging."
        )
    tips.append(
        "**Contact your care team if something changes** or does not feel right. "
        "It is always better to ask early."
    )
    lines.extend(f"- {tip}" for tip in tips[:4])
    lines.extend([
        "",
        "If anything here is unclear, please ask your nurse or doctor before you "
        "leave. There is no such thing as a silly question.",
        "",
        "*Please discuss anything in this note with your care team.*",
    ])
    return "\n".join(lines)


def _template_cohort_brief(grounded: dict[str, Any]) -> str:
    cohort = grounded["cohort"]
    population = grounded.get("population", [])
    lines = [
        "## Readmission patterns — service briefing",
        "",
        "**What the data shows**",
        "",
        f"Across {cohort['encounters analysed']} encounters covering "
        f"{cohort['distinct people']} distinct people, the 30-day readmission "
        f"rate is {cohort['overall 30-day readmission rate']}, with an average "
        f"stay of {cohort['average length of stay']}.",
        "",
    ]
    if population:
        lines.append("The factors most strongly associated with readmission:")
        lines.append("")
        for fact in population[:5]:
            lines.append(f"- {fact['statement']}.")
        lines.append("")
    lines.extend([
        "**Where to focus**",
        "",
        "- Prior hospital utilisation is the strongest signal available. "
        "Identifying frequent attenders at admission, rather than at discharge, "
        "would give the care-transitions team more time to act.",
        "- Follow-up capacity is the binding constraint on any risk model. The "
        "useful question is not 'who is at risk' but 'who are the top N we can "
        "actually follow up'.",
        "- Pharmacist review capacity is worth examining, given how strongly "
        "medication count tracks with readmission.",
        "",
        "**What this does not tell us**",
        "",
        f"- The data covers {cohort['cohort']}. It is historical and does not "
        "reflect current practice, current case mix, or any other country.",
        "- These are associations, not causes. None of the above establishes "
        "that changing a factor would change the outcome.",
        "- Service changes should be evaluated prospectively rather than assumed "
        "to work because a correlation exists.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def generate_report(
    row: pd.Series,
    df: pd.DataFrame,
    prediction: dict[str, Any] | None = None,
    template: str = "clinical_report",
    provider: str = "auto",
) -> dict[str, Any]:
    """The full Module 5 pipeline: facts -> prompt -> model -> validation.

    Always returns a report. If no provider is configured, or a call fails, the
    template generator produces the same content without a language model.
    """
    grounded = G.build_grounding(row, df, prediction)
    facts_text = G.grounding_to_text(grounded)
    prompt = P.build_prompt(facts_text, template=template)

    result = V.generate(prompt["system"], prompt["user"], provider=provider)

    if result["text"]:
        text = result["text"]
        source = result["provider"]
    else:
        text = template_report(grounded, template=template)
        source = "template"

    validation = validate_output(text, grounded, template=template)

    return {
        "text": text,
        "source": source,
        "used_fallback": result["used_fallback"],
        "provider_error": result["error"],
        "grounding": grounded,
        "facts_text": facts_text,
        "prompt": prompt,
        "validation": validation,
        "template": template,
    }
