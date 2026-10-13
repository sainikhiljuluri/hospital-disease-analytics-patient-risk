"""
MODULE 5, part 1: assembling the FACTS an AI is allowed to use.

THE CENTRAL IDEA: GROUNDED GENERATION
--------------------------------------
There are two ways to build an AI medical report generator, and only one is
safe.

**The unsafe way.** Send the model a patient and ask it to explain the risk. It
answers from its own training — which contains a great deal of medical text —
and produces something fluent, confident and partly invented. It will cite
statistics that do not exist. It will sound exactly as certain when it is wrong
as when it is right. In a clinical setting that is dangerous, and it is the
default behaviour if you do not design against it.

**The way this project does it.** Assemble a block of verified facts first —
measured from OUR dataset, computed by OUR model — and instruct the AI to write
using ONLY those facts. Its job becomes translation, not recall: turn numbers
into clear English for a busy clinician.

This file builds that fact block. It is the safety mechanism, and it is why
Module 2 (the statistics) and Module 4 (the model) had to come first.

THE FOUR SOURCES OF GROUNDING
-----------------------------
  1. THE PATIENT      values recorded for this specific admission
  2. THE PREDICTION   the model's score, band, and its own measured accuracy
  3. THE EXPLANATION  which features drove THIS score (from Module 4)
  4. THE POPULATION   measured lift for each risk factor (from Module 2)

Number 4 is what makes the output defensible. When the report says a factor
roughly doubles readmission risk, that number was computed from this dataset by
`metrics.risk_factor_summary()`. A clinician can challenge it and be shown the
calculation.

WHAT IS DELIBERATELY EXCLUDED
-----------------------------
No patient identifiers reach the AI. `patient_id` becomes a pseudonym before the
prompt is built, because sending identifiable medical records to a third-party
API is a data-protection problem regardless of how good the model is.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from ..analysis import metrics

# Facts the model must never be asked to comment on, because this dataset does
# not record them. Without this list an LLM will confidently invent a blood
# pressure reading, because reports of this kind normally contain one.
NOT_RECORDED = [
    "blood pressure", "BMI", "weight", "cholesterol", "smoking status",
    "admission or discharge dates", "vital signs", "symptoms",
]


def pseudonymise(patient_id: Any) -> str:
    """Turn an encounter ID into a stable pseudonym.

    Same input always gives the same pseudonym, so a clinician can discuss
    "Patient A4F2C1" across a conversation, but the original ID cannot be
    recovered from it.
    """
    digest = hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()
    return f"Patient {digest[:6].upper()}"


# ---------------------------------------------------------------------------
def patient_facts(row: pd.Series) -> dict[str, Any]:
    """The recorded facts for one admission, in plain language.

    Only fields this dataset actually contains. Anything absent is simply left
    out rather than reported as null — a blank invites the model to fill it in.
    """
    def get(key: str, default: Any = None) -> Any:
        value = row.get(key, default)
        if isinstance(value, float) and np.isnan(value):
            return default
        return value

    facts: dict[str, Any] = {}

    age_band = get("age_group")
    if age_band:
        facts["age band"] = str(age_band)
    if get("gender") and str(get("gender")) != "Unknown":
        facts["gender"] = str(get("gender"))

    if get("diagnosis"):
        facts["primary diagnosis group"] = str(get("diagnosis"))
    secondary = [
        str(get(k)) for k in ("diagnosis_secondary", "diagnosis_tertiary")
        if get(k) and str(get(k)) not in ("Other", "Unknown")
    ]
    if secondary:
        facts["other diagnosis groups"] = ", ".join(secondary)
    if get("num_diagnoses") is not None:
        facts["total diagnoses recorded"] = int(get("num_diagnoses"))

    if get("length_of_stay") is not None:
        facts["length of stay"] = f"{int(get('length_of_stay'))} days"
    if get("admission_type"):
        facts["admission type"] = str(get("admission_type"))
    if get("department") and str(get("department")) != "Unknown":
        facts["admitting speciality"] = str(get("department"))

    for key, label in [
        ("num_prior_admissions", "inpatient admissions in the prior year"),
        ("num_prior_emergency", "emergency visits in the prior year"),
        ("num_prior_outpatient", "outpatient visits in the prior year"),
    ]:
        if get(key) is not None:
            facts[label] = int(get(key))

    if get("encounter_seq") is not None:
        facts["this is recorded visit number"] = int(get("encounter_seq"))

    a1c = get("a1c_result")
    if a1c:
        facts["HbA1c"] = (
            "not measured during this stay" if str(a1c) == "Not Tested" else str(a1c)
        )
    glucose = get("glucose_serum")
    if glucose:
        facts["serum glucose"] = (
            "not measured during this stay" if str(glucose) == "Not Tested" else str(glucose)
        )

    if get("num_medications") is not None:
        facts["medications during stay"] = int(get("num_medications"))
    if get("medication_changed") is not None:
        facts["diabetes regimen changed"] = "yes" if int(get("medication_changed")) else "no"
    if get("on_diabetes_meds") is not None:
        facts["on diabetes medication"] = "yes" if int(get("on_diabetes_meds")) else "no"

    return facts


def population_facts(df: pd.DataFrame, top_n: int = 8) -> list[dict[str, Any]]:
    """Measured risk-factor lift for the whole cohort.

    This is the evidence base. Every statistic the AI is allowed to quote comes
    from here, and every one was computed by Module 2 from this dataset.
    """
    summary = metrics.risk_factor_summary(df)
    if summary.empty:
        return []

    facts = []
    for _, row in summary.head(top_n).iterrows():
        facts.append(
            {
                "factor": row["risk_factor"],
                "rate_with": round(float(row["readmit_rate_with"]), 1),
                "rate_without": round(float(row["readmit_rate_without"]), 1),
                "lift": round(float(row["lift"]), 2),
                "patients_with": int(row["patients_with_factor"]),
                "statement": (
                    f"{row['risk_factor']}: {row['readmit_rate_with']:.1f}% of "
                    f"patients with this factor were readmitted, against "
                    f"{row['readmit_rate_without']:.1f}% without it "
                    f"({row['lift']:.2f}x)"
                ),
            }
        )
    return facts


def cohort_facts(df: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers for the population this patient belongs to."""
    k = metrics.kpis(df)
    return {
        "encounters analysed": f"{k['total_patients']:,}",
        "distinct people": f"{k.get('unique_people', 0):,}",
        "overall 30-day readmission rate": f"{k['readmission_rate']:.1f}%",
        "average length of stay": f"{k['avg_length_of_stay']:.1f} days",
        "cohort": "diabetic inpatients, 130 US hospitals, 1999-2008",
    }


# ---------------------------------------------------------------------------
def build_grounding(
    row: pd.Series, df: pd.DataFrame, prediction: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Assemble everything the AI is permitted to use, and nothing else.

    Parameters
    ----------
    row
        One patient's cleaned record.
    df
        The full cohort, for population statistics.
    prediction
        Output of `ml.predict.explain_one()`. Optional — without a trained model
        the report still works, using measured lift alone, and says so.
    """
    grounding: dict[str, Any] = {
        "pseudonym": pseudonymise(row.get("patient_id", "unknown")),
        "patient": patient_facts(row),
        "population": population_facts(df),
        "cohort": cohort_facts(df),
        "not_recorded": NOT_RECORDED,
        "has_model": prediction is not None,
    }

    if prediction is not None:
        grounding["prediction"] = {
            "risk_score": f"{prediction['risk_pct']:.1f}",
            "risk_band": prediction["risk_band"],
            "recommended_action": prediction["recommended_action"],
            "model": prediction["model_name"],
            "model_roc_auc": round(float(prediction["model_roc_auc"]), 3),
            "calibration": prediction.get("calibration_note", ""),
        }

        contributions = prediction.get("contributions")
        if contributions is not None and not contributions.empty:
            grounding["drivers"] = [
                {
                    "feature": r["readable"],
                    "value": r["value"],
                    "direction": r["direction"],
                    "strength": round(abs(float(r["contribution"])), 4),
                }
                for _, r in contributions.iterrows()
            ]
        grounding["explanation_method"] = prediction.get("explanation_method", "")
        grounding["caveat"] = prediction.get("caveat", "")

    return grounding


def grounding_to_text(grounding: dict[str, Any]) -> str:
    """Format the fact block for insertion into a prompt.

    Plain labelled text, not JSON. Language models follow labelled prose more
    reliably than nested structures, and it stays readable when you print the
    prompt to check what was actually sent — which you should.
    """
    lines: list[str] = []

    lines.append(f"PATIENT: {grounding['pseudonym']}")
    lines.append("")
    lines.append("RECORDED FACTS FOR THIS ADMISSION")
    for key, value in grounding["patient"].items():
        lines.append(f"  - {key}: {value}")

    if grounding.get("prediction"):
        p = grounding["prediction"]
        lines.append("")
        lines.append("MODEL PREDICTION")
        lines.append(f"  - risk score: {p['risk_score']} (band: {p['risk_band']})")
        lines.append(f"  - model used: {p['model']} (ROC-AUC {p['model_roc_auc']})")
        lines.append(f"  - suggested action: {p['recommended_action']}")
        if p.get("calibration"):
            lines.append(f"  - NOTE: {p['calibration']}")

    if grounding.get("drivers"):
        lines.append("")
        lines.append("WHAT DROVE THIS SCORE (from the model, strongest first)")
        for driver in grounding["drivers"]:
            lines.append(
                f"  - {driver['feature']} (value {driver['value']}): "
                f"{driver['direction']}"
            )
        if grounding.get("explanation_method"):
            lines.append(f"  method: {grounding['explanation_method']}")

    if grounding.get("population"):
        lines.append("")
        lines.append("MEASURED POPULATION STATISTICS - the ONLY numbers you may quote")
        for fact in grounding["population"]:
            lines.append(f"  - {fact['statement']}")

    lines.append("")
    lines.append("COHORT CONTEXT")
    for key, value in grounding["cohort"].items():
        lines.append(f"  - {key}: {value}")

    lines.append("")
    lines.append("NOT RECORDED IN THIS DATASET - never mention or estimate these:")
    lines.append(f"  {', '.join(grounding['not_recorded'])}")

    if grounding.get("caveat"):
        lines.append("")
        lines.append(f"CAVEAT: {grounding['caveat']}")

    return "\n".join(lines)
