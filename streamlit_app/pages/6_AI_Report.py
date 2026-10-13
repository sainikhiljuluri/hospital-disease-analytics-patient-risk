"""
PAGE 6 - AI Medical Report Generator (Module 5).

WHAT THIS PAGE DOES
-------------------
Turns a risk score into an explanation a human can act on. A probability of 0.31
tells a nurse nothing; a paragraph naming the factors behind it, how much each
one matters in this population, and what the ward could do about each - that is
usable.

THE IDEA THAT MATTERS: GROUNDED GENERATION
------------------------------------------
There are two ways to build this and only one is safe.

  UNSAFE   "Here is a patient, explain the risk."
           The model answers from its own training knowledge and produces
           fluent, confident, INVENTED statistics.

  THIS     We compute the facts ourselves - the patient's recorded values, the
           model's probability, the per-feature contributions, and the measured
           readmission lift for each factor on THIS dataset - and pass them in.
           The system prompt forbids any number that was not supplied.

The language model's job is reduced to WORDING, not knowledge.

WORKS WITH NO API KEY
---------------------
With no provider configured, the same grounded facts are rendered by a
deterministic template. Same information, stiffer prose. That is worth
demonstrating rather than apologising for: it proves the grounding layer is
doing the work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402

from hospital_analytics.analysis import metrics  # noqa: E402
from hospital_analytics.genai import generator, prompts, providers  # noqa: E402
from hospital_analytics.ml import predict as P  # noqa: E402

ui.page_setup(
    "AI Medical Report",
    icon="📄",
    subtitle="Module 5 preview - turning numbers into a plain-language explanation",
)

patients, meds, source = ui.require_data()
df = ui.sidebar_filters(patients, source, key_prefix="ai_")
if ui.empty_guard(df):
    st.stop()

PROVIDERS = providers.available_providers()
ACTIVE = providers.best_available()
MODEL_READY = P.model_exists()

ui.section(
    "How this page works",
    "Facts are computed in Python. The AI only chooses the wording.",
)

pc1, pc2 = st.columns([2, 1])
with pc1:
    st.markdown(
        "1. **Ground** — collect the patient's recorded values, the model's "
        "prediction and the measured population lift for each risk factor.\n"
        "2. **Prompt** — wrap those facts in a system prompt that forbids using "
        "any number not supplied.\n"
        "3. **Generate** — send it to a language model, or render the same facts "
        "with a deterministic template if none is configured.\n"
        "4. **Validate** — check the output automatically for invented numbers, "
        "causal claims, prescribing and overconfidence before showing it."
    )
with pc2:
    st.markdown("**AI providers detected**")
    for name, info in PROVIDERS.items():
        if info.get("ready"):
            st.markdown(f"- **{name}** — ready")
        else:
            st.caption(f"- {name} — {info.get('reason', 'not configured')}")
    st.caption(f"Active: **{ACTIVE}**")

if ACTIVE == "ollama":
    st.success(
        "**Running Llama locally via Ollama.** No API key, no cost, and — the "
        "part that matters for clinical data — no patient record leaves this "
        "machine. There is no outbound request, so there is nothing to leak or "
        "log.",
        icon="🦙",
    )
elif ACTIVE == "template":
    st.info(
        "**The page works without an AI model, and this is worth demonstrating.** "
        "Reports below are rendered from the identical grounded facts by a "
        "Python template. If the report is still correct without a language "
        "model, the grounding layer is doing the work — which is the whole "
        "architectural claim of Module 5.",
        icon="🧩",
    )
    with st.expander("Add a free local Llama model (about 5 minutes, no API key)"):
        st.markdown(
            "**Ollama** runs Meta's Llama models on your own computer. It is "
            "free, needs no account, and keeps patient data on the machine.\n\n"
            "1. Download from [ollama.com/download](https://ollama.com/download)\n"
            "2. Open a terminal and run the command below\n"
            "3. Refresh this page — the project finds it automatically"
        )
        st.code("ollama pull llama3.2", language="bash")
        st.caption(
            f"Currently looking for Ollama at `{providers.OLLAMA_HOST}` and for "
            f"the model `{providers.OLLAMA_MODEL}`. Change either in your `.env` "
            f"file. No `pip install` is needed — the project talks to Ollama "
            f"using Python's built-in urllib."
        )
        ready, why = providers.ollama_status()
        st.caption(f"Status: {why}")

if not MODEL_READY:
    st.warning(
        "No trained model found, so reports will use measured population lift "
        "only and will say so. Run `python scripts/04_train_model.py` to add "
        "per-patient predictions and feature contributions.",
        icon="🤖",
    )

st.divider()

# ===========================================================================
ui.section("Pick a patient")

pid = st.selectbox("Patient", df["patient_id"].head(400).tolist(), key="ai_pid")
row = df[df["patient_id"] == pid].iloc[0]
population = metrics.risk_factor_summary(df)
lift_lookup = dict(zip(population["risk_factor"], population["lift"])) if not population.empty else {}

# ---------------------------------------------------------------------------
def collect_findings(r: pd.Series) -> list[dict]:
    """Gather the patient's risk factors, each with its measured population lift.

    THIS is the part that makes an LLM explanation trustworthy: every sentence the
    model writes will be traceable to a number computed from real data, rather
    than to the model's own recollection of medical literature.
    """
    checks = [
        ("Prior inpatient admission", r.get("num_prior_admissions", 0) >= 1,
         f"{int(r.get('num_prior_admissions', 0))} inpatient admission(s) in the prior year"),
        ("Prior emergency visit", r.get("num_prior_emergency", 0) >= 1,
         f"{int(r.get('num_prior_emergency', 0))} emergency visit(s) in the prior year"),
        ("Repeat patient (not first visit)", r.get("is_repeat_patient", 0) == 1,
         f"this is recorded visit number {int(r.get('encounter_seq', 1))} for this person"),
        ("Age 65 or over", r.get("age", 0) >= 65,
         f"in the {r.get('age_group', 'older')} age band"),
        ("Long stay (> 7 days)", r.get("length_of_stay", 0) > 7,
         f"hospital stay of {int(r.get('length_of_stay', 0))} days"),
        ("Complex case (9+ diagnoses)", r.get("num_diagnoses", 0) >= 9,
         f"{int(r.get('num_diagnoses', 0))} diagnoses recorded on this admission"),
        ("Polypharmacy (> 20 meds)", r.get("num_medications", 0) > 20,
         f"prescribed {int(r.get('num_medications', 0))} medications during the stay"),
        ("HbA1c high (> 7)", r.get("a1c_high", 0) == 1,
         f"HbA1c recorded as {r.get('a1c_result', 'elevated')}, indicating poor long-term control"),
        ("Serum glucose high (> 200)", r.get("glucose_high", 0) == 1,
         f"serum glucose recorded as {r.get('glucose_serum', 'elevated')}"),
        ("Medication regimen changed", r.get("medication_changed", 0) == 1,
         "diabetes medication was adjusted during this admission"),
        ("Discharged to care facility", r.get("to_care_facility", 0) == 1,
         f"discharged to {r.get('discharge_disposition', 'a care facility')}"),
    ]
    out = []
    for factor, fired, phrase in checks:
        if bool(fired):
            out.append({
                "factor": factor,
                "phrase": phrase,
                "lift": lift_lookup.get(factor, 1.0),
            })
    return sorted(out, key=lambda d: d["lift"], reverse=True)


findings = collect_findings(row)



ui.section("Generated report", "Choose an audience — the rules change with it.")

rc1, rc2, rc3 = st.columns([2, 2, 1])
with rc1:
    template_key = st.selectbox(
        "Report type",
        list(prompts.TEMPLATE_LABELS.keys()),
        format_func=lambda k: prompts.TEMPLATE_LABELS[k],
        key="ai_template",
    )
with rc2:
    provider_choice = st.selectbox(
        "Generate with",
        ["auto"] + list(PROVIDERS.keys()),
        format_func=lambda p: (
            f"{p} (recommended)" if p == "auto" else p
        ),
        key="ai_provider",
    )
with rc3:
    st.write("")
    st.write("")
    regenerate = st.button("Generate", type="primary", key="ai_go")

# Predict first, so the report can be grounded in the model's own output.
prediction = None
if MODEL_READY:
    try:
        prediction = P.explain_one(row)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Model unavailable for this patient ({type(exc).__name__}).")

result = generator.generate_report(
    row, df, prediction=prediction,
    template=template_key, provider=provider_choice,
)

c1, c2 = st.columns([2, 1])
with c1:
    if result["source"] == "template":
        st.caption(
            "Rendered from the grounded facts by a Python template — no language "
            "model involved. The content is identical to what an LLM receives."
        )
    else:
        st.caption(f"Written by **{result['source']}** from the grounded facts below.")

    if result["used_fallback"] and result["provider_error"]:
        st.warning(
            f"The AI provider failed, so the template produced this instead. "
            f"Reason: {result['provider_error']}",
            icon="⚠️",
        )

    st.markdown(result["text"])

    st.download_button(
        "Download this report",
        data=result["text"],
        file_name=f"report_{pid}_{template_key}.md",
        mime="text/markdown",
    )

    # -----------------------------------------------------------------------
    st.divider()
    ui.section(
        "Automatic validation",
        "The model is not trusted. It is checked, every time.",
    )
    validation = result["validation"]
    checks = validation.get("checks")
    if checks is not None and not checks.empty:
        vc1, vc2 = st.columns(2)
        for i, (_, chk) in enumerate(checks.iterrows()):
            target = vc1 if i % 2 == 0 else vc2
            with target:
                if chk["passed"]:
                    st.markdown(f"✅ **{chk['rule']}**")
                else:
                    st.markdown(f"❌ **{chk['rule']}**")
                    if str(chk.get("found", "")).strip():
                        st.caption(f"found: {chk['found']}")

    if validation.get("passed"):
        st.success(
            "All checks passed. In a deployed system this report would be "
            "released to the ward; a failure would block it instead.",
            icon="🛡️",
        )
    else:
        st.error(
            f"{validation.get('n_failed', 0)} check(s) failed. A deployed system "
            f"would BLOCK this report rather than display it. That is the whole "
            f"purpose of a validator — the language model is not trusted to "
            f"police itself.",
            icon="🛑",
        )

    st.caption(
        "The most important rule is **no invented facts**: every number in the "
        "text is matched against the grounded facts supplied. A figure that "
        "appears in the output but not the input is a hallucination, and it is "
        "caught mechanically rather than by someone noticing."
    )

    # -----------------------------------------------------------------------
    st.divider()
    with st.expander("See the grounded facts — everything the AI was allowed to use"):
        st.caption(
            "This is the complete input. Nothing else was supplied, and the "
            "system prompt forbids using anything beyond it."
        )
        st.code(result["facts_text"], language="text")

    with st.expander("See the exact prompt sent to the model"):
        st.markdown("**System prompt** — the rules")
        st.code(result["prompt"]["system"], language="text")
        st.markdown("**User prompt** — the facts and the request")
        st.code(result["prompt"]["user"], language="text")
        st.caption(
            "Being able to show this is the point. Every claim in the report "
            "above traces back to a number computed in `metrics.py`, and a "
            "clinician can challenge any of them and be shown the source."
        )

with c2:
    st.markdown("**Patient snapshot**")
    snap = {
        "Age band": row.get("age_group"),
        "Gender": row.get("gender"),
        "Diagnosis": row.get("diagnosis"),
        "Visit number": row.get("encounter_seq"),
        "HbA1c": row.get("a1c_result"),
        "Serum glucose": row.get("glucose_serum"),
        "Stay (days)": row.get("length_of_stay"),
        "Diagnoses": row.get("num_diagnoses"),
        "Prior inpatient": row.get("num_prior_admissions"),
        "Prior emergency": row.get("num_prior_emergency"),
        "Medications": row.get("num_medications"),
        "Discharged to": row.get("discharge_disposition"),
        "Readmitted (actual)": "Yes" if row.get("readmitted_30d") == 1 else "No",
    }
    ui.table(pd.DataFrame({"Field": list(snap.keys()),
                           "Value": [str(v) for v in snap.values()]}))

    st.markdown("**Risk factors detected**")
    st.metric("Count", len(findings))

st.divider()

# ===========================================================================
ui.section("The prompt Module 5 will send")
st.caption(
    "Notice that the prompt supplies the FACTS and forbids the model from "
    "inventing new ones. This is retrieval-grounded generation, and it is what "
    "keeps a clinical AI report safe."
)

facts = "\n".join(
    f"  - {f['factor']}: {f['phrase']} (measured population lift {f['lift']:.2f}x)"
    for f in findings
) or "  - No tracked risk factors present."

st.code(
    f'''SYSTEM
You are a clinical decision-support assistant writing for hospital
discharge planners. You must use ONLY the facts supplied below. Never
invent a measurement, a diagnosis or a statistic. If a fact is absent,
say it is not recorded. Do not give a definitive prognosis. End with a
reminder that a qualified clinician must review the plan.

USER
Encounter: {row.get("patient_id")}   Visit number: {row.get("encounter_seq")}
Age band: {row.get("age_group")}   Gender: {row.get("gender")}
Primary diagnosis group: {row.get("diagnosis")}  (speciality: {row.get("department")})
Secondary diagnoses: {row.get("diagnosis_secondary")}, {row.get("diagnosis_tertiary")}
Total diagnoses recorded: {row.get("num_diagnoses")}
Length of stay: {row.get("length_of_stay")} days
Prior year — inpatient: {row.get("num_prior_admissions")}, emergency: {row.get("num_prior_emergency")}, outpatient: {row.get("num_prior_outpatient")}
HbA1c: {row.get("a1c_result")}   Serum glucose: {row.get("glucose_serum")}
Medications during stay: {row.get("num_medications")}   Regimen changed: {"yes" if row.get("medication_changed") == 1 else "no"}
Discharged to: {row.get("discharge_disposition")}

NOTE: this dataset records no vital signs and no dates. Do not refer to blood
pressure, BMI, cholesterol or admission dates - they are not recorded.

Model prediction (from Module 4): <predicted probability>
Top contributing features (SHAP): <feature contributions>

Risk factors measured for this patient, with population-level lift:
{facts}

TASK
1. Write a 3-4 sentence clinical summary.
2. Explain, in plain language, why this patient is at elevated risk,
   citing the factors above and their measured lift values.
3. List 3-5 specific, actionable interventions.
4. Note explicitly which important information is missing from the record.
''',
    language="text",
)

st.divider()

with st.expander("🔒 Safety and ethics constraints for Module 5"):
    st.markdown(
        """
Writing this down now, before the LLM is connected, is deliberate — these
constraints shape the design rather than being bolted on afterwards.

- **Grounded generation only.** The model receives the computed facts and is
  instructed to use nothing else. This is the main defence against a plausible-
  sounding invented statistic.
- **Decision support, never diagnosis.** Output is advisory. Every report carries
  a clinician-review notice.
- **No identifiable data leaves the building.** If a hosted API is used, the
  patient identifier is replaced with a pseudonym before the call, or a local
  Hugging Face model is used instead. This matters legally, not just ethically.
- **Every claim is auditable.** Because each statement traces back to a number in
  `analysis/metrics.py`, a clinician can challenge any sentence and be shown the
  figure behind it.
- **Fairness carried forward.** The demographic audit on the Patient Explorer
  page must be repeated on the model's outputs, so the explanations do not
  reproduce a demographic bias.
- **Known limitation.** The data is real but historical — US hospital admissions
  from 1999–2008, covering diabetic inpatients only. Findings do not
  automatically transfer to current practice, other countries, or non-diabetic
  patients. Every generated output must say so, and must state that a qualified
  clinician has to review any recommendation.
        """
    )
