"""
STEP 5 - generate a grounded AI report for a patient.

RUN IT LIKE THIS
----------------
    python scripts/05_generate_report.py                 # a random high-risk patient
    python scripts/05_generate_report.py --patient 149190
    python scripts/05_generate_report.py --highest-risk
    python scripts/05_generate_report.py --template patient_letter
    python scripts/05_generate_report.py --show-prompt   # see exactly what the AI is sent
    python scripts/05_generate_report.py --provider template   # force the offline version

WHAT MODULE 5 ACTUALLY DOES
---------------------------
It answers the question a risk score cannot: **why is this patient at risk, and
what should we do about it?**

A probability of 0.31 tells a nurse nothing actionable. A paragraph saying which
factors drove that number, how much each one matters in this population, and
what the ward could do about each - that is usable.

THE ONE IDEA THAT MATTERS HERE: GROUNDED GENERATION
---------------------------------------------------
There are two ways to build this, and only one is safe.

  THE UNSAFE WAY
      "Here is a patient. Explain why they might be readmitted."
      The model answers from its own training knowledge. It will produce
      fluent, confident, medically-plausible statistics that are INVENTED.
      In a clinical setting that is dangerous, and in a viva it is
      indefensible, because you cannot show where any number came from.

  THE WAY THIS PROJECT WORKS
      We compute the facts ourselves - the patient's recorded values, the
      model's predicted probability, the per-feature contributions, and the
      readmission lift for each risk factor as measured on THIS dataset - and
      pass them in. The system prompt forbids the model from using any number
      that was not supplied.

The language model's job is reduced to WORDING, not knowledge. Every figure in
the output traces back to a line in `metrics.py`, and a clinician can challenge
any claim and be shown the number behind it.

WHY IT STILL WORKS WITH NO API KEY
----------------------------------
If no provider is configured, the same grounded facts are rendered by a
deterministic template. The report is slightly stiffer to read, and it contains
exactly the same information.

That fallback is worth demonstrating, not apologising for: it proves the
grounding layer is doing the work rather than the language model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hospital_analytics.analysis.data_access import load_patients  # noqa: E402
from hospital_analytics.genai import generator, providers  # noqa: E402

TEMPLATES = ["clinical_report", "patient_letter", "cohort_brief"]


def pick_patient(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    """Choose which encounter to report on."""
    if args.patient:
        match = df[df["patient_id"].astype(str) == str(args.patient)]
        if match.empty:
            raise SystemExit(
                f"\nNo encounter with id '{args.patient}'.\n"
                f"Example ids from your data: "
                f"{', '.join(df['patient_id'].astype(str).head(5))}\n"
            )
        return match.iloc[0]

    if args.highest_risk:
        # Highest risk according to the trained model, if there is one.
        try:
            from hospital_analytics.ml import predict as P

            if P.model_exists():
                scored = P.predict_batch(df)
                best = scored["readmission_probability"].idxmax()
                return df.loc[best]
        except Exception:  # noqa: BLE001
            pass
        # No model: fall back to the patient with the most prior admissions.
        return df.sort_values("num_prior_admissions", ascending=False).iloc[0]

    # Default: a patient who was actually readmitted, so the report has content.
    readmitted = df[df["readmitted_30d"] == 1]
    pool = readmitted if not readmitted.empty else df
    return pool.sample(1, random_state=args.seed).iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a grounded AI report for one patient."
    )
    parser.add_argument("--patient", help="encounter id to report on")
    parser.add_argument("--highest-risk", action="store_true",
                        help="pick the highest-risk encounter in the data")
    parser.add_argument("--template", default="clinical_report", choices=TEMPLATES,
                        help="which style of report to produce")
    parser.add_argument("--provider", default="auto",
                        help="auto | ollama | template | anthropic | openai | huggingface")
    parser.add_argument("--ollama-check", action="store_true",
                        help="test the local Llama connection and exit")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print the exact system and user prompt sent to the model")
    parser.add_argument("--show-facts", action="store_true",
                        help="print the grounded facts block")
    parser.add_argument("--save", metavar="PATH", help="write the report to a file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 68)
    print("MODULE 5  GROUNDED AI REPORT")
    print("=" * 68)

    if args.ollama_check:
        ready, why = providers.ollama_status()
        print(f"  host  : {providers.OLLAMA_HOST}")
        print(f"  model : {providers.OLLAMA_MODEL}")
        print(f"  status: {'READY' if ready else 'NOT READY'}")
        print(f"  {why}")
        if not ready:
            print()
            print("  To set up the free local Llama model:")
            print("    1. Download Ollama from https://ollama.com/download")
            print(f"    2. Run:  ollama pull {providers.OLLAMA_MODEL}")
            print("    3. Run this check again")
            print()
            print("  Until then, reports still generate from the template using")
            print("  the identical grounded facts.")
        return 0 if ready else 1

    # --- 1. what can we actually run? --------------------------------------
    available = providers.available_providers()
    print("  providers:")
    for name, info in available.items():
        mark = "available" if info.get("ready") else info.get("reason", "not configured")
        print(f"    {name:<14} {mark}")
    print(f"  using: {providers.best_available() if args.provider == 'auto' else args.provider}")
    print()

    # --- 2. load the data ---------------------------------------------------
    df, source = load_patients()
    print(f"  loaded {len(df):,} encounters from: {source}")

    row = pick_patient(df, args)
    print(f"  reporting on encounter: {row['patient_id']}")

    # --- 3. is there a model to explain? ------------------------------------
    prediction = None
    try:
        from hospital_analytics.ml import predict as P

        if P.model_exists():
            prediction = P.explain_one(row)
            print(
                f"  model prediction: {prediction['risk_pct']:.1f}% "
                f"({prediction['risk_band']} risk) "
                f"via {prediction['model_name']}"
            )
        else:
            print("  no trained model found - the report will use measured")
            print("  population lift only, and will say so.")
            print("  (run scripts/04_train_model.py to add predictions)")
    except Exception as exc:  # noqa: BLE001
        print(f"  model unavailable ({type(exc).__name__}: {exc}) - continuing without it")

    # --- 4. generate --------------------------------------------------------
    print()
    print("  generating...")
    result = generator.generate_report(
        row, df, prediction=prediction,
        template=args.template, provider=args.provider,
    )

    if args.show_facts:
        print()
        print("-" * 68)
        print("GROUNDED FACTS - everything the AI is permitted to use")
        print("-" * 68)
        print(result["facts_text"])

    if args.show_prompt:
        print()
        print("-" * 68)
        print("SYSTEM PROMPT - the rules the model must follow")
        print("-" * 68)
        print(result["prompt"]["system"])
        print()
        print("-" * 68)
        print("USER PROMPT - the facts and the request")
        print("-" * 68)
        print(result["prompt"]["user"])

    # --- 5. show it ---------------------------------------------------------
    print()
    print("=" * 68)
    print(f"REPORT   (source: {result['source']}, template: {result['template']})")
    print("=" * 68)
    print(result["text"])
    print("=" * 68)

    # --- 6. the validation layer -------------------------------------------
    # The validation layer is the part worth showing an evaluator. A generated
    # report is only trustworthy if something CHECKS it, automatically, every
    # time - and can explain what it checked.
    validation = result["validation"]
    print()
    print("VALIDATION - automatic checks run on the generated text")
    print("-" * 68)
    checks = validation.get("checks")
    if checks is not None and not checks.empty:
        for _, check in checks.iterrows():
            mark = "PASS" if check["passed"] else "FAIL"
            print(f"  [{mark}] {check['rule']}")
            if not check["passed"] and str(check.get("found", "")).strip():
                print(f"         found: {check['found']}")
    print()
    print(f"  overall: {'PASS' if validation.get('passed') else 'NEEDS REVIEW'}"
          f"  ({validation.get('n_failed', 0)} rule(s) failed)")
    if validation.get("failures"):
        print()
        print("  In a real deployment a failure here would BLOCK the report")
        print("  rather than show it. That is the point of a validator: the")
        print("  language model is not trusted, it is checked.")

    if result["used_fallback"]:
        print()
        print("  NOTE: the AI provider was unavailable, so the deterministic")
        print("        template produced this report from the same facts.")
        if result["provider_error"]:
            print(f"        reason: {result['provider_error']}")

    # --- 7. optionally save -------------------------------------------------
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["text"], encoding="utf-8")
        print()
        print(f"  saved to {out}")

    print()
    print("Try also:")
    print("  python scripts/05_generate_report.py --ollama-check    # test local Llama")
    print("  python scripts/05_generate_report.py --show-prompt     # see what the AI gets")
    print("  python scripts/05_generate_report.py --provider template  # compare")
    print("  streamlit run streamlit_app/app.py                     # the AI Report page")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
