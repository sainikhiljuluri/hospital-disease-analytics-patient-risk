"""
MODULE 5 - Generative AI: explaining a risk score in plain English.

WHAT IS IN HERE
---------------
    grounding.py   assembles ONLY verified facts (the safety mechanism)
    prompts.py     the system rules and report templates
    providers.py   Anthropic / OpenAI / HuggingFace / template fallback
    generator.py   runs the pipeline and validates what comes back

HOW TO USE IT
-------------
    python scripts/05_generate_report.py --patient-id 12345

    from hospital_analytics.genai import generator
    report = generator.generate_report(row, df, prediction)
    print(report["text"])

THE ONE IDEA THAT MATTERS: GROUNDED GENERATION
-----------------------------------------------
Never ask a language model to explain a prediction from its own knowledge. It
will produce a fluent, confident, invented statistic, and it will sound exactly
as certain when it is wrong as when it is right.

Instead: assemble the facts first - the patient's recorded values, the model's
score, the per-feature contributions, and the population lift measured from THIS
dataset - and instruct the model to use only those. Its job becomes translation,
not recall.

This is why Modules 2 and 4 had to come first. `metrics.risk_factor_summary()`
computes the statistics; `ml.predict.explain_one()` computes the drivers. Module
5 only phrases them.

IT WORKS WITHOUT AN API KEY
---------------------------
If no provider is configured, `generator.template_report()` builds the same
report from the same facts in plain Python. That is not a consolation prize - it
is a demonstration that the facts, not the language model, are the source of
truth. Run both in your demo: the LLM version reads better, the template version
is never wrong.

VALIDATION HAPPENS AFTER GENERATION
-----------------------------------
A safety rule in a prompt is a request, not a guarantee. `validate_output()`
checks the generated text for causal claims, prescribing, overconfidence and
invented facts, and reports which rules passed. Checking is what turns "I told
it not to" into "I verified it didn't".
"""

from . import generator, grounding, prompts, providers

__all__ = ["generator", "grounding", "prompts", "providers"]
