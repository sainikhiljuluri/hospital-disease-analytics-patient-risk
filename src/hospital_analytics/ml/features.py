"""
MODULE 4, part 1: turning a cleaned table into a matrix a model can learn from.

WHAT "FEATURE ENGINEERING" ACTUALLY MEANS
-----------------------------------------
A model cannot read a DataFrame. It needs a rectangle of numbers: one row per
encounter, one column per feature, no text, no gaps. This file builds that
rectangle and — just as importantly — decides what must be LEFT OUT.

THE MOST IMPORTANT DECISION IN THIS FILE: THE PREDICTION MOMENT
---------------------------------------------------------------
Before choosing features you must answer one question:

    "At what moment does this model run?"

Our answer: **the day before discharge**, so the care team still has time to act.
Everything follows from that. A feature is allowed only if a clinician could look
it up the day before discharge. That single rule is what prevents data leakage.

WHAT LEAKAGE IS, AND WHY IT IS THE CLASSIC BEGINNER TRAP
--------------------------------------------------------
Leakage is including information that would not exist yet at prediction time.
The model scores brilliantly in testing and is worthless in deployment, because
in the real world that column is blank.

This dataset has a genuine trap: `discharge_disposition` — where the patient
went afterwards. It is strongly related to readmission, and it is recorded AT
discharge. Including it would inflate our score and produce a model that cannot
run when we need it.

So it is in LEAKY_FEATURES below and never enters the matrix. `readmitted_any`
is excluded for the same reason: it is a different view of the answer.

The dashboard still SHOWS discharge destination, because it is useful for
understanding the population. Understanding and predicting are different jobs.

HOW CATEGORIES BECOME NUMBERS
-----------------------------
`diagnosis` holds text like "Circulatory". A model needs numbers, and you cannot
simply map Circulatory=1, Respiratory=2 — that would tell the model Respiratory
is "twice" Circulatory, which is meaningless and actively misleading.

Instead we use ONE-HOT ENCODING: one new 0/1 column per category.

    diagnosis                    diag_Circulatory  diag_Respiratory
    Circulatory        ->               1                 0
    Respiratory        ->               0                 1

No false ordering is implied. Each category gets its own switch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# THE FEATURE CONTRACT
# ---------------------------------------------------------------------------
# Numeric columns used as-is. Every one is knowable the day before discharge.
NUMERIC_FEATURES = [
    # Utilisation BEFORE this admission — the strongest family in the
    # readmission literature, and knowable on arrival.
    "num_prior_admissions",
    "num_prior_emergency",
    "num_prior_outpatient",
    "num_prior_visits_total",
    "encounter_seq",
    # This admission, knowable the day before discharge
    "length_of_stay",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "num_diagnoses",
    # Demographics
    "age",
    # Glycaemic control and treatment response
    "a1c_tested",
    "glucose_tested",
    "a1c_high",
    "glucose_high",
    "medication_changed",
    "on_diabetes_meds",
    # Engineered flags from cleaning.py
    "is_elderly",
    "is_long_stay",
    "is_polypharmacy",
    "has_prior_admission",
    "has_prior_emergency",
    "is_complex",
    "is_emergency",
]

# Text columns that become one-hot 0/1 columns.
CATEGORICAL_FEATURES = [
    "gender",
    "age_group",
    "diagnosis",
    "diagnosis_secondary",
    "admission_type",
    "admission_source",
    "a1c_result",
    "glucose_serum",
]

# NEVER features. Each is excluded for a stated reason — this list is the single
# best thing to be able to explain about the whole ML module.
LEAKY_FEATURES = {
    "discharge_disposition": (
        "Recorded AT discharge. Our model runs BEFORE discharge, so this value "
        "does not exist yet at prediction time. Classic data leakage."
    ),
    "discharged_home": "Derived from discharge_disposition — same leakage.",
    "to_care_facility": "Derived from discharge_disposition — same leakage.",
    "readmitted_any": (
        "A different view of the answer we are trying to predict. Including it "
        "would let the model read the target."
    ),
    "readmitted_30d": "This IS the target.",
    "patient_id": "An identifier, not a fact about the patient.",
    "person_id": "An identifier. Used for the SPLIT, never as a feature.",
    "icd9_primary": (
        "Hundreds of distinct codes. One-hot encoding it would create hundreds "
        "of near-empty columns. The grouped `diagnosis` column carries the "
        "signal without the sparsity."
    ),
}

# `race` is deliberately NOT in CATEGORICAL_FEATURES.
#
# It is in the dataset, and it does correlate with readmission — but that
# correlation reflects unequal access to care, not biology. A model that uses it
# would allocate follow-up resources by ethnicity, entrenching the very
# disparity the data records.
#
# We exclude it from the features and USE it in the fairness audit instead:
# the model must be checked for whether it performs equally across groups it
# cannot see. That is the standard approach, and the reasoning is worth being
# able to give out loud.
FAIRNESS_ATTRIBUTES = ["race", "gender", "age_group"]

TARGET = "readmitted_30d"
GROUP_COLUMN = "person_id"


# ---------------------------------------------------------------------------
def build_feature_matrix(
    df: pd.DataFrame, feature_names: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Turn a cleaned patients table into (X, y, groups).

    Parameters
    ----------
    df
        A cleaned patients DataFrame, as produced by `cleaning.clean_patients`.
    feature_names
        Pass the training-time column list when scoring NEW data. One-hot
        encoding produces different columns depending on which categories happen
        to appear, so a single unseen patient would otherwise yield a matrix of
        the wrong shape. Passing the list forces an exact match: missing columns
        are added as 0, unexpected ones dropped.

        This is a real, common bug — "the model was trained on 68 columns but
        got 41" — and this parameter is the fix.

    Returns
    -------
    X : the numeric feature matrix
    y : the target, or None if the column is absent (i.e. real unlabelled data)
    groups : person_id per row, used to keep one person out of both splits
    """
    df = df.copy()

    y = df[TARGET].astype(int) if TARGET in df.columns else None
    groups = df[GROUP_COLUMN].astype(str) if GROUP_COLUMN in df.columns else None

    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X_num = df[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if categorical:
        X_cat = pd.get_dummies(
            df[categorical].astype(str), prefix=categorical, prefix_sep="=", dtype=float
        )
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num

    X.columns = [str(c) for c in X.columns]

    # Align to the training-time columns when scoring new data.
    if feature_names is not None:
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_names]

    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    return X, y, groups


def feature_report(X: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: type, spread, and how often it is non-zero.

    Useful before training. A feature that is non-zero in 0.1% of rows carries
    almost no information and mostly adds noise.
    """
    rows = []
    for col in X.columns:
        series = X[col]
        is_binary = set(series.unique()).issubset({0.0, 1.0})
        rows.append(
            {
                "feature": col,
                "kind": "one-hot" if "=" in col else ("flag" if is_binary else "numeric"),
                "mean": round(float(series.mean()), 3),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "pct_nonzero": round(100 * float((series != 0).mean()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_nonzero", ascending=False).reset_index(drop=True)


def excluded_features_table() -> pd.DataFrame:
    """The leakage exclusions, as a table the dashboard can display.

    Showing what you deliberately left out — and why — is more convincing than
    showing what you put in.
    """
    rows = [{"column": k, "reason": v} for k, v in LEAKY_FEATURES.items()]
    rows.append(
        {
            "column": "race",
            "reason": (
                "Excluded on fairness grounds, not leakage. It correlates with "
                "the outcome because of unequal access to care, so using it "
                "would allocate resources by ethnicity. Used for auditing the "
                "model instead."
            ),
        }
    )
    return pd.DataFrame(rows)


def describe_prediction_moment() -> dict[str, Any]:
    """The design decision every feature choice follows from."""
    return {
        "moment": "The day before discharge",
        "why": (
            "Late enough that the stay is nearly complete and its details are "
            "known, but early enough that the care team can still act — arrange "
            "follow-up, review medication, involve a pharmacist."
        ),
        "rule": (
            "A feature is allowed only if a clinician could look it up the day "
            "before discharge."
        ),
        "excluded_count": len(LEAKY_FEATURES),
    }
