"""
Module 2 (part 2): every number the project reports is computed here.

WHY THIS FILE EXISTS
--------------------
Suppose "average length of stay" were computed inside the notebook AND again
inside the dashboard. Sooner or later one of them filters out day-zero stays and
the other does not, your two screens disagree, and an evaluator spots it. Every
statistic therefore has exactly one implementation, right here.

Nothing in this file imports plotly or streamlit - it is pure pandas. That is
what lets the FastAPI service in a later module reuse it without dragging a whole
UI framework along.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Columns used for the correlation matrix and (later) as ML features.
# These are the numeric columns this dataset actually contains. Note what is
# absent: no blood pressure, BMI or cholesterol, because the dataset is
# de-identified and records no vitals. Working with what exists rather than what
# you wish existed is most of applied data science.
NUMERIC_FEATURES = [
    "age", "length_of_stay", "num_lab_procedures", "num_procedures",
    "num_medications", "num_diagnoses", "num_prior_admissions",
    "num_prior_emergency", "num_prior_outpatient", "encounter_seq",
    "a1c_tested", "medication_changed", "readmitted_30d",
]


# ---------------------------------------------------------------------------
# Headline KPIs
# ---------------------------------------------------------------------------
def kpis(df: pd.DataFrame) -> dict[str, Any]:
    """The numbers that sit across the top of the dashboard."""
    total = len(df)
    if total == 0:
        return {}

    readm = pd.to_numeric(df.get("readmitted_30d"), errors="coerce")
    return {
        "total_patients": total,
        "unique_diagnoses": int(df["diagnosis"].nunique()) if "diagnosis" in df else 0,
        "readmission_rate": float(readm.mean() * 100) if readm is not None else np.nan,
        "readmitted_count": int(readm.sum()) if readm is not None else 0,
        "avg_age": float(df["age"].mean()) if "age" in df else np.nan,
        "median_age": float(df["age"].median()) if "age" in df else np.nan,
        "avg_length_of_stay": float(df["length_of_stay"].mean()) if "length_of_stay" in df else np.nan,
        "total_bed_days": int(df["length_of_stay"].sum()) if "length_of_stay" in df else 0,
        "avg_medications": float(df["num_medications"].mean())
        if "num_medications" in df else np.nan,
        "avg_lab_procedures": float(df["num_lab_procedures"].mean())
        if "num_lab_procedures" in df else np.nan,
        "avg_diagnoses": float(df["num_diagnoses"].mean()) if "num_diagnoses" in df else np.nan,
        "unique_people": int(df["person_id"].nunique()) if "person_id" in df else total,
        "pct_repeat_patients": float(
            pd.to_numeric(df["is_repeat_patient"], errors="coerce").mean() * 100
        ) if "is_repeat_patient" in df else np.nan,
        "pct_male": float((df["gender"] == "Male").mean() * 100) if "gender" in df else np.nan,
        "pct_female": float((df["gender"] == "Female").mean() * 100) if "gender" in df else np.nan,
        "pct_a1c_tested": float(
            pd.to_numeric(df["a1c_tested"], errors="coerce").mean() * 100
        ) if "a1c_tested" in df else np.nan,
        "pct_med_changed": float(
            pd.to_numeric(df["medication_changed"], errors="coerce").mean() * 100
        ) if "medication_changed" in df else np.nan,
        "pct_prior_admission": float(
            pd.to_numeric(df["has_prior_admission"], errors="coerce").mean() * 100
        ) if "has_prior_admission" in df else np.nan,
    }


# ---------------------------------------------------------------------------
# Distributions and group summaries
# ---------------------------------------------------------------------------
def disease_distribution(df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Patient count and share per diagnosis, biggest first."""
    out = (
        df.groupby("diagnosis", dropna=False)
        .size()
        .reset_index(name="patient_count")
        .sort_values("patient_count", ascending=False)
    )
    out["pct_of_total"] = (out["patient_count"] / len(df) * 100).round(2)
    return out.head(top_n) if top_n else out.reset_index(drop=True)


def readmission_by_group(df: pd.DataFrame, group_col: str, min_count: int = 20) -> pd.DataFrame:
    """Readmission rate per category of `group_col`.

    `min_count` guards against nonsense like "100% readmission" computed from a
    single patient - always state this filter when you present the chart.
    """
    if group_col not in df.columns or "readmitted_30d" not in df.columns:
        return pd.DataFrame()

    g = (
        df.groupby(group_col, dropna=False)
        .agg(
            total_patients=("readmitted_30d", "size"),
            readmissions=("readmitted_30d", "sum"),
            avg_length_of_stay=("length_of_stay", "mean"),
            avg_age=("age", "mean"),
        )
        .reset_index()
    )
    g = g[g["total_patients"] >= min_count]
    g["readmission_rate"] = (g["readmissions"] / g["total_patients"] * 100).round(2)
    g["avg_length_of_stay"] = g["avg_length_of_stay"].round(2)
    g["avg_age"] = g["avg_age"].round(1)
    return g.sort_values("readmission_rate", ascending=False).reset_index(drop=True)


def encounter_sequence_curve(df: pd.DataFrame, max_seq: int = 8) -> pd.DataFrame:
    """Readmission rate by which visit number this is for that person.

    This dataset has NO admission dates - it is fully de-identified. What it
    does have is `encounter_seq`: for each person, whether this was their first
    recorded visit, their second, and so on. That is a genuine ordering, and the
    closest thing to a time axis the data supports.

    Do not call this a time series in your demo. It is a sequence.
    """
    if "encounter_seq" not in df.columns or "readmitted_30d" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["seq_band"] = pd.to_numeric(d["encounter_seq"], errors="coerce").clip(upper=max_seq)
    out = (
        d.groupby("seq_band")
        .agg(patients=("patient_id", "count"), readmitted=("readmitted_30d", "sum"))
        .reset_index()
    )
    out = out[out["patients"] >= 20]
    if out.empty:
        return pd.DataFrame()
    out["readmission_rate"] = (100 * out["readmitted"] / out["patients"]).round(2)
    out["visit_number"] = out["seq_band"].astype(int).astype(str)
    out.loc[out["seq_band"] >= max_seq, "visit_number"] = f"{max_seq}+"
    return out.reset_index(drop=True)


def prior_utilisation_curve(
    df: pd.DataFrame, col: str = "num_prior_admissions", max_val: int = 6
) -> pd.DataFrame:
    """Readmission rate against a count of prior visits, capped for readability."""
    if col not in df.columns or "readmitted_30d" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["_band"] = pd.to_numeric(d[col], errors="coerce").clip(upper=max_val)
    out = (
        d.groupby("_band")
        .agg(patients=("patient_id", "count"), readmitted=("readmitted_30d", "sum"))
        .reset_index()
    )
    out = out[out["patients"] >= 20]
    if out.empty:
        return pd.DataFrame()
    out["readmission_rate"] = (100 * out["readmitted"] / out["patients"]).round(2)
    out["label"] = out["_band"].astype(int).astype(str)
    out.loc[out["_band"] >= max_val, "label"] = f"{max_val}+"
    return out.reset_index(drop=True)


def a1c_testing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Readmission rate by HbA1c test result.

    This is the exact question the dataset was published to answer: does
    measuring HbA1c during an admission relate to early readmission?
    Reproducing the published finding on your own pipeline is the strongest
    single validation that your cleaning is correct.
    """
    if "a1c_result" not in df.columns or "readmitted_30d" not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby("a1c_result")
        .agg(patients=("patient_id", "count"), readmitted=("readmitted_30d", "sum"))
        .reset_index()
    )
    out["readmission_rate"] = (100 * out["readmitted"] / out["patients"]).round(2)
    order = ["Not Tested", "Norm", ">7", ">8"]
    out["_o"] = out["a1c_result"].map({v: i for i, v in enumerate(order)}).fillna(99)
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def discharge_destination_summary(df: pd.DataFrame, min_count: int = 50) -> pd.DataFrame:
    """Readmission rate by where the patient went after discharge.

    Read this one carefully in your demo. Discharge destination is recorded AT
    discharge, so a model meant to run BEFORE discharge cannot use it - that
    would be data leakage. It is here because it genuinely helps you understand
    the population, and because naming the leakage risk out loud is better than
    quietly feeding the column to Module 4.
    """
    if "discharge_disposition" not in df.columns:
        return pd.DataFrame()
    return readmission_by_group(df, "discharge_disposition", min_count=min_count)


def correlation_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Pearson correlation between the numeric clinical variables.

    Two uses: spot multicollinearity (systolic vs diastolic BP move together, so
    a linear model does not need both), and rank which variables actually relate
    to readmission before any model is trained.
    """
    cols = cols or [c for c in NUMERIC_FEATURES if c in df.columns]
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True).round(3)


def target_correlations(df: pd.DataFrame, target: str = "readmitted_30d") -> pd.DataFrame:
    """Correlation of every numeric variable with the target, strongest first.

    This is a *feature-importance preview*: it tells you what the ML model in
    Module 4 is likely to lean on, before you have trained anything.
    """
    corr = correlation_matrix(df)
    if target not in corr.columns:
        return pd.DataFrame()
    s = corr[target].drop(labels=[target], errors="ignore")
    out = pd.DataFrame({"feature": s.index, "correlation": s.values})
    out["abs_correlation"] = out["correlation"].abs()
    return out.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def medication_usage(meds: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most-prescribed drugs. Drives pharmacy stock planning."""
    if meds.empty or "medication_name" not in meds.columns:
        return pd.DataFrame()
    out = (
        meds.groupby("medication_name")
        .agg(
            prescriptions=("patient_id", "count"),
            unique_patients=("patient_id", "nunique"),
        )
        .reset_index()
        .sort_values("prescriptions", ascending=False)
    )
    # This dataset records no dosage in mg. What it records instead is whether
    # the dose was held steady, increased or decreased during the stay - which
    # is arguably more useful, because a dose change means a clinician actively
    # judged the patient's control to be wrong.
    if "dose_status" in meds.columns:
        adjusted = (
            meds.assign(_adj=meds["dose_status"].isin(["Up", "Down"]).astype(int))
            .groupby("medication_name")["_adj"].mean()
            .mul(100).round(1).rename("pct_dose_adjusted")
        )
        out = out.merge(adjusted, on="medication_name", how="left")
    return out.head(top_n).reset_index(drop=True)


def drug_class_usage(meds: pd.DataFrame) -> pd.DataFrame:
    """Prescriptions grouped by therapeutic class."""
    if meds.empty or "drug_class" not in meds.columns:
        return pd.DataFrame()
    return (
        meds.groupby("drug_class")
        .size()
        .reset_index(name="prescriptions")
        .sort_values("prescriptions", ascending=False)
        .reset_index(drop=True)
    )


def polypharmacy_vs_readmission(df: pd.DataFrame) -> pd.DataFrame:
    """Readmission rate by number-of-medications band.

    "Polypharmacy" (many concurrent drugs) is a recognised readmission risk
    factor - it is a proxy for how many conditions a patient is managing at once.
    """
    if "num_medications" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["med_band"] = pd.cut(
        d["num_medications"],
        bins=[0, 5, 10, 15, 20, 100],
        labels=["1-5", "6-10", "11-15", "16-20", "20+"],
    )
    return readmission_by_group(d, "med_band", min_count=10)


def risk_factor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Readmission rate with vs without each binary risk factor, plus lift.

    `lift` = rate WITH the factor / rate WITHOUT it. A lift of 2.0 means the
    factor doubles the readmission rate. This table is the clearest single piece
    of evidence in the whole dashboard, and it is what Module 5's GenAI
    explanations will be grounded in.
    """
    if "readmitted_30d" not in df.columns:
        return pd.DataFrame()

    factors = {
        # Prior utilisation - consistently the strongest family of predictors
        # in the readmission literature, and this dataset records three kinds.
        "Prior inpatient admission": df["num_prior_admissions"] >= 1
        if "num_prior_admissions" in df else None,
        "Prior emergency visit": df["num_prior_emergency"] >= 1
        if "num_prior_emergency" in df else None,
        "Prior outpatient visit": df["num_prior_outpatient"] >= 1
        if "num_prior_outpatient" in df else None,
        "Repeat patient (not first visit)": df["is_repeat_patient"] == 1
        if "is_repeat_patient" in df else None,

        # Severity and complexity of this admission
        "Long stay (> 7 days)": df["length_of_stay"] > 7
        if "length_of_stay" in df else None,
        "Complex case (9+ diagnoses)": df["num_diagnoses"] >= 9
        if "num_diagnoses" in df else None,
        "Polypharmacy (> 20 meds)": df["num_medications"] > 20
        if "num_medications" in df else None,
        "Emergency admission": df["admission_type"] == "Emergency"
        if "admission_type" in df else None,

        # Demographics
        "Age 65 or over": df["age"] >= 65 if "age" in df else None,

        # Glycaemic control - this dataset's own research question
        "HbA1c tested during stay": df["a1c_tested"] == 1
        if "a1c_tested" in df else None,
        "HbA1c high (> 7)": df["a1c_high"] == 1 if "a1c_high" in df else None,
        "Serum glucose high (> 200)": df["glucose_high"] == 1
        if "glucose_high" in df else None,
        "Diabetes medication prescribed": df["on_diabetes_meds"] == 1
        if "on_diabetes_meds" in df else None,
        "Medication regimen changed": df["medication_changed"] == 1
        if "medication_changed" in df else None,

        # Discharge destination - a proxy for how well the patient was doing
        "Discharged to care facility": df["to_care_facility"] == 1
        if "to_care_facility" in df else None,
    }

    rows = []
    target = pd.to_numeric(df["readmitted_30d"], errors="coerce")
    for name, mask in factors.items():
        if mask is None:
            continue
        mask = mask.fillna(False).astype(bool)
        if mask.sum() < 20 or (~mask).sum() < 20:
            continue
        with_rate = float(target[mask].mean() * 100)
        without_rate = float(target[~mask].mean() * 100)
        rows.append(
            {
                "risk_factor": name,
                "patients_with_factor": int(mask.sum()),
                "readmit_rate_with": round(with_rate, 2),
                "readmit_rate_without": round(without_rate, 2),
                "difference_pp": round(with_rate - without_rate, 2),
                "lift": round(with_rate / without_rate, 2) if without_rate else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("lift", ascending=False).reset_index(drop=True) if len(out) else out


def department_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-department operational view: volume, bed days, readmission rate."""
    if "department" not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby("department")
        .agg(
            patients=("patient_id", "count"),
            avg_los=("length_of_stay", "mean"),
            total_bed_days=("length_of_stay", "sum"),
            readmission_rate=("readmitted_30d", "mean"),
            avg_age=("age", "mean"),
        )
        .reset_index()
    )
    out["avg_los"] = out["avg_los"].round(2)
    out["readmission_rate"] = (out["readmission_rate"] * 100).round(2)
    out["avg_age"] = out["avg_age"].round(1)
    return out.sort_values("patients", ascending=False).reset_index(drop=True)


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """A tidied-up `df.describe()` for the numeric clinical columns."""
    cols = [c for c in NUMERIC_FEATURES if c in df.columns and c != "readmitted_30d"]
    out = df[cols].describe().T.round(2)
    out.index.name = "variable"
    return out.reset_index()
