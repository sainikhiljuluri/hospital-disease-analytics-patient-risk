"""
Module 2 (part 1): turn the raw public dataset into a trustworthy analysis table.

WHY THIS FILE EXISTS
--------------------
Real data is messy in specific, documented ways, and this dataset is no
exception. Roughly 97% of `weight` values are missing. About half the
`medical_specialty` values are missing. Missing values are encoded as the string
"?" rather than as blanks, so pandas reads them as ordinary text unless you say
otherwise. None of that is a flaw in the dataset - it is what real clinical
extracts look like, and handling it is the job.

DESIGN RULE THAT MATTERS
------------------------
`clean_patients()` returns BOTH the cleaned DataFrame and a report dictionary.
The report is not thrown away - the Streamlit "Data Quality" page displays it,
so the dashboard can PROVE what was fixed instead of just asserting it.

THE DEFECTS IN THIS DATASET, AND WHAT WE DO ABOUT EACH
------------------------------------------------------
 1. "?" used as the missing-value marker      -> converted to real nulls
 2. `weight` ~97% missing                     -> DROPPED, with the number shown
 3. `medical_specialty` ~49% missing          -> kept as an explicit "Unknown"
 4. `payer_code` ~40% missing                 -> dropped (not clinically useful)
 5. Duplicate encounters                      -> removed on encounter id
 6. Repeat patients across rows               -> KEPT, and flagged (see below)
 7. Age only as 10-year bands                 -> midpoint used, band retained
 8. Gender "Unknown/Invalid"                  -> treated as missing
 9. Deceased/hospice patients                 -> removed in the adapter
10. Impossible numerics after conversion      -> nulled, then median-imputed

WHY REPEAT PATIENTS ARE KEPT BUT FLAGGED
----------------------------------------
The same person can appear many times, because each row is an ENCOUNTER. Those
are not duplicates - they are separate hospital visits, and each one is a real
observation. But they do break the assumption that rows are independent, which
matters when Module 4 splits the data: if the same person lands in both the
training and test sets, the model can memorise them and the score is inflated.
So we flag `is_repeat_patient` and keep `person_id`, and Module 4 must group its
split by person. Deleting them would throw away the strongest signal in the
dataset; ignoring the issue would produce a dishonest accuracy number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import CLINICAL_RANGES

# Maps possible source spellings onto canonical names. Supporting a new dataset
# means adding entries here (plus an adapter), not editing analysis code.
COLUMN_ALIASES: dict[str, str] = {
    "encounter_id": "patient_id",
    "patientid": "patient_id",
    "patient_nbr": "person_id",
    "sex": "gender",
    "time_in_hospital": "length_of_stay",
    "stay_days": "length_of_stay",
    "number_inpatient": "num_prior_admissions",
    "number_emergency": "num_prior_emergency",
    "number_outpatient": "num_prior_outpatient",
    "number_diagnoses": "num_diagnoses",
    "readmitted": "readmitted_30d",
    "medical_specialty": "department",
    "a1cresult": "a1c_result",
    "max_glu_serum": "glucose_serum",
}

NUMERIC_COLS = [
    "age", "length_of_stay", "num_lab_procedures", "num_procedures",
    "num_medications", "num_diagnoses", "num_prior_admissions",
    "num_prior_emergency", "num_prior_outpatient", "num_prior_visits_total",
    "encounter_seq", "is_repeat_patient", "a1c_tested", "glucose_tested",
    "medication_changed", "on_diabetes_meds", "readmitted_30d", "readmitted_any",
]

TEXT_COLS = [
    "gender", "race", "diagnosis", "diagnosis_secondary", "diagnosis_tertiary",
    "admission_type", "admission_source", "discharge_disposition", "department",
    "a1c_result", "glucose_serum", "age_group", "icd9_primary",
]

# Columns so incomplete that keeping them would mislead more than help.
DROP_IF_PRESENT = ["weight", "payer_code"]

REQUIRED_COLS = ["patient_id", "diagnosis", "readmitted_30d"]

# Terms that a naive .title() call would mangle.
ACRONYM_FIXES = {
    "Icu": "ICU", "Snf": "SNF", "Hmo": "HMO", "Iv": "IV",
    "Obstetricsandgynecology": "Obstetrics and Gynaecology",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip and alias every column name."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    return df


def _fix_acronyms(series: pd.Series) -> pd.Series:
    out = series
    for wrong, right in ACRONYM_FIXES.items():
        out = out.str.replace(rf"\b{wrong}\b", right, regex=True)
    return out


def _clean_text(series: pd.Series) -> pd.Series:
    """Trim whitespace and unify every flavour of 'missing' into a real null."""
    out = series.astype("string").str.strip()
    return out.replace(
        {
            "?": pd.NA, "": pd.NA, "nan": pd.NA, "NULL": pd.NA,
            "Unknown/Invalid": pd.NA, "Not Available": pd.NA, "Not Mapped": pd.NA,
        }
    )


def _split_camel_case(series: pd.Series) -> pd.Series:
    """`InternalMedicine` -> `Internal Medicine`.

    The dataset writes speciality names without spaces. Left alone they render
    unreadably on chart axes.
    """
    return series.str.replace(r"(?<=[a-z])(?=[A-Z])", " ", regex=True)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per numeric column: how many values are impossible vs merely unusual.

    The two are handled differently and the distinction is the single best thing
    to be able to explain about this pipeline:

      * OUT OF RANGE means the value contradicts the dataset's own documented
        limits - a 40-day stay when only 1-14 day stays were included. That is a
        corrupted file, so the value is removed.
      * IQR OUTLIER means statistically unusual but entirely possible - a patient
        on 70 medications. That is a genuinely complex case, so it is KEPT.
        Deleting these would teach the model that severe cases do not exist.
    """
    rows = []
    for col, (low, high) in CLINICAL_RANGES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "column": col,
                "valid_range": f"{low} - {high}",
                "out_of_range": int((~series.between(low, high) & series.notna()).sum()),
                "iqr_lower": round(float(lo_f), 2),
                "iqr_upper": round(float(hi_f), 2),
                "iqr_outliers": int(((series < lo_f) | (series > hi_f)).sum()),
            }
        )
    return pd.DataFrame(rows)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: how many values are missing, and what share."""
    total = len(df)
    rows = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing": n_missing,
                "missing_pct": round(100 * n_missing / total, 2) if total else 0.0,
                "dtype": str(df[col].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values("missing", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The main cleaning routine
# ---------------------------------------------------------------------------
def clean_patients(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the adapted patients table and report every change made."""
    report: dict[str, Any] = {}
    df = _normalise_columns(raw)

    report["rows_in"] = len(df)
    report["cols_in"] = len(df.columns)
    report["missing_before"] = missing_value_report(df)
    report["outliers_before"] = outlier_report(df)

    # --- 1. Drop columns too incomplete to be trustworthy -----------------
    dropped = []
    for col in DROP_IF_PRESENT:
        if col in df.columns:
            pct = round(100 * float(df[col].isna().mean()), 1)
            dropped.append({"column": col, "missing_pct": pct})
            df = df.drop(columns=[col])
    report["columns_dropped"] = dropped

    # --- 2. Text tidy-up ---------------------------------------------------
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = _clean_text(df[col])

    if "department" in df.columns:
        dept = df["department"].fillna("Unknown").astype("string")
        dept = _split_camel_case(dept)
        df["department"] = _fix_acronyms(dept.str.strip()).fillna("Unknown")

    # --- 3. Numeric coercion ----------------------------------------------
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 4. Duplicate encounters ------------------------------------------
    before = len(df)
    if "patient_id" in df.columns:
        df = df.drop_duplicates(subset=["patient_id"], keep="first")
    report["duplicate_encounters_removed"] = before - len(df)

    # Repeat visits by the same person are NOT duplicates - report, don't drop.
    if "person_id" in df.columns:
        report["unique_people"] = int(df["person_id"].nunique())
        report["repeat_encounters"] = int(len(df) - df["person_id"].nunique())

    # --- 5. Impossible values ---------------------------------------------
    # Out-of-range here means a corrupted file, not an unusual patient, because
    # the dataset's own inclusion criteria already bound these columns.
    impossible = 0
    range_hits: dict[str, int] = {}
    for col, (low, high) in CLINICAL_RANGES.items():
        if col not in df.columns:
            continue
        bad = ~df[col].between(low, high) & df[col].notna()
        n_bad = int(bad.sum())
        if n_bad:
            range_hits[col] = n_bad
            impossible += n_bad
            df.loc[bad, col] = np.nan
    report["impossible_values_nulled"] = impossible
    report["impossible_by_column"] = range_hits

    # --- 6. Impute remaining numeric gaps with the MEDIAN -----------------
    # Median, not mean: these distributions are right-skewed (a few patients
    # have 60+ medications), and the mean would be dragged upward by them.
    imputed: dict[str, dict[str, float]] = {}
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = float(df[col].median())
            df[col] = df[col].fillna(median)
            imputed[col] = {"count": n_missing, "value": round(median, 2)}
    report["imputed"] = imputed

    # --- 7. Categorical gaps become an explicit 'Unknown' -----------------
    # We do NOT guess someone's gender or ethnicity. "Unknown" is the truthful
    # value, and it stays visible in every chart so nobody mistakes an imputed
    # value for a recorded one.
    cat_filled: dict[str, int] = {}
    for col in TEXT_COLS:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                df[col] = df[col].fillna("Unknown")
                cat_filled[col] = n_missing
    report["categorical_unknown_filled"] = cat_filled

    # --- 8. Engineered features -------------------------------------------
    df = add_derived_features(df)

    # --- 9. Drop rows still missing something essential -------------------
    before = len(df)
    present_required = [c for c in REQUIRED_COLS if c in df.columns]
    if present_required:
        df = df.dropna(subset=present_required)
    report["rows_dropped_missing_key"] = before - len(df)

    df = df.reset_index(drop=True)
    report["rows_out"] = len(df)
    report["cols_out"] = len(df.columns)
    report["missing_after"] = missing_value_report(df)
    report["outliers_after"] = outlier_report(df)
    report["readmission_rate"] = (
        round(100 * float(df["readmitted_30d"].mean()), 2)
        if "readmitted_30d" in df.columns and len(df) else None
    )
    return df, report


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns the analysis and dashboard rely on.

    Kept separate so Module 4 can call it on new incoming data and get exactly
    the same features the model was trained on. Feature drift between training
    and serving is one of the commonest ways a deployed model quietly breaks.
    """
    df = df.copy()

    if "age" in df.columns:
        df["is_elderly"] = (df["age"] >= 65).astype(int)

    if "length_of_stay" in df.columns:
        df["stay_band"] = pd.cut(
            df["length_of_stay"],
            bins=[0, 3, 7, 100],
            labels=["Short (1-3d)", "Medium (4-7d)", "Long (8d+)"],
        ).astype("string")
        df["is_long_stay"] = (df["length_of_stay"] > 7).astype(int)

    if "num_medications" in df.columns:
        df["med_band"] = pd.cut(
            df["num_medications"],
            bins=[0, 10, 20, 30, 200],
            labels=["1-10", "11-20", "21-30", "31+"],
        ).astype("string")
        df["is_polypharmacy"] = (df["num_medications"] > 20).astype(int)

    if "num_prior_admissions" in df.columns:
        df["has_prior_admission"] = (df["num_prior_admissions"] > 0).astype(int)
        df["prior_admission_band"] = pd.cut(
            df["num_prior_admissions"],
            bins=[-1, 0, 1, 3, 1000],
            labels=["0", "1", "2-3", "4+"],
        ).astype("string")

    if "num_prior_emergency" in df.columns:
        df["has_prior_emergency"] = (df["num_prior_emergency"] > 0).astype(int)

    if "num_diagnoses" in df.columns:
        df["is_complex"] = (df["num_diagnoses"] >= 9).astype(int)

    # HbA1c is this dataset's headline research question. "Not Tested" is a
    # meaningful third state, not a missing value.
    if "a1c_result" in df.columns:
        df["a1c_high"] = df["a1c_result"].isin([">7", ">8"]).astype(int)

    if "glucose_serum" in df.columns:
        df["glucose_high"] = df["glucose_serum"].isin([">200", ">300"]).astype(int)

    if "discharge_disposition" in df.columns:
        df["discharged_home"] = (
            df["discharge_disposition"].astype(str).str.startswith("Home")
        ).astype(int)
        df["to_care_facility"] = df["discharge_disposition"].isin(
            ["Skilled Nursing Facility", "Intermediate Care Facility",
             "Nursing Facility", "Long Term Care Hospital", "Rehabilitation Facility"]
        ).astype(int)

    if "admission_type" in df.columns:
        df["is_emergency"] = (df["admission_type"] == "Emergency").astype(int)

    return df


def clean_medications(
    meds: pd.DataFrame, valid_patient_ids: pd.Series | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tidy the long medications table and enforce the foreign key."""
    report: dict[str, Any] = {"rows_in": len(meds)}
    df = _normalise_columns(meds)

    for col in ("medication_name", "drug_class", "dose_status"):
        if col in df.columns:
            df[col] = _clean_text(df[col]).fillna("Unknown")

    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    if valid_patient_ids is not None and "patient_id" in df.columns:
        allowed = set(pd.Series(valid_patient_ids).astype(str))
        before = len(df)
        df = df[df["patient_id"].isin(allowed)]
        # Orphans would violate the MySQL foreign key. Removing them here means
        # the database load never fails halfway through.
        report["orphans_removed"] = before - len(df)

    df = df.reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report
