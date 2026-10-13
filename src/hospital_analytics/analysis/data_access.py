"""
The one function every consumer calls to get data: `load_patients()`.

WHY THIS FILE EXISTS
--------------------
The notebook, the Streamlit dashboard and (in Module 4/5) the FastAPI service all
need the same analysis table. Without this file each of them would write its own
SQL, and they would slowly drift apart. With it, there is exactly one definition
of "the patient dataset".

THE FALLBACK CHAIN - the most important idea in this file
--------------------------------------------------------
    1. Try MySQL          (the real, intended source)
    2. Fall back to the cleaned CSV in data/processed/
    3. Fall back to cleaning the raw CSV on the fly

Reason: on demo day, MySQL will be the thing that breaks - wrong password, service
not started, laptop is not the one you set it up on. The dashboard degrades
gracefully instead of showing a stack trace, and it tells you which source it
used, so you are never accidentally presenting stale data without knowing.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import (
    PROCESSED_MEDS_CSV,
    PROCESSED_PATIENTS_CSV,
    RAW_DIR,
    TBL_DISEASE_REF,
    TBL_MEDICATIONS,
    TBL_PATIENTS,
)
from ..db import DatabaseUnavailable, read_sql, table_exists

logger = logging.getLogger(__name__)

def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee correct dtypes and derived columns whatever the source.

    MySQL and CSV return slightly different types for the same column, so both
    paths are normalised here. That is why a chart written against
    `load_patients()` behaves identically no matter which fallback fired.

    Note there is no date handling: this dataset is de-identified and contains
    no admission dates at all. `encounter_seq` is the ordering column instead.
    """
    df = df.copy()

    numeric = [
        "age", "length_of_stay", "num_lab_procedures", "num_procedures",
        "num_medications", "num_diagnoses", "num_prior_admissions",
        "num_prior_emergency", "num_prior_outpatient", "num_prior_visits_total",
        "encounter_seq", "is_repeat_patient", "a1c_tested", "glucose_tested",
        "a1c_high", "glucose_high", "medication_changed", "on_diabetes_meds",
        "is_elderly", "is_long_stay", "is_polypharmacy", "has_prior_admission",
        "has_prior_emergency", "is_complex", "discharged_home", "to_care_facility",
        "is_emergency", "is_chronic", "readmitted_30d", "readmitted_any",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("patient_id", "person_id"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Re-derive engineered features if they are absent, so the dashboard works
    # even against a database loaded by an older version of the pipeline.
    try:
        from ..ingestion.cleaning import add_derived_features

        if "is_elderly" not in df.columns and "age" in df.columns:
            df = add_derived_features(df)
    except Exception:
        pass

    return df


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------
def load_patients_from_mysql() -> pd.DataFrame:
    """Read patients joined to the disease reference lookup.

    The LEFT JOIN is why we bothered normalising the schema: disease_category and
    is_chronic live in one place, and every query gets them for free.
    """
    if not table_exists(TBL_PATIENTS):
        raise DatabaseUnavailable(f"Table '{TBL_PATIENTS}' does not exist yet.")

    join = ""
    select_extra = ""
    if table_exists(TBL_DISEASE_REF):
        select_extra = ", d.disease_category, d.icd10_code, d.is_chronic"
        join = f"LEFT JOIN {TBL_DISEASE_REF} d ON p.diagnosis = d.diagnosis"

    return read_sql(f"SELECT p.*{select_extra} FROM {TBL_PATIENTS} p {join}")


def load_patients_from_csv() -> pd.DataFrame:
    """Read the cleaned CSV, adapting and cleaning the raw download if needed.

    Three tiers, in order of preference:
      1. the cleaned CSV written by the ETL (fast, and what usually happens)
      2. the raw download, adapted and cleaned on the fly
      3. a clear error explaining how to get the dataset
    """
    if PROCESSED_PATIENTS_CSV.exists():
        df = pd.read_csv(PROCESSED_PATIENTS_CSV)
    else:
        from ..ingestion.adapters import uci_diabetes
        from ..ingestion.cleaning import clean_patients

        # Raises DatasetNotFoundError with download instructions if absent.
        raw, _, _, _ = uci_diabetes.convert(RAW_DIR)
        df, _ = clean_patients(raw)

    # Attach the lookup columns so the CSV path matches the MySQL path exactly.
    if "disease_category" not in df.columns:
        from ..ingestion.adapters import uci_diabetes

        ref = uci_diabetes.build_disease_reference()
        # `department` exists on both sides; keep the encounter's own value.
        ref = ref.drop(columns=[c for c in ("department",) if c in df.columns])
        df = df.merge(ref, on="diagnosis", how="left")
    return df


def load_patients(prefer: str = "mysql") -> tuple[pd.DataFrame, str]:
    """Load the patient analysis table.

    Parameters
    ----------
    prefer : "mysql" | "csv"

    Returns
    -------
    (DataFrame, source_label)
        source_label is shown in the dashboard sidebar, e.g. "MySQL" or
        "CSV fallback", so you always know what you are looking at.
    """
    if prefer == "mysql":
        try:
            df = load_patients_from_mysql()
            if len(df):
                return _post_process(df), "MySQL"
            logger.warning("MySQL table is empty; falling back to CSV.")
        except (DatabaseUnavailable, Exception) as exc:  # noqa: BLE001
            logger.warning("MySQL unavailable (%s); falling back to CSV.", exc)

    return _post_process(load_patients_from_csv()), "CSV fallback"


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------
def load_medications(prefer: str = "mysql") -> tuple[pd.DataFrame, str]:
    """Load the medications table, with the same fallback chain."""
    if prefer == "mysql":
        try:
            if table_exists(TBL_MEDICATIONS):
                df = read_sql(f"SELECT * FROM {TBL_MEDICATIONS}")
                if len(df):
                    return df, "MySQL"
        except Exception as exc:  # noqa: BLE001
            logger.warning("MySQL medications unavailable (%s).", exc)

    for path in (PROCESSED_MEDS_CSV,):
        if path.exists():
            return pd.read_csv(path), "CSV fallback"
    return pd.DataFrame(columns=["patient_id", "medication_name", "drug_class", "dose_status"]), "empty"


# ---------------------------------------------------------------------------
# Example of pushing work INTO the database
# ---------------------------------------------------------------------------
def readmission_rate_by_diagnosis_sql() -> pd.DataFrame:
    """Same answer as a pandas groupby, but computed by MySQL.

    Keep this in your demo: it proves you can express analytics in SQL as well as
    in pandas, and it is the pattern you would use at real scale (aggregate 50
    million rows in the database, ship 12 rows to Python).
    """
    return read_sql(
        f"""
        SELECT  p.diagnosis,
                d.disease_category,
                COUNT(*)                          AS total_patients,
                SUM(p.readmitted_30d)             AS readmissions,
                ROUND(AVG(p.readmitted_30d) * 100, 2) AS readmission_rate_pct,
                ROUND(AVG(p.length_of_stay), 2)   AS avg_length_of_stay,
                ROUND(AVG(p.age), 1)              AS avg_age
        FROM        {TBL_PATIENTS} p
        LEFT JOIN   {TBL_DISEASE_REF} d ON p.diagnosis = d.diagnosis
        GROUP BY    p.diagnosis, d.disease_category
        ORDER BY    readmission_rate_pct DESC
        """
    )
