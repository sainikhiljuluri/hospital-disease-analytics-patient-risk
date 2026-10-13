"""
Module 1 (part 2): the ETL job. Raw CSV -> clean -> MySQL.

ETL = Extract, Transform, Load. That is literally what `run_pipeline()` does:
  EXTRACT   read the CSV files from data/raw/
  TRANSFORM run them through ingestion/cleaning.py
  LOAD      write them into MySQL, and also save a cleaned CSV copy

WHY THE CLEANED CSV COPY?
-------------------------
So the dashboard has something to fall back to when MySQL is unreachable. It
costs one line and saves your demo.

LOAD ORDER MATTERS
------------------
disease_reference -> patients -> medications.
Foreign keys enforce that order: a patient row cannot reference a diagnosis that
does not exist yet, and a medication cannot reference a missing patient. Load
them out of order and MySQL rejects the rows.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from ..config import (
    DATASET_NAME,
    DATASET_SOURCE_FILE,
    RAW_DIR,
    PROCESSED_MEDS_CSV,
    PROCESSED_PATIENTS_CSV,
    RAW_FILES,
    TBL_DISEASE_REF,
    TBL_LOAD_LOG,
    TBL_MEDICATIONS,
    TBL_PATIENTS,
)
from ..db import DatabaseUnavailable, get_engine, row_count, table_exists
from .adapters import uci_diabetes
from .cleaning import clean_medications, clean_patients

# Only these columns are written to MySQL - they must match schema.sql exactly.
# Extra derived columns (admission_month, etc.) are recreated on read instead of
# stored, so the database holds facts and the app holds presentation logic.
PATIENT_DB_COLUMNS = [
    # identity
    "patient_id", "person_id", "encounter_seq", "is_repeat_patient",
    # demographics
    "age", "age_group", "gender", "race",
    # diagnosis
    "diagnosis", "diagnosis_secondary", "diagnosis_tertiary", "icd9_primary",
    "num_diagnoses",
    # admission context
    "admission_type", "admission_source", "discharge_disposition", "department",
    # utilisation this stay
    "length_of_stay", "num_lab_procedures", "num_procedures", "num_medications",
    # utilisation before this stay
    "num_prior_admissions", "num_prior_emergency", "num_prior_outpatient",
    "num_prior_visits_total",
    # glycaemic control
    "a1c_result", "glucose_serum", "a1c_tested", "glucose_tested",
    # treatment response
    "medication_changed", "on_diabetes_meds",
    # targets
    "readmitted_30d", "readmitted_any",
]

MEDICATION_DB_COLUMNS = [
    "patient_id", "medication_name", "drug_class", "dose_status",
]

REFERENCE_DB_COLUMNS = [
    "diagnosis", "icd9_range", "disease_category", "is_chronic", "department",
]


def _log_load(
    source_file: str, table: str, rows_in: int, rows_loaded: int,
    rows_rejected: int, status: str, message: str = "",
) -> None:
    """Append one row to data_load_log. Never allowed to break the pipeline."""
    try:
        entry = pd.DataFrame([{
            "source_file": str(source_file)[:255],
            "target_table": table,
            "rows_in_file": int(rows_in),
            "rows_loaded": int(rows_loaded),
            "rows_rejected": int(rows_rejected),
            "status": status,
            "message": str(message)[:2000],
        }])
        entry.to_sql(TBL_LOAD_LOG, get_engine(), if_exists="append", index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not write to {TBL_LOAD_LOG}: {exc})")


def _truncate(table: str) -> None:
    """Empty a table before a fresh load, so re-running is safe (idempotent).

    Foreign keys are disabled for the moment of truncation, because MySQL will
    not let you truncate a parent table while children reference it.
    """
    from sqlalchemy import text

    with get_engine().begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        conn.execute(text(f"TRUNCATE TABLE `{table}`"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))


def _write(df: pd.DataFrame, table: str, chunksize: int = 1000) -> int:
    """Insert a DataFrame into MySQL in chunks.

    Chunking matters: a single INSERT with 10,000 rows can exceed MySQL's
    max_allowed_packet and fail. 1,000-row batches are safe and still fast.
    """
    df.to_sql(table, get_engine(), if_exists="append", index=False,
              chunksize=chunksize, method="multi")
    return len(df)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def run_pipeline(to_mysql: bool = True, verbose: bool = True) -> dict[str, Any]:
    """Run the full ETL. Returns a summary dict.

    `to_mysql=False` runs Extract + Transform only and just writes the cleaned
    CSVs - useful for testing the cleaning rules without a database.
    """
    summary: dict[str, Any] = {}

    def say(msg: str = "") -> None:
        if verbose:
            print(msg)

    # ---------------- EXTRACT ----------------
    # The adapter reads the real downloaded dataset and reshapes it into the
    # project's three-table schema. If the download is missing it raises with
    # step-by-step instructions rather than a bare stack trace.
    say("=" * 66)
    say("STEP 1/4  EXTRACT - reading the source dataset")
    say("=" * 66)
    say(f"  dataset: {DATASET_NAME}")
    say(f"  file:    {RAW_DIR / DATASET_SOURCE_FILE}")

    raw_patients, raw_meds, ref, adapt_report = uci_diabetes.convert(RAW_DIR)
    summary["adapter"] = adapt_report

    say(f"  encounters read       {adapt_report['rows_in']:>7,}")
    say(f"  deceased/hospice removed {adapt_report['expired_or_hospice_removed']:>4,}")
    say(f"  encounters kept       {adapt_report['rows_out']:>7,}")
    say(f"  unique people         {adapt_report['unique_people']:>7,}")
    say(f"  prescriptions (melted){adapt_report['medication_rows']:>7,}")
    say(f"  reference groups      {len(ref):>7,}")
    for col in adapt_report.get("incomplete_columns", []):
        note = "DROPPED" if col["column"] in ("weight", "payer_code") else "kept as Unknown"
        say(f"    {col['column']:<20} {col['missing_pct']:>5.1f}% missing  -> {note}")

    # ---------------- TRANSFORM ----------------
    say("")
    say("=" * 66)
    say("STEP 2/4  TRANSFORM - cleaning and validating")
    say("=" * 66)
    patients, prep = clean_patients(raw_patients)
    meds, mrep = clean_medications(raw_meds, valid_patient_ids=patients["patient_id"])

    for dropped in prep["columns_dropped"]:
        say(f"  dropped column {dropped['column']:<16} ({dropped['missing_pct']}% missing)")
    say(f"  duplicate encounters removed {prep['duplicate_encounters_removed']:>7,}")
    say(f"  unique people                {prep.get('unique_people', 0):>7,}")
    say(f"  repeat encounters kept       {prep.get('repeat_encounters', 0):>7,}")
    say(f"  impossible values nulled     {prep['impossible_values_nulled']:>7,}")
    for col, info in prep["imputed"].items():
        say(f"    imputed {col:<20} {info['count']:>5,} -> median {info['value']}")
    for col, n in prep["categorical_unknown_filled"].items():
        say(f"    '{col}' set to Unknown    {n:>5,} rows")
    say(f"  rows dropped (missing key)   {prep['rows_dropped_missing_key']:>7,}")
    say(f"  clean encounter rows         {prep['rows_out']:>7,}")
    say(f"  30-day readmission rate      {prep['readmission_rate']:>7}%")
    say(f"  orphan medications removed   {mrep['orphans_removed']:>7,}")
    say(f"  clean medication rows        {mrep['rows_out']:>7,}")
    summary["patients_report"] = prep
    summary["medications_report"] = mrep

    # ---------------- SAVE CLEANED CSV (the fallback source) ----------------
    patients.to_csv(PROCESSED_PATIENTS_CSV, index=False)
    meds.to_csv(PROCESSED_MEDS_CSV, index=False)
    say("")
    say(f"  saved  {PROCESSED_PATIENTS_CSV}")
    say(f"  saved  {PROCESSED_MEDS_CSV}")

    if not to_mysql:
        say("\n--to-mysql not requested: stopping after cleaning.")
        summary["loaded_to_mysql"] = False
        return summary

    # ---------------- LOAD ----------------
    say("")
    say("=" * 66)
    say("STEP 3/4  LOAD - writing to MySQL")
    say("=" * 66)
    try:
        get_engine()
    except DatabaseUnavailable as exc:
        say(f"  SKIPPED: {exc}")
        say("  The cleaned CSV files are ready, so the dashboard will still work.")
        summary["loaded_to_mysql"] = False
        return summary

    # Pre-flight check: confirm the tables exist BEFORE touching them. Without
    # this the first TRUNCATE fails with a raw driver error naming one table,
    # which does not tell you that the real problem is a missing schema.
    required = [TBL_DISEASE_REF, TBL_PATIENTS, TBL_MEDICATIONS, TBL_LOAD_LOG]
    absent = [t for t in required if not table_exists(t)]
    if absent:
        say("  SKIPPED: the database tables have not been created yet.")
        say(f"           missing: {', '.join(absent)}")
        say("")
        say("  Run this first:")
        say("      python scripts/02_setup_database.py")
        say("")
        say("  The cleaned CSV files are ready, so the dashboard works either way.")
        summary["loaded_to_mysql"] = False
        summary["missing_tables"] = absent
        return summary

    try:
        # 1. lookup table (parent of patients)
        _truncate(TBL_MEDICATIONS)
        _truncate(TBL_PATIENTS)
        _truncate(TBL_DISEASE_REF)

        n = _write(ref, TBL_DISEASE_REF)
        say(f"  {TBL_DISEASE_REF:<20} {n:>7,} rows")
        _log_load(DATASET_SOURCE_FILE, TBL_DISEASE_REF, len(ref), n, 0, "SUCCESS")

        # 2. patients
        db_patients = patients[[c for c in PATIENT_DB_COLUMNS if c in patients.columns]].copy()
        # MySQL wants NULL, not pandas NaT/NA
        db_patients = db_patients.astype(object).where(pd.notna(db_patients), None)
        n = _write(db_patients, TBL_PATIENTS)
        say(f"  {TBL_PATIENTS:<20} {n:>7,} rows")
        _log_load(DATASET_SOURCE_FILE, TBL_PATIENTS,
                  len(raw_patients), n, len(raw_patients) - n, "SUCCESS")

        # 3. medications
        db_meds = meds[[c for c in MEDICATION_DB_COLUMNS if c in meds.columns]].copy()
        db_meds = db_meds.astype(object).where(pd.notna(db_meds), None)
        n = _write(db_meds, TBL_MEDICATIONS)
        say(f"  {TBL_MEDICATIONS:<20} {n:>7,} rows")
        _log_load(DATASET_SOURCE_FILE, TBL_MEDICATIONS,
                  len(raw_meds), n, mrep["orphans_removed"], "SUCCESS")

        summary["loaded_to_mysql"] = True
    except Exception as exc:  # noqa: BLE001
        say(f"  LOAD FAILED: {exc}")
        _log_load("multiple", "multiple", 0, 0, 0, "FAILED", str(exc))
        summary["loaded_to_mysql"] = False
        summary["error"] = str(exc)
        return summary

    # ---------------- VERIFY ----------------
    say("")
    say("=" * 66)
    say("STEP 4/4  VERIFY - counting rows in MySQL")
    say("=" * 66)
    for table in (TBL_DISEASE_REF, TBL_PATIENTS, TBL_MEDICATIONS, TBL_LOAD_LOG):
        say(f"  {table:<20} {row_count(table):>7,} rows")

    say("")
    say("Pipeline complete. Next:  streamlit run streamlit_app/app.py")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Load hospital data into MySQL.")
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Clean the data and write CSVs, but skip the MySQL load.",
    )
    args = parser.parse_args()
    run_pipeline(to_mysql=not args.csv_only)


if __name__ == "__main__":
    main()
