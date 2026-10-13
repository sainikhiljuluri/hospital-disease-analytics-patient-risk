"""
Adapter for the "Diabetes 130-US Hospitals for Years 1999-2008" dataset.

THE DATASET
-----------
    Source   : UCI Machine Learning Repository, dataset 296
               (also mirrored on Kaggle as `brandao/diabetes`)
    Citation : Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ,
               Clore JN. "Impact of HbA1c Measurement on Hospital Readmission
               Rates: Analysis of 70,000 Clinical Database Patient Records."
               BioMed Research International, vol. 2014, Article ID 781670.
    DOI      : 10.24432/C5230J
    Licence  : Creative Commons Attribution 4.0 (CC BY 4.0) - free to use and
               adapt for any purpose, including coursework, with credit.
    Size     : 101,766 encounters x 50 columns (18.3 MB uncompressed)

Ten years of real inpatient care from 130 US hospitals. Every row is one
hospital ENCOUNTER (an admission), not one person - the same patient can appear
several times, which matters and is handled below.

WHAT THIS FILE DOES
-------------------
Translates the dataset's vocabulary into the project's vocabulary, and splits
one wide 50-column table into the three normalised tables the schema expects:

    diabetic_data.csv  ->  patients.csv           (one row per encounter)
                       ->  medications.csv        (one row per drug given)
                       ->  disease_reference.csv  (one row per ICD-9 group)

WHY THE MEDICATION SPLIT MATTERS
--------------------------------
The raw file stores 23 drugs as 23 SEPARATE COLUMNS, each holding "No",
"Steady", "Up" or "Down". That is called wide format, and it is unusable: you
cannot ask "how many patients received insulin?" without naming the column.

This adapter melts those 23 columns into a long table with one row per
(patient, drug) pair. That is what makes a one-to-many relationship queryable,
and it is why the project has a separate `medications` table at all.

HONEST LIMITATIONS - KNOW THESE BEFORE YOUR DEMO
------------------------------------------------
This is real data, so it does NOT contain everything a textbook project wants:

  * NO admission dates.  The dataset is fully de-identified, so there are no
    calendar dates at all. Monthly-trend and seasonality analysis are therefore
    impossible, and Module 4 cannot use a time-based train/test split.
    Compensation: `encounter_seq` below numbers each patient's visits in order,
    which gives a genuine (if coarse) sense of sequence.
  * NO blood pressure, NO cholesterol, NO BMI.  `weight` exists but is missing
    in ~97% of rows, which is itself a finding worth showing.
  * Glucose and HbA1c are CATEGORICAL, not numeric ("Norm", ">200", ">300").
  * Every patient is diabetic - it is a diabetes study. So `diagnosis` here
    means the PRIMARY condition treated during the stay, not the only one.

Do not hide these. An evaluator who spots a limitation you already named thinks
you understand your data. One who spots a limitation you missed thinks you do
not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. FILES THIS ADAPTER EXPECTS
# ---------------------------------------------------------------------------
# The download is a .zip containing exactly these two files. Only the first is
# required; the second is a small lookup table we already have built in below.
EXPECTED_FILES = {
    "main": "diabetic_data.csv",
    "mapping": "IDS_mapping.csv",  # optional
}

DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
KAGGLE_URL = "https://www.kaggle.com/datasets/brandao/diabetes"

# In this dataset a missing value is literally the string "?".
MISSING_TOKENS = ["?", "", "NULL", "Unknown/Invalid", "Not Available", "Not Mapped"]


# ---------------------------------------------------------------------------
# 2. LOOKUP TABLES
# ---------------------------------------------------------------------------
# The raw file stores admission type, discharge destination and admission source
# as bare integers. The meanings live in IDS_mapping.csv, but that file has an
# awkward three-tables-stacked-in-one-file layout, so the maps are built in here
# for reliability. The adapter will still read the file if it is present.

ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Unknown", 6: "Unknown", 7: "Trauma Centre", 8: "Unknown",
}

ADMISSION_SOURCE_MAP = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from Hospital", 5: "Transfer from SNF",
    6: "Transfer from Other Facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Unknown", 10: "Transfer from Other Facility",
    11: "Normal Delivery", 12: "Premature Delivery", 13: "Sick Baby",
    14: "Extramural Birth", 15: "Unknown", 17: "Unknown",
    18: "Transfer from Other Facility", 19: "Readmission to Same Facility",
    20: "Unknown", 21: "Unknown", 22: "Transfer from Rehab Facility",
    23: "Born Inside Facility", 24: "Born Outside Facility",
    25: "Transfer from Ambulatory Surgery", 26: "Transfer from Hospice",
}

DISCHARGE_MAP = {
    1: "Home", 2: "Short-term Hospital", 3: "Skilled Nursing Facility",
    4: "Intermediate Care Facility", 5: "Other Inpatient Care",
    6: "Home Health Service", 7: "Left Against Medical Advice",
    8: "Home IV Provider", 9: "Readmitted Inpatient", 10: "Neonatal Transfer",
    11: "Expired", 12: "Still Patient", 13: "Hospice - Home",
    14: "Hospice - Facility", 15: "Medicare Swing Bed",
    16: "Outpatient Services - Other", 17: "Outpatient Services - This Facility",
    18: "Unknown", 19: "Expired", 20: "Expired", 21: "Expired",
    22: "Rehabilitation Facility", 23: "Long Term Care Hospital",
    24: "Nursing Facility", 25: "Unknown", 26: "Unknown",
    27: "Federal Health Care Facility", 28: "Psychiatric Hospital",
    29: "Critical Access Hospital", 30: "Other Health Care Institution",
}

# Patients who died cannot be readmitted. Leaving them in the dataset would drag
# the readmission rate down for a reason that has nothing to do with care
# quality, and would teach a model that "expired" predicts "not readmitted" -
# true, but useless and grotesque. Standard practice in the published literature
# is to exclude them. Hospice is excluded for the same reason.
EXCLUDED_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

# The 23 diabetes medication columns, grouped into drug classes. Grouping turns
# 23 thin columns into a handful of clinically meaningful categories, which is
# what makes the medication analysis readable.
DRUG_CLASSES = {
    "metformin": "Biguanide",
    "repaglinide": "Meglitinide",
    "nateglinide": "Meglitinide",
    "chlorpropamide": "Sulfonylurea",
    "glimepiride": "Sulfonylurea",
    "acetohexamide": "Sulfonylurea",
    "glipizide": "Sulfonylurea",
    "glyburide": "Sulfonylurea",
    "tolbutamide": "Sulfonylurea",
    "pioglitazone": "Thiazolidinedione",
    "rosiglitazone": "Thiazolidinedione",
    "acarbose": "Alpha-glucosidase Inhibitor",
    "miglitol": "Alpha-glucosidase Inhibitor",
    "troglitazone": "Thiazolidinedione",
    "tolazamide": "Sulfonylurea",
    "examide": "Other",
    "citoglipton": "Other",
    "insulin": "Insulin",
    "glyburide-metformin": "Combination",
    "glipizide-metformin": "Combination",
    "glimepiride-pioglitazone": "Combination",
    "metformin-rosiglitazone": "Combination",
    "metformin-pioglitazone": "Combination",
}

# ICD-9 diagnosis codes grouped into clinical categories. This is the exact
# grouping used by Strack et al. in the paper that accompanies the dataset, so
# your analysis is directly comparable to published work - which is a strong
# thing to be able to say in a demo.
ICD9_GROUPS = [
    # (label, category, is_chronic, numeric ranges, exact-prefix codes)
    ("Circulatory",    "Cardiovascular", 1, [(390, 459)], ["785"]),
    ("Respiratory",    "Respiratory",    1, [(460, 519)], ["786"]),
    ("Digestive",      "Digestive",      0, [(520, 579)], ["787"]),
    ("Diabetes",       "Endocrine",      1, [(250, 250)], []),
    ("Injury",         "Musculoskeletal", 0, [(800, 999)], []),
    ("Musculoskeletal", "Musculoskeletal", 1, [(710, 739)], []),
    ("Genitourinary",  "Renal",          1, [(580, 629)], ["788"]),
    ("Neoplasms",      "Oncology",       1, [(140, 239)], []),
]

ICD9_CATEGORY_META = {
    "Circulatory": ("Cardiovascular", 1, "390-459", "Cardiology"),
    "Respiratory": ("Respiratory", 1, "460-519", "Pulmonology"),
    "Digestive": ("Digestive", 0, "520-579", "Gastroenterology"),
    "Diabetes": ("Endocrine", 1, "250", "Endocrinology"),
    "Injury": ("Musculoskeletal", 0, "800-999", "Orthopaedics"),
    "Musculoskeletal": ("Musculoskeletal", 1, "710-739", "Orthopaedics"),
    "Genitourinary": ("Renal", 1, "580-629", "Nephrology"),
    "Neoplasms": ("Oncology", 1, "140-239", "Oncology"),
    "Other": ("Other", 0, "other", "General Medicine"),
}


# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------
def _icd9_to_group(code: Any) -> str:
    """Map one ICD-9 code to a clinical group name.

    ICD-9 codes are mostly numeric ("428" = heart failure) but some start with
    a letter: "V" codes are supplementary factors, "E" codes are external causes
    of injury. Both go to "Other".
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Other"
    text = str(code).strip()
    if text in ("", "?", "nan"):
        return "Other"
    if text[0].upper() in ("V", "E"):
        return "Other"
    try:
        value = float(text)
    except ValueError:
        return "Other"

    # Diabetes is 250.xx - check it before the broad ranges.
    if 250 <= value < 251:
        return "Diabetes"

    whole = int(value)
    for label, _cat, _chronic, ranges, prefixes in ICD9_GROUPS:
        for low, high in ranges:
            if low <= whole <= high:
                return label
        if str(whole) in prefixes:
            return label
    return "Other"


def _age_bracket_to_midpoint(bracket: Any) -> float:
    """Turn the dataset's age bracket into a usable number.

    Age is recorded as a 10-year band like "[70-80)" for de-identification. Most
    statistics need a number, so we take the midpoint: "[70-80)" -> 75.

    Be honest about this in your demo: it is an approximation, and it means the
    project's "average age" is an average of midpoints, not of true ages.
    """
    if not isinstance(bracket, str):
        return np.nan
    text = bracket.strip().strip("[]()")
    if "-" not in text:
        return np.nan
    low, _, high = text.partition("-")
    try:
        return (float(low) + float(high)) / 2.0
    except ValueError:
        return np.nan


def _find_input_file(raw_dir: Path, filename: str) -> Path | None:
    """Look for a file in raw_dir, tolerating case and nesting.

    Beginners often unzip into a subfolder, or the file arrives capitalised
    differently. Rather than failing with 'file not found', search a bit.
    """
    direct = raw_dir / filename
    if direct.exists():
        return direct
    matches = list(raw_dir.rglob(filename))
    if matches:
        return matches[0]
    lowered = filename.lower()
    for candidate in raw_dir.rglob("*.csv"):
        if candidate.name.lower() == lowered:
            return candidate
    return None


class DatasetNotFoundError(FileNotFoundError):
    """Raised with beginner-readable download instructions."""


def _missing_file_message(raw_dir: Path) -> str:
    return (
        f"\nCould not find '{EXPECTED_FILES['main']}' in:\n"
        f"    {raw_dir}\n\n"
        "This project uses a REAL public dataset, which you download once.\n\n"
        "  Option A - UCI (no account needed):\n"
        f"      {DOWNLOAD_URL}\n"
        "      Unzip it, then move 'diabetic_data.csv' into the folder above.\n\n"
        "  Option B - Kaggle (free account needed):\n"
        f"      {KAGGLE_URL}\n"
        "      Click Download, unzip, move 'diabetic_data.csv' into the folder above.\n\n"
        "Then run this script again.\n"
    )


# ---------------------------------------------------------------------------
# 4. THE ADAPTER
# ---------------------------------------------------------------------------
def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Read diabetic_data.csv, with a helpful error if it is not there."""
    path = _find_input_file(raw_dir, EXPECTED_FILES["main"])
    if path is None:
        raise DatasetNotFoundError(_missing_file_message(raw_dir))

    # keep_default_na=False stops pandas turning the string "None" (a real
    # value in max_glu_serum meaning 'test not ordered') into a missing value.
    df = pd.read_csv(path, na_values=MISSING_TOKENS, keep_default_na=False, low_memory=False)
    df = df.replace({"": np.nan})
    return df


def build_disease_reference() -> pd.DataFrame:
    """One row per ICD-9 clinical group - the project's lookup table."""
    rows = []
    for label, (category, chronic, icd_range, department) in ICD9_CATEGORY_META.items():
        rows.append(
            {
                "diagnosis": label,
                "icd9_range": icd_range,
                "disease_category": category,
                "is_chronic": chronic,
                "department": department,
            }
        )
    return pd.DataFrame(rows)


def to_patients(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Translate the raw encounters table into the project's `patients` table.

    Returns the DataFrame plus a report of what happened, so the Data Quality
    page can show the transformation rather than just claim it.
    """
    report: dict[str, Any] = {"rows_in": len(raw)}
    df = raw.copy()

    # --- Identity ---------------------------------------------------------
    # encounter_id is the row key. patient_nbr identifies the PERSON, and is
    # kept because the same person appearing twice is a real signal.
    df["patient_id"] = df["encounter_id"].astype(str)
    df["person_id"] = df["patient_nbr"].astype(str)

    # --- Remove patients who could not possibly be readmitted -------------
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)].copy()
    report["expired_or_hospice_removed"] = before - len(df)

    # --- Demographics -----------------------------------------------------
    df["age_group"] = df["age"].astype(str).str.strip()
    df["age"] = df["age_group"].map(_age_bracket_to_midpoint)
    df["gender"] = df["gender"].replace({"Unknown/Invalid": np.nan})

    # --- Stay and utilisation --------------------------------------------
    df["length_of_stay"] = pd.to_numeric(df["time_in_hospital"], errors="coerce")

    for col in ("number_outpatient", "number_emergency", "number_inpatient"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # The single most predictive feature in the readmission literature:
    # how much hospital care this person already used in the prior year.
    df["num_prior_admissions"] = df["number_inpatient"].astype(int)
    df["num_prior_emergency"] = df["number_emergency"].astype(int)
    df["num_prior_outpatient"] = df["number_outpatient"].astype(int)
    df["num_prior_visits_total"] = (
        df["number_inpatient"] + df["number_emergency"] + df["number_outpatient"]
    ).astype(int)

    df["num_lab_procedures"] = pd.to_numeric(df["num_lab_procedures"], errors="coerce")
    df["num_procedures"] = pd.to_numeric(df["num_procedures"], errors="coerce")
    df["num_medications"] = pd.to_numeric(df["num_medications"], errors="coerce")
    df["num_diagnoses"] = pd.to_numeric(df["number_diagnoses"], errors="coerce")

    # --- Encounter sequence: the closest thing to a date this dataset has --
    # Sorting by encounter_id and numbering each person's visits gives a real
    # ordering, even without calendar dates. Visit 3 genuinely happened after
    # visit 2 for that person.
    df = df.sort_values("encounter_id")
    df["encounter_seq"] = df.groupby("person_id").cumcount() + 1
    df["is_repeat_patient"] = (df["encounter_seq"] > 1).astype(int)

    # --- Categorical lookups ---------------------------------------------
    df["admission_type"] = (
        pd.to_numeric(df["admission_type_id"], errors="coerce")
        .map(ADMISSION_TYPE_MAP).fillna("Unknown")
    )
    df["admission_source"] = (
        pd.to_numeric(df["admission_source_id"], errors="coerce")
        .map(ADMISSION_SOURCE_MAP).fillna("Unknown")
    )
    df["discharge_disposition"] = (
        pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
        .map(DISCHARGE_MAP).fillna("Unknown")
    )

    # medical_specialty is the admitting physician's speciality - the closest
    # thing this dataset has to a hospital department.
    df["department"] = (
        df["medical_specialty"].fillna("Unknown").astype(str).str.replace("-", " ", regex=False)
    )

    # --- Diagnosis --------------------------------------------------------
    # diag_1 is the primary diagnosis. diag_2/diag_3 are secondary, kept
    # because comorbidity count is a known risk factor.
    df["diagnosis"] = df["diag_1"].map(_icd9_to_group)
    df["diagnosis_secondary"] = df["diag_2"].map(_icd9_to_group)
    df["diagnosis_tertiary"] = df["diag_3"].map(_icd9_to_group)
    df["icd9_primary"] = df["diag_1"].astype(str)

    # --- Lab results (categorical in this dataset) ------------------------
    # "None" here means the test was NOT ORDERED, which is different from a
    # missing value - the absence of a test is itself information about how
    # closely the patient was monitored.
    df["a1c_result"] = df["A1Cresult"].fillna("Not Tested").replace({"None": "Not Tested"})
    df["glucose_serum"] = df["max_glu_serum"].fillna("Not Tested").replace({"None": "Not Tested"})
    df["a1c_tested"] = (df["a1c_result"] != "Not Tested").astype(int)
    df["glucose_tested"] = (df["glucose_serum"] != "Not Tested").astype(int)

    # --- Medication behaviour --------------------------------------------
    df["medication_changed"] = (df["change"].astype(str).str.strip() == "Ch").astype(int)
    df["on_diabetes_meds"] = (df["diabetesMed"].astype(str).str.strip() == "Yes").astype(int)

    # --- THE TARGET -------------------------------------------------------
    # The raw column has three values: "<30", ">30" and "NO".
    # Our question is specifically 30-DAY readmission, so ">30" counts as 0.
    # This is a modelling decision, not a data cleaning one - say so if asked.
    readmit_raw = df["readmitted"].astype(str).str.strip()
    df["readmitted_30d"] = (readmit_raw == "<30").astype(int)
    df["readmitted_any"] = readmit_raw.isin(["<30", ">30"]).astype(int)
    report["readmit_class_counts"] = readmit_raw.value_counts().to_dict()

    # --- Weight: kept only to demonstrate a real missingness problem ------
    # Columns dropped for being too incomplete to trust. Recorded rather than
    # silently discarded, because the Data Quality page shows the decision and
    # the reasoning behind it is worth defending in a demo.
    dropped = []
    for col in ("weight", "payer_code", "medical_specialty"):
        if col in df.columns:
            pct = round(float(df[col].isna().mean()) * 100, 2)
            dropped.append({"column": col, "missing_pct": pct})
    report["incomplete_columns"] = dropped
    weight = next((d for d in dropped if d["column"] == "weight"), None)
    report["weight_missing_pct"] = weight["missing_pct"] if weight else None

    keep = [
        "patient_id", "person_id", "encounter_seq", "is_repeat_patient",
        "age", "age_group", "gender", "race",
        "diagnosis", "diagnosis_secondary", "diagnosis_tertiary", "icd9_primary",
        "num_diagnoses", "admission_type", "admission_source",
        "discharge_disposition", "department",
        "length_of_stay", "num_lab_procedures", "num_procedures", "num_medications",
        "num_prior_admissions", "num_prior_emergency", "num_prior_outpatient",
        "num_prior_visits_total",
        "a1c_result", "glucose_serum", "a1c_tested", "glucose_tested",
        "medication_changed", "on_diabetes_meds",
        "readmitted_30d", "readmitted_any",
    ]
    out = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    report["rows_out"] = len(out)
    report["columns_out"] = len(out.columns)
    report["unique_people"] = int(out["person_id"].nunique())
    report["readmission_rate"] = round(float(out["readmitted_30d"].mean()) * 100, 2)
    return out, report


def to_medications(raw: pd.DataFrame, valid_patient_ids: pd.Series | None = None) -> pd.DataFrame:
    """Melt the 23 wide drug columns into a long one-row-per-prescription table.

    Wide (unusable):
        encounter_id | metformin | insulin | glipizide | ...
        2278392      | No        | Steady  | No        | ...

    Long (queryable):
        patient_id | medication_name | drug_class | dose_status
        2278392    | insulin         | Insulin    | Steady
    """
    drug_cols = [c for c in DRUG_CLASSES if c in raw.columns]
    if not drug_cols:
        return pd.DataFrame(
            columns=["patient_id", "medication_name", "drug_class", "dose_status"]
        )

    subset = raw[["encounter_id"] + drug_cols].copy()
    long = subset.melt(
        id_vars="encounter_id", var_name="medication_name", value_name="dose_status"
    )

    # "No" means the drug was not prescribed - those rows are not prescriptions.
    long["dose_status"] = long["dose_status"].astype(str).str.strip()
    long = long[~long["dose_status"].isin(["No", "nan", "", "?"])].copy()

    long["patient_id"] = long["encounter_id"].astype(str)
    long["drug_class"] = long["medication_name"].map(DRUG_CLASSES).fillna("Other")

    if valid_patient_ids is not None:
        # Enforce the foreign key: no medication may reference a patient we
        # dropped (expired/hospice). MySQL would reject these anyway.
        allowed = set(valid_patient_ids.astype(str))
        long = long[long["patient_id"].isin(allowed)]

    out = long[["patient_id", "medication_name", "drug_class", "dose_status"]]
    return out.sort_values(["patient_id", "medication_name"]).reset_index(drop=True)


def convert(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Full conversion: raw download -> (patients, medications, reference, report)."""
    raw = load_raw(raw_dir)
    patients, report = to_patients(raw)
    medications = to_medications(raw, valid_patient_ids=patients["patient_id"])
    reference = build_disease_reference()
    report["medication_rows"] = len(medications)
    report["source_file"] = EXPECTED_FILES["main"]
    report["citation"] = "Strack et al. (2014), UCI ML Repository, DOI 10.24432/C5230J"
    report["licence"] = "CC BY 4.0"
    return patients, medications, reference, report
