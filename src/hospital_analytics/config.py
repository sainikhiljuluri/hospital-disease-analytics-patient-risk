"""
Central configuration for the whole project.

WHY THIS FILE EXISTS
--------------------
Every other file in this project needs to know two kinds of things:
  1. Where the data folders live on disk.
  2. How to connect to the MySQL database.

If each file hard-coded those values, changing your MySQL password would mean
editing 10 files. Instead everything imports from here, and this file reads the
values from a `.env` file (a plain text file of settings you keep private).

Nothing secret is stored in this file itself - only safe defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# 1. PROJECT FOLDERS
# ---------------------------------------------------------------------------
# __file__ is this file. .parents[2] walks up 2 folders:
#   config.py  ->  [0] hospital_analytics  ->  [1] src  ->  [2] PROJECT ROOT
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"            # untouched input CSV files
PROCESSED_DIR: Path = DATA_DIR / "processed"  # cleaned output CSV files
NOTEBOOK_DIR: Path = PROJECT_ROOT / "notebooks"
MODEL_DIR: Path = PROJECT_ROOT / "models"   # trained models live here (Module 4)
MODELS_DIR: Path = MODEL_DIR                # alias; both names work

# Create the folders if they do not exist yet, so no script ever crashes with
# "No such file or directory".
for _folder in (RAW_DIR, PROCESSED_DIR, MODEL_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# WHICH DATASET THIS PROJECT USES
# ---------------------------------------------------------------------------
# This project runs on a REAL, PUBLIC dataset that you download once. Nothing is
# invented or simulated.
#
#   Name    : Diabetes 130-US Hospitals for Years 1999-2008
#   Source  : UCI Machine Learning Repository (dataset 296), mirrored on Kaggle
#   Size    : 101,766 hospital encounters x 50 columns
#   Licence : CC BY 4.0 - free to download, use and adapt, with credit
#   Cite    : Strack et al. (2014), BioMed Research International, 781670.
#             DOI 10.24432/C5230J
#
# You place ONE file, `diabetic_data.csv`, into data/raw/. The adapter in
# ingestion/adapters/uci_diabetes.py translates it into the project's schema.
DATASET_NAME = "Diabetes 130-US Hospitals (1999-2008)"
DATASET_ADAPTER = "uci_diabetes"
DATASET_SOURCE_FILE = "diabetic_data.csv"
DATASET_UCI_URL = (
    "https://archive.ics.uci.edu/dataset/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008"
)
DATASET_KAGGLE_URL = "https://www.kaggle.com/datasets/brandao/diabetes"
DATASET_LICENCE = "CC BY 4.0"
DATASET_CITATION = (
    "Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN "
    "(2014). Impact of HbA1c Measurement on Hospital Readmission Rates. "
    "BioMed Research International, 781670. DOI 10.24432/C5230J"
)

# The raw download the pipeline reads.
RAW_FILES = {
    "source": RAW_DIR / DATASET_SOURCE_FILE,
    "id_mapping": RAW_DIR / "IDS_mapping.csv",  # optional extra file in the zip
}

# Intermediate files written by the adapter, in the project's own schema.
INTERIM_FILES = {
    "patients": RAW_DIR / "patients.csv",
    "medications": RAW_DIR / "medications.csv",
    "disease_reference": RAW_DIR / "disease_reference.csv",
}

# ---------------------------------------------------------------------------
# DATASET CAPABILITIES
# ---------------------------------------------------------------------------
# Real data does not contain everything you might want. Rather than let pages
# crash or silently show empty charts, the project declares up front which
# analyses this dataset can support. Charts and pages check these flags.
#
# Being explicit about what you CANNOT do is a feature, not an apology - it is
# the difference between "I didn't do seasonality" and "this dataset is
# de-identified with no calendar dates, so seasonality is not computable".
DATASET_PROFILE = {
    "has_dates": False,          # fully de-identified: no admission dates at all
    "has_vitals": False,         # no blood pressure, no BMI, no cholesterol
    "has_numeric_labs": False,   # HbA1c and glucose are categorical bands
    "has_smoking_status": False,
    "has_encounter_sequence": True,   # visits CAN be ordered per person
    "has_lab_test_flags": True,       # whether HbA1c/glucose was ordered at all
    "has_medication_detail": True,    # 23 drugs, with dose change direction
    "has_comorbidity_count": True,    # number_diagnoses, diag_1/2/3
    "one_row_per": "encounter",       # NOT per patient - people repeat
}

# The cleaned, analysis-ready file. The Streamlit app can fall back to this if
# MySQL is not running, so the project is ALWAYS demoable.
PROCESSED_PATIENTS_CSV: Path = PROCESSED_DIR / "patients_clean.csv"
PROCESSED_MEDS_CSV: Path = PROCESSED_DIR / "medications_clean.csv"


# ---------------------------------------------------------------------------
# 2. LOAD .env (optional dependency, so the project still runs without it)
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    """Read PROJECT_ROOT/.env into environment variables, if the file exists."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:  # preferred: the python-dotenv library
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:  # fallback: tiny hand-written parser
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()


# ---------------------------------------------------------------------------
# 3. DATABASE SETTINGS
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DBSettings:
    """All MySQL connection details in one small object."""

    host: str = os.getenv("MYSQL_HOST", "localhost")
    port: int = int(os.getenv("MYSQL_PORT", "3306"))
    user: str = os.getenv("MYSQL_USER", "root")
    password: str = os.getenv("MYSQL_PASSWORD", "")
    database: str = os.getenv("MYSQL_DATABASE", "hospital_analytics")

    @property
    def url(self) -> str:
        """SQLAlchemy connection string INCLUDING the database name."""
        pwd = quote_plus(self.password)  # escapes @ / # / etc. in passwords
        return (
            f"mysql+pymysql://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )

    @property
    def server_url(self) -> str:
        """Connection string WITHOUT a database name.

        Needed once, to run `CREATE DATABASE hospital_analytics`, because you
        cannot connect to a database that does not exist yet.
        """
        pwd = quote_plus(self.password)
        return (
            f"mysql+pymysql://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/?charset=utf8mb4"
        )


DB = DBSettings()

# Table names kept as constants so a typo becomes an ImportError (loud + early)
# instead of a silent empty query result.
TBL_PATIENTS = "patients"
TBL_MEDICATIONS = "medications"
TBL_DISEASE_REF = "disease_reference"
TBL_LOAD_LOG = "data_load_log"

# ---------------------------------------------------------------------------
# 4. ANALYSIS CONSTANTS
# ---------------------------------------------------------------------------
RANDOM_SEED = 42   # used for sampling in scatter plots and, later, ML splits

# Plausible ranges for the columns this dataset actually contains. The cleaning
# module flags anything outside these, and the Data Quality page shows the rules
# so a viewer can see the standard being applied.
#
# The dataset's own documented limits: stays are 1-14 days by construction
# (that was an inclusion criterion), ages are 10-year bands from [0-10) to
# [90-100). Values outside these indicate a corrupted download, not a sick
# patient - which is exactly what a validation gate is for.
CLINICAL_RANGES = {
    "age": (0, 100),
    "length_of_stay": (1, 14),
    "num_lab_procedures": (0, 150),
    "num_procedures": (0, 10),
    "num_medications": (1, 90),
    "num_diagnoses": (1, 20),
    "num_prior_admissions": (0, 30),
    "num_prior_emergency": (0, 80),
    "num_prior_outpatient": (0, 50),
}

# Columns that must be present after adapting, or something has gone wrong.
REQUIRED_AFTER_ADAPT = [
    "patient_id", "age", "gender", "diagnosis",
    "length_of_stay", "num_prior_admissions", "readmitted_30d",
]
