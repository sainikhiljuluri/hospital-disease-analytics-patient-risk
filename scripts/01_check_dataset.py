"""
STEP 1 of the pipeline: check that the real dataset has been downloaded.

WHAT THIS SCRIPT DOES
---------------------
This project uses a REAL public dataset. Nothing is invented or simulated. You
download one file, once, and this script confirms it is where the pipeline
expects it and that it looks correct.

HOW TO RUN IT
-------------
    python scripts/01_check_dataset.py

If the file is missing, the script prints exactly where to get it and where to
put it. It does not download anything itself - partly because the download
sometimes needs a Kaggle login, and partly because a script that silently pulls
100,000 medical records off the internet is not a script you want to run without
knowing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hospital_analytics.config import (  # noqa: E402
    DATASET_CITATION,
    DATASET_KAGGLE_URL,
    DATASET_LICENCE,
    DATASET_NAME,
    DATASET_SOURCE_FILE,
    DATASET_UCI_URL,
    RAW_DIR,
)
from hospital_analytics.ingestion.adapters import uci_diabetes  # noqa: E402

# The 8 columns whose absence means you have downloaded the wrong file.
CRITICAL_COLUMNS = [
    "encounter_id", "patient_nbr", "age", "time_in_hospital",
    "number_inpatient", "diag_1", "A1Cresult", "readmitted",
]

EXPECTED_ROWS = 101_766
EXPECTED_COLS = 50


def main() -> int:
    print("=" * 68)
    print("STEP 1  CHECK THE DATASET")
    print("=" * 68)
    print(f"  dataset : {DATASET_NAME}")
    print(f"  licence : {DATASET_LICENCE}")
    print(f"  looking in: {RAW_DIR}")
    print()

    path = uci_diabetes._find_input_file(RAW_DIR, DATASET_SOURCE_FILE)
    if path is None:
        print("  NOT FOUND")
        print(uci_diabetes._missing_file_message(RAW_DIR))
        print("  Direct links:")
        print(f"    UCI    (no account) : {DATASET_UCI_URL}")
        print(f"    Kaggle (free login) : {DATASET_KAGGLE_URL}")
        return 1

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  FOUND: {path}")
    print(f"  size : {size_mb:.1f} MB")
    print()

    # Read only the header first - checking 50 column names does not require
    # loading 18 MB into memory.
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    print(f"  columns: {len(cols)} (expected {EXPECTED_COLS})")

    missing = [c for c in CRITICAL_COLUMNS if c not in cols]
    if missing:
        print()
        print("  PROBLEM: required columns are missing:")
        for col in missing:
            print(f"    - {col}")
        print()
        print("  This usually means a different file was downloaded. You want")
        print(f"  '{DATASET_SOURCE_FILE}' from inside the zip - not a cleaned or")
        print("  pre-processed copy from someone else's notebook.")
        return 1
    print("  all critical columns present")

    # Now read for real and sanity-check the contents.
    df = pd.read_csv(path, low_memory=False)
    print(f"  rows   : {len(df):,} (expected {EXPECTED_ROWS:,})")
    if len(df) != EXPECTED_ROWS:
        print("  NOTE: row count differs from the published figure. Not fatal -")
        print("        some mirrors ship a subset - but say so in your write-up.")

    outcomes = df["readmitted"].value_counts()
    print()
    print("  target variable `readmitted`:")
    for label, count in outcomes.items():
        print(f"    {str(label):>4} : {count:>7,}  ({100 * count / len(df):5.2f}%)")
    print()
    print(f"  30-day readmission rate: {100 * (df['readmitted'] == '<30').mean():.2f}%")

    q_marks = int((df == "?").sum().sum())
    print(f"  '?' placeholders to clean: {q_marks:,}")
    print()
    print("  CITE THIS in your report and slides:")
    print(f"    {DATASET_CITATION}")
    print()
    print("  Dataset looks good. Next:")
    print("    python scripts/02_setup_database.py")
    print("    python scripts/03_load_data.py")
    print("  or just:  python scripts/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
