"""
STEP 3 OF 3 - clean the raw data and load it into MySQL.

RUN IT LIKE THIS:
    python scripts/03_load_data.py

OR, if you have not set up MySQL yet and just want the dashboard to work:
    python scripts/03_load_data.py --csv-only

WHAT IT DOES
    EXTRACT   reads data/raw/*.csv
    TRANSFORM cleans it (duplicates, outliers, missing values, date formats)
    LOAD      writes it to MySQL and saves cleaned copies to data/processed/

Safe to run repeatedly: it empties the tables first, so you never get duplicated
rows from running it twice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hospital_analytics.ingestion.load_to_mysql import main  # noqa: E402

if __name__ == "__main__":
    main()
