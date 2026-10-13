"""
CONVENIENCE SCRIPT - runs steps 1, 2 and 3 back to back.

RUN IT LIKE THIS:
    python scripts/run_all.py

Use this when you are setting the project up on a new machine, or the night
before a demo to guarantee everything is fresh.

If MySQL is not available it keeps going and finishes with cleaned CSV files, so
the dashboard still runs. It tells you clearly which happened.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # the exact python running this file (respects your venv)

STEPS = [
    ("Check the dataset", [PY, str(ROOT / "scripts" / "01_check_dataset.py")], True),
    ("Set up MySQL", [PY, str(ROOT / "scripts" / "02_setup_database.py")], False),
    ("Clean and load data", [PY, str(ROOT / "scripts" / "03_load_data.py")], True),
    # Module 4. Not "must succeed": if scikit-learn is missing, the dashboard
    # still works on the rule-based baseline, so a failure here should not stop
    # the pipeline.
    ("Train the ML model", [PY, str(ROOT / "scripts" / "04_train_model.py")], False),
]


def main() -> int:
    for i, (label, cmd, must_succeed) in enumerate(STEPS, start=1):
        print()
        print("#" * 66)
        print(f"#  {i}/{len(STEPS)}  {label}")
        print("#" * 66)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            if must_succeed:
                print(f"\nStopped: '{label}' failed. Fix the error above and re-run.")
                return 1
            print(f"\n'{label}' failed, but it is optional. Continuing without MySQL;")
            print("the dashboard will read the cleaned CSV files instead.")

    print()
    print("=" * 66)
    print("ALL DONE")
    print("=" * 66)
    print("Launch the dashboard with:")
    print("    streamlit run streamlit_app/app.py")
    print()
    print("Open the analysis notebook with:")
    print("    jupyter notebook notebooks/01_eda_hospital_analytics.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
