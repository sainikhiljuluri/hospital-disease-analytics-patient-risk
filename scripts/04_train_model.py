"""
STEP 4 - train the readmission prediction model (Module 4).

RUN IT LIKE THIS:
    python scripts/04_train_model.py

Options:
    --test-size 0.25    share of PEOPLE held back for testing (default 0.25)
    --quiet             less output

WHAT IT DOES
    1. Loads the cleaned data (MySQL if available, else the CSV).
    2. Builds the feature matrix, excluding leaky and fairness-sensitive columns.
    3. Splits train/test GROUPED BY PERSON so nobody appears in both halves.
    4. Trains Logistic Regression, Random Forest and a boosted-tree model.
    5. Compares them on ROC-AUC and keeps the best.
    6. Runs a fairness audit and a first-time-patient stress test.
    7. Saves everything to models/.

HOW LONG IT TAKES
    About 30-90 seconds on the full dataset. No downloads, no GPU, no API key.

RUN IT AGAIN ANY TIME
    Re-running overwrites the saved model. The random seed is fixed, so the same
    data gives the same result every time.

IF IT FAILS
    "No data found"  -> run python scripts/03_load_data.py first
    "No module named sklearn" -> pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hospital_analytics.analysis.data_access import load_patients  # noqa: E402
from hospital_analytics.ml import evaluate as E  # noqa: E402
from hospital_analytics.ml import features as F  # noqa: E402
from hospital_analytics.ml import train as T  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the readmission model.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    # --- 1. load ----------------------------------------------------------
    try:
        df, source = load_patients()
    except FileNotFoundError as exc:
        print("Could not load any data.\n")
        print(exc)
        print("\nRun this first:  python scripts/03_load_data.py")
        return 1

    if verbose:
        print(f"Loaded {len(df):,} encounters from: {source}\n")

    if "readmitted_30d" not in df.columns:
        print("The data has no 'readmitted_30d' column, so there is nothing to")
        print("learn from. Re-run: python scripts/03_load_data.py")
        return 1

    # --- 2. what is being excluded, and why -------------------------------
    if verbose:
        print("EXCLUDED FROM THE MODEL (this is the important part)")
        print("-" * 66)
        for _, row in F.excluded_features_table().iterrows():
            print(f"  {row['column']}")
            print(f"      {row['reason']}")
        print()
        moment = F.describe_prediction_moment()
        print(f"PREDICTION MOMENT: {moment['moment']}")
        print(f"  {moment['why']}")
        print(f"  Rule: {moment['rule']}")
        print()

    # --- 3. train ---------------------------------------------------------
    try:
        result = T.train_all(df, test_size=args.test_size, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        print(f"Training failed: {type(exc).__name__}: {exc}")
        return 1

    split = result["split"]
    model = result["best_model"]
    y_test = split["y_test"]
    y_pred = model.predict(split["X_test"])
    y_proba = model.predict_proba(split["X_test"])[:, 1]
    df_test = df.iloc[split["idx_test"]]

    # --- 4. what the numbers mean -----------------------------------------
    print("WHAT THESE NUMBERS MEAN")
    print("-" * 66)
    for line in E.interpret_scores(result["best_scores"], result["baseline"]):
        print(f"  {line}")
    print()

    print("CONFUSION MATRIX")
    print("-" * 66)
    print(E.confusion_table(result["best_scores"]).to_string())
    print()
    print("  A FALSE NEGATIVE (bottom-left) is a patient who came back and was")
    print("  not flagged - they went home without support. That is the costly")
    print("  error, which is why the model is tuned toward recall.")
    print()

    # --- 5. threshold trade-off -------------------------------------------
    thresholds = E.threshold_table(model, split["X_test"], y_test)
    if not thresholds.empty:
        print("CHOOSING A CUT-OFF  (0.5 is a convention, not a law)")
        print("-" * 66)
        print(thresholds.to_string(index=False))
        print()
        print("  Pick the row matching your hospital's follow-up capacity.")
        print("  Lower threshold = catch more, investigate more.")
        print()

    # --- 6. fairness -------------------------------------------------------
    audit = E.fairness_audit(df_test, y_test, y_pred)
    if not audit.empty:
        print("FAIRNESS AUDIT  (on the model's PREDICTIONS, not the input data)")
        print("-" * 66)
        print(audit.to_string(index=False))
        print()
        spread = audit.groupby("attribute")["recall"].agg(["min", "max"])
        for attr, row in spread.iterrows():
            gap = row["max"] - row["min"]
            verdict = "acceptable" if gap < 0.15 else "INVESTIGATE - large gap"
            print(f"  {attr}: recall ranges {row['min']:.2f} to {row['max']:.2f} "
                  f"(gap {gap:.2f}) - {verdict}")
        print()
        print("  The model never sees ethnicity. It can still perform unequally,")
        print("  because other features correlate with it. That is why this audit")
        print("  runs on predictions rather than on inputs.")
        print()

    # --- 7. the honest stress test ----------------------------------------
    subgroup = E.subgroup_check(df_test, y_test, y_proba)
    if not subgroup.empty:
        print("STRESS TEST  first-time patients vs everyone else")
        print("-" * 66)
        print(subgroup.to_string(index=False))
        print()
        print("  Prior-visit counts are the strongest features - and they are")
        print("  ZERO for a first-time patient, which is exactly when a clinician")
        print("  has least information. If ROC-AUC drops sharply in the top row,")
        print("  the model is less useful than one overall number suggests.")
        print("  Report both figures, not just the headline.")
        print()

    # --- 8. save -----------------------------------------------------------
    paths = T.save_model(result)
    print("SAVED")
    print("-" * 66)
    for label, path in paths.items():
        print(f"  {label:<12} {path}")
    print()
    print("Next:")
    print("  streamlit run streamlit_app/app.py      # Risk Prediction page is now live")
    print("  python scripts/05_generate_report.py    # Module 5: AI explanations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
