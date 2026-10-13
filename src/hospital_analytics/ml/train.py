"""
MODULE 4, part 2: training and comparing readmission models.

WHAT GETS TRAINED
-----------------
Three models, deliberately chosen to be different in kind:

  1. LOGISTIC REGRESSION - the interpretable baseline. Fits one coefficient per
     feature, so you can read off exactly how each one pushes risk up or down.
     Only finds LINEAR relationships.
  2. RANDOM FOREST - many decision trees, each on a random slice of the data,
     voting together. Catches interactions the linear model cannot, at the cost
     of being harder to interpret.
  3. GRADIENT BOOSTING - trees built in sequence, each correcting the previous
     one's mistakes. Usually the strongest of the three. Uses XGBoost when
     installed, otherwise scikit-learn's own HistGradientBoosting, which is
     comparable and always available.

Training all three is the point. You cannot claim a model is good without
something to compare it against.

THE TWO DECISIONS THAT MATTER MOST
----------------------------------
**1. The split is GROUPED BY PERSON.**

One row is one ENCOUNTER, and the same patient appears many times. A normal
random split would put Mrs Patel's January admission in training and her March
admission in testing. The model would recognise her rather than learn the
pattern, and the test score would be a lie.

`GroupShuffleSplit` keeps every row belonging to one person on the SAME side of
the split. This usually LOWERS the reported score — and that lower number is the
honest one.

**2. The classes are IMBALANCED.**

Only about one encounter in nine ends in readmission. Left alone, a model
discovers that predicting "no" every time is ~89% accurate, so it does exactly
that and catches nobody.

`class_weight="balanced"` makes each positive case count roughly eight times as
much as a negative one during training, so missing one is expensive.

WHY ACCURACY IS THE WRONG METRIC HERE
-------------------------------------
Always predicting "not readmitted" scores ~89% accuracy and helps zero patients.
Any accuracy figure must be reported against that baseline or it is meaningless.

What we actually care about:
  * RECALL      - of the patients who WILL come back, how many did we flag?
                  Missing one means someone goes home without support.
  * ROC-AUC     - can the model rank a readmitted patient above a non-readmitted
                  one? Unaffected by imbalance, which is why it leads.
  * PR-AUC      - like ROC-AUC but focused on the rare positive class. On
                  heavily imbalanced data it is the more honest of the two.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MODELS_DIR, RANDOM_SEED
from . import features as F
from .evaluate import evaluate_model, majority_baseline

# XGBoost is optional. It is often the strongest model, but it needs a compiler
# on some systems and this project must run without it.
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

MODEL_PATH = MODELS_DIR / "readmission_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"


# ---------------------------------------------------------------------------
def split_by_person(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series | None, test_size: float = 0.25
) -> dict[str, Any]:
    """Split into train/test so no PERSON appears on both sides.

    Returns the split plus a report proving no person leaked across, which the
    dashboard displays — the claim is worth more when it is checked.
    """
    if groups is None:
        # No person column: fall back to a stratified random split and say so
        # loudly rather than pretending the split is grouped.
        from sklearn.model_selection import train_test_split

        idx_train, idx_test = train_test_split(
            np.arange(len(X)), test_size=test_size,
            random_state=RANDOM_SEED, stratify=y,
        )
        report = {"grouped": False, "warning": "No person_id column — split is NOT grouped."}
    else:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=RANDOM_SEED
        )
        idx_train, idx_test = next(splitter.split(X, y, groups=groups))
        train_people = set(groups.iloc[idx_train])
        test_people = set(groups.iloc[idx_test])
        overlap = train_people & test_people
        report = {
            "grouped": True,
            "people_train": len(train_people),
            "people_test": len(test_people),
            "people_in_both": len(overlap),   # must be 0
            "leak_free": len(overlap) == 0,
        }

    report.update(
        {
            "rows_train": int(len(idx_train)),
            "rows_test": int(len(idx_test)),
            "positive_rate_train": round(float(y.iloc[idx_train].mean()) * 100, 2),
            "positive_rate_test": round(float(y.iloc[idx_test].mean()) * 100, 2),
        }
    )

    return {
        "X_train": X.iloc[idx_train], "X_test": X.iloc[idx_test],
        "y_train": y.iloc[idx_train], "y_test": y.iloc[idx_test],
        "idx_train": idx_train, "idx_test": idx_test,
        "report": report,
    }


# ---------------------------------------------------------------------------
def build_models(scale_pos_weight: float) -> dict[str, Any]:
    """Create the candidate models, each configured for imbalanced data.

    Logistic regression is wrapped in a Pipeline with StandardScaler because it
    is sensitive to feature scale: `num_lab_procedures` runs 1-132 while the
    one-hot flags are 0-1, and without scaling the large-range feature dominates
    the fit for no good reason. Tree models are scale-invariant and need no such
    step — a difference worth knowing rather than cargo-culting a scaler onto
    everything.
    """
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=None,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,      # guards against memorising single patients
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,   # XGBoost's imbalance handling
            eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1,
        )
    else:
        # Always-available stand-in so the comparison is never a table of two.
        models["Gradient Boosting"] = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            class_weight="balanced", random_state=RANDOM_SEED,
        )

    return models


# ---------------------------------------------------------------------------
def train_all(
    df: pd.DataFrame, test_size: float = 0.25, verbose: bool = True
) -> dict[str, Any]:
    """Train every candidate, compare them, and keep the best.

    "Best" is decided on ROC-AUC, not accuracy — see the module docstring.
    """
    def say(msg: str = "") -> None:
        if verbose:
            print(msg)

    X, y, groups = F.build_feature_matrix(df)
    if y is None:
        raise ValueError(
            f"No '{F.TARGET}' column found. Training needs labelled data — run "
            "scripts/03_load_data.py first."
        )

    say("=" * 66)
    say("MODULE 4  TRAIN A READMISSION MODEL")
    say("=" * 66)
    say(f"  encounters        {len(X):,}")
    say(f"  features built    {X.shape[1]}")
    say(f"  positive class    {y.mean() * 100:.2f}%  ({int(y.sum()):,} readmitted)")
    say(f"  features excluded {len(F.LEAKY_FEATURES)} (leakage) + 1 (fairness)")
    say()

    split = split_by_person(X, y, groups, test_size=test_size)
    rep = split["report"]
    say("SPLIT  (grouped by person, so nobody is in both halves)")
    if rep.get("grouped"):
        say(f"  people in train   {rep['people_train']:,}")
        say(f"  people in test    {rep['people_test']:,}")
        say(f"  people in BOTH    {rep['people_in_both']}   <- must be 0")
    else:
        say(f"  WARNING: {rep.get('warning')}")
    say(f"  rows train/test   {rep['rows_train']:,} / {rep['rows_test']:,}")
    say(f"  positive rate     {rep['positive_rate_train']}% / {rep['positive_rate_test']}%")
    say()

    y_train = split["y_train"]
    n_neg = int((y_train == 0).sum())
    n_pos = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = n_neg / n_pos

    baseline = majority_baseline(split["y_test"])
    say(f"BASELINE  always predict 'not readmitted'")
    say(f"  accuracy          {baseline['accuracy'] * 100:.2f}%")
    say(f"  recall            {baseline['recall'] * 100:.2f}%   <- catches nobody")
    say("  Any model must be judged against this, not against zero.")
    say()

    models = build_models(scale_pos_weight)
    if not HAS_XGBOOST:
        say("  note: xgboost not installed — using scikit-learn's")
        say("        HistGradientBoosting instead (comparable performance).")
        say()

    results: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}

    for name, model in models.items():
        say(f"Training {name}...")
        started = time.time()
        model.fit(split["X_train"], y_train)
        elapsed = time.time() - started

        scores = evaluate_model(model, split["X_test"], split["y_test"])
        scores["model"] = name
        scores["train_seconds"] = round(elapsed, 1)
        results.append(scores)
        trained[name] = model

        say(f"  ROC-AUC {scores['roc_auc']:.3f} | PR-AUC {scores['pr_auc']:.3f} "
            f"| recall {scores['recall']:.3f} | precision {scores['precision']:.3f} "
            f"| {elapsed:.1f}s")

    comparison = pd.DataFrame(results)[
        ["model", "roc_auc", "pr_auc", "recall", "precision", "f1",
         "accuracy", "train_seconds"]
    ].sort_values("roc_auc", ascending=False).reset_index(drop=True)

    best_name = str(comparison.iloc[0]["model"])
    best_model = trained[best_name]
    best_scores = next(r for r in results if r["model"] == best_name)

    say()
    say("COMPARISON  (ranked by ROC-AUC)")
    say(comparison.to_string(index=False))
    say()
    say(f"WINNER: {best_name}")
    say(f"  ROC-AUC  {best_scores['roc_auc']:.3f}   (0.5 = coin flip, 1.0 = perfect)")
    say(f"  Recall   {best_scores['recall']:.3f}   vs {baseline['recall']:.3f} baseline")
    say(f"  Accuracy {best_scores['accuracy']:.3f}   vs {baseline['accuracy']:.3f} baseline")
    say()
    if best_scores["accuracy"] < baseline["accuracy"]:
        say("  NOTE: accuracy is BELOW the baseline, and that is expected.")
        say("  Balanced class weights make the model flag more patients, which")
        say("  costs accuracy and buys recall. Catching at-risk patients is the")
        say("  entire point; a 89%-accurate model that flags nobody is useless.")
        say()

    return {
        "best_name": best_name,
        "best_model": best_model,
        "best_scores": best_scores,
        "comparison": comparison,
        "split_report": rep,
        "baseline": baseline,
        "feature_names": list(X.columns),
        "all_models": trained,
        "split": split,
    }


# ---------------------------------------------------------------------------
def save_model(result: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Path]:
    """Persist the winning model plus everything needed to reuse it.

    Saving the model alone is not enough. Without `feature_names` in the same
    file, scoring a new patient produces a matrix with different one-hot columns
    and the prediction silently becomes nonsense.
    """
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Store band cut-offs derived from THIS model's own output distribution.
    # Without them the fixed thresholds put nearly every patient in HIGH — see
    # the calibration note in predict.py.
    from .predict import compute_band_thresholds

    split = result["split"]
    test_proba = result["best_model"].predict_proba(split["X_test"])[:, 1]

    bundle = {
        "band_thresholds": compute_band_thresholds(test_proba),
        "probability_range": [
            round(float(test_proba.min()), 4), round(float(test_proba.max()), 4)
        ],
        "model": result["best_model"],
        "model_name": result["best_name"],
        "feature_names": result["feature_names"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scores": result["best_scores"],
        "baseline": result["baseline"],
        "split_report": result["split_report"],
        "prediction_moment": F.describe_prediction_moment(),
    }
    joblib.dump(bundle, MODEL_PATH)

    metadata = {k: v for k, v in bundle.items() if k != "model"}
    metadata["n_features"] = len(result["feature_names"])
    metadata["training_rows"] = int(result["split_report"]["rows_train"])
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    result["comparison"].to_csv(COMPARISON_PATH, index=False)

    return {"model": MODEL_PATH, "metadata": METADATA_PATH, "comparison": COMPARISON_PATH}
