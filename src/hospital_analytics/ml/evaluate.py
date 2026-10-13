"""
MODULE 4, part 3: measuring whether a model is any good.

THE ONE THING TO UNDERSTAND
---------------------------
Because only about one encounter in nine ends in readmission, a model that
always says "no" is roughly 89% accurate and catches nobody. Accuracy is
therefore close to useless here, and quoting it without the baseline beside it
is misleading.

THE CONFUSION MATRIX, IN THIS PROJECT'S TERMS
---------------------------------------------
                        Actually readmitted   Actually not
    Model said YES      True Positive (TP)    False Positive (FP)
    Model said NO       False Negative (FN)   True Negative (TN)

  * FALSE NEGATIVE - a patient who WILL come back and we did not flag. They go
    home without extra support. **This is the expensive mistake.**
  * FALSE POSITIVE - a patient we flagged who was fine anyway. Costs a follow-up
    phone call. Annoying, cheap.

Because FN is far costlier than FP, we tune toward RECALL and accept more false
positives. That trade-off is a clinical decision, not a statistical one — which
is exactly the kind of judgement worth stating out loud in a demo.

THE METRICS
-----------
  RECALL (sensitivity)  TP / (TP + FN)
      Of the patients who came back, what share did we catch? Leads here.
  PRECISION             TP / (TP + FP)
      Of those we flagged, what share genuinely came back? Guards against
      flagging everybody.
  F1                    the harmonic mean of the two — one number when you need
      a single figure, but always show both underneath.
  ROC-AUC               Can the model rank a readmitted patient above a
      non-readmitted one? 0.5 = coin flip, 1.0 = perfect. Unaffected by class
      imbalance, which is why it decides the winner.
  PR-AUC                Like ROC-AUC but focused on the rare positive class. On
      heavily imbalanced data it is the more honest of the two, and it is
      compared against the positive rate rather than against 0.5.

REALISTIC EXPECTATIONS
----------------------
Published models on this dataset reach ROC-AUC around 0.64-0.70. If you see
0.95, you have leaked something — go and look for it. Knowing what a plausible
score looks like is itself a skill.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def majority_baseline(y_true: pd.Series) -> dict[str, float]:
    """Scores for a model that always predicts the majority class.

    Compute this FIRST. It is the number every model must beat, and reporting a
    result without it is how student projects overclaim.
    """
    y_true = np.asarray(y_true).astype(int)
    majority = 1 if y_true.mean() > 0.5 else 0
    y_pred = np.full_like(y_true, majority)
    return {
        "strategy": f"always predict {majority}",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
    }


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Every headline metric for one trained model."""
    y_test = np.asarray(y_test).astype(int)
    y_pred = model.predict(X_test).astype(int)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
        "flagged": int(tp + fp),
        "missed": int(fn),
    }


def confusion_table(scores: dict[str, Any]) -> pd.DataFrame:
    """The confusion matrix as a labelled table, in this project's language."""
    return pd.DataFrame(
        {
            "Actually readmitted": [scores["true_positives"], scores["false_negatives"]],
            "Actually not readmitted": [scores["false_positives"], scores["true_negatives"]],
        },
        index=["Model flagged as at-risk", "Model said not at-risk"],
    )


def interpret_scores(scores: dict[str, Any], baseline: dict[str, float]) -> list[str]:
    """Plain-English sentences about what the numbers mean.

    The dashboard prints these. A metric with no interpretation next to it is a
    number an evaluator can ask about and you cannot answer.
    """
    lines: list[str] = []
    total_positive = scores["true_positives"] + scores["false_negatives"]

    lines.append(
        f"Of {total_positive:,} patients in the test set who were genuinely "
        f"readmitted, the model flagged {scores['true_positives']:,} — a recall "
        f"of {scores['recall'] * 100:.1f}%. The always-say-no baseline catches 0."
    )
    lines.append(
        f"It missed {scores['false_negatives']:,}. Each miss is a patient who "
        f"went home without extra support, which is the expensive kind of error."
    )
    lines.append(
        f"It flagged {scores['flagged']:,} patients in total, of whom "
        f"{scores['precision'] * 100:.1f}% were genuinely readmitted. The rest "
        f"cost a follow-up call that turned out to be unnecessary."
    )
    lines.append(
        f"ROC-AUC is {scores['roc_auc']:.3f}. Given a random readmitted and a "
        f"random non-readmitted patient, the model ranks them correctly "
        f"{scores['roc_auc'] * 100:.1f}% of the time (50% would be a coin flip)."
    )
    lines.append(
        f"PR-AUC is {scores['pr_auc']:.3f} against a positive rate of "
        f"{baseline['positive_rate'] * 100:.1f}% — that rate, not 0.5, is the "
        f"floor for this metric."
    )

    if scores["accuracy"] < baseline["accuracy"]:
        lines.append(
            f"Accuracy ({scores['accuracy'] * 100:.1f}%) is BELOW the baseline "
            f"({baseline['accuracy'] * 100:.1f}%), and that is expected, not a "
            f"failure. Balanced class weights make the model flag more patients: "
            f"it trades accuracy for recall on purpose. A 89%-accurate model "
            f"that flags nobody helps nobody."
        )
    return lines


def roc_points(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Points for plotting the ROC curve."""
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame()
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(np.asarray(y_test).astype(int), y_proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def threshold_table(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series,
    thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
) -> pd.DataFrame:
    """How recall and precision trade off as the decision threshold moves.

    The default cut-off of 0.5 is a CONVENTION, not a law. A hospital with
    capacity to follow up 20% of discharges would pick the threshold that flags
    roughly that many. This table is what that conversation is based on.
    """
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame()
    y_true = np.asarray(y_test).astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        flagged = int(y_pred.sum())
        rows.append(
            {
                "threshold": t,
                "patients_flagged": flagged,
                "pct_flagged": round(100 * flagged / len(y_true), 1),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
                "missed": int(((y_true == 1) & (y_pred == 0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def fairness_audit(
    df_test: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray,
    attributes: list[str] | None = None, min_count: int = 50,
) -> pd.DataFrame:
    """Does the model perform equally well across demographic groups?

    Run on the model's PREDICTIONS, not on the input data. A model can be built
    from features that exclude ethnicity and still perform worse for one group,
    because other features correlate with it. Checking the inputs is not enough.

    Read the `recall` column: if the model catches 60% of at-risk patients in one
    group and 35% in another, it is distributing care unequally even though it
    never saw the attribute.
    """
    attributes = attributes or ["gender", "race", "age_group"]
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    rows = []
    for attr in attributes:
        if attr not in df_test.columns:
            continue
        for value, mask in df_test.groupby(attr).groups.items():
            positions = df_test.index.get_indexer(mask)
            if len(positions) < min_count:
                continue
            yt, yp = y_true[positions], y_pred[positions]
            if yt.sum() == 0:
                continue
            rows.append(
                {
                    "attribute": attr,
                    "group": str(value),
                    "patients": int(len(positions)),
                    "actual_readmission_rate": round(100 * float(yt.mean()), 2),
                    "flagged_rate": round(100 * float(yp.mean()), 2),
                    "recall": round(float(recall_score(yt, yp, zero_division=0)), 3),
                    "precision": round(float(precision_score(yt, yp, zero_division=0)), 3),
                }
            )
    return pd.DataFrame(rows)


def subgroup_check(
    df_test: pd.DataFrame, y_true: pd.Series, y_proba: np.ndarray,
    column: str = "num_prior_admissions",
) -> pd.DataFrame:
    """Performance on first-time patients versus everyone else.

    This is the honest stress test for this project. Prior-utilisation features
    are the strongest predictors — but they are ZERO for a genuinely first-time
    patient, which is exactly when a clinician has least information and would
    most value a prediction.

    If ROC-AUC collapses in the `no prior admissions` row, the model is much less
    useful than one overall number suggests. Report both.
    """
    if column not in df_test.columns:
        return pd.DataFrame()

    y_true = np.asarray(y_true).astype(int)
    has_prior = (pd.to_numeric(df_test[column], errors="coerce").fillna(0) > 0).to_numpy()

    rows = []
    for label, mask in [("no prior admissions", ~has_prior), ("has prior admissions", has_prior)]:
        if mask.sum() < 30 or len(np.unique(y_true[mask])) < 2:
            continue
        rows.append(
            {
                "subgroup": label,
                "patients": int(mask.sum()),
                "actual_readmission_rate": round(100 * float(y_true[mask].mean()), 2),
                "roc_auc": round(float(roc_auc_score(y_true[mask], y_proba[mask])), 3),
            }
        )
    return pd.DataFrame(rows)
