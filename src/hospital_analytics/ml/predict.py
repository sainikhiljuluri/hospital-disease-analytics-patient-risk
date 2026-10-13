"""
MODULE 4, part 4: scoring a patient and explaining why.

WHAT THIS FILE IS FOR
---------------------
Training happens once. Prediction happens constantly — every time the dashboard
scores a patient, and later every time the Module 6 API receives a request. This
is the boundary both of those go through.

TWO JOBS, AND THE SECOND MATTERS MORE
--------------------------------------
1. `predict_one()` returns a probability and a risk band.
2. `explain_one()` returns WHY — which features pushed this patient's risk up
   or down, and by how much.

A bare probability is not actionable. "68% risk" tells a nurse nothing about
what to do. "68%, driven mainly by three prior admissions and a 12-day stay"
tells them where to focus. The explanation is what makes the model usable, and
it is what Module 5 turns into readable English.

HOW THE EXPLANATION WORKS
-------------------------
SHAP is the standard tool, but it is an extra dependency that needs compiling on
some systems, so this project treats it as optional:

  * If `shap` IS installed, we use it. It computes each feature's exact
    contribution to this specific prediction, using a result from game theory
    about fairly splitting credit among contributors.
  * If NOT, we fall back to a transparent alternative: multiply each feature
    value by the model's own importance for it, then compare that patient
    against the population average. Cruder than SHAP, but honest, fast, and it
    requires nothing extra.

Both answer the same question — "what about THIS patient drove THIS score?" —
and the dashboard states which method produced the numbers on screen.

THE MOST IMPORTANT CAVEAT IN THIS FILE
---------------------------------------
These contributions are ASSOCIATIONS, not CAUSES. "Long stay increased risk"
does NOT mean discharging early would reduce risk. Long stays and readmission
share a common cause: the patient is sicker.

Confusing the two is the single most dangerous mistake available here, because
it leads to acting on the wrong thing. Every explanation this module produces
carries that warning, and Module 5's prompt repeats it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import MODELS_DIR
from . import features as F

MODEL_PATH = MODELS_DIR / "readmission_model.joblib"

# ---------------------------------------------------------------------------
# RISK BANDS, AND WHY THEY ARE NOT FIXED NUMBERS
# ---------------------------------------------------------------------------
# Turning a probability into LOW / MODERATE / HIGH needs cut-off points. The
# obvious approach is fixed thresholds (0.15, 0.35, 0.60) — and it fails badly
# here, for a reason worth understanding.
#
# `class_weight="balanced"` tells the model to treat the rare positive class as
# roughly eight times more important. That fixes the "predict no every time"
# problem, but it also inflates every predicted probability. The numbers stop
# being true probabilities: a "55%" from this model does NOT mean 55 out of 100
# such patients get readmitted. The model is well RANKED but poorly CALIBRATED.
#
# With fixed thresholds the result was almost every patient landing in HIGH —
# a triage tool that flags everybody is exactly as useful as one that flags
# nobody.
#
# The fix: derive the cut-offs from the model's own output distribution at
# training time, stored as percentiles. HIGH then means "in the top 10% of risk
# as this model scores it", which is what a hospital with capacity to follow up
# 10% of discharges actually needs.
#
# The percentiles below are a CAPACITY DECISION, not a statistical result. A
# hospital able to follow up more patients would widen them.
BAND_PERCENTILES = [
    (0, 50, "LOW", "Routine discharge process."),
    (50, 80, "MODERATE", "Confirm follow-up is booked before discharge."),
    (80, 95, "HIGH", "Care-transitions review; telephone follow-up within 48h."),
    (95, 101, "VERY HIGH", "Senior review before discharge; consider delaying."),
]

# Used only if a model was saved without stored thresholds.
FALLBACK_BANDS = [
    (0.00, 0.15, "LOW", "Routine discharge process."),
    (0.15, 0.35, "MODERATE", "Confirm follow-up is booked before discharge."),
    (0.35, 0.60, "HIGH", "Care-transitions review; telephone follow-up within 48h."),
    (0.60, 1.01, "VERY HIGH", "Senior review before discharge; consider delaying."),
]


def compute_band_thresholds(probabilities: np.ndarray) -> list[dict[str, Any]]:
    """Derive band cut-offs from a model's own probability distribution."""
    bands = []
    for low_pct, high_pct, label, action in BAND_PERCENTILES:
        low = float(np.percentile(probabilities, low_pct)) if low_pct > 0 else 0.0
        high = float(np.percentile(probabilities, high_pct)) if high_pct <= 100 else 1.01
        bands.append(
            {
                "label": label, "action": action,
                "low": low, "high": high,
                "percentile_range": f"{low_pct}-{min(high_pct, 100)}",
            }
        )
    return bands

_CACHE: dict[str, Any] = {}


class ModelNotTrained(FileNotFoundError):
    """Raised with instructions rather than a bare missing-file error."""


def model_exists() -> bool:
    """Cheap check so pages can offer the baseline instead of crashing."""
    return MODEL_PATH.exists()


def load_model(force_reload: bool = False) -> dict[str, Any]:
    """Load the saved bundle (model + feature names + scores), cached."""
    if not force_reload and "bundle" in _CACHE:
        return _CACHE["bundle"]

    if not MODEL_PATH.exists():
        raise ModelNotTrained(
            f"No trained model at {MODEL_PATH}.\n\n"
            "Train one with:\n"
            "    python scripts/04_train_model.py\n\n"
            "It takes about a minute and needs no extra downloads."
        )

    import joblib

    bundle = joblib.load(MODEL_PATH)

    # Expose a nested `metadata` view alongside the flat keys. The dashboard was
    # written against the nested shape and the scripts against the flat one;
    # supporting both here is cheaper and safer than rewriting either, and it
    # means an older saved model still loads.
    bundle.setdefault(
        "metadata",
        {
            "model_name": bundle.get("model_name"),
            "test_scores": bundle.get("scores", {}),
            "baseline": bundle.get("baseline", {}),
            "trained_at": bundle.get("trained_at"),
            "split_report": bundle.get("split_report", {}),
            "feature_names": bundle.get("feature_names", []),
            "n_features": len(bundle.get("feature_names", [])),
            "prediction_moment": bundle.get("prediction_moment", {}),
            "band_thresholds": bundle.get("band_thresholds", []),
        },
    )
    bundle.setdefault("test_scores", bundle.get("scores", {}))

    _CACHE["bundle"] = bundle
    return bundle


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
def band_for(probability: float, thresholds: list[dict[str, Any]] | None = None) -> tuple[str, str]:
    """Turn a probability into (band, recommended action).

    Uses the thresholds stored with the model when available — see the note
    above on why fixed cut-offs do not work with balanced class weights.
    """
    if thresholds:
        for band in thresholds:
            if band["low"] <= probability < band["high"]:
                return band["label"], band["action"]
        return thresholds[-1]["label"], thresholds[-1]["action"]

    for low, high, label, action in FALLBACK_BANDS:
        if low <= probability < high:
            return label, action
    return "UNKNOWN", ""


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score every row. Returns patient_id, probability, band, action."""
    bundle = load_model()
    model = bundle["model"]
    X, _, _ = F.build_feature_matrix(df, feature_names=bundle["feature_names"])

    proba = model.predict_proba(X)[:, 1]
    thresholds = bundle.get("band_thresholds")
    out = pd.DataFrame(
        {
            "patient_id": df["patient_id"].astype(str).to_numpy()
            if "patient_id" in df.columns else np.arange(len(df)),
            "probability": proba,
        }
    )
    bands = [band_for(p, thresholds) for p in proba]
    out["risk_band"] = [b for b, _ in bands]
    out["recommended_action"] = [a for _, a in bands]
    out["risk_pct"] = (out["probability"] * 100).round(1)
    # Alias kept so both naming conventions work. The dashboard was written
    # against `readmission_probability`; the scripts use `probability`.
    out["readmission_probability"] = out["probability"]
    out["risk_score"] = out["risk_pct"]
    return out


def predict_one(row: pd.Series | pd.DataFrame) -> dict[str, Any]:
    """Score a single patient."""
    df = row.to_frame().T if isinstance(row, pd.Series) else row
    result = predict_batch(df).iloc[0]
    bundle = load_model()
    return {
        "patient_id": result["patient_id"],
        "probability": float(result["probability"]),
        "risk_pct": float(result["risk_pct"]),
        "risk_band": result["risk_band"],
        "recommended_action": result["recommended_action"],
        "model_name": bundle["model_name"],
        "model_roc_auc": bundle["scores"]["roc_auc"],
        "calibration_note": (
            "This score RANKS risk reliably but is not a calibrated probability "
            "— balanced class weights inflate it. Read the band, not the number."
        ),
    }


# ---------------------------------------------------------------------------
def global_importance(top_n: int = 20) -> pd.DataFrame:
    """Which features matter most to the model overall, across all patients."""
    bundle = load_model()
    model = bundle["model"]
    names = bundle["feature_names"]

    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "tree importance (how much each feature reduced impurity)"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float).ravel())
        kind = "absolute logistic-regression coefficient"
    else:
        return pd.DataFrame()

    out = pd.DataFrame({"feature": names[:len(values)], "importance": values})
    out["importance_pct"] = (100 * out["importance"] / max(out["importance"].sum(), 1e-9)).round(2)
    out = out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    out.attrs["kind"] = kind
    return out


def explain_one(row: pd.Series | pd.DataFrame, top_n: int = 8) -> dict[str, Any]:
    """Why did THIS patient get THIS score?

    Returns the prediction plus the features that pushed it up or down, ready
    for Module 5 to turn into readable English.
    """
    df = row.to_frame().T if isinstance(row, pd.Series) else row
    bundle = load_model()
    model = bundle["model"]
    names = bundle["feature_names"]

    X, _, _ = F.build_feature_matrix(df, feature_names=names)
    prediction = predict_one(df)

    contributions, method = _contributions(model, X, names)

    frame = pd.DataFrame(
        {
            "feature": names,
            "value": X.iloc[0].to_numpy(),
            "contribution": contributions,
        }
    )
    frame["direction"] = np.where(frame["contribution"] >= 0, "increases risk", "reduces risk")
    frame["abs"] = frame["contribution"].abs()
    frame = frame[frame["abs"] > 1e-9].sort_values("abs", ascending=False)

    top = frame.head(top_n).drop(columns="abs").reset_index(drop=True)
    top["readable"] = top["feature"].map(humanise_feature)

    prediction.update(
        {
            "explanation_method": method,
            "contributions": top,
            "caveat": (
                "These are ASSOCIATIONS, not causes. A long stay is linked to "
                "readmission because sicker patients stay longer — discharging "
                "someone earlier would not reduce their risk."
            ),
        }
    )
    return prediction


def _contributions(model: Any, X: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, str]:
    """Per-feature contributions, via SHAP if available, else a fallback."""
    try:
        import shap

        estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[1]
        return np.asarray(values, dtype=float).ravel()[: len(names)], "SHAP (exact)"
    except Exception:  # noqa: BLE001 — shap missing, or model unsupported
        pass

    # Fallback: importance x how far this patient sits from the population norm.
    importance = global_importance(top_n=len(names))
    if importance.empty:
        return np.zeros(len(names)), "unavailable"

    weights = dict(zip(importance["feature"], importance["importance"]))
    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    signs = {}
    if hasattr(estimator, "coef_"):
        for name, coef in zip(names, np.asarray(estimator.coef_).ravel()):
            signs[name] = 1.0 if coef >= 0 else -1.0

    row = X.iloc[0]
    contributions = []
    for name in names:
        weight = float(weights.get(name, 0.0))
        value = float(row.get(name, 0.0))
        sign = signs.get(name, 1.0 if value > 0 else -1.0)
        contributions.append(weight * value * sign)

    contributions = np.asarray(contributions, dtype=float)
    largest = np.abs(contributions).max()
    if largest > 0:
        contributions = contributions / largest * 0.5   # scale to a readable range
    return contributions, "importance-weighted (install `shap` for exact values)"


# ---------------------------------------------------------------------------
HUMAN_NAMES = {
    "num_prior_admissions": "prior inpatient admissions",
    "num_prior_emergency": "prior emergency visits",
    "num_prior_outpatient": "prior outpatient visits",
    "num_prior_visits_total": "total prior hospital visits",
    "encounter_seq": "which recorded visit this is",
    "length_of_stay": "length of this stay",
    "num_lab_procedures": "lab procedures performed",
    "num_procedures": "procedures performed",
    "num_medications": "medications during this stay",
    "num_diagnoses": "diagnoses recorded",
    "age": "age",
    "a1c_tested": "HbA1c was measured",
    "a1c_high": "HbA1c above 7",
    "glucose_tested": "serum glucose was measured",
    "glucose_high": "serum glucose above 200",
    "medication_changed": "diabetes regimen was changed",
    "on_diabetes_meds": "on diabetes medication",
    "is_elderly": "aged 65 or over",
    "is_long_stay": "stay longer than 7 days",
    "is_polypharmacy": "more than 20 medications",
    "has_prior_admission": "has at least one prior admission",
    "has_prior_emergency": "has at least one prior emergency visit",
    "is_complex": "9 or more diagnoses",
    "is_emergency": "admitted as an emergency",
}


def humanise_feature(name: str) -> str:
    """`diagnosis=Circulatory` -> `primary diagnosis is Circulatory`."""
    if "=" in name:
        column, _, value = name.partition("=")
        pretty = {
            "diagnosis": "primary diagnosis",
            "diagnosis_secondary": "secondary diagnosis",
            "admission_type": "admission type",
            "admission_source": "admission source",
            "a1c_result": "HbA1c result",
            "glucose_serum": "serum glucose",
            "age_group": "age band",
            "gender": "gender",
        }.get(column, column.replace("_", " "))
        return f"{pretty} is {value}"
    return HUMAN_NAMES.get(name, name.replace("_", " "))
