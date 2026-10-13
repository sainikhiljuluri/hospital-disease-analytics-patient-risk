"""
PAGE 5 - Risk Prediction (Module 4).

WHAT THIS PAGE DOES
-------------------
If a trained model exists in `models/`, this page serves it: per-patient
probabilities, per-feature explanations, evaluation against a baseline, a
threshold explorer and a fairness audit.

If no model has been trained yet, it falls back to the transparent rule-based
score - and tells you how to train one. Neither state is an error.

WHY THE RULE-BASED BASELINE IS STILL HERE
-----------------------------------------
It has not been deleted, and it should not be. It is the comparison point: a
model that cannot beat ten hand-weighted rules is not worth the complexity it
adds. Showing both side by side is what turns "my model got 0.63 ROC-AUC" into
evidence rather than a number with no context.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402

from hospital_analytics.analysis import metrics  # noqa: E402
from hospital_analytics.ml import predict as P  # noqa: E402

ui.page_setup(
    "Patient Risk Prediction",
    icon="🤖",
    subtitle="Module 4 - trained model predictions, explained per patient",
)

patients, meds, source = ui.require_data()
df = ui.sidebar_filters(patients, source, key_prefix="risk_")
if ui.empty_guard(df):
    st.stop()

# ===========================================================================
MODEL_READY = P.model_exists()

if MODEL_READY:
    meta = P.load_model()["metadata"]
    scores = meta["test_scores"]
    baseline = meta["baseline"]

    ui.section(
        f"Trained model: {meta['model_name']}",
        "Evaluated on encounters from people the model never saw during training.",
    )

    ui.kpi_row([
        ("ROC-AUC", f"{scores['roc_auc']:.3f}", "0.5 = coin flip"),
        ("Recall", f"{scores['recall'] * 100:.0f}%", "of readmissions caught"),
        ("Precision", f"{scores['precision'] * 100:.0f}%", "of flags that were right"),
        ("PR-AUC", f"{scores['pr_auc']:.3f}", f"vs {baseline['positive_rate']:.3f} by chance"),
    ])

    st.warning(
        f"**Read the accuracy trap before quoting any number here.** This model "
        f"scores **{scores['accuracy'] * 100:.1f}%** accuracy. A model that simply "
        f"predicts 'not readmitted' for everyone scores "
        f"**{baseline['accuracy'] * 100:.1f}%** — while catching nobody at all. "
        f"Accuracy is nearly meaningless on imbalanced data, which is why recall "
        f"and ROC-AUC lead the table above.",
        icon="⚠️",
    )
else:
    ui.roadmap_note(
        "No trained model found",
        "Everything below runs on the transparent rule-based baseline. To train "
        "the real model, run `python scripts/04_train_model.py` in your terminal "
        "and refresh this page.",
        "the cleaned table, the feature list in `ml/features.py`, and a "
        "train/test split grouped by `person_id` so the same person cannot "
        "appear in both halves.",
    )

st.divider()

# ===========================================================================
ui.section(
    "Interim: a transparent rule-based risk score",
    "Points assigned from the risk factors measured in Module 2 - no model involved.",
)

st.markdown(
    """
Each factor below scores points, weighted by the **lift** measured on this
dataset (how many times more likely readmission is when the factor is present).
This is a *baseline*: it is completely transparent, needs no training, and gives
us a number the trained model has to beat before the extra complexity is
justified. Reporting a baseline is standard practice and it is what separates a
result from a claim.

One of these factors — **discharge destination** — is only known once discharge
has happened. It is included here because this is a descriptive scorer, not a
deployable one. Module 4 must decide whether the model runs *before* discharge,
and if so drop that feature. Using it would be textbook data leakage.
    """
)

# Weights are proportional to the lift each factor showed in Module 2 on THIS
# dataset. They are not tuned, and they are not learned - that is the point.
WEIGHTS = {
    "Prior inpatient admission": 25,
    "2+ prior inpatient admissions": 12,
    "Prior emergency visit": 12,
    "Discharged to a care facility": 12,
    "Stay longer than 7 days": 10,
    "Age 65 or over": 8,
    "Complex case (9+ diagnoses)": 7,
    "More than 20 medications": 6,
    "HbA1c above 7": 5,
    "Emergency admission": 3,
}


def score_row(row: pd.Series) -> tuple[int, list[str]]:
    """Return (score out of 100, list of factors that fired)."""
    hits: list[str] = []
    total = 0
    checks = [
        ("Prior inpatient admission", row.get("num_prior_admissions", 0) >= 1),
        ("2+ prior inpatient admissions", row.get("num_prior_admissions", 0) >= 2),
        ("Prior emergency visit", row.get("num_prior_emergency", 0) >= 1),
        ("Discharged to a care facility", row.get("to_care_facility", 0) == 1),
        ("Stay longer than 7 days", row.get("length_of_stay", 0) > 7),
        ("Age 65 or over", row.get("age", 0) >= 65),
        ("Complex case (9+ diagnoses)", row.get("num_diagnoses", 0) >= 9),
        ("More than 20 medications", row.get("num_medications", 0) > 20),
        ("HbA1c above 7", row.get("a1c_high", 0) == 1),
        ("Emergency admission", row.get("is_emergency", 0) == 1),
    ]
    for name, fired in checks:
        if bool(fired):
            total += WEIGHTS[name]
            hits.append(name)
    return total, hits


def band(score: int) -> tuple[str, str]:
    if score >= 45:
        return "HIGH", "🔴"
    if score >= 25:
        return "MODERATE", "🟠"
    return "LOW", "🟢"


# ---------------------------------------------------------------------------
_tab_labels = ["🧍 Score one patient", "📊 Does the score work?"]
if MODEL_READY:
    _tab_labels = [
        "🤖 Model prediction", "📈 Model evaluation", "🎚️ Choose a threshold",
        "⚖️ Fairness audit",
    ] + _tab_labels

_tabs = st.tabs(_tab_labels)
if MODEL_READY:
    tab_model, tab_eval, tab_thresh, tab_fair, tab_one, tab_all = _tabs
else:
    tab_one, tab_all = _tabs

with tab_one:
    pid = st.selectbox("Patient", df["patient_id"].head(400).tolist(), key="risk_pid")
    row = df[df["patient_id"] == pid].iloc[0]
    score, hits = score_row(row)
    label, emoji = band(score)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk score", f"{score}/110", f"{label} risk")
        st.progress(min(score / 110, 1.0))
        st.markdown(f"### {emoji} {label}")
        actual = row.get("readmitted_30d")
        if actual is not None:
            st.caption(
                f"Recorded outcome: **{'was readmitted' if actual == 1 else 'was not readmitted'}** "
                "within 30 days."
            )
    with c2:
        st.markdown("**Risk factors present in this patient**")
        if hits:
            ui.table(
                pd.DataFrame({
                    "Risk factor": hits,
                    "Points": [WEIGHTS[h] for h in hits],
                })
            )
        else:
            st.success("No listed risk factors are present.", icon="✅")

        absent = [h for h in WEIGHTS if h not in hits]
        if absent:
            with st.expander(f"{len(absent)} factors NOT present"):
                st.write(", ".join(absent))

    st.info(
        "**This is not machine learning.** The thresholds and weights were chosen "
        "by hand from the population statistics in Module 2. It cannot capture "
        "interactions between factors, it cannot output a calibrated probability, "
        "and it treats every patient above a threshold identically. Those three "
        "limitations are exactly what Module 4's trained model is for.",
        icon="⚠️",
    )

with tab_all:
    ui.section("Baseline validation", "Does a higher score actually mean higher readmission?")

    sample = df.head(4000).copy()
    scored = sample.apply(lambda r: score_row(r)[0], axis=1)
    sample["risk_score"] = scored
    sample["risk_band"] = sample["risk_score"].apply(lambda s: band(s)[0])

    summary = (
        sample.groupby("risk_band")
        .agg(patients=("patient_id", "count"), readmission_rate=("readmitted_30d", "mean"))
        .reset_index()
    )
    summary["readmission_rate"] = (summary["readmission_rate"] * 100).round(2)
    order = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    summary["_o"] = summary["risk_band"].map(order)
    summary = summary.sort_values("_o").drop(columns="_o")

    import plotly.express as px

    from hospital_analytics.analysis.charts import THEME

    fig = px.bar(
        summary,
        x="risk_band",
        y="readmission_rate",
        text="readmission_rate",
        color="risk_band",
        color_discrete_map={"LOW": THEME["success"], "MODERATE": THEME["warning"],
                            "HIGH": THEME["danger"]},
        labels={"risk_band": "Risk band", "readmission_rate": "Actual readmission rate (%)"},
        title="Actual Readmission Rate by Baseline Risk Band",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(template="plotly_white", height=420, showlegend=False)
    ui.chart(fig, key="risk_bands")

    ui.table(summary.rename(columns={
        "risk_band": "Risk band", "patients": "Patients",
        "readmission_rate": "Actual readmission rate (%)",
    }))

    if len(summary) >= 2:
        hi = summary[summary["risk_band"] == "HIGH"]["readmission_rate"]
        lo = summary[summary["risk_band"] == "LOW"]["readmission_rate"]
        if len(hi) and len(lo) and lo.iloc[0] > 0:
            st.success(
                f"The score separates the population: patients in the HIGH band are "
                f"readmitted at **{hi.iloc[0]:.1f}%** against **{lo.iloc[0]:.1f}%** in "
                f"the LOW band - a **{hi.iloc[0] / lo.iloc[0]:.1f}x** difference. "
                f"That is the bar Module 4's model has to clear.",
                icon="🎯",
            )

    st.caption(
        "Scored on the first 4,000 filtered patients to keep the page responsive."
    )

st.divider()

ui.section("The full risk factor evidence base")
ui.table(metrics.risk_factor_summary(df))
st.caption(
    "The weights in the score above are derived from the `lift` column here, so "
    "the baseline is grounded in this dataset rather than guessed."
)


# ===========================================================================
# MODULE 4 TABS - only rendered when a trained model exists
# ===========================================================================
if MODEL_READY:
    import plotly.express as px
    import plotly.graph_objects as go

    from hospital_analytics.analysis.charts import THEME, _style
    from hospital_analytics.ml import evaluate as E

    bundle = P.load_model()
    meta = bundle["metadata"]

    # -----------------------------------------------------------------------
    with tab_model:
        ui.section(
            "Predict for one patient",
            "The probability, the band, and — crucially — why.",
        )

        col_pick, col_mode = st.columns([2, 1])
        with col_mode:
            mode = st.radio(
                "Choose by", ["Highest risk first", "Patient ID"],
                key="mdl_mode",
            )

        scored = P.predict_batch(df)
        ranked = scored.sort_values("readmission_probability", ascending=False)

        with col_pick:
            if mode == "Highest risk first":
                options = ranked["patient_id"].head(200).tolist()
            else:
                options = sorted(df["patient_id"].astype(str).tolist())[:400]
            mpid = st.selectbox("Encounter", options, key="mdl_pid")

        mrow = df[df["patient_id"].astype(str) == str(mpid)].iloc[0]
        exp = P.explain_one(mrow)

        c1, c2 = st.columns([1, 2])
        with c1:
            colour = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "🟢"}
            st.metric("Readmission risk", f"{exp['risk_pct']:.1f}%",
                      f"{exp['risk_band']} band")
            st.progress(min(exp["probability"], 1.0))
            st.markdown(f"### {colour.get(exp['risk_band'], '⚪')} {exp['risk_band']}")
            st.info(exp["recommended_action"], icon="🩺")

            actual = mrow.get("readmitted_30d")
            if pd.notna(actual):
                st.caption(
                    f"Recorded outcome: **{'was readmitted' if actual == 1 else 'not readmitted'}** "
                    "within 30 days. One patient proves nothing either way — "
                    "judge the model on the evaluation tab, not on cherry-picked cases."
                )

        with c2:
            st.markdown("**Why this score**")
            contrib = exp["contributions"]
            if contrib is not None and not contrib.empty:
                plot = contrib.head(8).iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=plot["contribution"], y=plot["readable"], orientation="h",
                    marker_color=[
                        THEME["danger"] if d == "increases risk" else THEME["success"]
                        for d in plot["direction"]
                    ],
                    hovertemplate="%{y}<br>value: %{customdata}<extra></extra>",
                    customdata=plot["value"],
                ))
                fig.update_layout(
                    title="Feature contributions to this prediction",
                    xaxis_title="Pushes risk down  ←→  pushes risk up",
                    yaxis_title="", height=380,
                )
                ui.chart(_style(fig, legend=False), key="mdl_contrib")

        st.caption(f"Explanation method: {exp['explanation_method']}")
        st.warning(exp["caveat"], icon="🧭")
        st.caption(exp["calibration_note"])

    # -----------------------------------------------------------------------
    with tab_eval:
        ui.section("How good is it, really?",
                   "Every number here is measured on people the model never saw.")

        scores, baseline = meta["test_scores"], meta["baseline"]

        comparison = pd.DataFrame([
            {"Approach": "Always say 'not readmitted'",
             "Accuracy": f"{baseline['accuracy'] * 100:.1f}%",
             "Recall": "0.0%", "Precision": "n/a", "ROC-AUC": "0.500"},
            {"Approach": f"Trained model ({meta['model_name']})",
             "Accuracy": f"{scores['accuracy'] * 100:.1f}%",
             "Recall": f"{scores['recall'] * 100:.1f}%",
             "Precision": f"{scores['precision'] * 100:.1f}%",
             "ROC-AUC": f"{scores['roc_auc']:.3f}"},
        ])
        ui.table(comparison)

        st.markdown(
            "**Read the two rows together.** The baseline wins on accuracy and is "
            "worthless — it never identifies a single at-risk patient. The model "
            "trades some accuracy for recall, which is the trade a hospital "
            "actually wants: missing a patient who is readmitted costs far more "
            "than an unnecessary follow-up call."
        )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix**")
            cm = E.confusion_table(scores)
            ui.table(cm)
            st.caption(
                "Rows are what actually happened, columns are what the model said. "
                "The bottom-left cell — readmitted patients the model missed — is "
                "the one that matters clinically."
            )
        with c2:
            st.markdown("**What the numbers mean**")
            for line in E.interpret_scores(scores, baseline):
                st.markdown(f"- {line}")

        st.divider()
        ui.section("Model comparison")
        comp_path = meta.get("comparison_csv")
        if comp_path and Path(comp_path).exists():
            ui.table(pd.read_csv(comp_path))
            st.caption(
                "All candidates were trained on the same split with the same "
                "features. Ranked by ROC-AUC, because accuracy would rank them "
                "all as roughly equal and equally useless."
            )

    # -----------------------------------------------------------------------
    with tab_thresh:
        ui.section(
            "The threshold is a policy decision, not a technical one",
            "Move it and you are choosing who gets missed.",
        )
        st.markdown(
            "A model outputs a probability. Turning that into a yes/no flag needs "
            "a cut-off — and there is no mathematically correct value. It depends "
            "on how many follow-up calls your ward can make, and how much you "
            "care about missing someone.\n\n"
            "**Lower threshold** → catch more readmissions, more false alarms.\n"
            "**Higher threshold** → fewer wasted calls, more missed patients."
        )

        table = meta.get("threshold_table")
        if table:
            tdf = pd.DataFrame(table)
            fig = go.Figure()
            fig.add_scatter(x=tdf["threshold"], y=tdf["recall"] * 100,
                            name="Recall (% caught)", mode="lines+markers",
                            line=dict(color=THEME["danger"], width=3))
            fig.add_scatter(x=tdf["threshold"], y=tdf["precision"] * 100,
                            name="Precision (% flags correct)", mode="lines+markers",
                            line=dict(color=THEME["primary"], width=3))
            fig.add_scatter(x=tdf["threshold"], y=tdf["flagged_pct"],
                            name="% of patients flagged", mode="lines",
                            line=dict(color=THEME["muted"], width=2, dash="dot"))
            fig.update_layout(
                title="The trade-off, made explicit",
                xaxis_title="Decision threshold", yaxis_title="Percent",
                height=440,
            )
            ui.chart(_style(fig), key="mdl_thresh")
            ui.table(tdf)

        st.info(
            "**This is the slide to put in your report.** It shows you understand "
            "that a model is deployed into a system with finite capacity, not "
            "evaluated in a vacuum.",
            icon="💡",
        )

    # -----------------------------------------------------------------------
    with tab_fair:
        ui.section(
            "Fairness audit — run on predictions, not on inputs",
            "The model never sees ethnicity. That does not make it fair.",
        )
        st.markdown(
            "Ethnicity is deliberately excluded from the feature set, so the model "
            "cannot use it directly. But other features correlate with it — prior "
            "hospital use reflects access to care, which is not equally "
            "distributed. A model can therefore reproduce a disparity it was never "
            "shown.\n\n"
            "The only way to find out is to measure **recall by group**: of the "
            "patients in each group who were actually readmitted, what share did "
            "the model catch? If that differs sharply, some groups are being "
            "under-served by the very tool meant to help them."
        )

        audit = meta.get("fairness")
        if audit:
            adf = pd.DataFrame(audit)
            ui.table(adf)

            for attribute in adf["attribute"].unique():
                sub = adf[adf["attribute"] == attribute]
                gap = sub["recall"].max() - sub["recall"].min()
                if gap > 0.15:
                    st.warning(
                        f"**{attribute}**: recall ranges "
                        f"{sub['recall'].min():.2f} to {sub['recall'].max():.2f} "
                        f"(gap {gap:.2f}). Worth investigating — though small "
                        f"groups produce unstable rates, so check the patient "
                        f"counts before drawing conclusions.",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        f"**{attribute}**: recall gap {gap:.2f} across groups — "
                        f"no large disparity detected.",
                        icon="✅",
                    )

        stress = meta.get("subgroup")
        if stress:
            st.divider()
            ui.section("Stress test: first-time patients")
            ui.table(pd.DataFrame(stress))
            st.markdown(
                "Prior-visit counts are the strongest features — and they are all "
                "zero for someone with no recorded history. That is exactly when a "
                "clinician has least information and would most value a "
                "prediction. **Report both rows, not just the headline number.**"
            )
