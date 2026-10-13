"""
PAGE 3 - Correlation Lab.

The bridge between EDA (Module 2) and machine learning (Module 4): which
variables carry signal about readmission, and which merely duplicate each other.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
import ui  # noqa: E402

from hospital_analytics.analysis import charts, metrics  # noqa: E402

ui.page_setup(
    "Correlation Lab",
    icon="🔗",
    subtitle="Feature relationships, multicollinearity and a preview of feature importance",
)

patients, meds, source = ui.require_data()
df = ui.sidebar_filters(patients, source, key_prefix="corr_")
if ui.empty_guard(df):
    st.stop()

# ---------------------------------------------------------------------------
ui.section("What tracks readmission?", "Ranked by strength of linear relationship.")
ui.chart(charts.target_correlation_bar(df), key="corr_target")

tc = metrics.target_correlations(df)
if not tc.empty:
    top3 = tc.head(3)
    names = ", ".join(f"`{r.feature}` ({r.correlation:+.3f})" for r in top3.itertuples())
    st.info(
        f"**The three strongest signals are {names}.** These will very likely be "
        f"the top-ranked features once the model is trained in Module 4 - so if it "
        f"reports something wildly different, that is a signal to go looking for a "
        f"bug rather than a discovery.\n\n"
        f"**A caution to state out loud:** correlation only detects *linear* "
        f"relationships. A variable with near-zero correlation can still matter "
        f"enormously if its effect is non-linear or only appears in combination "
        f"with another variable. That is precisely why we will use tree-based "
        f"models such as Random Forest and XGBoost, which find those interactions "
        f"automatically, rather than relying on this table alone.",
        icon="🧭",
    )
    with st.expander("Full correlation-with-target table"):
        ui.table(tc)
        ui.download_button(tc, "target_correlations.csv")

ui.insight_box("correlation_heatmap", show_finding=False)

st.divider()

# ---------------------------------------------------------------------------
ui.section("Full correlation matrix", "Read the off-diagonal cells to spot redundant variables.")
ui.chart(charts.correlation_heatmap(df), key="corr_heat")

corr = metrics.correlation_matrix(df)
if not corr.empty:
    # Find variable pairs that are strongly related to each other.
    pairs = []
    cols = [c for c in corr.columns if c != "readmitted_30d"]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = corr.loc[a, b]
            if abs(val) >= 0.30:
                pairs.append({"variable_1": a, "variable_2": b, "correlation": round(float(val), 3)})

    if pairs:
        import pandas as pd

        pdf = pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False)
        st.warning(
            "**Multicollinearity check.** These variable pairs move together with "
            "|r| ≥ 0.30. They carry overlapping information, so a linear model "
            "given both will produce unstable, hard-to-interpret coefficients. "
            "In Module 4 the plan is to keep both for the tree-based models "
            "(which are unaffected) and drop one from the logistic regression.",
            icon="⚠️",
        )
        ui.table(pdf)
    else:
        st.success("No variable pair exceeds |r| = 0.30 - no multicollinearity concern.", icon="✅")

st.divider()

# ---------------------------------------------------------------------------
ui.section("Compare any two variables yourself")
st.caption(
    "Pick two variables to see how they relate within each readmission outcome. "
    "Colouring by outcome is the point: it shows whether the *combination* of the "
    "two separates readmitted patients better than either does alone."
)

numeric_cols = [c for c in metrics.NUMERIC_FEATURES if c in df.columns and c != "readmitted_30d"]
c1, c2, c3 = st.columns(3)
with c1:
    xcol = st.selectbox("X axis", numeric_cols, index=0, key="corr_x")
with c2:
    ycol = st.selectbox(
        "Y axis", numeric_cols,
        index=min(4, len(numeric_cols) - 1), key="corr_y",
    )
with c3:
    trend = st.selectbox("Trend line", ["none", "ols", "lowess"], key="corr_trend")

if xcol == ycol:
    st.warning("Choose two different variables.")
else:
    import plotly.express as px

    sample = df.sample(min(2500, len(df)), random_state=42).copy()
    sample["Outcome"] = sample["readmitted_30d"].map({0: "Not readmitted", 1: "Readmitted"})
    try:
        fig = px.scatter(
            sample,
            x=xcol,
            y=ycol,
            color="Outcome",
            opacity=0.55,
            trendline=None if trend == "none" else trend,
            color_discrete_map={
                "Not readmitted": charts.THEME["primary"],
                "Readmitted": charts.THEME["danger"],
            },
            title=f"{ycol} against {xcol}",
        )
    except Exception:  # noqa: BLE001 - trendlines need statsmodels installed
        st.caption("Trend lines require the `statsmodels` package; showing plain points.")
        fig = px.scatter(
            sample, x=xcol, y=ycol, color="Outcome", opacity=0.55,
            color_discrete_map={
                "Not readmitted": charts.THEME["primary"],
                "Readmitted": charts.THEME["danger"],
            },
            title=f"{ycol} against {xcol}",
        )
    fig.update_layout(template="plotly_white", height=520)
    ui.chart(fig, key="corr_custom")

    r = df[[xcol, ycol]].corr().iloc[0, 1]
    strength = (
        "very strong" if abs(r) >= 0.7 else
        "strong" if abs(r) >= 0.5 else
        "moderate" if abs(r) >= 0.3 else
        "weak" if abs(r) >= 0.1 else
        "negligible"
    )
    st.caption(
        f"Pearson r = **{r:+.3f}** - a {strength} "
        f"{'positive' if r >= 0 else 'negative'} linear relationship. "
        f"Correlation is not causation: both variables may simply be driven by a "
        f"third, such as age."
    )

st.divider()

ui.roadmap_note(
    "Module 4 - Machine learning",
    "Train Logistic Regression, Random Forest and XGBoost to predict "
    "`readmitted_30d`; compare them on accuracy, precision, recall, F1 and "
    "ROC-AUC; and replace the correlation ranking on this page with real model "
    "feature importances and SHAP values.",
    "the engineered features from `ingestion/cleaning.py`, the numeric feature "
    "list already defined in `analysis/metrics.NUMERIC_FEATURES`, and this page's "
    "multicollinearity findings to decide which columns each model receives.",
)
