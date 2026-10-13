"""
THE DASHBOARD ENTRY POINT - this is the file you run.

    streamlit run streamlit_app/app.py

Streamlit's multi-page convention: this file is the home page, and every .py
file inside `streamlit_app/pages/` automatically becomes a page in the sidebar
navigation, ordered by the number at the start of its filename. You do not write
any routing code - the folder layout IS the navigation.
"""

import ui  # noqa: F401  (must be first: it puts `src` on the import path)
import streamlit as st

from hospital_analytics.analysis import charts, metrics

# ---------------------------------------------------------------------------
ui.page_setup(
    "Hospital Disease Analytics",
    icon="🏥",
    subtitle=(
        "AI-Based Hospital Disease Analytics & Patient Risk Prediction  ·  "
        "Modules 1-3: data collection, analysis and visualisation"
    ),
)

patients, meds, source = ui.require_data()
df = ui.sidebar_filters(patients, source, key_prefix="home_")

if ui.empty_guard(df):
    st.stop()

k = metrics.kpis(df)

# ---------------------------------------------------------------------------
# HEADLINE KPIs
# ---------------------------------------------------------------------------
ui.section(
    "Hospital at a glance",
    "Every figure recalculates against the sidebar filters.",
)

ui.kpi_row([
    ("Encounters", f"{k['total_patients']:,}", f"{k['unique_people']:,} distinct people"),
    ("30-day readmission rate", f"{k['readmission_rate']:.1f}%", f"{k['readmitted_count']:,} encounters"),
    ("Average age", f"{k['avg_age']:.0f} yrs", "from 10-year bands"),
    ("Average stay", f"{k['avg_length_of_stay']:.1f} days", f"{k['total_bed_days']:,} bed days"),
])

st.write("")

ui.kpi_row([
    ("Diagnosis groups", f"{k['unique_diagnoses']}", "ICD-9 categories"),
    ("Had a prior admission", f"{k['pct_prior_admission']:.0f}%", "in the year before"),
    ("HbA1c tested", f"{k['pct_a1c_tested']:.0f}%", "during the stay"),
    ("Medications changed", f"{k['pct_med_changed']:.0f}%", "regimen adjusted"),
])

st.caption(
    f"{k['pct_male']:.0f}% male / {k['pct_female']:.0f}% female  ·  "
    f"{k['pct_repeat_patients']:.0f}% are a repeat visit by someone already in the data  ·  "
    f"average {k['avg_medications']:.0f} medications and "
    f"{k['avg_lab_procedures']:.0f} lab procedures per stay"
)

st.divider()

# ---------------------------------------------------------------------------
# THE HEADLINE FINDING
# ---------------------------------------------------------------------------
ui.section(
    "The single most important finding",
    "Which patient characteristics actually change the readmission rate?",
)

ui.chart(charts.risk_factor_lift(df), key="home_risk")
ui.insight_box("risk_factors")

risk = metrics.risk_factor_summary(df)
if not risk.empty:
    top = risk.iloc[0]
    st.success(
        f"**Strongest driver: {top['risk_factor']}.**  "
        f"{top['readmit_rate_with']:.1f}% of these patients are readmitted within "
        f"30 days, against {top['readmit_rate_without']:.1f}% of everyone else - "
        f"a {top['lift']:.2f}x increase in risk. This is the first feature the "
        f"Module 4 model will be built around.",
        icon="🎯",
    )
    with st.expander("See the full risk factor table"):
        ui.table(risk)
        ui.download_button(risk, "risk_factors.csv")

st.divider()

# ---------------------------------------------------------------------------
# DISEASE MIX AND OUTCOME BALANCE
# ---------------------------------------------------------------------------
ui.section("Disease mix and outcome balance")

left, right = st.columns([1.35, 1])
with left:
    ui.chart(charts.disease_distribution(df), key="home_disease")
with right:
    ui.chart(charts.readmission_pie(df), key="home_pie")
    st.caption(
        "Class balance drives how the ML model is evaluated - see the note below."
    )

ui.insight_box("disease_distribution")
ui.insight_box("readmission_pie")

st.divider()

# ---------------------------------------------------------------------------
# PRIOR ADMISSIONS
# ---------------------------------------------------------------------------
ui.section("Prior hospital use predicts future hospital use")
ui.chart(charts.prior_admissions_curve(df), key="home_prior")
ui.insight_box("prior_admissions")

st.divider()

# ---------------------------------------------------------------------------
# WHERE TO GO NEXT
# ---------------------------------------------------------------------------
ui.section("Explore further")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**📈 Disease Trends**\n\nSeasonality, monthly volume and length of stay by condition.")
with c2:
    st.markdown("**🧍 Patient Explorer**\n\nDemographics, vitals and the searchable record table.")
with c3:
    st.markdown("**🔗 Correlation Lab**\n\nWhich variables move together, and which track readmission.")
with c4:
    st.markdown("**🧪 Data Quality**\n\nProof of what the cleaning pipeline fixed.")

st.caption("Use the sidebar to switch pages.")

st.divider()
with st.expander("ℹ️  About this project and where it is going"):
    st.markdown(
        """
**Architecture built so far**

    data/raw/*.csv
        -> ingestion/cleaning.py        (validate, dedupe, impute, engineer)
        -> ingestion/load_to_mysql.py   (ETL into MySQL, with an audit log)
        -> analysis/data_access.py      (one loader, MySQL with CSV fallback)
        -> analysis/metrics.py          (every statistic, computed once)
        -> analysis/charts.py           (every figure, defined once)
        -> notebooks/  and  streamlit_app/   (two consumers, same library)

**Why it is laid out this way.** The notebook and this dashboard import the same
`metrics` and `charts` modules, so a number shown here can never disagree with
the notebook. The next two modules plug into the same library rather than
replacing it:

| Module | Status | Plugs in at |
|---|---|---|
| 1 · Data collection into MySQL | ✅ Done | `ingestion/` |
| 2 · Cleaning, EDA and statistics | ✅ Done | `ingestion/cleaning.py`, `analysis/metrics.py` |
| 3 · Visualisation dashboard | ✅ Done | `analysis/charts.py`, `streamlit_app/` |
| 4 · ML readmission model | 🚧 Next | `src/hospital_analytics/ml/` - trains on the features engineered in Module 2 |
| 5 · GenAI explanation report | 🚧 Planned | `src/hospital_analytics/genai/` - explains Module 4's output using the risk-factor table above |
| 6 · FastAPI service + React UI | 🚧 Planned | reuses `analysis/` unchanged |
        """
    )
