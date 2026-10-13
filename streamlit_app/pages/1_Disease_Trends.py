"""
PAGE 1 - Disease and glycaemic-control patterns.

Answers: which conditions dominate, does long-term blood-sugar control relate to
readmission, and how does risk build across repeat visits?

NOTE ON WHAT IS NOT HERE
------------------------
There is no monthly trend or seasonality chart, because this dataset is fully
de-identified and contains no admission dates at all. Rather than fabricate a
time axis, the page uses `encounter_seq` - the order of a person's visits -
which is a genuine ordering the data does support.
"""

import sys
from pathlib import Path

# Pages live one folder deeper, so add streamlit_app/ to the path to find `ui`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
import ui  # noqa: E402

from hospital_analytics.analysis import charts, metrics  # noqa: E402

ui.page_setup(
    "Disease Trends",
    icon="📈",
    subtitle="Condition mix, glycaemic control and repeat-visit patterns",
)

patients, meds, source = ui.require_data()
df = ui.sidebar_filters(patients, source, key_prefix="trends_")
if ui.empty_guard(df):
    st.stop()

# ---------------------------------------------------------------------------
ui.section(
    "Glycaemic control and readmission",
    "The question this dataset was published to answer.",
)

ui.chart(charts.a1c_readmission_bar(df), key="trend_a1c")
ui.insight_box("a1c_testing")

a1c = metrics.a1c_testing_summary(df)
if not a1c.empty:
    not_tested = a1c[a1c["a1c_result"] == "Not Tested"]
    if not not_tested.empty:
        share = 100 * not_tested["patients"].iloc[0] / a1c["patients"].sum()
        st.info(
            f"**{share:.0f}%** of these diabetic admissions involved no HbA1c "
            f"measurement at all. That is the headline finding of the original "
            f"study: the decision not to test is the most common decision, and "
            f"whether the test was ordered carries information about how "
            f"thoroughly the patient's diabetes was managed during the stay.",
            icon="🩸",
        )
    with st.expander("HbA1c result table"):
        ui.table(a1c)
        ui.download_button(a1c, "a1c_readmission.csv")

st.divider()

# ---------------------------------------------------------------------------
ui.section(
    "Risk across repeat visits",
    "A sequence, not a time series - there are no dates in this dataset.",
)
ui.chart(charts.encounter_sequence_chart(df), key="trend_seq")
ui.insight_box("encounter_sequence")

st.divider()

# ---------------------------------------------------------------------------
ui.section("Length of stay by diagnosis", "Bed-days are the hospital's costliest resource.")
ui.chart(charts.los_box_by_diagnosis(df), key="trend_los")
ui.insight_box("los_box")

st.divider()

# ---------------------------------------------------------------------------
ui.section("Readmission rate by diagnosis")

by_diag = metrics.readmission_by_group(df, "diagnosis")
if not by_diag.empty:
    ui.table(
        by_diag.rename(
            columns={
                "diagnosis": "Diagnosis",
                "total_patients": "Patients",
                "readmissions": "Readmitted",
                "readmission_rate": "Readmission rate (%)",
                "avg_length_of_stay": "Avg stay (days)",
                "avg_age": "Avg age",
            }
        )
    )
    ui.download_button(by_diag, "readmission_by_diagnosis.csv")

    worst, best = by_diag.iloc[0], by_diag.iloc[-1]
    st.info(
        f"**{worst['diagnosis']}** has the highest readmission rate at "
        f"**{worst['readmission_rate']:.1f}%**, versus **{best['readmission_rate']:.1f}%** "
        f"for **{best['diagnosis']}** - a {worst['readmission_rate'] / max(best['readmission_rate'], 0.01):.1f}x gap. "
        f"The pattern is systematic rather than random: chronic, relapsing conditions "
        f"sit at the top and self-limiting acute conditions at the bottom. That is why "
        f"`is_chronic` is joined in from the reference table as an explicit feature.",
        icon="🔍",
    )

st.divider()

# ---------------------------------------------------------------------------
ui.section("Departmental load", "Volume plotted against outcome, to separate 'busy' from 'struggling'.")
ui.chart(charts.department_scatter(df), key="trend_dept")
ui.insight_box("department")

dept = metrics.department_summary(df)
if not dept.empty:
    with st.expander("Department table"):
        ui.table(dept)
        ui.download_button(dept, "department_summary.csv")
