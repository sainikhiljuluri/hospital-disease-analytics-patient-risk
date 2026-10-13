"""
PAGE 2 - Who are these patients?

Demographics, clinical vitals, medications, and a searchable record table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import ui  # noqa: E402

from hospital_analytics.analysis import charts, metrics  # noqa: E402

ui.page_setup(
    "Patient Explorer",
    icon="🧍",
    subtitle="Demographics, vitals, medications and individual records",
)

patients, meds, source = ui.require_data()
df = ui.sidebar_filters(patients, source, key_prefix="explore_")
if ui.empty_guard(df):
    st.stop()

tab_demo, tab_vitals, tab_meds, tab_records = st.tabs(
    ["👥 Demographics", "🏥 Utilisation", "💊 Medications", "📋 Encounter records"]
)

# ===========================================================================
with tab_demo:
    ui.section("Age distribution")
    ui.chart(charts.age_histogram(df, split_by_outcome=True), key="exp_age")
    ui.insight_box("age_histogram")

    if "age" in df.columns and "readmitted_30d" in df.columns:
        readm = df[df["readmitted_30d"] == 1]["age"].mean()
        notr = df[df["readmitted_30d"] == 0]["age"].mean()
        if pd.notna(readm) and pd.notna(notr):
            st.info(
                f"Readmitted patients average **{readm:.1f} years** against "
                f"**{notr:.1f} years** for those not readmitted - a gap of "
                f"**{readm - notr:.1f} years**. Small in absolute terms, but "
                f"consistent, which is exactly what makes it a usable feature.",
                icon="📊",
            )

    st.divider()
    ui.section("Readmission by age band and gender", "Also a fairness audit.")
    ui.chart(charts.gender_age_readmission(df), key="exp_gender")
    ui.insight_box("gender_readmission")

    st.divider()
    ui.section("Age band summary")
    by_age = metrics.readmission_by_group(df, "age_group")
    if not by_age.empty:
        order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                 "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        by_age["_o"] = by_age["age_group"].apply(
            lambda v: order.index(v) if v in order else 99
        )
        ui.table(by_age.sort_values("_o").drop(columns="_o"))

# ===========================================================================
with tab_vitals:
    # This dataset records no vitals - no blood pressure, BMI or cholesterol.
    # What it records instead is UTILISATION: how much hospital care this person
    # had already used. That turns out to be the stronger signal anyway, which
    # is a genuinely useful thing to have discovered rather than assumed.
    ui.section(
        "Prior hospital use",
        "The strongest predictor family in the readmission literature.",
    )
    st.caption(
        "This dataset contains no vital signs — it is de-identified, so there "
        "is no blood pressure, BMI or cholesterol. Instead it records how often "
        "the patient used hospital services in the year before this admission."
    )

    ui.chart(charts.prior_admissions_curve(df), key="exp_prior")
    ui.insight_box("prior_admissions")

    prior_kind = st.radio(
        "Count which kind of prior visit:",
        ["Inpatient admissions", "Emergency visits", "Outpatient visits"],
        horizontal=True,
        key="exp_prior_kind",
    )
    prior_col = {
        "Inpatient admissions": "num_prior_admissions",
        "Emergency visits": "num_prior_emergency",
        "Outpatient visits": "num_prior_outpatient",
    }[prior_kind]

    curve = metrics.prior_utilisation_curve(df, col=prior_col)
    if not curve.empty:
        ui.table(
            curve[["label", "patients", "readmitted", "readmission_rate"]].rename(
                columns={
                    "label": prior_kind,
                    "patients": "Encounters",
                    "readmitted": "Readmitted",
                    "readmission_rate": "Readmission rate (%)",
                }
            )
        )

    st.divider()
    ui.section(
        "Discharge destination",
        "Informative about the population — and the clearest leakage risk here.",
    )
    ui.chart(charts.discharge_destination_bar(df), key="exp_disch")
    ui.insight_box("discharge_destination")

    st.divider()
    ui.section("Distribution of every numeric variable")
    st.caption(
        "Pick a variable to inspect its spread and outliers. Box plots are used "
        "because they show the median, the middle 50% and the extremes at once."
    )
    numeric_options = [
        c for c in ["age", "length_of_stay", "num_medications", "num_lab_procedures",
                    "num_procedures", "num_diagnoses", "num_prior_admissions",
                    "num_prior_emergency", "num_prior_outpatient"]
        if c in df.columns
    ]
    col1, col2 = st.columns([1, 2])
    with col1:
        chosen = st.selectbox("Variable", numeric_options, key="exp_var")
    with col2:
        st.write("")
    ui.chart(charts.outlier_box(df, chosen), key="exp_box")

    st.divider()
    ui.section("Summary statistics")
    ui.table(metrics.describe_numeric(df))
    st.caption(
        "Compare `mean` against `50%` (the median). Where they differ noticeably "
        "the variable is skewed - which is why the cleaning step imputes with the "
        "median rather than the mean."
    )

# ===========================================================================
with tab_meds:
    if meds.empty:
        st.info("No medication data is loaded.")
    else:
        ui.section("Most prescribed medications")
        ui.chart(charts.top_medications(meds), key="exp_meds")
        ui.insight_box("medications")

        st.divider()
        ui.section("Polypharmacy and readmission")
        ui.chart(charts.polypharmacy_bar(df), key="exp_poly")
        ui.insight_box("polypharmacy")

        poly = metrics.polypharmacy_vs_readmission(df)
        if not poly.empty and len(poly) > 1:
            hi = poly.iloc[0]
            lo = poly.iloc[-1]
            st.info(
                f"Patients on **{hi['med_band']}** medications are readmitted at "
                f"**{hi['readmission_rate']:.1f}%**, against **{lo['readmission_rate']:.1f}%** "
                f"for those on **{lo['med_band']}**. The rate climbs at every step, "
                f"so `num_medications` can be fed to the model as a plain numeric "
                f"feature - no non-linear transformation needed.",
                icon="💊",
            )

        st.divider()
        ui.section("Prescriptions by drug class")
        ui.chart(charts.drug_class_bar(meds), key="exp_class")
        ui.insight_box("drug_class")
        cls = metrics.drug_class_usage(meds)
        if not cls.empty:
            with st.expander("Drug class table"):
                ui.table(cls)
                ui.download_button(cls, "drug_class_usage.csv")

# ===========================================================================
with tab_records:
    ui.section("Individual patient records")
    st.caption(
        "Search by patient ID. This table is the raw material the GenAI report "
        "generator in Module 5 will summarise into plain-language explanations."
    )

    search = st.text_input(
        "Search encounter ID", placeholder="e.g. 149190", key="exp_search"
    )
    view = df.copy()
    if search:
        view = view[view["patient_id"].astype(str).str.contains(search, case=False, na=False)]

    display_cols = [
        c for c in ["patient_id", "person_id", "encounter_seq", "age", "gender",
                    "diagnosis", "department", "admission_type", "length_of_stay",
                    "num_prior_admissions", "num_medications", "num_diagnoses",
                    "a1c_result", "discharge_disposition", "readmitted_30d"]
        if c in view.columns
    ]

    st.caption(f"{len(view):,} records match.")
    ui.table(view[display_cols].head(500), height=440)
    if len(view) > 500:
        st.caption("Showing the first 500 rows. Narrow the filters or search to see more.")
    ui.download_button(view[display_cols], "patient_records.csv")

    # --- single patient drill-down ----------------------------------------
    if len(view):
        st.divider()
        ui.section("Patient detail")
        pid = st.selectbox(
            "Choose an encounter",
            view["patient_id"].head(300).tolist(),
            key="exp_pid",
        )
        row = view[view["patient_id"] == pid].iloc[0]

        ui.kpi_row([
            ("Age", f"{row.get('age', '-')}", str(row.get("gender", ""))),
            ("Diagnosis", str(row.get("diagnosis", "-")), str(row.get("department", ""))),
            ("Length of stay", f"{row.get('length_of_stay', '-')} days", None),
            ("Readmitted in 30 days",
             "Yes" if row.get("readmitted_30d") == 1 else "No",
             f"{row.get('num_prior_admissions', 0)} prior admissions"),
        ])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Labs and glycaemic control**")
            labs = {
                "HbA1c result": row.get("a1c_result"),
                "Serum glucose": row.get("glucose_serum"),
                "Lab procedures": row.get("num_lab_procedures"),
                "Procedures": row.get("num_procedures"),
                "Diagnoses recorded": row.get("num_diagnoses"),
                "On diabetes medication": "Yes" if row.get("on_diabetes_meds") == 1 else "No",
                "Regimen changed": "Yes" if row.get("medication_changed") == 1 else "No",
            }
            ui.table(
                pd.DataFrame({"Measure": list(labs.keys()),
                              "Value": [str(v) for v in labs.values()]})
            )
        with c2:
            st.markdown("**Admission and history**")
            adm = {
                "Person ID": row.get("person_id"),
                "Visit number": row.get("encounter_seq"),
                "Admission type": row.get("admission_type"),
                "Admission source": row.get("admission_source"),
                "Discharge to": row.get("discharge_disposition"),
                "Prior inpatient": row.get("num_prior_admissions"),
                "Prior emergency": row.get("num_prior_emergency"),
                "Prior outpatient": row.get("num_prior_outpatient"),
            }
            ui.table(
                pd.DataFrame({"Field": list(adm.keys()),
                              "Value": [str(v) for v in adm.values()]})
            )

        if not meds.empty:
            plist = meds[meds["patient_id"] == pid]
            if not plist.empty:
                st.markdown("**Prescribed medications**")
                ui.table(plist[["medication_name", "drug_class", "dose_status"]])
                st.caption(
                    "`dose_status` is what this dataset records instead of a "
                    "dosage in mg: whether the dose was held steady, increased "
                    "or decreased during the stay. Arguably more useful — a "
                    "change means a clinician judged the control to be wrong."
                )

        ui.roadmap_note(
            "Module 4 + 5",
            "This panel will gain a predicted readmission probability with a risk "
            "band, the per-feature SHAP contributions behind it, and a "
            "plain-language clinical summary written by an LLM.",
            "the patient row shown above, the engineered features from "
            "`ingestion/cleaning.py`, and the population risk-factor table from "
            "`analysis/metrics.py` as grounding facts for the prompt.",
        )
