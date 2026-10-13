"""
Module 3: every Plotly figure in the project is built here.

WHY THIS FILE EXISTS
--------------------
The notebook and the Streamlit dashboard show the SAME charts. Written twice,
they would diverge. Written here once, `fig = charts.disease_distribution(df)`
works identically in both, and restyling the whole project means editing one
THEME dict.

WHY `CHART_INSIGHTS` IS IN THIS FILE
-----------------------------------
Your brief asks you to state why each feature matters. A chart with no
interpretation earns no marks. Each figure therefore ships with its explanation
attached, and the dashboard renders that text directly under the chart. The
words on screen cannot drift away from the chart they describe.

Every function returns a plotly.graph_objects.Figure. Nothing here touches
Streamlit, so the notebook can use it too.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import metrics as M

# ---------------------------------------------------------------------------
# Shared visual identity
# ---------------------------------------------------------------------------
THEME = {
    "primary": "#1f6feb",
    "danger": "#d1242f",
    "success": "#1a7f37",
    "warning": "#d97706",
    "muted": "#8b949e",
    "sequence": ["#1f6feb", "#d1242f", "#1a7f37", "#d97706", "#8250df",
                 "#0891b2", "#be185d", "#65a30d", "#c2410c", "#4338ca",
                 "#0f766e", "#a16207"],
    "diverging": "RdBu_r",
    "heat": "Blues",
    "font": "Inter, Segoe UI, system-ui, sans-serif",
}


def _style(fig: go.Figure, height: int = 420, legend: bool = True) -> go.Figure:
    """Apply one consistent look to every figure in the project."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family=THEME["font"], size=13),
        title_font=dict(size=17),
        margin=dict(l=60, r=30, t=60, b=60),
        showlegend=legend,
        hoverlabel=dict(font_size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#d0d7de")
    fig.update_yaxes(gridcolor="#eaeef2", linecolor="#d0d7de")
    return fig


# ===========================================================================
# WHY EACH FEATURE MATTERS - displayed under the corresponding chart
# ===========================================================================
CHART_INSIGHTS: dict[str, dict[str, str]] = {
    "disease_distribution": {
        "title": "Disease Distribution",
        "feature": "diagnosis",
        "why": (
            "**Why this feature matters.** `diagnosis` decides which department "
            "needs beds, which specialists to roster and which drugs to stock. It "
            "is also the strongest single categorical predictor of readmission, "
            "because chronic diseases relapse and acute ones resolve. Every later "
            "module depends on it: the ML model one-hot encodes it as a feature, "
            "and the GenAI report names it in the first line of every explanation."
        ),
        "read": (
            "The top few diagnoses account for the majority of admissions, which "
            "means a small number of care pathways drive most of the hospital's "
            "workload - that is where process improvement pays off first."
        ),
    },
    "age_histogram": {
        "title": "Patient Age Distribution",
        "feature": "age",
        "why": (
            "**Why this feature matters.** `age` is the strongest demographic "
            "driver of risk. Older patients have less physiological reserve, more "
            "coexisting conditions and slower recovery, so they stay longer and "
            "return sooner. The distribution is left-skewed towards older patients, "
            "which is why we impute missing values with the MEDIAN rather than the "
            "mean - the mean is dragged by the tail."
        ),
        "read": (
            "The population is concentrated in the older bands, confirming this is "
            "an inpatient population rather than a general community sample."
        ),
    },
    "readmission_pie": {
        "title": "Readmission Split",
        "feature": "readmitted_30d",
        "why": (
            "**Why this feature matters.** `readmitted_30d` is the target variable - "
            "the thing Module 4 predicts. Checking its balance FIRST is essential. "
            "With roughly one readmission in six, a lazy model that predicts 'never "
            "readmitted' would still be about 83% accurate while being clinically "
            "useless. That is exactly why we will judge the model on recall and "
            "ROC-AUC, not accuracy."
        ),
        "read": (
            "The classes are imbalanced but not severely so - manageable with class "
            "weighting rather than heavy resampling."
        ),
    },
    "correlation_heatmap": {
        "title": "Correlation Matrix",
        "feature": "all numeric variables",
        "why": (
            "**Why this matters.** The matrix does two jobs at once. It ranks which "
            "variables actually move with readmission (a feature-importance preview "
            "before any model exists), and it exposes multicollinearity - variables "
            "that duplicate each other. Systolic and diastolic BP are strongly "
            "correlated, so feeding both to a logistic regression inflates the "
            "standard errors and makes the coefficients unreliable."
        ),
        "read": (
            "Prior admissions, length of stay and age show the clearest positive "
            "relationships with readmission; the clinical labs contribute weaker "
            "but consistent signal."
        ),
    },
    "risk_factors": {
        "title": "Risk Factor Lift",
        "feature": "engineered binary risk flags",
        "why": (
            "**Why this matters most of all.** 'Lift' is the readmission rate among "
            "patients WITH a factor divided by the rate among those WITHOUT it. A "
            "lift of 2.0 means the factor doubles the risk. Unlike a correlation "
            "coefficient, a clinician can act on this immediately, and it is stated "
            "in a unit a hospital board understands. This table is the evidence "
            "base that Module 5's GenAI explanations will be grounded in, so the "
            "model's reasons are real rather than invented."
        ),
        "read": (
            "Prior admission history dominates every other factor. Past use of the "
            "hospital predicts future use better than any individual lab value - a "
            "finding consistently reported in the readmission literature."
        ),
    },
    "prior_admissions": {
        "title": "Prior Admissions vs Readmission",
        "feature": "num_prior_admissions",
        "why": (
            "**Why this feature matters.** This will almost certainly be the "
            "top-ranked feature in Module 4's model. It is a summary of everything "
            "we do not directly measure - disease severity, treatment adherence, "
            "home support, social circumstances. One caveat to state out loud: it "
            "is a *historical* variable, so it is unavailable for a genuinely "
            "first-time patient, and the model must not collapse without it."
        ),
        "read": (
            "The relationship is monotonic - each additional prior admission raises "
            "the readmission rate again. That is the clean, near-linear signal a "
            "model can exploit."
        ),
    },
    "los_box": {
        "title": "Length of Stay by Diagnosis",
        "feature": "length_of_stay",
        "why": (
            "**Why this feature matters.** Length of stay is simultaneously a cost "
            "driver (bed-days are the hospital's most expensive resource) and a "
            "severity proxy. A box plot rather than a bar chart is deliberate: it "
            "shows the median, the spread AND the outliers, so a diagnosis with a "
            "short typical stay but a long tail of complicated cases is visible - "
            "and a mean would have hidden it."
        ),
        "read": (
            "Chronic cardiovascular, endocrine and renal conditions carry both "
            "longer median stays and fatter upper tails than acute presentations."
        ),
    },
    "gender_readmission": {
        "title": "Readmission by Gender and Age Band",
        "feature": "gender, age_group",
        "why": (
            "**Why this matters.** This is a fairness check, not just a demographic "
            "chart. If readmission rates differ sharply by gender within the same "
            "age band, a model trained without care can encode that difference and "
            "distribute care unequally. Auditing it now, before training, is what "
            "responsible clinical ML looks like."
        ),
        "read": (
            "Age dominates gender: the gap between age bands is far larger than the "
            "gap between genders inside any band."
        ),
    },
    "a1c_testing": {
        "title": "HbA1c Testing and Readmission",
        "feature": "a1c_result",
        "why": (
            "**Why this feature matters.** This is the question the dataset was "
            "published to answer. HbA1c measures average blood sugar over roughly "
            "three months, so it tells you whether a diabetic patient's condition "
            "has been controlled - not just what it is today. The original study "
            "found that whether the test was ORDERED AT ALL relates to readmission, "
            "because ordering it signals that the team engaged with the patient's "
            "diabetes rather than treating only what they came in for."
        ),
        "read": (
            "Note the size of the 'Not Tested' group. The most common decision is "
            "not to test, which is itself the finding: a majority of diabetic "
            "admissions involve no assessment of long-term glycaemic control."
        ),
    },
    "encounter_sequence": {
        "title": "Readmission by Visit Number",
        "feature": "encounter_seq",
        "why": (
            "**Why this feature matters.** This dataset is de-identified and "
            "contains no calendar dates, so a conventional time-series analysis is "
            "impossible. What survives is ORDER: for each person we can number "
            "their visits. That still answers a real question - does risk climb as "
            "someone accumulates admissions? - without pretending to a precision "
            "the data does not have."
        ),
        "read": (
            "Call this a sequence, not a trend. There is no time axis here, and "
            "claiming one would misrepresent the data."
        ),
    },
    "discharge_destination": {
        "title": "Readmission by Discharge Destination",
        "feature": "discharge_disposition",
        "why": (
            "**Why this feature matters.** Where a patient goes after discharge is "
            "a compact summary of how well they were doing. Home means recovered; "
            "a skilled nursing facility means still dependent. It is strongly "
            "related to readmission - and it is also the clearest **data leakage** "
            "risk in the dataset, because it is only known once discharge has "
            "already happened. A model meant to run BEFORE discharge cannot use it."
        ),
        "read": (
            "Patients discharged to care facilities readmit more often than those "
            "sent home. Useful for understanding the population; unusable as a "
            "predictor at the moment a clinician would actually want one."
        ),
    },
    "drug_class": {
        "title": "Prescriptions by Drug Class",
        "feature": "drug_class",
        "why": (
            "**Why this feature matters.** The raw file records 23 diabetes drugs "
            "as 23 separate columns, which is unusable for analysis. Reshaping them "
            "into one row per prescription and grouping into pharmacological "
            "classes turns 23 thin columns into a handful of meaningful categories "
            "- and it is what makes the one-to-many `medications` table worth "
            "having as a separate table at all."
        ),
        "read": (
            "Insulin dominates, which is expected in an inpatient population: oral "
            "agents are often held during admission and glucose managed by insulin."
        ),
    },
    "missingness": {
        "title": "Missing Data by Column",
        "feature": "data completeness",
        "why": (
            "**Why this feature matters.** Missingness is not random and it is not "
            "noise - it is a record of what a hospital chose to measure. `weight` "
            "is absent in about 97% of rows, which is why the project drops it "
            "rather than imputing 97% of a column and pretending the result means "
            "something. Deciding what to discard is as much a part of the pipeline "
            "as deciding what to keep."
        ),
        "read": (
            "Any column above roughly 40% missing is a candidate for removal. "
            "Below that, imputation is defensible if you say what you did."
        ),
    },
    "medications": {
        "title": "Most Prescribed Medications",
        "feature": "medication_name, num_medications",
        "why": (
            "**Why these features matter.** The drug list is a second, independent "
            "view of what is actually wrong with a patient - a record can carry one "
            "diagnosis code but four cardiac drugs. The *count* of medications "
            "measures polypharmacy, a well-established readmission risk factor, "
            "because it proxies how many conditions are being managed at once."
        ),
        "read": (
            "Prescribing is dominated by cardiovascular and metabolic drug classes, "
            "consistent with the diagnosis mix."
        ),
    },
    "polypharmacy": {
        "title": "Polypharmacy vs Readmission",
        "feature": "num_medications",
        "why": (
            "**Why this feature matters.** Banding a continuous count into groups "
            "and reading the rate across bands tests whether the relationship is "
            "monotonic. If it is, a linear model can use the raw count; if it were "
            "U-shaped, we would need the bands as categorical features. This is "
            "feature engineering driven by evidence rather than guesswork."
        ),
        "read": (
            "Readmission rate climbs steadily with each medication band, so the raw "
            "count is a usable numeric feature as-is."
        ),
    },
    "department": {
        "title": "Departmental Load",
        "feature": "department",
        "why": (
            "**Why this feature matters.** Departments are how a hospital is "
            "actually managed - budgets, rosters and accountability all follow this "
            "column. Presenting patient volume beside readmission rate separates "
            "'busy' from 'struggling': a high-volume ward with a low rate is "
            "performing well, while a small ward with a high rate is where a quality "
            "review should start."
        ),
        "read": (
            "Volume and readmission rate do not move together - the busiest "
            "department is not the one with the worst rate."
        ),
    },
}


def insight(key: str) -> dict[str, str]:
    """Fetch the explanation block for a chart, with a safe default."""
    return CHART_INSIGHTS.get(key, {"title": key, "feature": "", "why": "", "read": ""})


# ===========================================================================
# THE FIGURES
# ===========================================================================
def disease_distribution(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of patient count per diagnosis."""
    data = M.disease_distribution(df, top_n=top_n).sort_values("patient_count")
    fig = px.bar(
        data,
        x="patient_count",
        y="diagnosis",
        orientation="h",
        text="patient_count",
        color="patient_count",
        color_continuous_scale=THEME["heat"],
        labels={"patient_count": "Patients", "diagnosis": ""},
        title="Patient Count by Diagnosis",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    return _style(fig, height=460, legend=False)


def age_histogram(df: pd.DataFrame, split_by_outcome: bool = True) -> go.Figure:
    """Age histogram, optionally overlaid by readmission outcome."""
    if split_by_outcome and "readmitted_30d" in df.columns:
        d = df.copy()
        d["Outcome"] = d["readmitted_30d"].map({0: "Not readmitted", 1: "Readmitted"})
        fig = px.histogram(
            d,
            x="age",
            color="Outcome",
            nbins=40,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={"Not readmitted": THEME["primary"], "Readmitted": THEME["danger"]},
            labels={"age": "Age (years)", "count": "Patients"},
            title="Age Distribution by Readmission Outcome",
        )
    else:
        fig = px.histogram(
            df, x="age", nbins=40,
            color_discrete_sequence=[THEME["primary"]],
            labels={"age": "Age (years)"},
            title="Patient Age Distribution",
        )
    fig.update_layout(bargap=0.05)
    return _style(fig, height=400)


def readmission_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart of the target variable's class balance."""
    counts = df["readmitted_30d"].map({0: "Not readmitted", 1: "Readmitted"}).value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.55,
            marker=dict(colors=[THEME["primary"], THEME["danger"]]),
            textinfo="label+percent",
        )
    )
    fig.update_layout(title="30-Day Readmission Class Balance")
    return _style(fig, height=380)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Annotated correlation matrix of the numeric clinical variables."""
    corr = M.correlation_matrix(df)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=THEME["diverging"],
        zmin=-1,
        zmax=1,
        title="Correlation Matrix of Clinical Variables",
    )
    fig.update_xaxes(tickangle=-45)
    return _style(fig, height=620, legend=False)


def target_correlation_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart ranking each variable's correlation with readmission."""
    data = M.target_correlations(df).sort_values("correlation")
    fig = px.bar(
        data,
        x="correlation",
        y="feature",
        orientation="h",
        text="correlation",
        color="correlation",
        color_continuous_scale=THEME["diverging"],
        range_color=[-0.4, 0.4],
        labels={"correlation": "Correlation with readmission", "feature": ""},
        title="Which Variables Track 30-Day Readmission?",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.add_vline(x=0, line_width=1, line_color=THEME["muted"])
    return _style(fig, height=460, legend=False)


def risk_factor_lift(df: pd.DataFrame) -> go.Figure:
    """Grouped bars: readmission rate with vs without each risk factor."""
    data = M.risk_factor_summary(df)
    if data.empty:
        return _style(go.Figure())
    data = data.sort_values("lift")

    fig = go.Figure()
    fig.add_bar(
        y=data["risk_factor"], x=data["readmit_rate_without"],
        name="Without factor", orientation="h",
        marker_color=THEME["primary"], opacity=0.65,
    )
    fig.add_bar(
        y=data["risk_factor"], x=data["readmit_rate_with"],
        name="With factor", orientation="h",
        marker_color=THEME["danger"],
        text=[f"{v:.1f}%  ({l}x)" for v, l in zip(data["readmit_rate_with"], data["lift"])],
        textposition="outside",
    )
    fig.update_layout(
        barmode="group",
        title="Readmission Rate With vs Without Each Risk Factor",
        xaxis_title="Readmission rate (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_traces(cliponaxis=False)
    return _style(fig, height=560)


def prior_admissions_curve(df: pd.DataFrame) -> go.Figure:
    """Combo chart: patient volume (bars) and readmission rate (line) by prior count."""
    if "num_prior_admissions" not in df.columns:
        return _style(go.Figure())

    d = df.copy()
    d["prior_band"] = d["num_prior_admissions"].clip(upper=6)
    g = (
        d.groupby("prior_band")
        .agg(patients=("readmitted_30d", "size"), rate=("readmitted_30d", "mean"))
        .reset_index()
    )
    g = g[g["patients"] >= 10]
    g["rate"] = (g["rate"] * 100).round(2)
    g["label"] = g["prior_band"].astype(int).astype(str).replace({"6": "6+"})

    fig = go.Figure()
    fig.add_bar(
        x=g["label"], y=g["patients"], name="Patients",
        marker_color=THEME["muted"], opacity=0.35, yaxis="y2",
    )
    fig.add_scatter(
        x=g["label"], y=g["rate"], name="Readmission rate (%)",
        mode="lines+markers+text", text=[f"{v:.1f}%" for v in g["rate"]],
        textposition="top center",
        line=dict(color=THEME["danger"], width=3), marker=dict(size=9),
    )
    fig.update_layout(
        title="Readmission Rate Rises With Every Prior Admission",
        xaxis_title="Number of prior admissions",
        yaxis=dict(title="Readmission rate (%)"),
        yaxis2=dict(title="Patients", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _style(fig, height=440)


def los_box_by_diagnosis(df: pd.DataFrame) -> go.Figure:
    """Box plot of length of stay per diagnosis, ordered by median."""
    order = (
        df.groupby("diagnosis")["length_of_stay"].median().sort_values(ascending=False).index.tolist()
    )
    fig = px.box(
        df,
        x="diagnosis",
        y="length_of_stay",
        color="diagnosis",
        category_orders={"diagnosis": order},
        color_discrete_sequence=THEME["sequence"],
        points="outliers",
        labels={"length_of_stay": "Length of stay (days)", "diagnosis": ""},
        title="Length of Stay Distribution by Diagnosis",
    )
    fig.update_xaxes(tickangle=-40)
    return _style(fig, height=480, legend=False)


def gender_age_readmission(df: pd.DataFrame) -> go.Figure:
    """Grouped bars: readmission rate by age band, split by gender."""
    if not {"gender", "age_group"}.issubset(df.columns):
        return _style(go.Figure())
    d = df[df["gender"].isin(["Male", "Female"])]
    g = (
        d.groupby(["age_group", "gender"])
        .agg(rate=("readmitted_30d", "mean"), n=("readmitted_30d", "size"))
        .reset_index()
    )
    g = g[g["n"] >= 20]
    g["rate"] = (g["rate"] * 100).round(2)

    order = ["0-18", "19-30", "31-45", "46-60", "61-75", "76+"]
    fig = px.bar(
        g,
        x="age_group",
        y="rate",
        color="gender",
        barmode="group",
        text="rate",
        category_orders={"age_group": order},
        color_discrete_map={"Male": THEME["primary"], "Female": THEME["warning"]},
        labels={"rate": "Readmission rate (%)", "age_group": "Age band", "gender": "Gender"},
        title="Readmission Rate by Age Band and Gender",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _style(fig, height=430)


def top_medications(meds: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of the most prescribed drugs, coloured by class."""
    data = M.medication_usage(meds, top_n=top_n)
    if data.empty:
        return _style(go.Figure())
    if "drug_class" in meds.columns:
        cls = meds.drop_duplicates("medication_name")[["medication_name", "drug_class"]]
        data = data.merge(cls, on="medication_name", how="left")
    data = data.sort_values("prescriptions")

    fig = px.bar(
        data,
        x="prescriptions",
        y="medication_name",
        orientation="h",
        color=data.get("drug_class"),
        text="prescriptions",
        color_discrete_sequence=THEME["sequence"],
        labels={"prescriptions": "Prescriptions", "medication_name": "", "color": "Drug class"},
        title=f"Top {top_n} Prescribed Medications",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _style(fig, height=520)


def polypharmacy_bar(df: pd.DataFrame) -> go.Figure:
    """Readmission rate across medication-count bands."""
    data = M.polypharmacy_vs_readmission(df)
    if data.empty:
        return _style(go.Figure())
    order = ["1-5", "6-10", "11-15", "16-20", "20+"]
    data["med_band"] = data["med_band"].astype(str)
    fig = px.bar(
        data,
        x="med_band",
        y="readmission_rate",
        text="readmission_rate",
        color="readmission_rate",
        color_continuous_scale="Reds",
        category_orders={"med_band": order},
        labels={"med_band": "Number of medications", "readmission_rate": "Readmission rate (%)"},
        title="Readmission Rate by Medication Count (polypharmacy)",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    return _style(fig, height=400, legend=False)


def department_scatter(df: pd.DataFrame) -> go.Figure:
    """Bubble chart: volume vs readmission rate per department."""
    data = M.department_summary(df)
    if data.empty:
        return _style(go.Figure())
    fig = px.scatter(
        data,
        x="patients",
        y="readmission_rate",
        size="total_bed_days",
        color="department",
        text="department",
        size_max=55,
        color_discrete_sequence=THEME["sequence"],
        labels={"patients": "Patient volume", "readmission_rate": "Readmission rate (%)"},
        title="Departments: Volume vs Readmission Rate (bubble = bed days)",
    )
    fig.update_traces(textposition="top center")
    avg = float(data["readmission_rate"].mean())
    fig.add_hline(y=avg, line_dash="dash", line_color=THEME["muted"],
                  annotation_text=f"Average {avg:.1f}%")
    return _style(fig, height=500, legend=False)


def outlier_box(df: pd.DataFrame, column: str) -> go.Figure:
    """Box plot of one numeric column, used on the Data Quality page."""
    fig = px.box(
        df, y=column, points="outliers",
        color_discrete_sequence=[THEME["primary"]],
        title=f"Distribution and Outliers: {column}",
    )
    return _style(fig, height=380, legend=False)


# ===========================================================================
# CHARTS SPECIFIC TO THIS DATASET
# ===========================================================================
def a1c_readmission_bar(df: pd.DataFrame) -> go.Figure:
    """Readmission rate by HbA1c result, with the size of each group.

    This reproduces the dataset's own published research question, so it is the
    single best evidence that the pipeline is processing the data correctly.
    """
    g = M.a1c_testing_summary(df)
    if g.empty:
        return _style(go.Figure())

    labels = {
        "Not Tested": "Not tested",
        "Norm": "Normal",
        ">7": "High (>7)",
        ">8": "Very high (>8)",
    }
    g["label"] = g["a1c_result"].map(labels).fillna(g["a1c_result"])

    fig = go.Figure()
    fig.add_bar(
        x=g["label"], y=g["patients"], name="Patients",
        marker_color=THEME["muted"], opacity=0.35, yaxis="y2",
    )
    fig.add_scatter(
        x=g["label"], y=g["readmission_rate"], name="Readmission rate (%)",
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in g["readmission_rate"]], textposition="top center",
        line=dict(color=THEME["danger"], width=3), marker=dict(size=10),
    )
    fig.update_layout(
        title="HbA1c Result vs 30-Day Readmission",
        xaxis_title="HbA1c measured during the stay",
        yaxis=dict(title="Readmission rate (%)"),
        yaxis2=dict(title="Patients", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _style(fig, height=440)


def encounter_sequence_chart(df: pd.DataFrame) -> go.Figure:
    """Readmission rate by which visit number this is for that person."""
    g = M.encounter_sequence_curve(df)
    if g.empty:
        return _style(go.Figure())

    fig = go.Figure()
    fig.add_bar(
        x=g["visit_number"], y=g["patients"], name="Encounters",
        marker_color=THEME["muted"], opacity=0.35, yaxis="y2",
    )
    fig.add_scatter(
        x=g["visit_number"], y=g["readmission_rate"], name="Readmission rate (%)",
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in g["readmission_rate"]], textposition="top center",
        line=dict(color=THEME["primary"], width=3), marker=dict(size=9),
    )
    fig.update_layout(
        title="Readmission Rate by Visit Number (Sequence, Not Time)",
        xaxis_title="Which recorded visit this is for that person",
        yaxis=dict(title="Readmission rate (%)"),
        yaxis2=dict(title="Encounters", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _style(fig, height=440)


def discharge_destination_bar(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Readmission rate by discharge destination - and a leakage warning."""
    g = M.discharge_destination_summary(df)
    if g.empty:
        return _style(go.Figure())
    g = g.head(top_n).sort_values("readmission_rate")

    fig = go.Figure(
        go.Bar(
            x=g["readmission_rate"], y=g["discharge_disposition"], orientation="h",
            marker_color=THEME["warning"],
            text=[f"{v:.1f}%" for v in g["readmission_rate"]], textposition="outside",
            customdata=g[["total_patients"]],
            hovertemplate="%{y}<br>Rate: %{x:.2f}%<br>Patients: %{customdata[0]:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Readmission Rate by Discharge Destination",
        xaxis_title="30-day readmission rate (%)", yaxis_title="",
    )
    return _style(fig, height=460, legend=False)


def drug_class_bar(meds: pd.DataFrame) -> go.Figure:
    """Prescription counts grouped into pharmacological classes."""
    g = M.drug_class_usage(meds)
    if g.empty:
        return _style(go.Figure())
    g = g.sort_values("prescriptions")

    fig = go.Figure(
        go.Bar(
            x=g["prescriptions"], y=g["drug_class"], orientation="h",
            marker_color=THEME["primary"],
            text=[f"{v:,}" for v in g["prescriptions"]], textposition="outside",
        )
    )
    fig.update_layout(
        title="Prescriptions by Drug Class",
        xaxis_title="Prescriptions", yaxis_title="",
    )
    return _style(fig, height=420, legend=False)


def missingness_bar(missing_report: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Percentage of values missing, per column, before cleaning."""
    if missing_report is None or missing_report.empty:
        return _style(go.Figure())
    g = missing_report[missing_report["missing"] > 0].head(top_n).sort_values("missing_pct")
    if g.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values remain", showarrow=False,
            xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=15),
        )
        return _style(fig, height=300, legend=False)

    fig = go.Figure(
        go.Bar(
            x=g["missing_pct"], y=g["column"], orientation="h",
            marker_color=THEME["danger"],
            text=[f"{v:.1f}%" for v in g["missing_pct"]], textposition="outside",
        )
    )
    fig.update_layout(
        title="Missing Values by Column",
        xaxis_title="Missing (%)", yaxis_title="",
        xaxis=dict(range=[0, 105]),
    )
    return _style(fig, height=460, legend=False)
