"""
Shared building blocks for every page of the dashboard.

WHY THIS FILE EXISTS
--------------------
Streamlit apps rot fast. Without a shared module, each page re-loads the data, 
re-declares the sidebar filters and re-styles its own headers, and after five 
pages nothing matches. Every page here imports `ui` and gets:

    ui.page_setup(...)      page title, icon, CSS
    ui.require_data()       cached data loading, with setup help if data is absent
    ui.sidebar_filters(df)  the identical filter panel on every page
    ui.kpi_row([...])       the metric cards
    ui.insight_box(key)     the "why this feature matters" panel
    ui.chart(fig)           render a plotly figure
    ui.table(df)            render a dataframe

CACHING - the single most important performance idea in Streamlit
----------------------------------------------------------------
Streamlit re-runs your whole script top to bottom on EVERY click. Without
caching, every checkbox tick would re-query MySQL and re-clean 10,000 rows, and
the app would feel broken. `@st.cache_data` makes the data load happen once.
"""

from __future__ import annotations

# --- bootstrap: make the project's `src` folder importable ------------------
# Streamlit runs this file from an arbitrary working directory, so we resolve
# paths from __file__ rather than trusting the current directory.
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _p in (str(PROJECT_ROOT / "src"), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from hospital_analytics.analysis import charts, metrics  # noqa: E402
from hospital_analytics.analysis.data_access import load_medications, load_patients  # noqa: E402
from hospital_analytics.db import test_connection  # noqa: E402

# ---------------------------------------------------------------------------
# Streamlit version compatibility
# ---------------------------------------------------------------------------
# Streamlit renamed the "fill the container" argument in v1.49. Detecting it once
# here means the app runs on both older and newer installs, and there is exactly
# one line to change if it ever moves again.
def _version_tuple() -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:  # noqa: BLE001
        return (1, 32)


_NEW_WIDTH_API = _version_tuple() >= (1, 49)


def chart(fig, key: str | None = None) -> None:
    """Render a Plotly figure at full container width."""
    if _NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df: pd.DataFrame, height: int | None = None, hide_index: bool = True) -> None:
    """Render a DataFrame at full container width."""
    kwargs = {"hide_index": hide_index}
    if height:
        kwargs["height"] = height
    if _NEW_WIDTH_API:
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)


# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------
CSS = """
<style>
  .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
  h1 {font-size: 2rem !important; font-weight: 700 !important;}
  h2 {font-size: 1.35rem !important; font-weight: 650 !important; margin-top: 1.6rem !important;}
  h3 {font-size: 1.1rem !important; font-weight: 600 !important;}

  /* KPI metric cards */
  div[data-testid="stMetric"] {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #1f6feb;
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(16,24,40,.05);
  }
  div[data-testid="stMetricLabel"] {
      font-size: .78rem !important;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: #64748b !important;
  }
  div[data-testid="stMetricValue"] {font-size: 1.6rem !important; font-weight: 700 !important;}

  /* "Why this feature matters" panel */
  .insight-box {
      background: #f6f9fe;
      border: 1px solid #d7e3f7;
      border-left: 4px solid #1f6feb;
      border-radius: 8px;
      padding: 14px 18px;
      margin: 6px 0 22px 0;
      font-size: .93rem;
      line-height: 1.6;
  }
  .insight-box .tag {
      display: inline-block;
      background: #1f6feb;
      color: #fff;
      font-size: .68rem;
      font-weight: 700;
      letter-spacing: .06em;
      padding: 2px 8px;
      border-radius: 4px;
      margin-bottom: 8px;
  }
  .finding-box {
      background: #f0fdf4;
      border-left: 4px solid #1a7f37;
      border-radius: 6px;
      padding: 10px 16px;
      margin: 4px 0 20px 0;
      font-size: .92rem;
  }
  .roadmap {
      background:#fffbeb; border:1px solid #fde68a; border-left:4px solid #d97706;
      border-radius:8px; padding:14px 18px; font-size:.92rem; line-height:1.6;
  }
  section[data-testid="stSidebar"] {background: #fbfcfe;}
</style>
"""


def page_setup(title: str, icon: str = "🏥", subtitle: str | None = None) -> None:
    """Call this as the FIRST Streamlit command on every page."""
    st.set_page_config(
        page_title=f"{title} | Hospital Analytics",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(f"{icon}  {title}")
    if subtitle:
        st.caption(subtitle)


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading hospital data...", ttl=3600)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load patients + medications once and reuse across pages and reruns.

    Returns (patients, medications, source_label).
    `ttl=3600` re-checks the source every hour, so a fresh MySQL load appears
    without restarting the app.
    """
    patients, source = load_patients(prefer="mysql")
    meds, _ = load_medications(prefer="mysql")
    return patients, meds, source


def require_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load the data, or show setup instructions instead of a stack trace.

    A missing dataset is the single most likely first-run problem, and a raw
    FileNotFoundError in the middle of a Streamlit page is not something a
    beginner can act on. This turns it into instructions.
    """
    try:
        return get_data()
    except FileNotFoundError as exc:
        st.error("No data found yet — the dataset has not been set up.")
        st.markdown(
            "This project runs on a **real public dataset** that you download "
            "once. It is not included in the repository because it is 18 MB and "
            "has its own permanent home."
        )
        st.markdown(
            "**1. Download it** (either source, same file):\n\n"
            "- [UCI — no account needed]"
            "(https://archive.ics.uci.edu/dataset/296/"
            "diabetes+130-us+hospitals+for+years+1999-2008)\n"
            "- [Kaggle — free login]"
            "(https://www.kaggle.com/datasets/brandao/diabetes)\n\n"
            "**2. Unzip it** and move `diabetic_data.csv` into the `data/raw/` "
            "folder of this project.\n\n"
            "**3. Run these two commands** in your terminal:"
        )
        st.code(
            "python scripts/01_check_dataset.py\n"
            "python scripts/03_load_data.py",
            language="bash",
        )
        st.caption("Then refresh this page.")
        with st.expander("Technical detail"):
            st.code(str(exc))
        st.stop()
        raise


@st.cache_data(ttl=3600)
def get_quality_report() -> dict:
    """Re-run the cleaning rules on the RAW file to report what was fixed.

    Cached because it re-reads and re-processes the raw CSV, which is the
    slowest thing the app does.
    """
    from hospital_analytics.config import RAW_DIR
    from hospital_analytics.ingestion.adapters import uci_diabetes
    from hospital_analytics.ingestion.cleaning import clean_patients

    try:
        raw, _, _, adapt_report = uci_diabetes.convert(RAW_DIR)
    except FileNotFoundError:
        # The raw download is not present. The dashboard still works from the
        # cleaned CSV; only this before/after comparison is unavailable.
        return {}
    _, report = clean_patients(raw)
    report["adapter"] = adapt_report
    return report


def clear_cache() -> None:
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar_filters(df: pd.DataFrame, source: str, key_prefix: str = "") -> pd.DataFrame:
    """Draw the shared filter panel and return the filtered DataFrame.

    Filters live in the sidebar so they persist as the user moves between pages
    (Streamlit keeps widget state keyed by the widget's `key`).
    """
    sb = st.sidebar
    sb.markdown("### 🏥 Hospital Analytics")

    # --- data source badge -------------------------------------------------
    if source == "MySQL":
        sb.success(f"Data source: **{source}**", icon="🗄️")
    else:
        sb.warning(f"Data source: **{source}**", icon="📄")
        with sb.expander("Why the fallback?"):
            ok, msg = test_connection()
            st.caption(msg)
            st.caption(
                "The dashboard is reading the cleaned CSV in data/processed/. "
                "Run `python scripts/03_load_data.py` after starting MySQL to "
                "switch to the database."
            )

    sb.divider()
    sb.markdown("#### Filters")

    filtered = df.copy()

    # --- visit number ------------------------------------------------------
    # This dataset has no admission dates (it is de-identified), so there is no
    # date-range filter. `encounter_seq` is the ordering column instead.
    if "encounter_seq" in df.columns and df["encounter_seq"].notna().any():
        only_first = sb.checkbox(
            "First recorded visit only",
            value=False,
            key=f"{key_prefix}firstvisit",
            help=(
                "Each row is one hospital encounter, and the same person can "
                "appear several times. Tick this to keep one row per person."
            ),
        )
        if only_first:
            filtered = filtered[filtered["encounter_seq"] == 1]

    # --- diagnosis ---------------------------------------------------------
    if "diagnosis" in df.columns:
        options = sorted(df["diagnosis"].dropna().unique().tolist())
        chosen = sb.multiselect(
            "Diagnosis",
            options,
            default=[],
            placeholder="All diagnoses",
            key=f"{key_prefix}diagnosis",
        )
        if chosen:
            filtered = filtered[filtered["diagnosis"].isin(chosen)]

    # --- department --------------------------------------------------------
    if "department" in df.columns:
        options = sorted(df["department"].dropna().unique().tolist())
        chosen = sb.multiselect(
            "Department",
            options,
            default=[],
            placeholder="All departments",
            key=f"{key_prefix}department",
        )
        if chosen:
            filtered = filtered[filtered["department"].isin(chosen)]

    # --- gender ------------------------------------------------------------
    if "gender" in df.columns:
        options = [g for g in ["Male", "Female", "Unknown"] if g in df["gender"].unique()]
        chosen = sb.multiselect(
            "Gender", options, default=options, key=f"{key_prefix}gender"
        )
        if chosen:
            filtered = filtered[filtered["gender"].isin(chosen)]

    # --- age ---------------------------------------------------------------
    if "age" in df.columns and df["age"].notna().any():
        lo, hi = int(df["age"].min()), int(df["age"].max())
        picked = sb.slider("Age range", lo, hi, (lo, hi), key=f"{key_prefix}age")
        filtered = filtered[filtered["age"].between(picked[0], picked[1])]

    # --- outcome -----------------------------------------------------------
    if "readmitted_30d" in df.columns:
        choice = sb.radio(
            "Readmission outcome",
            ["All patients", "Readmitted only", "Not readmitted only"],
            key=f"{key_prefix}outcome",
        )
        if choice == "Readmitted only":
            filtered = filtered[filtered["readmitted_30d"] == 1]
        elif choice == "Not readmitted only":
            filtered = filtered[filtered["readmitted_30d"] == 0]

    # --- feedback on the filter's effect -----------------------------------
    sb.divider()
    pct = len(filtered) / max(len(df), 1) * 100
    sb.metric("Patients in view", f"{len(filtered):,}", f"{pct:.0f}% of {len(df):,}")
    if len(filtered) == 0:
        sb.error("No patients match these filters.")
    elif len(filtered) < 50:
        sb.warning("Very few patients selected - percentages will be unstable.")

    reset_kwargs = {"width": "stretch"} if _NEW_WIDTH_API else {"use_container_width": True}
    if sb.button("Reset filters", **reset_kwargs):
        for k in list(st.session_state.keys()):
            if k.startswith(key_prefix):
                del st.session_state[k]
        st.rerun()

    sb.divider()
    sb.caption("Module 1-3 complete · Modules 4-5 in progress")
    return filtered


# ---------------------------------------------------------------------------
# Display components
# ---------------------------------------------------------------------------
def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of metric cards.

    `items` is a list of (label, value, delta_or_help) tuples.
    """
    cols = st.columns(len(items))
    for col, (label, value, extra) in zip(cols, items):
        with col:
            st.metric(label, value, extra)


def insight_box(key: str, show_finding: bool = True) -> None:
    """Render the 'why this feature matters' panel beneath a chart.

    The text comes from charts.CHART_INSIGHTS, the same dictionary the notebook
    reads, so the explanation on screen is identical to the one in your analysis.
    """
    info = charts.insight(key)
    if not info.get("why"):
        return

    feature = info.get("feature", "")
    body = info["why"]
    st.markdown(
        f"""<div class="insight-box">
              <span class="tag">WHY IT MATTERS</span>
              {"<br><code>" + feature + "</code><br>" if feature else ""}
              {body}
            </div>""",
        unsafe_allow_html=True,
    )
    if show_finding and info.get("read"):
        st.markdown(
            f"""<div class="finding-box">
                  <strong>What this chart shows:</strong> {info["read"]}
                </div>""",
            unsafe_allow_html=True,
        )


def section(title: str, description: str | None = None) -> None:
    """A consistent section header."""
    st.markdown(f"## {title}")
    if description:
        st.caption(description)


def download_button(df: pd.DataFrame, filename: str, label: str = "Download as CSV") -> None:
    """Let the viewer export whatever table is on screen."""
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def roadmap_note(module: str, what: str, uses: str) -> None:
    """Placeholder panel for a module that is not built yet.

    Shipping a visible, honest placeholder is better than hiding future work: it
    shows the evaluator you have an architecture, not just a screen.
    """
    st.markdown(
        f"""<div class="roadmap">
              <strong>🚧 {module} - not built yet (next development cycle)</strong><br><br>
              <strong>What it will do:</strong> {what}<br><br>
              <strong>What it will reuse from the current version:</strong> {uses}
            </div>""",
        unsafe_allow_html=True,
    )


def empty_guard(df: pd.DataFrame) -> bool:
    """Show a friendly message and return True if there is nothing to plot."""
    if len(df) == 0:
        st.info("No patients match the current filters. Widen them in the sidebar.")
        return True
    return False


# Re-export so pages can do `ui.metrics` / `ui.charts` without another import.
__all__ = [
    "APP_DIR", "PROJECT_ROOT", "chart", "table", "page_setup", "get_data",
    "get_quality_report", "clear_cache", "sidebar_filters", "kpi_row",
    "insight_box", "section", "download_button", "roadmap_note", "empty_guard",
    "charts", "metrics",
]
