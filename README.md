# AI-Based Hospital Disease Analytics & Patient Risk Prediction

**Modules 1–5 complete and runnable.**

An end-to-end pipeline over ten years of real hospital admission records:
loading into MySQL, cleaning and exploratory analysis, an interactive Streamlit
dashboard, a machine learning model predicting 30-day readmission, and a
generative AI layer that explains each prediction in plain language using a
**locally-hosted Llama model** — grounded in measured statistics rather than in
the language model's own knowledge, and with no patient data leaving the
machine.

Module 6 (FastAPI + React) is designed for and reserved in the codebase.

**Dataset:** Diabetes 130-US Hospitals, 1999–2008 — 101,766 real encounters from
130 US hospitals, published by Strack et al. (2014), CC BY 4.0.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [What is built and what is not](#2-what-is-built-and-what-is-not)
3. [Terminal basics — read this if you have never used a terminal](#3-terminal-basics)
4. [Install the prerequisites](#4-install-the-prerequisites)
5. [Set up the project — 7 steps](#5-set-up-the-project)
6. [Run the project](#6-run-the-project)
7. [Every file, and what it is for](#7-every-file-and-what-it-is-for)
8. [How the pieces fit together](#8-how-the-pieces-fit-together)
9. [Troubleshooting](#9-troubleshooting)
10. [Swapping in a different dataset](#10-swapping-in-a-different-dataset)
11. [Roadmap for the next cycles](#11-roadmap)

---

## 0. Which document do I need?

| I want to... | Read |
|---|---|
| Set it up, having never used a terminal | **`GETTING_STARTED.md`** |
| Understand how the ML and AI modules work | **`ML_AND_AI_EXPLAINED.md`** |
| Write my report / understand the design | **`SDD.md`** |
| Present it to an evaluator | **`DEMO_GUIDE.md`** |
| Look something up quickly | this file |

---

## 1. What this project does

Hospitals generate thousands of patient records a day. This system:

- **collects** those records into a MySQL database
- **cleans** them (real hospital data is messy)
- **analyses** them to find which patients come back within 30 days, and why
- **presents** the findings in an interactive web dashboard
- *(next cycle)* **predicts** readmission risk with machine learning
- *(next cycle)* **explains** each prediction in plain English using an LLM

The central question the whole project answers: **which patients are likely to be
readmitted within 30 days of discharge, and what can the hospital do about it?**

That matters because a readmission usually means something went wrong — a
medication misunderstood, a follow-up missed, a condition not stabilised — and
because readmissions are expensive and, in many health systems, financially
penalised.

---

## 2. What is built and what is not

| # | Module | Status | Where it lives |
|---|---|---|---|
| 1 | Data collection into MySQL | ✅ **Built** | `src/hospital_analytics/ingestion/` |
| 2 | Cleaning, EDA, statistics | ✅ **Built** | `ingestion/cleaning.py`, `analysis/metrics.py`, `notebooks/` |
| 3 | Visualisation dashboard | ✅ **Built** | `analysis/charts.py`, `streamlit_app/` |
| 4 | ML readmission model | ✅ **Built** | `src/hospital_analytics/ml/`, `scripts/04_train_model.py` |
| 5 | GenAI report generator (local Llama) | ✅ **Built** | `src/hospital_analytics/genai/`, `scripts/05_generate_report.py` |
| 6 | FastAPI + React front end | 🚧 Planned | reuses `analysis/` unchanged |

**Everything in this repository runs today**, and every module degrades rather
than crashing. No MySQL → CSV fallback. No trained model → the transparent
rule-based baseline, with the command to train one. No Ollama → the same
grounded facts rendered by a deterministic template.

**The AI runs locally.** Module 5 uses **Llama 3.2 via Ollama** — free, no API
key, no account. More importantly, no patient record ever leaves the machine,
which for clinical data is an architectural decision rather than a cost saving.

That last one is worth demonstrating rather than apologising for: if the report
is still correct without a language model, the grounding layer is doing the work
— which is the whole architectural claim of Module 5.

---

## 3. Terminal basics

The "terminal" (also called Command Prompt, PowerShell, or shell) is a window
where you type commands instead of clicking. You need about five commands.

### Opening it

| System | How |
|---|---|
| **Windows** | Press `Win`, type `cmd`, press Enter |
| **macOS** | Press `Cmd + Space`, type `terminal`, press Enter |
| **Linux** | Press `Ctrl + Alt + T` |

### The five commands you need

| Command | What it means | Example |
|---|---|---|
| `cd folder` | **c**hange **d**irectory — go into a folder | `cd Downloads` |
| `cd ..` | go **up** one folder | `cd ..` |
| `dir` (Windows) / `ls` (Mac, Linux) | list what is in this folder | `ls` |
| `python file.py` | run a Python file | `python scripts/01_check_dataset.py` |
| `pip install x` | install a Python package | `pip install pandas` |

### The one thing that trips everyone up

**Almost every error a beginner hits is because the terminal is in the wrong
folder.** Every command in this guide must be run from the project's **top
folder** — the one containing `README.md` and `requirements.txt`.

To get there, type `cd `  (with a space), then **drag the project folder from
your file explorer onto the terminal window** and press Enter. The path fills in
automatically. Then check you are in the right place:

```bash
# Windows
dir
# macOS / Linux
ls
```

You should see `README.md`, `requirements.txt`, `data`, `scripts`, `src`,
`notebooks`, `streamlit_app`. If you do not, you are in the wrong folder.

> **Note on `python` vs `python3`:** on macOS and Linux, use `python3` and `pip3`.
> On Windows, use `python` and `pip`. If one says "command not found", try the
> other. This guide writes `python` — substitute as needed for your system.

---

## 4. Install the prerequisites

### 4.1 Python 3.10 or newer

Check whether you already have it:

```bash
python --version
```

If you see `Python 3.10.x` or higher, you are done. Otherwise download it from
[python.org/downloads](https://www.python.org/downloads/).

> **Windows users: this step matters.** During installation, tick
> **"Add Python to PATH"** on the very first screen. If you miss it, the terminal
> will not find Python and you will have to reinstall.

### 4.2 MySQL

Download **MySQL Community Server** from
[dev.mysql.com/downloads/mysql](https://dev.mysql.com/downloads/mysql/).

During installation you will be asked to set a **root password**.
**Write it down.** You need it in step 3 below.

| System | Easier alternative |
|---|---|
| Windows | Use the **MySQL Installer**, choose "Developer Default" |
| macOS | `brew install mysql` then `brew services start mysql` |
| Linux | `sudo apt install mysql-server` |

Check MySQL is running:

```bash
mysql --version
```

> ### You can skip MySQL entirely for now
>
> The project is designed to fall back to CSV files if MySQL is not available.
> Everything — the notebook and every dashboard chart — still works. The
> dashboard's sidebar will simply show "Data source: CSV fallback" instead of
> "MySQL".
>
> This is deliberate: it means a database problem on demo day cannot stop your
> presentation. If you are short of time, get the project running with CSV first,
> then add MySQL.

---

## 5. Set up the project

Seven steps. Run every command from the project's top folder.

### Step 1 — Go to the project folder

```bash
cd path/to/hospital-analytics
```

(Remember the drag-and-drop trick from section 3.)

### Step 2 — Create a virtual environment

A **virtual environment** is a private box of Python packages just for this
project. Without it, installing a package for this project can break a different
project on your computer.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now start with `(.venv)`. **That prefix is how you
know the environment is active.** You must re-activate it (repeat the second
command above) every time you open a new terminal window.

### Step 3 — Install the packages

```bash
pip install -r requirements.txt
```

This takes a few minutes. It reads `requirements.txt` and installs pandas,
plotly, streamlit, SQLAlchemy and the rest.

### Step 4 — Create your `.env` file

`.env` is a small text file holding your MySQL password. It exists so your
password never appears in any `.py` file and never reaches GitHub.

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Now **open `.env` in any text editor** (Notepad, TextEdit, VS Code) and change
this one line to your real MySQL password:

```
MYSQL_PASSWORD=your_mysql_password_here
```

If you skipped MySQL, leave the file exactly as it is and carry on.

### Step 5 — Download the dataset

This project uses a **real public dataset**. Nothing is invented or simulated.
You download one file, once.

**Diabetes 130-US Hospitals for Years 1999-2008** — ten years of real inpatient
care from 130 US hospitals. 101,766 encounters, 50 columns, licensed CC BY 4.0
(free to use and adapt, with credit).

Get it from **either** source — the file is identical:

| | Link | Account needed? |
|---|---|---|
| **UCI** (easier) | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 | No |
| **Kaggle** | https://www.kaggle.com/datasets/brandao/diabetes | Free login |

On the UCI page, click the **Download (3 MB)** button. You get a `.zip`.

**Unzip it, and move `diabetic_data.csv` into the `data/raw/` folder inside this
project.** That is the only file you need — you can ignore `IDS_mapping.csv`.

Then check it landed correctly:

```bash
python scripts/01_check_dataset.py
```

Expected output:

```
  FOUND: .../data/raw/diabetic_data.csv
  size : 18.3 MB
  columns: 50 (expected 50)
  all critical columns present
  rows   : 101,766 (expected 101,766)

  target variable `readmitted`:
      NO :  54,864  (53.91%)
     >30 :  35,545  (34.93%)
     <30 :  11,357  (11.16%)

  30-day readmission rate: 11.16%
  '?' placeholders to clean: 192,849
```

Those `?` marks are how this dataset records missing values — nearly 200,000 of
them. Step 7 handles them, and they are the reason the cleaning module exists.

**If it says NOT FOUND**, the script prints exactly where to put the file. The
usual cause is leaving it inside the unzipped subfolder instead of moving it into
`data/raw/`.

### Step 6 — Create the database and tables

```bash
python scripts/02_setup_database.py
```

Expected output ends with:

```
  Connected to MySQL database 'hospital_analytics' on localhost:3306
```

If it fails, jump to [Troubleshooting](#9-troubleshooting) — or skip to Step 7
with `--csv-only` and come back to this later.

### Step 7 — Clean the data and load it into MySQL

```bash
python scripts/03_load_data.py
```

**No MySQL? Use this instead:**

```bash
python scripts/03_load_data.py --csv-only
```

Expected output (abbreviated):

```
STEP 1/4  EXTRACT - reading the source dataset
  encounters read       101,766
  deceased/hospice removed  1,642
  encounters kept        100,124
  unique people           69,990
  prescriptions (melted) ~180,000

STEP 2/4  TRANSFORM - cleaning and validating
  dropped column weight            (96.9% missing)
  dropped column payer_code        (39.6% missing)
  duplicate encounters removed           0
  repeat encounters kept            30,134
  'race' set to Unknown              2,273 rows
  clean encounter rows             100,124
  30-day readmission rate            11.3%

STEP 4/4  VERIFY - counting rows in MySQL
  patients             100,124 rows
  medications         ~180,000 rows
```

**Read those numbers — they are your evidence.** Note what the cleaning work
actually consists of here: this is published research data, so it arrives in
good shape. The value is in the **decisions** — which columns to discard, what
an absent value means, and which rows are duplicates versus genuine repeat
visits — not in repairing corruption.

Two of those lines are worth being able to explain:

- **deceased/hospice removed** — these patients cannot be readmitted. Leaving
  them in would teach a model that "expired" predicts "not readmitted": true,
  useless, and grim. Excluding them is standard in the published literature.
- **repeat encounters kept** — the same person appearing twice is *not* a
  duplicate. Those are separate real admissions. They are kept and flagged,
  because Module 4 must split train/test by person.

(Your exact figures may differ slightly; read them off your own run.)

### Shortcut: all of it in one command

```bash
python scripts/run_all.py
```

This runs steps 5, 6 and 7 in order. If MySQL fails it keeps going with CSV and
tells you so.

---

## 6. Run the project

### 6.1 The dashboard

```bash
streamlit run streamlit_app/app.py
```

Your browser opens at `http://localhost:8501`. If it does not, open that address
manually.

**Six pages, in the left sidebar:**

| Page | What it shows |
|---|---|
| 🏥 **Home** | Headline KPIs and the single most important finding |
| 📈 **Disease Trends** | Monthly volume, the seasonality heatmap, length of stay |
| 🧍 **Patient Explorer** | Demographics, vitals, medications, individual records |
| 🔗 **Correlation Lab** | What tracks readmission; multicollinearity check |
| 🧪 **Data Quality** | Proof of what the cleaning pipeline fixed |
| 🤖 **Risk Prediction** | The rule-based baseline Module 4's model must beat |
| 📄 **AI Report** | The report structure and prompt Module 5 will use |

To stop the dashboard, click the terminal window and press `Ctrl + C`.

### 6.2 The notebook

```bash
jupyter notebook notebooks/01_eda_hospital_analytics.ipynb
```

Then click **Cell → Run All** (or `Kernel → Restart & Run All`).

The notebook has 88 cells that walk through the entire analysis, and every chart
is followed by a written explanation of **why that feature matters to the
project**. This is where your understanding of the data comes from — read it, do
not just run it.

> **If charts do not appear in the notebook:** find the line
> `pio.renderers.default = "notebook"` in cell 2 and change `"notebook"` to
> `"iframe"`, then re-run.

---

## 7. Every file, and what it is for

```
hospital-analytics/
│
├── README.md                  ← this file
├── DEMO_GUIDE.md              ← how to present the project to your evaluator
├── requirements.txt           ← the list of Python packages to install
├── .env.example               ← template for your database password
├── .env                       ← YOUR password (you create this; never share it)
├── .gitignore                 ← tells Git to never upload .env or the CSVs
│
├── data/
│   ├── raw/                   ← input CSVs, never edited by any code
│   │   ├── patients.csv           10,000 admissions (with deliberate defects)
│   │   └── diabetic_data.csv      ← THE FILE YOU DOWNLOAD (not in the repo)
│   └── processed/             ← cleaned output; the dashboard's fallback source
│       ├── patients_clean.csv
│       └── medications_clean.csv
│
├── scripts/                   ← THE FILES YOU RUN
│   ├── 01_check_dataset.py        verifies your download is the right file
│   ├── check_schema.py            validates schema.sql (no database needed)
│   ├── 04_train_model.py          Module 4: trains the ML model
│   ├── 05_generate_report.py      Module 5: generates a grounded AI report
│   ├── 02_setup_database.py       creates the MySQL database and tables
│   ├── 03_load_data.py            cleans the data and loads it into MySQL
│   └── run_all.py                 runs all three, in order
│
├── notebooks/
│   └── 01_eda_hospital_analytics.ipynb   the full analysis, with explanations
│
├── src/hospital_analytics/     ← THE SHARED LIBRARY (the project's brain)
│   ├── config.py                  all settings and file paths in one place
│   ├── db.py                      the only file that talks to MySQL
│   ├── sql/schema.sql             the CREATE TABLE statements
│   │
│   ├── ingestion/              ← Module 1: getting data in
│   │   ├── adapters/
│   │   │   └── uci_diabetes.py       translates the real dataset into our schema
│   │   ├── cleaning.py               all cleaning and feature engineering rules
│   │   └── load_to_mysql.py          the ETL job (extract → transform → load)
│   │
│   ├── analysis/               ← Modules 2 and 3: understanding the data
│   │   ├── data_access.py            loads data (MySQL first, CSV fallback)
│   │   ├── metrics.py                every statistic in the project
│   │   └── charts.py                 every chart, plus the "why it matters" text
│   │
│   ├── ml/                     ← MODULE 4: machine learning
│   │   ├── features.py           builds the matrix; excludes leakage
│   │   ├── train.py              grouped split, 3 models, saves the winner
│   │   ├── evaluate.py           baseline, thresholds, fairness, subgroups
│   │   └── predict.py            load model, predict, explain per patient
│   └── genai/                  ← MODULE 5: grounded AI reports
│       ├── grounding.py          collects the permitted facts
│       ├── prompts.py            system + user prompts, 3 templates
│       ├── providers.py          Ollama/Llama (default) / Claude / GPT / template
│       └── generator.py          orchestration + output validation
│
└── streamlit_app/              ← the dashboard
    ├── app.py                     the home page — this is the file you run
    ├── ui.py                      shared helpers: caching, filters, KPI cards
    └── pages/                     one file per page; the folder IS the navigation
        ├── 1_Disease_Trends.py
        ├── 2_Patient_Explorer.py
        ├── 3_Correlation_Lab.py
        ├── 4_Data_Quality.py
        ├── 5_Risk_Prediction.py
        └── 6_AI_Report.py
```

### The five files worth understanding properly

If you only read five files, read these — and be ready to talk about them.

**`src/hospital_analytics/config.py`**
Every setting in one place: folder paths, MySQL credentials (read from `.env`),
and the clinically valid range for each measurement. Change your MySQL password
here-ish (actually in `.env`) and the whole project follows. Nothing else
hard-codes a path or a password.

**`src/hospital_analytics/ingestion/cleaning.py`**
The quality gate. Ten ordered rules: normalise column names, tidy text, parse
mixed date formats, coerce numerics, standardise categories, remove duplicates,
null out impossible values, impute with the median, engineer new features, final
integrity check. It returns the cleaned data **and a report of what it changed**,
which is what the Data Quality dashboard page displays.

**`src/hospital_analytics/analysis/metrics.py`**
Every number the project reports. Pure pandas — it imports no plotting or web
framework, which is why the FastAPI service in Module 6 will be able to reuse it
untouched.

**`src/hospital_analytics/analysis/charts.py`**
Every chart. Also holds `CHART_INSIGHTS`, a dictionary explaining why each
feature matters — the notebook and the dashboard both read from it, so the
explanation on screen can never drift from the one in your analysis.

**`src/hospital_analytics/analysis/data_access.py`**
The fallback chain: try MySQL → fall back to the cleaned CSV → fall back to
cleaning the raw CSV. This is the file that makes the project demo-proof.

---

## 8. How the pieces fit together

```
        data/raw/*.csv
              │
              ▼
    ingestion/cleaning.py          validate · dedupe · impute · engineer features
              │
              ▼
    ingestion/load_to_mysql.py     ETL, with an audit log of every run
              │
              ▼
         MySQL database             patients · medications · disease_reference
              │
              ▼
    analysis/data_access.py        ONE loader — MySQL first, CSV fallback
              │
      ┌───────┴────────┐
      ▼                ▼
analysis/metrics.py   analysis/charts.py
 (every statistic)     (every figure + why it matters)
      │                │
      └───────┬────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
  notebooks/       streamlit_app/
 (investigate)      (present)
```

### Why it is structured this way

**One definition of every number.** The notebook and the dashboard import the
same `metrics.py`. It is therefore *impossible* for the dashboard to show an
average length of stay that disagrees with the notebook — a common and
embarrassing failure in projects that copy-paste analysis code between files.

**The next modules plug in, they do not replace.** Module 4 will import the same
cleaned features. Module 6's FastAPI service will import the same `metrics.py`.
Nothing built in this cycle has to be rewritten, which is exactly what the brief
asked for.

**One door to the database.** Only `db.py` knows the connection details. If the
project migrated to PostgreSQL, one file would change.

**The app degrades instead of dying.** If MySQL is unreachable the dashboard
loads the cleaned CSV and says so in the sidebar. Nothing silently shows stale
data, and nothing crashes.

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'pandas'` (or streamlit, plotly...)

The virtual environment is not active, or the packages are not installed.

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'hospital_analytics'`

You are running the command from the wrong folder. Go to the project's top folder
(the one with `README.md`) and try again.

### `Table 'hospital_analytics.medications' doesn't exist`

The database exists but its tables were not created. Run:

```bash
python scripts/02_setup_database.py
```

That script now verifies each of the four tables after creating them, and names
any that are missing. If it reports a failure, the message identifies the exact
SQL statement that failed.

To check `schema.sql` itself without needing MySQL running:

```bash
python scripts/check_schema.py
```

### `Could not find 'diabetic_data.csv'`

The dataset has not been downloaded, or it is not in the right folder.

Download it from
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
(no account needed), unzip it, and move `diabetic_data.csv` into `data/raw/`.

The commonest mistake is leaving the file inside the unzipped subfolder. You
want `data/raw/diabetic_data.csv`, not
`data/raw/diabetes+130-us+hospitals.../diabetic_data.csv` — though the script
will search one level down and find it either way.

Check with `python scripts/01_check_dataset.py`.

### `required columns are missing`

You downloaded a *different* file. Several people on Kaggle publish cleaned or
pre-processed copies of this dataset with renamed columns.

You want the original `diabetic_data.csv` with all 50 columns, from the UCI link
above or from `kaggle.com/datasets/brandao/diabetes`.

### `can't open file 'scripts/01_check_dataset.py'`

Same cause: wrong folder. Run `ls` (or `dir`) — if you cannot see `README.md`,
you are in the wrong place.

### `Access denied for user 'root'@'localhost'`

Wrong MySQL password in `.env`. Open `.env` and fix `MYSQL_PASSWORD`.
Test the password by hand with `mysql -u root -p`.

### `Can't connect to MySQL server on 'localhost'`

MySQL is not running. Start it:

| System | Command |
|---|---|
| Windows | Press `Win`, type `services.msc`, find `MySQL80`, right-click → Start |
| macOS | `brew services start mysql` |
| Linux | `sudo systemctl start mysql` |

Or skip it: `python scripts/03_load_data.py --csv-only`

### `RuntimeError: 'cryptography' package is required`

MySQL 8 uses an authentication method that needs it:

```bash
pip install cryptography
```

### The dashboard says "Data source: CSV fallback"

That is the fallback working as designed, not a bug. To use MySQL: start it, run
`python scripts/02_setup_database.py`, then `python scripts/03_load_data.py`, then
press **Refresh data** in the dashboard sidebar.

### `ImportError: cannot import name 'iteritems'` when importing cufflinks

Cufflinks has been unmaintained since 2021 and calls a pandas method removed in
pandas 2.0. Cell 2 of the notebook already patches this. If you hit it elsewhere,
add these two lines before importing cufflinks:

```python
pd.DataFrame.iteritems = pd.DataFrame.items
pd.Series.iteritems = pd.Series.items
```

Cufflinks is entirely optional — every chart works without it.

### Charts do not appear in the notebook

In cell 2, change `pio.renderers.default = "notebook"` to `"iframe"` and re-run
all cells.

### `Port 8501 is already in use`

A dashboard is already running. Either use that browser tab, or:

```bash
streamlit run streamlit_app/app.py --server.port 8502
```

### The dashboard shows old data after reloading MySQL

Streamlit caches data for speed. Click **Refresh data (clear cache)** on the Data
Quality page, or press `C` then `Enter` in the browser.

---

## 10. Swapping in a different dataset

The project already runs on real data. If you ever need to point it at a
*different* dataset — MIMIC-III, a hospital's own extract, another Kaggle set —
the architecture is built for it.

**You write one new adapter. Nothing else changes.**

An adapter lives in `src/hospital_analytics/ingestion/adapters/` and does exactly
one job: turn someone else's column names and layout into this project's. The
existing one, `uci_diabetes.py`, is the worked example — copy it.

An adapter must provide three functions:

```python
def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Read the source file(s)."""

def to_patients(raw) -> tuple[pd.DataFrame, dict]:
    """Return one row per encounter, using THIS project's column names,
    plus a report dict describing what happened."""

def to_medications(raw, valid_patient_ids) -> pd.DataFrame:
    """Return one row per prescription."""
```

Then register it in `adapters/__init__.py` and point `DATASET_ADAPTER` in
`config.py` at it.

**Also update `DATASET_PROFILE` in `config.py`.** That dictionary declares what
the dataset can and cannot support — whether it has dates, vitals, numeric labs.
Charts and pages read those flags, which is why the dashboard degrades honestly
instead of crashing when a column is absent.

### Why this design is worth pointing out in your demo

The alternative — scattering `if "bmi" in df.columns` checks through every chart
and page — is how projects become unmaintainable. Isolating dataset-specific
knowledge in one file means the analysis code never has to know which dataset it
is looking at.

That is the same reason `metrics.py` imports no plotting or web framework: it
gets reused unchanged by the notebook, the dashboard, and later the FastAPI
service.

## 11. Roadmap

### Cycle 2 — Module 4: machine learning

- `ml/features.py` — build the feature matrix, encode categoricals, guard
  against leakage
- `ml/train.py` — Logistic Regression, Random Forest, XGBoost
- `ml/evaluate.py` — accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- `ml/explain.py` — SHAP values for per-feature contributions
- The dashboard's **Risk Prediction** page swaps its rule-based baseline for real
  predictions; the baseline stays visible as the comparison

Every design decision is already justified by this cycle's analysis and written
down in `src/hospital_analytics/ml/__init__.py`.

### Cycle 3 — Module 5: GenAI, and Module 6: API

- `genai/` — grounded LLM explanations. The prompt is already designed and
  visible on the dashboard's AI Report page
- `api/` — FastAPI service exposing `/predict` and `/explain`, importing the same
  `analysis/` library
- React front end consuming the API
- Docker + deployment

---

## A note on the dataset

The data is **real**: ten years of inpatient care from 130 US hospitals,
published by Strack et al. (2014) and hosted by the UCI Machine Learning
Repository under a CC BY 4.0 licence. Cite it in your report:

> Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN (2014).
> *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000
> Clinical Database Patient Records.* BioMed Research International, 781670.
> DOI 10.24432/C5230J

Three limitations to state before an evaluator finds them:

- **No dates.** The data is de-identified, so seasonality and time-based
  train/test splitting are impossible. `encounter_seq` provides ordering instead.
- **No vital signs.** No blood pressure, BMI or cholesterol. `weight` exists but
  is ~97% missing, so it is dropped rather than imputed.
- **Rows are encounters, not people.** The same patient recurs. Module 4 must
  split train/test grouped by `person_id`, or the score is inflated.

Naming these yourself reads as understanding your data. Being caught out by them
reads as not having looked.
