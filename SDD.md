# Software Design Document

**AI-Based Hospital Disease Analytics & Patient Risk Prediction**

| | |
|---|---|
| Version | 2.0 |
| Status | Modules 1–5 implemented; Module 6 (API + web app) planned |
| Dataset | Diabetes 130-US Hospitals, 1999–2008 (UCI dataset 296, CC BY 4.0) |
| Language | Python 3.10+ |
| Database | MySQL 8 |

---

## 1. Purpose and scope

### 1.1 The problem

Hospitals readmit a significant share of patients within 30 days of discharge.
Each readmission usually signals that something in the discharge process failed —
a medication misunderstood, a follow-up never booked, a deterioration not
anticipated. Readmissions are expensive, are increasingly penalised by payers,
and are partly preventable.

The obstacle is not a shortage of data. A hospital records everything. The
obstacle is that nobody has time to read it: identifying which of forty patients
being discharged today is most likely to return is a question no ward has the
capacity to answer manually.

### 1.2 What this system does

1. Ingests real hospital admission records into a normalised relational database.
2. Cleans them and computes the statistics that describe readmission risk.
3. Presents those findings through an interactive dashboard.
4. Trains a machine learning model to predict 30-day readmission for an
   individual patient.
5. Generates a plain-language explanation of each prediction, grounded in
   measured statistics rather than in a language model's own knowledge.

### 1.3 What it explicitly does not do

- It does **not** diagnose. Output is a prioritisation aid for clinicians.
- It does **not** make autonomous decisions. Every report requires human review.
- It does **not** claim causation. Every relationship it reports is an
  association measured on historical data.

Stating the boundaries is part of the design, not a disclaimer bolted on at the
end. A system that oversells itself in a clinical setting is a liability.

---

## 2. The dataset

### 2.1 Source

> Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN (2014).
> *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000
> Clinical Database Patient Records.* BioMed Research International, vol. 2014,
> Article ID 781670. DOI 10.24432/C5230J.

Ten years of inpatient care from 130 US hospitals. 101,766 encounters, 50
columns. Licensed CC BY 4.0 — free to use and adapt with attribution.

Chosen over MIMIC-III (requires credentialed access and ethics training) and over
generating synthetic data (proves nothing about a cleaning pipeline). Critically,
its published research question is the same as this project's target variable,
which gives a way to validate the pipeline against peer-reviewed work.

### 2.2 Properties that shape the entire design

These are not incidental. Each one forces a specific architectural decision.

| Property | Consequence |
|---|---|
| One row = one **encounter**, not one patient | Train/test split must be grouped by `person_id` |
| **No dates** — fully de-identified | Time-based splitting impossible; no seasonality analysis |
| **No vital signs** — no BP, BMI, cholesterol | Feature set built from utilisation, not physiology |
| `weight` ~97% missing | Column dropped, not imputed |
| Missing values encoded as the string `"?"` | Must be handled at read time or counts read as zero |
| 23 medications stored as 23 wide columns | Requires reshaping into a long table |
| HbA1c is categorical, including "not tested" | "Not tested" is a third state, never a null |
| Target is ~11% positive | Accuracy is misleading; recall and ROC-AUC lead |

---

## 3. Architecture

### 3.1 Layered design

```
                        data/raw/diabetic_data.csv
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │  INGESTION            ingestion/                      │
        │    adapters/uci_diabetes.py   dataset → our schema    │
        │    cleaning.py                quality rules           │
        │    load_to_mysql.py           ETL + audit log         │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
                          MySQL: 4 tables
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │  ACCESS               analysis/data_access.py         │
        │    MySQL → cleaned CSV → raw file  (3-tier fallback)  │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  ANALYSIS             analysis/                       │
        │    metrics.py    pure pandas, no UI, no plotting      │
        │    charts.py     plotly figures + CHART_INSIGHTS      │
        └──────┬─────────────────────┬──────────────────┬───────┘
               │                     │                  │
        ┌──────▼──────┐      ┌───────▼───────┐   ┌──────▼──────┐
        │  MODULE 4   │      │   MODULE 5    │   │ PRESENTATION│
        │  ml/        │─────▶│   genai/      │   │ notebook    │
        │  features   │      │   grounding   │   │ streamlit   │
        │  train      │      │   prompts     │   │             │
        │  evaluate   │      │   providers   │   │             │
        │  predict    │      │   generator   │   │             │
        └─────────────┘      └───────────────┘   └─────────────┘
```

### 3.2 The three rules that hold it together

**Rule 1 — one definition of every statistic.**
`metrics.py` imports no plotting library and no web framework. The notebook, the
dashboard, the ML module and the GenAI module all import it. A number therefore
cannot mean one thing in the analysis and another on screen.

**Rule 2 — dataset knowledge lives in exactly one file.**
Everything specific to this dataset — ICD-9 groupings, the 23 drug columns, the
`?` convention — is confined to `adapters/uci_diabetes.py`. Supporting a
different dataset means writing one new adapter, not editing every chart.

**Rule 3 — every layer degrades instead of crashing.**
MySQL down → cleaned CSV. No cleaned CSV → adapt the raw file. No raw file →
on-screen download instructions. No trained model → rule-based baseline. No API
key → deterministic template. At every level the system reports which path it
took, so nobody unknowingly presents stale or degraded output.

---

## 4. Data design

### 4.1 Schema

Four tables, InnoDB, utf8mb4.

**`disease_reference`** — ICD-9 group lookup. `diagnosis` (PK), `icd9_range`,
`disease_category`, `is_chronic`, `department`.
*Rationale:* group metadata lives in one row, not repeated across 100,000.

**`patients`** — one row per encounter, 34 columns. PK `patient_id`
(the source's `encounter_id`); `person_id` indexed because the ML split depends
on it. FK to `disease_reference`.

**`medications`** — one row per prescription. `patient_id` FK with
`ON DELETE CASCADE`, `medication_name`, `drug_class`, `dose_status`.
*Rationale:* a patient has many medications. This is the one-to-many side, and
it is what makes "how many patients received any sulfonylurea?" answerable.

**`data_load_log`** — audit trail: source file, target table, rows in, rows
loaded, rows rejected, status, timestamp.

### 4.2 Cleaning rules

| # | Rule | Rationale |
|---|---|---|
| 1 | `"?"` → null | Otherwise pandas reads ~193,000 missing values as text |
| 2 | Drop `weight` (~97% missing) | Imputing 97% invents data |
| 3 | Drop `payer_code` (~40%) | Not clinically relevant |
| 4 | Keep `medical_specialty` as "Unknown" | Absence indicates no assigned consultant — that is information |
| 5 | Remove deceased/hospice discharges | They cannot be readmitted |
| 6 | Deduplicate on `patient_id` | True duplicate encounters |
| 7 | **Keep** repeat visits by the same person | Separate real admissions, flagged via `person_id` |
| 8 | Null values outside documented ranges | Contradicts the dataset's own inclusion criteria |
| 9 | Impute numerics with the **median** | Counts are right-skewed |
| 10 | Categorical gaps → explicit "Unknown" | Never guess gender or ethnicity |

Two outlier types are handled differently and deliberately: **out-of-range**
values indicate a corrupted file and are removed; **statistical (IQR)** outliers
are real complex patients and are kept. Removing the second kind would teach the
model that severe cases do not exist.

---

## 5. Module 4 — machine learning

### 5.1 Design decisions and their evidence

Every decision below came from Module 2's analysis, not from a tutorial.

| Decision | Driven by |
|---|---|
| Prediction moment = **day before discharge** | Only useful if staff can still act |
| Split **grouped by `person_id`** | People repeat; ungrouped split inflates scores |
| **No time-based split** | Dataset has no dates |
| `class_weight="balanced"` | ~11% positive class |
| Lead on **recall and ROC-AUC** | Baseline accuracy is ~88% while catching nobody |
| Exclude discharge features | Leakage — recorded after the prediction moment |
| Exclude `race` from features, **keep for audit** | Fairness: cannot check a disparity you did not measure |
| Report first-time-patient subgroup separately | Strongest features are zero for them |

### 5.2 Pipeline

```
cleaned encounters
   │
   ├─ features.build_feature_matrix()   numeric + binary + one-hot categorical
   │     excludes: leakage, identifiers, race
   │
   ├─ train.split_by_person()           GroupShuffleSplit, verifies 0 overlap
   │
   ├─ train.train_all()                 Logistic Regression, Random Forest,
   │                                    Gradient Boosting (XGBoost if installed)
   │
   ├─ evaluate.*                        baseline, confusion matrix, ROC,
   │                                    thresholds, fairness, subgroup
   │
   └─ train.save_model()                models/readmission_model.joblib
                                        models/model_metadata.json
                                        models/model_comparison.csv
```

The saved bundle contains the model **and** the exact feature recipe. At
prediction time the recipe is replayed, guaranteeing identical columns in
identical order. Feature drift between training and serving is silent and is one
of the commonest ways a deployed model quietly breaks.

### 5.3 Evaluation philosophy

The system never reports a model score without the baseline beside it. A model
achieving 88% accuracy on this data may be doing nothing at all, because always
predicting "not readmitted" achieves the same. The threshold explorer presents
the recall/precision trade-off as a **capacity decision** — how many follow-up
calls can the ward actually make — rather than as a number to optimise.

---

## 6. Module 5 — generative AI

### 6.1 The core design decision: grounded generation

There are two ways to build an explanation layer.

**Ungrounded (rejected).** Send the patient record and ask the model to explain
the risk. The model answers from its training knowledge and produces fluent,
confident, invented statistics. Indefensible clinically and unverifiable
academically.

**Grounded (implemented).** Compute the facts in Python — the patient's recorded
values, the model's probability, per-feature contributions, and the measured
readmission lift for each factor **on this dataset** — and supply them. The
system prompt forbids using any number that was not provided.

The language model's role is reduced to **wording**, not knowledge. Every figure
traces back to a line in `metrics.py` and can be challenged.

### 6.2 Four-stage pipeline

```
grounding.py   build the fact block; pseudonymise the patient identifier
     ▼
prompts.py     system prompt (rules) + user prompt (facts + request)
     ▼
providers.py   Ollama/Llama → Anthropic → OpenAI → HuggingFace → template
               never raises; a provider failure falls through
     ▼
generator.py   validate the output, then return it
```

### 6.2.1 Provider selection

The default is **Ollama running Llama 3.2 locally**.

| Provider | Key | Cost | Data leaves machine |
|---|---|---|---|
| **Ollama / Llama** (default) | No | Free | **No** |
| Anthropic (Claude) | Yes | Paid | Yes |
| OpenAI (GPT) | Yes | Paid | Yes |
| Hugging Face | No | Free | No |
| Template | No | Free | No |

**Rationale.** Sending clinical records to a hosted API is a legal problem under
GDPR and HIPAA; pseudonymising the identifier is a mitigation, not a solution,
since the clinical detail remains identifying in combination. A locally-hosted
model eliminates the outbound request entirely. This is an architectural
decision about data residency, not a cost optimisation.

**Implementation.** Ollama exposes a plain HTTP API on `localhost:11434`. The
provider calls it via the standard library (`urllib`), adding **no new Python
dependencies**. Generation temperature is fixed at 0.2 — the task is factual
rewording of a supplied fact block, not open-ended writing.

**Configuration** (`.env`): `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`,
`GENAI_PROVIDER`.

### 6.3 Validation layer

Generated text is checked automatically before display:

| Rule | What it catches |
|---|---|
| No invented facts | Numbers in the output absent from the grounded facts |
| No causal claims | "caused by", "leads to" — the data shows association only |
| No prescribing | Specific drugs or doses |
| No overconfidence | "will be readmitted", "certainly" |
| Requires clinician review | Absence of the mandatory notice |
| States uncertainty | Absence of hedging |

A failure would block the report in deployment. The design principle: **the
model is not trusted, it is checked.**

### 6.4 Safety and privacy

- **The default provider is local.** Llama runs on the machine via Ollama, so
  no patient data is transmitted anywhere. Hosted APIs are supported but opt-in.
- Patient identifiers are pseudonymised before any external API call, for the
  case where a user deliberately selects a hosted provider.
- Every report carries a clinician-review notice and a statement that the
  underlying data is historical (1999–2008) and covers diabetic inpatients only.
- The demographic audit from Module 4 must be repeated on generated text.

---

## 7. Presentation layer

Six Streamlit pages, all reading the same shared library:

| Page | Purpose |
|---|---|
| Home | KPIs and the risk-factor lift chart |
| Disease Trends | Condition mix, HbA1c finding, visit-sequence analysis |
| Patient Explorer | Filterable records, utilisation, demographics audit |
| Correlation Lab | Feature ranking, multicollinearity check |
| Data Quality | Before/after cleaning evidence, outlier policy |
| Risk Prediction | Model predictions, evaluation, thresholds, fairness |
| AI Report | Grounded report generation, prompt inspection, validation |

The last two adapt to what is available: with no trained model the Risk page
shows the rule-based baseline; with no API key the AI page uses the template.

---

## 8. Error handling

| Failure | Behaviour |
|---|---|
| Dataset not downloaded | On-screen download links and instructions |
| MySQL unavailable | Fall back to cleaned CSV, badge shows the source |
| Tables not created | Named list of missing tables and the fix command |
| Schema statement fails | Error identifies the exact failing statement |
| No trained model | Rule-based baseline, with the training command |
| Ollama not installed | Install instructions with the exact `ollama pull` command |
| Ollama running, model missing | Names the missing model and the command to fetch it |
| AI provider fails or times out | Deterministic template, reason displayed |
| Filters match nothing | `empty_guard()` stops the page cleanly |

---

## 9. Known limitations

Stated deliberately. An evaluator who finds a limitation you already named
concludes you understand your data.

1. **Historical data.** 1999–2008 US practice. May not transfer to current
   practice or other health systems.
2. **Diabetic cohort only.** Findings do not generalise to general inpatients.
3. **Age is banded.** Ten-year bands; the midpoint is used, so "average age" is
   an average of midpoints.
4. **No vitals.** Physiological deterioration cannot be modelled.
5. **No dates.** No seasonality; no time-based validation.
6. **Prior-utilisation dependence.** The strongest features are zero for
   first-time patients — precisely when prediction is most valuable. Reported as
   a separate subgroup rather than hidden in an average.
7. **Scores rank, they are not calibrated.** Balanced class weights distort
   absolute probabilities. Read the band, not the number.

---

## 10. Roadmap

**Module 6 — API and web application.** FastAPI service importing the same
`metrics.py` and `ml.predict`; React front end; Docker; deployment to Render or
Railway. Not a rewrite: the analysis layer is already UI-agnostic, which is why
it imports no plotting or web framework.

**Model calibration.** Add Platt scaling or isotonic regression so probabilities
can be read as probabilities.

**Threshold tuning against real capacity.** The current explorer shows the
trade-off; a deployment would fix the threshold from actual follow-up capacity.
