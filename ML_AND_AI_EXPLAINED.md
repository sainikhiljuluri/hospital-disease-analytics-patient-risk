# How the ML and AI modules actually work

Written to be read in order. By the end you should be able to explain both
modules without notes, and answer the follow-up questions.

---

# PART 1 — MODULE 4: THE MACHINE LEARNING MODEL

## 1.1 The question, stated precisely

> *Given everything known about a patient the day before they are discharged,
> what is the probability they will be readmitted within 30 days?*

Every word in that sentence is a design decision.

**"the day before they are discharged"** — the prediction moment. Choose it
first, before anything else, because it determines which data you are allowed to
use.

**"within 30 days"** — the target definition. The raw data has three outcomes:
`<30`, `>30`, `NO`. We fold `>30` into the negative class. A readmission at 45
days is real; we are simply not predicting it.

**"probability"** — not yes/no. A probability can be ranked, and a ward with
capacity for ten follow-up calls can take the top ten.

---

## 1.2 Step one: choosing features, and what leakage means

**Leakage** is the classic beginner trap. It means using a feature that contains
information unavailable at the moment you actually need the prediction. The model
scores brilliantly in testing and is useless in deployment.

This dataset contains a textbook example: `discharge_disposition` — where the
patient went after discharge. It predicts readmission strongly. Patients sent to
a nursing facility come back more often than those sent home.

But it is **recorded at discharge**. Our model runs the day *before*. On the
afternoon the ward wants a prediction, that value does not exist.

Including it would raise every metric and destroy the model's usefulness.

### What is excluded, and why

| Excluded | Reason |
|---|---|
| `discharge_disposition` | **Leakage** — recorded after the prediction moment |
| `discharged_home`, `to_care_facility` | Derived from it — same leakage |
| `readmitted_any` | A different view of the answer |
| `patient_id`, `person_id` | Identifiers — memorisable, not clinical |
| `icd9_primary` | Hundreds of codes; one-hot encoding creates hundreds of near-empty columns. The grouped `diagnosis` carries the signal |
| `race` | **Fairness, not leakage** — see 1.7 |

### What is included

**Numeric (11):** age, length of stay, lab procedures, procedures, medications,
diagnoses recorded, prior inpatient/emergency/outpatient counts, total prior
visits, visit number.

**Binary flags (15):** is_elderly, is_long_stay, is_polypharmacy,
has_prior_admission, is_complex, a1c_tested, a1c_high, medication_changed,
on_diabetes_meds, is_emergency, is_chronic, and others.

**Categorical, one-hot encoded (8):** gender, age band, diagnosis group, disease
category, admission type, admission source, HbA1c result, glucose result.

Around 69 columns after encoding.

### One-hot encoding, explained

A model cannot read the word "Emergency". So a column with four values becomes
four columns of 0/1:

```
admission_type          admission_type__Emergency  __Elective  __Urgent  __Other
Emergency          →              1                    0           0        0
Elective           →              0                    1           0        0
```

Categories appearing fewer than 50 times are folded into `_Other`. A category
with three examples teaches the model nothing and adds a column that is almost
entirely zeros.

---

## 1.3 Step two: the split — the most important thing in Module 4

You must test a model on data it has never seen. Otherwise you are measuring
memory, not learning.

**The naive approach is wrong here.** A random 75/25 split would be a serious
methodological error, and it is the mistake this dataset is designed to catch.

**Why:** one row is one **encounter**, not one patient. The same person appears
several times — roughly 30,000 of the 100,000 rows are return visits.

A random split puts Mrs Smith's January admission in training and her March
admission in testing. The model does not learn "patients like this return" — it
learns *Mrs Smith specifically*. Then it recognises her in the test set and looks
brilliant. In deployment, facing genuinely new patients, it collapses.

**The fix: group the split by person.**

```python
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=person_id))
```

Every encounter belonging to one person lands entirely on one side. The training
script proves it:

```
SPLIT  (grouped by person, so nobody is in both halves)
  people in train   4,843
  people in test    1,615
  people in BOTH        0   <- must be 0
```

That line exists to be pointed at during a demo.

> **Note on time-based splitting.** Ideally you would train on earlier data and
> test on later, since that mirrors deployment. **This dataset has no dates** — it
> is fully de-identified. Knowing *why* you cannot do the ideal thing is worth as
> much as doing it.

---

## 1.4 Step three: the imbalance problem

About 11% of encounters end in readmission. That single fact reshapes everything.

### The accuracy trap

A model that ignores its input entirely and always answers "not readmitted"
achieves **~89% accuracy**.

It identifies zero at-risk patients. It is completely worthless. And it beats
most poorly-designed real models on accuracy.

This is why the training script prints the baseline before it prints any model
score. **A model score without a baseline is not a result, it is a number.**

### Two responses

**1. Balanced class weights.** Tell the model that missing a readmission costs
more than a false alarm:

```python
LogisticRegression(class_weight="balanced")
```

Internally each of the ~11% positive cases is weighted roughly 8× a negative
case, so the model cannot minimise its loss by ignoring them.

**2. Judge on the right metrics.**

| Metric | Meaning | Why it matters here |
|---|---|---|
| **Recall** | Of patients who *were* readmitted, what share did we flag? | The clinical priority — a missed patient is the expensive failure |
| **Precision** | Of patients we flagged, what share were readmitted? | The cost side — every flag is a follow-up call |
| **ROC-AUC** | Ability to rank a random positive above a random negative | Threshold-independent; 0.5 = coin flip |
| **PR-AUC** | Precision/recall trade-off across thresholds | More informative than ROC on imbalanced data |
| Accuracy | Share of all predictions correct | **Nearly meaningless here** |

---

## 1.5 Step four: three models, and why each

| Model | Why it is in the comparison |
|---|---|
| **Logistic Regression** | Interpretable — each coefficient is a readable weight. The honest baseline for "would something simple do?" |
| **Random Forest** | Many decision trees voting. Captures interactions automatically; unaffected by multicollinearity |
| **Gradient Boosting** | Trees built sequentially, each correcting the last. Usually strongest on tabular data |

All three train on the same split with the same features, and are ranked by
ROC-AUC.

> **On XGBoost:** if installed, it is used. If not, scikit-learn's
> `HistGradientBoostingClassifier` is used instead — comparable performance, no
> extra install. The script says which it used. Do not claim XGBoost if it fell
> back.

### Why multicollinearity only matters sometimes

When two features carry nearly the same information, a **logistic regression**
cannot decide which deserves the credit. Its coefficients become unstable — and
interpretability was the reason to use it.

**Tree models are unaffected.** They pick one at each split and move on.

So the decision is conditional: drop one of each redundant pair for the logistic
regression only, keep both for the trees.

---

## 1.6 Step five: reading the results

### The confusion matrix

|  | Predicted: no | Predicted: yes |
|---|---|---|
| **Actually not readmitted** | True negative | False positive |
| **Actually readmitted** | **False negative** | True positive |

**The bottom-left cell is the one that matters.** Those are at-risk patients the
model missed — the failure with a clinical cost. A false positive costs one
unnecessary phone call.

### The threshold is a policy decision, not a technical one

The model outputs a probability. Turning it into a yes/no flag needs a cut-off,
and **there is no mathematically correct value**.

- **Lower threshold** → catch more readmissions, more false alarms
- **Higher threshold** → fewer wasted calls, more missed patients

The right value depends on how many follow-up calls the ward can actually make.
The dashboard's threshold explorer presents this as a capacity question, because
that is what it is.

This is the strongest slide you can put in a report: it shows you understand a
model is deployed into a system with finite resources, not evaluated in a vacuum.

---

## 1.7 Step six: the fairness audit

`race` is **excluded from training**. Using it would let the model allocate care
differently by ethnicity.

But it is **retained for evaluation** — because you cannot check for a disparity
you did not measure.

### Why excluding it is not sufficient

The model never sees ethnicity. It can still discriminate, because other features
correlate with it. Prior hospital utilisation reflects access to care, which is
not equally distributed. A model trained only on "neutral" features can reproduce
a disparity it was never shown.

**The only way to find out is to measure recall by group** — of the patients in
each group who were actually readmitted, what share did the model catch?

The training script prints exactly that, and flags any group where the recall gap
exceeds 0.15.

> Check the patient counts before drawing conclusions. A group of 60 people
> produces an unstable rate, and a large gap there may be noise.

### The stress test

Prior-visit counts are the strongest features — **and they are all zero for a
first-time patient.** That is precisely when a clinician has least information and
would most value a prediction.

So performance is reported separately for both groups. If ROC-AUC collapses for
first-time patients, the model is less useful than one headline number suggests.
Report both rows.

---

## 1.8 What gets saved

```
models/readmission_model.joblib    the model + the exact feature recipe
models/model_metadata.json         scores, baseline, fairness, thresholds
models/model_comparison.csv        all three models ranked
```

The bundle contains the model **and the recipe**. At prediction time the recipe is
replayed, guaranteeing identical columns in identical order.

If those two ever drift apart, the model receives columns in a different order and
produces confident nonsense — **with no error message**. That silent failure is
one of the commonest ways a deployed model quietly breaks.

---

# PART 2 — MODULE 5: THE GENERATIVE AI LAYER

## 2.1 The problem Module 4 leaves behind

The model outputs `0.31`.

A nurse cannot act on `0.31`. They need to know *why*, and *what to do*.

That is Module 5's job: convert a number into an explanation a human can act on.

---

## 2.2 The central decision: grounded generation

There are two ways to build this. Only one is safe, and understanding the
difference is the single most valuable thing in this module.

### The unsafe way

> *"Here is a patient. Explain why they might be readmitted."*

The model answers from its own training knowledge. It will produce fluent,
confident, medically-plausible text containing **invented statistics** —
"patients with this profile have a 34% readmission rate" where 34% came from
nowhere.

It reads perfectly. It is indefensible clinically, and unverifiable
academically, because you cannot show where any number came from.

### The way this project works

We compute the facts ourselves and **supply** them:

- the patient's recorded values
- the model's predicted probability and risk band
- the per-feature contributions behind that prediction
- the **measured readmission lift** for each risk factor, on *this* dataset

The system prompt then forbids the model from using any number that was not
supplied.

**The language model's job is reduced to wording, not knowledge.**

Every figure in the output traces back to a line in `metrics.py`. A clinician can
challenge any sentence and be shown the number behind it.

> This is also why Module 2 had to come first. `metrics.risk_factor_summary()`
> already computes exactly the facts the prompt needs.

---

## 2.3 The four-stage pipeline

```
1. GROUND     grounding.py    collect the permitted facts
                              pseudonymise the patient identifier
        ▼
2. PROMPT     prompts.py      system prompt (the rules)
                              user prompt (the facts + the request)
        ▼
3. GENERATE   providers.py    Anthropic → OpenAI → HuggingFace → template
                              never raises; failure falls through
        ▼
4. VALIDATE   generator.py    check the output before anyone sees it
```

### Stage 1: Grounding

Builds the complete fact block. Nothing outside it may be used.

The patient identifier is **pseudonymised** before anything leaves the machine —
`564693` becomes `590DC7`. This matters legally, not just ethically.

### Stage 2: Prompting

Two parts:

- **System prompt** — the rules. Use only the supplied facts. Never state a
  cause. Never prescribe. Always require clinician review. State uncertainty.
- **User prompt** — the facts, plus what to write.

Three templates, each with a different audience and different rules:

| Template | Audience | Tone |
|---|---|---|
| `clinical_report` | Ward staff | Technical, cites lift values |
| `patient_letter` | The patient | Plain English, no jargon, no numbers |
| `cohort_brief` | A manager | Service-level, aggregate |

The patient letter has *stricter* rules — it must not contain risk percentages at
all. Handing someone a "31% chance of readmission" is not informative, it is
frightening.

### Stage 3: Generation — a local Llama model

The default provider is **Ollama running Llama 3.2 on your own machine**.

| Provider | Key needed | Cost | Data leaves machine |
|---|---|---|---|
| **Ollama / Llama** (default) | No | Free | **No** |
| Anthropic (Claude) | Yes | Paid | Yes |
| OpenAI (GPT) | Yes | Paid | Yes |
| Hugging Face | No | Free | No (but slower, bigger) |
| Template | No | Free | No |

**Why local is the correct choice here, not merely the cheap one:**

Sending real clinical records to a hosted API is a genuine legal problem under
GDPR and HIPAA. Pseudonymising the patient ID first is a *mitigation*, not a
solution — the clinical detail is still identifying in combination.

A locally-hosted model removes the question entirely. There is no outbound
request, so there is nothing to leak, log, or subpoena. For a project about
patient records, that is an architectural decision, not a cost saving.

It is also free, needs no account, and keeps working when the venue wifi does
not.

**How the project talks to it.** Ollama exposes a plain HTTP API on
`localhost:11434`. The project calls it with Python's built-in `urllib` — so
this provider adds **zero new dependencies**. Nothing to `pip install`, one less
thing to break on a marker's machine.

```python
payload = {
    "model": "llama3.2",
    "messages": [{"role": "system", ...}, {"role": "user", ...}],
    "options": {"temperature": 0.2},   # low: stick to the supplied facts
}
```

`temperature` is deliberately low. This is a factual summary assembled from
supplied numbers, not creative writing — we want the model close to its facts
rather than embellishing them.

**Failure never raises.** If Ollama is not installed, or the model is still
downloading, or the request times out, the pipeline falls through to the next
provider and finally to the template. A demo should not die because a local
service was not running.

### Stage 4: Validation

**This is the part worth showing an evaluator.**

Generated text is checked automatically before display:

| Rule | Catches |
|---|---|
| **No invented facts** | Numbers in the output absent from the supplied facts |
| No causal claims | "caused by", "leads to" — the data shows association only |
| No prescribing | Named drugs or doses |
| No overconfidence | "will be readmitted", "certainly" |
| Requires clinician review | Missing mandatory notice |
| States uncertainty | Missing hedging |

A failure **blocks** the report in deployment rather than displaying it.

The principle: **the language model is not trusted. It is checked.**

---

## 2.4 Why it still works with no AI at all

If Ollama is not installed either, the same grounded facts are rendered by a
deterministic Python template. Identical information, slightly stiffer prose.

**Do not apologise for this in a demo. Demonstrate it.**

It proves the grounding layer is doing the work rather than the language model —
which is the entire architectural claim of Module 5. If the report is still
correct and complete without an LLM, then the LLM was only ever handling wording,
exactly as designed.

---

## 2.5 The safety constraints, and why they were written first

1. **Grounded generation only** — the main defence against invented statistics.
2. **Decision support, never diagnosis** — output is advisory.
3. **No identifiable data leaves the building** — pseudonymised before any API
   call; local Hugging Face model supported where nothing may leave at all.
4. **Every claim is auditable** — each statement traces to a number in
   `metrics.py`.
5. **Fairness carried forward** — the Module 4 audit must be repeated on
   generated text, so explanations do not reproduce a bias in different words.
6. **Historical-data limitation stated on every output** — 1999–2008, US,
   diabetic inpatients only.

These were written **before** the LLM was connected. That ordering is deliberate:
constraints written first shape the design; constraints written afterwards are
apologies.

---

# PART 3 — QUESTIONS YOU WILL BE ASKED

**"Why is your accuracy only 63%?"**

> "Accuracy is the wrong metric here and I'd argue the 63% is the more honest
> number. Always predicting 'not readmitted' scores 89% while identifying nobody.
> My model trades accuracy for recall — it catches roughly half of actual
> readmissions. For a hospital that trade is correct: missing a patient who comes
> back costs far more than an unnecessary phone call."

**"How do you know the model isn't cheating?"**

> "Three defences. The split is grouped by person, and the script prints 'people
> in both: 0' as proof. Discharge features are excluded because they're recorded
> after the prediction moment. And identifiers are excluded entirely. The
> excluded-features table with reasons is printed every time it trains."

**"Why did you exclude ethnicity but still report it?"**

> "Training on it would let the model allocate care by ethnicity. But excluding
> it isn't sufficient — other features correlate with it, so the model can
> reproduce a disparity it was never shown. The only way to know is to measure
> recall by group, and you can't measure what you didn't keep. Exclude from
> training, include in evaluation."

**"How do you stop the AI hallucinating?"**

> "Grounded generation plus a validator. I compute the facts in Python and the
> system prompt forbids using any number I didn't supply. Then every generated
> report is checked automatically — the strictest rule matches every number in
> the output against the supplied facts, so a hallucinated figure is caught
> mechanically rather than by someone noticing. In deployment a failure blocks
> the report."

**"Is the AI part just a template?"**

> "No — it runs Llama 3.2 locally through Ollama. But it *also* works as a
> template, and that's deliberate. If the report is still correct without the
> LLM, that proves the grounding layer is doing the work — which is the
> architectural claim. The LLM only ever handles wording. I can show you both
> side by side."

**"Why a local model instead of ChatGPT or Claude?"**

> "Three reasons, and the third is the real one. It's free, so it works on a
> marker's machine without my API key. It works offline, so it doesn't fail on
> demo day when the wifi does.
>
> But mainly: patient records never leave the machine. Sending real clinical
> data to a hosted API is a genuine legal problem under GDPR and HIPAA, and
> pseudonymising the ID is a mitigation rather than a solution — the clinical
> detail is still identifying in combination. Running locally removes the
> question entirely. There's no outbound request, so there's nothing to leak or
> log. For patient data that's an architectural decision, not a cost saving."

**"Isn't a small local model worse than GPT-4?"**

> "At open-ended writing, yes. At this task, it barely matters — and that's a
> consequence of the design rather than luck. The model isn't supplying any
> facts; it's rewording a fact block that Modules 2 and 4 computed. It's a
> short, heavily-constrained writing task with a validator checking the output
> afterwards. That's exactly the kind of job a small model does fine.
>
> If the design *needed* a frontier model to be correct, that would mean the
> model was supplying content — which is the failure mode I built the grounding
> layer to prevent."

**"What would you do next?"**

> "Three things. Calibrate the probabilities — balanced class weights distort
> them, so the scores rank reliably but shouldn't be read as literal
> probabilities. Fix the threshold against a ward's real follow-up capacity
> rather than leaving it at 0.5. And investigate performance on first-time
> patients specifically, since my strongest features are zero for them and that's
> exactly when a prediction is most valuable."
