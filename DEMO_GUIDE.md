# How to present this project

A 10-minute demo script, the findings to memorise, and prepared answers to the
questions an evaluator is likely to ask.

---

## Before you start

**The night before:**

```bash
python scripts/run_all.py
streamlit run streamlit_app/app.py
```

Click through all six pages. Write your own numbers into the table below — read
them off your dashboard rather than trusting the figures here.

**Ten minutes before:**

1. Activate the virtual environment.
2. Start the dashboard. Leave it running in a browser tab.
3. Open the notebook in a second tab, already run (`Kernel → Restart & Run All`).
4. Open your editor in a third tab, showing `adapters/uci_diabetes.py`.

Never run `pip install` or `run_all.py` in front of an evaluator.

---

## The one-sentence summary

> "It's an end-to-end pipeline that loads ten years of real hospital admission
> records into MySQL, cleans and analyses them to find which patients get
> readmitted within 30 days and why, and presents the findings in an interactive
> dashboard — built so the machine learning and GenAI modules plug into the same
> library without a rewrite."

## The 30-second version

> "The data is real: 101,766 hospital encounters from 130 US hospitals over ten
> years, published by Strack and colleagues in 2014 and hosted by UCI under a
> Creative Commons licence.
>
> Modules 1 to 3 are complete. Module 1 adapts that file into a normalised MySQL
> schema — including reshaping 23 medication columns into a proper one-to-many
> table. Module 2 cleans it and computes the statistics. Module 3 is a six-page
> Streamlit dashboard.
>
> The headline finding is that **prior hospital utilisation dominates** — how
> much care someone already used predicts readmission better than any single
> clinical measurement in this dataset. And the number I'd point at first is the
> **baseline**: because only about 11% of encounters end in readmission, a model
> that always guesses 'no' scores 89% accuracy while helping nobody. That's why
> Module 4 will be judged on recall and ROC-AUC."

---

## The 10-minute demo script

### Minute 0–1 · The problem and the data

> "Hospitals readmit a substantial share of patients within 30 days. Each
> readmission usually means something went wrong — a medication misunderstood, a
> follow-up missed. They're expensive and often financially penalised. Identify
> the at-risk patients before discharge and you can intervene.
>
> I'm using the Diabetes 130-US Hospitals dataset: ten years of real inpatient
> care, 101,766 encounters, 50 columns. It's the standard benchmark for this
> problem and it's licensed CC BY 4.0."

### Minute 1–3 · The architecture

Show the diagram in `README.md` section 8, and make two points:

> "The notebook and the dashboard import the *same* `metrics.py` and `charts.py`.
> The dashboard cannot show a number that disagrees with my analysis — there is
> only one definition of each statistic in the project.
>
> And the dataset-specific knowledge is isolated in one adapter file. The
> analysis code never knows which dataset it's looking at. Pointing this at
> MIMIC-III or a hospital's own extract means writing one new adapter, not
> editing every chart."

### Minute 3–4 · Data quality — go here early

Open the **Data Quality** page.

> "Most projects claim they cleaned the data. This page proves it.
>
> The first thing to notice: this dataset encodes missing values as the string
> question-mark, not as blanks. Read it naively and pandas treats them as
> ordinary text, and every missing-value count comes out as zero. There are
> nearly 200,000 of them.
>
> The second thing is that the interesting work here is *decisions*, not repairs.
> This is published research data, so it arrives in good shape. What I had to
> decide was: `weight` is 97% missing, so I drop it — imputing 97% of a column is
> inventing data, not cleaning it. `medical_specialty` is about half missing, but
> I keep it as an explicit 'Unknown', because an unrecorded speciality usually
> means the admission had no assigned consultant. That absence is information."

Then the outlier table:

> "Two kinds of outlier, handled differently. An out-of-range value contradicts
> the dataset's own inclusion criteria — a 40-day stay when only 1-to-14-day
> stays were included. That's a corrupted file, so I remove it. A *statistical*
> outlier is a patient on 70 medications with 9 diagnoses: unusual but real, so I
> keep it. Removing those would teach the model that complex cases don't exist,
> and they're exactly the patients we most need to predict."

### Minute 4–5 · Two decisions worth defending

> "Two things I do that aren't obvious.
>
> First, I remove patients who died or went to hospice. They cannot be
> readmitted. Leaving them in drags the rate down for a reason unrelated to care
> quality, and would teach a model that 'expired' predicts 'not readmitted' —
> true, useless, and grim. Excluding them is standard in the literature.
>
> Second — and this is the one that matters most — **one row is one encounter,
> not one person.** The same patient recurs many times. Those aren't duplicates,
> they're separate real admissions, so I keep them. But it means Module 4 must
> split train and test *grouped by person*. If the same individual lands on both
> sides, the model can memorise them and the accuracy is inflated. I flag
> `person_id` specifically so that split is possible."

### Minute 5–6 · The headline finding

Go to **Home**, scroll to the risk-factor chart.

> "This is the most important chart in the project. Lift is the readmission rate
> among patients with a factor, divided by the rate among those without it.
>
> Prior hospital utilisation comes out on top — inpatient admissions, emergency
> visits, outpatient visits in the year before. Past hospital use predicts future
> hospital use better than any individual clinical measurement here, which is
> consistent with the published readmission literature.
>
> I use lift rather than a correlation coefficient because a clinician and a
> hospital board both understand 'doubles the risk'. A correlation of 0.05 means
> nothing to either of them — and on real data, correlations are small even when
> the effect is operationally meaningful. That's worth saying out loud: real
> effects are subtler than textbook examples."

### Minute 6–7 · The finding that validates the pipeline

Go to **Disease Trends** → the HbA1c chart.

> "This is the question the dataset was published to answer, so it's my strongest
> validation. HbA1c measures average blood sugar over about three months — it
> tells you whether a diabetic patient's condition has been *controlled*, not
> just what it is today.
>
> What I find is that the majority of these diabetic admissions involved no HbA1c
> measurement at all. That's the original study's headline too: the most common
> decision is not to test, and whether the test was ordered carries information
> about how thoroughly the team engaged with the patient's diabetes.
>
> Reproducing a peer-reviewed finding on my own pipeline is the best evidence I
> have that the cleaning is correct. Nothing else in the project validates my
> work against published research."

### Minute 7–8 · Honest about what's missing

> "Two things this dataset does not have, and I'd rather name them than be caught
> out.
>
> There are no dates — it's fully de-identified. So seasonality analysis is
> impossible, and Module 4 can't use a time-based split. What survives is order:
> `encounter_seq` numbers each person's visits. I call it a sequence, never a
> trend, because claiming a time axis I don't have would be misleading.
>
> And there are no vital signs — no blood pressure, BMI or cholesterol. My
> original plan assumed those. Working with what the data actually contains
> rather than what I wished it contained is most of applied data science."

### Minute 8–9 · Leakage and the baseline

Show **Correlation Lab**, then **Risk Prediction**.

> "Correlation Lab ranks which variables track readmission — a feature-importance
> preview before any model exists. If the trained model ranks them completely
> differently, that tells me to look for a bug rather than celebrate a discovery.
>
> There's also a leakage trap in this data. `discharge_disposition` predicts
> strongly — but it's recorded *at* discharge. If the model is meant to run
> before discharge so staff can intervene, using it is textbook leakage. I've
> flagged it in the ML module notes rather than quietly feeding it in.
>
> And the Risk Prediction page is deliberately not faked. What's there is a
> rule-based score whose weights come from the lift values, and the page says
> clearly it isn't machine learning. It's a *baseline* — the bar the trained
> model has to clear. Without it, an 89% accuracy sounds impressive, when in fact
> always guessing 'not readmitted' scores about that on this data."

That last point is the strongest thing you can say. Make sure you say it.

### Minute 9–10 · Where it goes next

Show **AI Report**.

> "Module 5 will generate these explanations with an LLM, and the key decision is
> already made: **grounded generation**. Instead of asking a model to explain a
> prediction from its own medical knowledge — where it will produce a fluent,
> confident, invented statistic — I supply the lift values measured on this
> dataset and instruct it to use only those.
>
> That's why Module 2 had to come first. `metrics.risk_factor_summary()` already
> computes exactly the facts the prompt needs. Every sentence in a generated
> report will trace back to a number a clinician can challenge."

---

## Findings to memorise

**Fill these in from your own dashboard before you present.** The dataset is
fixed, so your numbers will be stable — but read them rather than trusting a
table you didn't generate.

| Finding | Your number |
|---|---|
| Encounters after cleaning | ~100,000 |
| Distinct people | ~70,000 |
| 30-day readmission rate | ~11% |
| Majority-class baseline accuracy | ~89% |
| Strongest risk factor | prior inpatient admissions |
| Encounters with no HbA1c test | large majority |
| `weight` missing | ~97% |
| `?` placeholders in the raw file | ~193,000 |

### The five insights, in words

1. **Prior hospital use dominates.** How much care someone already consumed
   predicts future consumption better than any single clinical measurement here.
2. **Rows are encounters, not people.** This one fact determines how Module 4
   must be evaluated. Getting it wrong inflates every score.
3. **The baseline is the headline.** Always guessing "not readmitted" scores
   ~89% while catching nobody. Recall and ROC-AUC are the honest metrics.
4. **Absence is data.** An HbA1c never ordered, a speciality never recorded —
   those describe the care process, not a broken file.
5. **Effects on real data are small.** Correlations of 0.05 that still matter
   operationally. Textbook datasets have neat correlations; real ones don't.

---

## Questions you will be asked

### "Where did this data come from?"

> "The UCI Machine Learning Repository, dataset 296 — also mirrored on Kaggle.
> It's the Diabetes 130-US Hospitals dataset, ten years of real inpatient care
> from 130 US hospitals, published alongside a 2014 paper by Strack and
> colleagues in BioMed Research International.
>
> It's licensed CC BY 4.0, so it's free to use and adapt with credit, and the
> citation is in my report. It's the standard benchmark for readmission
> prediction, which means my results are comparable to published work."

### "Why this dataset and not MIMIC-III?"

> "MIMIC-III needs credentialed access and ethics training — weeks of lead time.
> This one is immediately downloadable, it's peer-reviewed, and critically its
> published research question is exactly my target variable: 30-day readmission.
>
> That last point gave me something most student projects don't have — a way to
> check my pipeline against a known result rather than just hoping it's right."

### "Why MySQL? Why not just use the CSV?"

> "Three reasons. Concurrency — many users can read at once, which a CSV can't
> support. Integrity — the foreign keys mean I physically cannot store a
> medication for an encounter that doesn't exist. And scale — I can aggregate in
> the database and ship only the summary to Python, which is the only approach
> that works at millions of rows.
>
> There's a demo of that on the Data Quality page: the same readmission-by-
> diagnosis analysis, computed in SQL instead of pandas."

### "Why three tables instead of one?"

> "Normalisation, and this dataset makes the case for me.
>
> The raw file stores 23 medications as 23 separate columns, each holding
> 'No', 'Steady', 'Up' or 'Down'. That's wide format and it's unusable — you
> can't ask 'how many patients received insulin?' without naming the column, and
> you can't ask 'how many received any sulfonylurea?' at all.
>
> My adapter melts those into one row per prescription. That's what makes the
> one-to-many relationship queryable, and it's why `medications` deserves its own
> table. The `disease_reference` table then holds the ICD-9 group metadata in one
> place — reclassify a group and you edit one row, not 100,000."

### "How did you handle the missing values?"

> "First I had to find them. This dataset encodes missing as the string
> question-mark, so a naive read reports zero missing values across the board.
> There are about 193,000.
>
> Then three different treatments, by column. `weight` is 97% missing so I drop
> it — imputing 97% of a column is inventing data. `payer_code` is 40% missing
> and not clinically meaningful, so it goes too. `medical_specialty` is about
> half missing but I keep it as an explicit 'Unknown', because an unrecorded
> speciality usually means the admission had no assigned consultant.
>
> For the numeric columns I impute with the median rather than the mean, because
> the counts are right-skewed — a few patients on 60+ medications would drag the
> mean up and I'd overstate the typical stay.
>
> And I never impute gender or ethnicity. 'Unknown' is the truthful value, and
> keeping it visible means nobody mistakes an invented value for a recorded one."

### "How do you know your cleaning didn't destroy real signal?"

> "That's the right question, and it's why the Data Quality page shows before and
> after rather than just asserting the data is clean.
>
> Two safeguards. I only remove values that contradict the dataset's own
> documented inclusion criteria. Statistical outliers I keep, deliberately. And
> imputation uses the median, which doesn't shift the distribution's centre.
>
> The strongest evidence is that I reproduce the published HbA1c finding after
> cleaning. If my pipeline had destroyed signal, I'd expect that to disappear."

### "Why haven't you built the ML model yet?"

> "Because the analysis had to come first, and it determined the design. Several
> decisions came out of Module 2 rather than out of a tutorial:
>
> The target is about 11% positive, so I'll use balanced class weights and judge
> on recall and ROC-AUC, not accuracy. Rows are encounters and people repeat, so
> the split must be grouped by person. There are no dates, so a time-based split
> is impossible — which I'd have assumed was the right approach if I hadn't
> looked. And `discharge_disposition` is a leakage risk, so the prediction moment
> has to be fixed before feature selection.
>
> They're written down in `src/hospital_analytics/ml/__init__.py`, along with the
> function signatures the dashboard already codes against."

### "What's the biggest weakness of your analysis?"

Have this ready — being asked and having nothing is much worse than naming a
real limitation.

> "The dependence on prior-utilisation features. They're my strongest predictors,
> but they're *historical* — they don't exist for a genuinely first-time patient.
> And that's exactly the case where a clinician has least information and would
> most value a prediction.
>
> So in Module 4 I need to check performance specifically on the subgroup where
> prior visits are zero, not just report one overall AUC. If the model collapses
> there, it's much less useful than the headline number suggests.
>
> The second weakness is that this is a *diabetes* cohort — every patient has
> diabetes recorded. So my findings generalise to diabetic inpatients, not to
> hospital populations in general. I'd want to be careful about that claim."

### "What is data leakage, and does your project have any?"

> "Leakage is when a feature contains information that wouldn't be available at
> the moment you actually need the prediction. The model scores brilliantly in
> testing and is useless in deployment.
>
> There's a real candidate here: `discharge_disposition`, where the patient went
> after discharge. It's strongly related to readmission and it's recorded *at*
> discharge. If the model is meant to run before discharge so staff can
> intervene, using it is leakage.
>
> I've flagged it in the ML module notes. The decision is to fix the prediction
> moment first, then include only features available at that moment."

### "Have you thought about bias or fairness?"

> "Yes — there's a demographic audit on the Patient Explorer page, and it's
> deliberately positioned *before* any model exists.
>
> This dataset records race and gender, so I check readmission rates across both.
> If they differed sharply within the same age band, a carelessly trained model
> could encode that and distribute care unequally. I've recorded the result
> either way, because auditing and writing down the finding is the point, not
> hoping it comes out clean.
>
> Module 4 must repeat that check on the model's *predictions*, not just the
> input data."

### "Which AI are you using, and why?"

> "Llama 3.2, running locally on this machine through Ollama. It's free and
> needs no API key, which matters practically — it works on your machine without
> my credentials, and it doesn't fail if the wifi drops.
>
> But the real reason is data residency. Sending clinical records to a hosted
> API is a genuine legal problem under GDPR and HIPAA, and pseudonymising the
> patient ID is a mitigation rather than a solution — the clinical detail is
> still identifying in combination. Running locally removes the question: there
> is no outbound request, so there is nothing to leak or log.
>
> Claude and GPT are supported as opt-in alternatives, but for patient data the
> local model is the correct default rather than the cheap one."

### "Isn't a small local model worse than GPT-4?"

> "At open-ended writing, yes. At this task it barely matters, and that's by
> design. The model supplies no facts — it rewords a fact block that Modules 2
> and 4 computed, under a validator that checks the output. That's a short,
> heavily-constrained task a small model handles fine.
>
> If my design needed a frontier model to be *correct*, that would mean the
> model was supplying content — which is exactly the failure mode the grounding
> layer prevents."

### "How would you stop the GenAI module hallucinating?"

> "Grounded generation. There are two ways to build it and only one is safe.
>
> The unsafe version says 'here's a patient, explain the risk' — the model
> answers from its own knowledge and produces a fluent, confident, invented
> statistic.
>
> The version I've designed supplies the facts: the patient's recorded values,
> the model's predicted probability, the SHAP contributions, and the population
> lift for each risk factor as measured on *this* dataset. The system prompt
> instructs it to use only those and to say when something isn't recorded.
>
> I've also told it explicitly that this dataset has no vitals and no dates, so
> it can't invent a blood pressure reading. Every sentence traces back to a
> number in `metrics.py`."

### "Why Streamlit and not React?"

> "Speed of iteration for the analysis phase. Streamlit let me build six
> interactive pages in Python without touching JavaScript, which was the right
> trade-off while the focus was on understanding the data.
>
> React is in the roadmap for cycle 3, against a FastAPI backend. And that's not
> a rewrite: the API will import the same `analysis/metrics.py` this dashboard
> uses. That's why `metrics.py` imports no plotting or web framework — it's
> deliberately UI-agnostic so it can be reused."

### "Explain your caching."

> "Streamlit re-runs the entire script top to bottom on every interaction.
> Without caching, every checkbox tick would re-query MySQL and re-clean 100,000
> rows, and the app would feel broken.
>
> `@st.cache_data` on `get_data()` means the load happens once and is reused
> across pages and reruns. There's a one-hour TTL so a fresh MySQL load appears
> without restarting, and a manual cache-clear button on the Data Quality page."

### "What happens if MySQL is down?"

> "The dashboard falls back to the cleaned CSV and displays 'Data source: CSV
> fallback' in the sidebar. Every chart still works.
>
> Two reasons that's deliberate. Practically, a database problem on demo day
> can't stop the presentation. But more importantly it *tells you* which source
> it used — so you're never unknowingly presenting stale data. Silent fallback
> would be worse than crashing."

### "Why did you use cufflinks if it's broken?"

> "The brief specified it, so I wanted to use it properly rather than ignore the
> requirement. Cufflinks adds `.iplot()` directly to pandas DataFrames, which is
> genuinely convenient.
>
> But it's been unmaintained since 2021 and calls `DataFrame.iteritems()`, which
> pandas removed in version 2.0 — so it crashes on import on any modern setup.
>
> The notebook patches that method back before importing it, and falls back to
> Plotly Express if the import still fails. So it works either way. I'd argue
> recognising an abandoned dependency and handling it is the more useful skill."

---

## If something breaks live

**Stay calm and narrate.** An evaluator watching you debug calmly learns more
about you than one watching a flawless click-through.

| Problem | Say this, then do that |
|---|---|
| Dashboard won't start | "Let me check the environment is active." → activate venv, re-run |
| MySQL error | "This is what the CSV fallback is for." → `--csv-only`, keep going |
| Dataset not found | "The raw file isn't in `data/raw` — the cleaned CSV still is." → carry on from the CSV |
| A chart is empty | "The filters are too narrow — let me widen them." → Reset filters |
| Notebook cell errors | "Let me restart the kernel and run from the top." |
| Total failure | Switch to the notebook, already run, with all output saved |

**Always have the notebook open in a second tab with its outputs saved.** It is
your parachute: every chart and finding is visible there even if nothing else
runs.

---

## Cite this

In your report, your slides, and anywhere you show a number from it:

> Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN (2014).
> *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000
> Clinical Database Patient Records.* BioMed Research International, vol. 2014,
> Article ID 781670. DOI 10.24432/C5230J. Available from the UCI Machine Learning
> Repository under a CC BY 4.0 licence.

Attribution is a *licence condition*, not a courtesy. Leaving it out is the one
thing here that would actually be a problem.
