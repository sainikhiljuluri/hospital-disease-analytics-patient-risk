# Getting Started — the complete walkthrough

This guide assumes **you have never opened a terminal**. Nothing is skipped.

Read it top to bottom the first time. Total time: about 30 minutes, most of it
waiting for downloads.

---

## Contents

1. [What you are building](#1-what-you-are-building)
2. [The terminal, explained](#2-the-terminal-explained)
3. [Install Python](#3-install-python)
4. [Install MySQL (optional)](#4-install-mysql-optional)
5. [Get the project onto your computer](#5-get-the-project-onto-your-computer)
6. [Create a virtual environment](#6-create-a-virtual-environment)
7. [Install the libraries](#7-install-the-libraries)
8. [Download the dataset](#8-download-the-dataset)
9. [Set your password file](#9-set-your-password-file)
10. [Build the database](#10-build-the-database)
11. [Train the machine learning model](#11-train-the-machine-learning-model)
12. [Generate an AI report](#12-generate-an-ai-report)
13. [Run the dashboard](#13-run-the-dashboard)
14. [Run the notebook](#14-run-the-notebook)
15. [Doing it all again tomorrow](#15-doing-it-all-again-tomorrow)
16. [When something goes wrong](#16-when-something-goes-wrong)

---

## 1. What you are building

Five connected pieces:

| Module | What it does |
|---|---|
| 1 | Loads 100,000 real hospital records into a database |
| 2 | Cleans them and works out what predicts readmission |
| 3 | Shows the findings in an interactive dashboard |
| 4 | Trains a model to predict which patients will return |
| 5 | Writes a plain-English explanation using a local Llama model |

You will run each one with a single command.

---

## 2. The terminal, explained

The terminal is a window where you type commands instead of clicking. Everything
below happens there.

### Opening it

| System | How |
|---|---|
| **Windows** | Press `Windows key`, type `cmd`, press Enter |
| **Mac** | Press `Cmd + Space`, type `terminal`, press Enter |
| **Linux** | `Ctrl + Alt + T` |

A window opens with a blinking cursor. That is the prompt. You type a command and
press Enter.

### The five commands you need

| Command | Meaning | Example |
|---|---|---|
| `cd folder` | go **into** a folder | `cd hospital-analytics` |
| `cd ..` | go **up** one folder | `cd ..` |
| `ls` (Mac/Linux) `dir` (Windows) | list what is here | `ls` |
| `pwd` (Mac/Linux) `cd` (Windows) | where am I? | `pwd` |
| `python file.py` | run a Python file | `python scripts/run_all.py` |

### The single most common beginner error

**Being in the wrong folder.**

Almost every "file not found" error means the terminal is somewhere other than
your project folder. When something fails, first run `ls` (or `dir`) and check
you can see `README.md` and `scripts`. If you cannot, you are in the wrong place.

### The drag-and-drop trick

Instead of typing a long folder path: type `cd` then a space, then **drag the
folder from your file explorer into the terminal window** and press Enter. The
path is filled in for you.

### One more thing

If `python` gives "command not found", try `python3` instead. Mac and Linux
usually need `python3`. Wherever this guide says `python`, use whichever works on
your machine — consistently.

---

## 3. Install Python

Go to **python.org/downloads** and install version 3.10 or newer.

> **Windows users: tick the box that says "Add Python to PATH"** on the first
> screen of the installer. It is easy to miss and everything fails without it.
> If you missed it, re-run the installer and choose Modify.

Check it worked. Open a **new** terminal window and type:

```bash
python --version
```

You should see something like `Python 3.12.1`. If not, try `python3 --version`.

---

## 4. Install MySQL (optional)

**You can skip this entire section.** The project works without MySQL — it falls
back to CSV files automatically, and every feature still functions.

Install it if your brief requires a database (most do).

**Windows / Mac:** download MySQL Community Server from `dev.mysql.com/downloads`
and run the installer. **Write down the root password you set.** You cannot
recover it easily.

**Mac with Homebrew:** `brew install mysql` then `brew services start mysql`

**Linux:** `sudo apt install mysql-server`

---

## 5. Get the project onto your computer

Unzip `hospital-analytics.zip` somewhere you will remember — `Documents` is fine.

> **Windows:** right-click → *Extract All*. Do not skip this. Windows lets you
> browse *inside* a zip without extracting, but Python cannot run from there.

Now point your terminal at the folder:

```bash
cd Documents/hospital-analytics
```

(Or use the drag-and-drop trick from section 2.)

Check you are in the right place:

```bash
ls          # Windows: dir
```

You should see `README.md`, `requirements.txt`, `scripts`, `src`,
`streamlit_app`. If you see a *single* folder also called `hospital-analytics`,
go one level deeper: `cd hospital-analytics`.

**Every remaining command in this guide is run from this folder.**

---

## 6. Create a virtual environment

A virtual environment is a private box of libraries for this project. Without
one, installing libraries here could break a different project on your computer.

Create it (once, ever):

```bash
python -m venv .venv
```

Activate it (**every time you open a new terminal**):

```bash
# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. That is how you know it worked.

> **If you forget this step**, you will get `ModuleNotFoundError: No module named
> 'pandas'` later. That error almost always means the environment is not active.

---

## 7. Install the libraries

```bash
pip install -r requirements.txt
```

This downloads pandas, plotly, streamlit, scikit-learn and the rest. It takes
2–5 minutes and prints a lot of text. That is normal.

When it finishes you should see something like `Successfully installed ...`.

---

## 8. Download the dataset

The project uses **real hospital data**. It is not included in the zip, because
it is 18 MB and has its own permanent home.

**Get it from either link — the file is identical:**

- **UCI (no account needed):**
  `archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008`
- **Kaggle (free login):** `kaggle.com/datasets/brandao/diabetes`

Click Download. You get a `.zip`.

**Unzip it, then move the file `diabetic_data.csv` into the `data/raw/` folder
inside your project.** Ignore the other file (`IDS_mapping.csv`).

Check it landed correctly:

```bash
python scripts/01_check_dataset.py
```

You should see:

```
  FOUND: .../data/raw/diabetic_data.csv
  size : 18.3 MB
  columns: 50 (expected 50)
  all critical columns present
  rows   : 101,766 (expected 101,766)
  30-day readmission rate: 11.16%
  '?' placeholders to clean: 192,849
```

Those `?` marks are how this dataset records missing values — nearly 200,000 of
them. Your cleaning module handles them.

> **If it says NOT FOUND**, the file is probably still inside the unzipped
> subfolder. Move `diabetic_data.csv` itself into `data/raw/`.

---

## 9. Set your password file

Copy the template:

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open the new `.env` file in any text editor (Notepad is fine) and change one
line:

```
MYSQL_PASSWORD=whatever_you_set_when_installing_mysql
```

Save and close.

**Skipping MySQL?** Leave the file as it is. The project will use CSV files.

> **Why a separate file?** So your password never appears in a `.py` file and
> never reaches GitHub. `.gitignore` already excludes `.env`.

---

## 10. Build the database

Two commands:

```bash
python scripts/02_setup_database.py
python scripts/03_load_data.py
```

The first creates the database and its four tables, then verifies each one
exists. The second cleans the data and loads it.

Watch the output of the second command — this is your evidence:

```
STEP 1/4  EXTRACT
  encounters read        101,766
  deceased/hospice removed  1,642
  encounters kept        100,124
  unique people           69,990
    weight              96.9% missing  -> DROPPED
    payer_code          39.6% missing  -> DROPPED
    medical_specialty   49.2% missing  -> kept as Unknown

STEP 2/4  TRANSFORM
  repeat encounters kept    30,134
  clean encounter rows     100,124
  30-day readmission rate    11.3%
```

Three of those lines are worth being able to explain:

- **deceased/hospice removed** — they cannot be readmitted. Leaving them in would
  teach a model that "expired" predicts "not readmitted": true, useless, grim.
- **repeat encounters kept** — the same person appearing twice is *not* a
  duplicate. Those are separate real admissions.
- **weight DROPPED** — 97% missing. Imputing 97% of a column is inventing data.

---

## 11. Train the machine learning model

```bash
python scripts/04_train_model.py
```

Takes about a minute. It trains three models and picks the best.

**The most important thing it prints:**

```
BASELINE  always predict 'not readmitted'
  accuracy          88.84%
  recall             0.00%   <- catches nobody
```

Read that carefully. A model that does **nothing** — that says "not readmitted"
for every single patient — scores nearly 89% accuracy. That is why this project
judges models on **recall** (what share of at-risk patients did we catch) and
**ROC-AUC**, not accuracy.

If an evaluator asks you one question about your model, it will probably be
about this. Have the answer ready.

It also prints:

- `people in BOTH: 0` — proof the train/test split kept each person on one side
- a fairness audit across gender, ethnicity and age
- a stress test on first-time patients

Results are saved to `models/`.

---

## 12. Install Llama (free, local, no API key)

Module 5 writes the plain-English explanations. It uses **Llama**, running on
your own computer through a tool called **Ollama**.

**Free. No account. No credit card. No API key.**

### Why local rather than a cloud AI

Say this in your demo, because it is the strongest point in Module 5:

> *"Patient records never leave the machine. Sending real clinical data to a
> hosted API is a genuine legal problem under GDPR and HIPAA. Running the model
> locally removes the question entirely — there is no outbound request, so there
> is nothing to leak or log."*

### Installing it

1. Go to **ollama.com/download** and install it like any other program.
2. Open a terminal and run:

```bash
ollama pull llama3.2
```

That downloads about 2 GB. It happens once.

3. Check the project can see it:

```bash
python scripts/05_generate_report.py --ollama-check
```

You want:

```
  host  : http://localhost:11434
  model : llama3.2
  status: READY
```

> **Note:** there is nothing to `pip install`. The project talks to Ollama using
> Python's built-in tooling, so Ollama is the only thing you install.

### Choosing a model size

| Command | Size | Notes |
|---|---|---|
| `ollama pull llama3.2:1b` | ~1.3 GB | Fastest; weaker writing |
| `ollama pull llama3.2` | ~2.0 GB | **Default** — fine on any modern laptop |
| `ollama pull llama3.1:8b` | ~4.7 GB | Best writing; wants ~8 GB RAM |

To switch, change `OLLAMA_MODEL` in your `.env` file.

A risk summary is a short, tightly-constrained writing task — the facts come
from Modules 2 and 4, so a small model is genuinely enough here.

---

## 12b. Generate an AI report

```bash
python scripts/05_generate_report.py
```

This produces a plain-English explanation of one patient's risk.

**If you skipped Ollama, this still works.** The same facts are rendered by a
Python template instead. That is not a downgrade — it demonstrates that the
*grounding* layer is doing the work, not the language model.

Useful variations:

```bash
python scripts/05_generate_report.py --show-prompt        # see what the AI is sent
python scripts/05_generate_report.py --show-facts         # see the grounded facts
python scripts/05_generate_report.py --highest-risk       # riskiest patient
python scripts/05_generate_report.py --template patient_letter
```

`--show-prompt` is the one to demonstrate. It shows the AI is given the facts and
forbidden from inventing others.

**Comparing Llama against the template** is a good demo move:

```bash
python scripts/05_generate_report.py --provider ollama
python scripts/05_generate_report.py --provider template
```

The Llama version reads better. The template version is never wrong. Both
contain the same facts — which is the point.

**Hosted APIs (Claude, GPT) are also supported** but are not the default,
because they cost money and send data off your machine. If you want one:
`pip install anthropic` and set `ANTHROPIC_API_KEY` in `.env`.

---

## 13. Run the dashboard

```bash
streamlit run streamlit_app/app.py
```

Your browser opens automatically at `http://localhost:8501`.

Six pages down the left side. Click through all of them.

**To stop it:** click the terminal window and press `Ctrl + C`.

> The terminal stays busy while the dashboard runs. That is normal — it is not
> frozen. Open a second terminal window if you need one.

---

## 14. Run the notebook

```bash
jupyter notebook notebooks/01_eda_hospital_analytics.ipynb
```

Then in the menu: **Kernel → Restart & Run All**. Wait for every cell to finish.

**Do this before any presentation.** A notebook with its outputs saved is your
parachute — if the dashboard fails during a demo, every chart and finding is
still visible there.

---

## 15. Doing it all again tomorrow

Once set up, you only need:

```bash
cd Documents/hospital-analytics
source .venv/bin/activate          # Windows: .venv\Scripts\activate
streamlit run streamlit_app/app.py
```

The data and model are already built. To rebuild everything from scratch:

```bash
python scripts/run_all.py
```

That runs steps 1–4 in order.

---

## 16. When something goes wrong

### `ModuleNotFoundError: No module named 'pandas'`

The virtual environment is not active. Look for `(.venv)` at the start of your
prompt. If it is missing, run the activate command from section 6.

### `ModuleNotFoundError: No module named 'hospital_analytics'`

You are in the wrong folder. Run `ls` — you should see `src` and `scripts`. If
not, `cd` into your project folder.

### `can't open file 'scripts/...'`

Same cause. Wrong folder.

### `Could not find 'diabetic_data.csv'`

Section 8. The file is probably still inside the unzipped subfolder.

### `Access denied for user 'root'@'localhost'`

Wrong MySQL password in `.env`. Test it by hand: `mysql -u root -p`

### `Can't connect to MySQL server`

MySQL is not running.
Windows: open Services, start MySQL80.
Mac: `brew services start mysql`.
Linux: `sudo systemctl start mysql`.

### `Table 'hospital_analytics.medications' doesn't exist`

Run `python scripts/02_setup_database.py`. To check the schema without needing
MySQL running: `python scripts/check_schema.py`.

### Dashboard says "Data source: CSV fallback"

MySQL is unreachable, so the project used CSV files instead. **Everything still
works.** Fix MySQL if your brief requires it; otherwise carry on.

### `status: NOT READY` from `--ollama-check`

Ollama is not running. Install it from **ollama.com/download**, then run
`ollama pull llama3.2`. On Mac and Windows it starts automatically after
install; on Linux you may need `ollama serve`.

If it says the model is missing rather than the server, run the `pull` command.

Reports still generate from the template in the meantime.

### "No trained model found" on the Risk Prediction page

Run `python scripts/04_train_model.py`, then refresh the page.

### `Port 8501 is already in use`

A dashboard is already running. Close the other terminal, or:
`streamlit run streamlit_app/app.py --server.port 8502`

### Charts look stale after reloading data

Data Quality page → **Clear cache**. Or restart the dashboard.

---

## What to do next

1. Read **`DEMO_GUIDE.md`** — a minute-by-minute presentation script and
   prepared answers to fourteen likely questions.
2. Read **`ML_AND_AI_EXPLAINED.md`** — how Modules 4 and 5 actually work.
3. Read **`SDD.md`** — the full design document for your report.

And cite the dataset. Attribution is a licence condition, not a courtesy:

> Strack B, et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission
> Rates.* BioMed Research International, 781670. DOI 10.24432/C5230J.
