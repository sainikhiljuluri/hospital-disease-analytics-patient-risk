"""
MODULE 4 - Machine learning: predicting 30-day readmission.

WHAT IS IN HERE
---------------
    features.py   which columns become model inputs, and which are BANNED
                  (leakage and fairness exclusions live here)
    train.py      trains and compares three models, keeps the best
    evaluate.py   recall, precision, ROC-AUC, PR-AUC, fairness, subgroups
    predict.py    scores a patient and explains why

HOW TO USE IT
-------------
    python scripts/04_train_model.py       # train once, saves to models/

    from hospital_analytics.ml import predict
    result = predict.explain_one(patient_row)

THE FOUR DECISIONS THAT DEFINE THIS MODULE
------------------------------------------
1. PREDICTION MOMENT: the day before discharge. Every feature must be knowable
   then. This is what makes leakage exclusion decidable rather than a guess.

2. GROUPED SPLIT: one row is one ENCOUNTER and people repeat, so train/test is
   split by `person_id`. A plain random split would put the same patient on both
   sides and inflate the score.

3. IMBALANCE: only ~11% of encounters end in readmission, so class weights are
   balanced and the model is judged on recall and ROC-AUC. Accuracy alone is
   meaningless when always saying "no" scores ~89%.

4. EXPLAINABILITY: every prediction ships with per-feature contributions. A bare
   probability is not actionable, and Module 5 needs those contributions as
   grounding facts.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not claim causation. Contributions are associations. It does not use
ethnicity as a feature. It does not report accuracy without the baseline beside
it. Each of those is a choice, and each is documented where it is made.
"""

from . import evaluate, features, predict, train

__all__ = ["evaluate", "features", "predict", "train"]
