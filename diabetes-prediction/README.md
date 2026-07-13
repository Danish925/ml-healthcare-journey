# Diabetes Prediction — Healthcare AI

Predicting diabetes onset from clinical measurements using the 
Pima Indians Diabetes dataset, with a focus on medically-informed 
preprocessing and honest evaluation.

## Overview

- **Dataset:** Pima Indians Diabetes Database (UCI/Kaggle), 768 patients, 8 clinical features
- **Best Model:** Random Forest
- **Accuracy:** 77.27%
- **Sensitivity (Recall):** 0.61 — catches 61% of diabetic patients
- **Specificity:** 0.86
- **Precision:** 0.70 | **F1-Score:** 0.65

## Medical Domain Approach

Several features (Glucose, BloodPressure, SkinThickness, Insulin, 
BMI) contain zeros that are clinically impossible for a living 
patient — these were identified using domain knowledge and treated 
as missing values rather than real measurements, before any 
modeling began.

## A Data Leakage Lesson

An earlier version of this pipeline imputed missing values using 
the median grouped by the target label (`Outcome`), calculated 
before the train/test split. Since Insulin was missing in ~49% of 
rows, this leaked the target into the features, inflating accuracy 
to 88.31% and making Insulin appear artificially dominant in 
feature importance.

This was identified and fixed: the pipeline now splits first, 
then imputes using only training-set statistics with no reference 
to the target. The corrected, honest accuracy is 77.27% — lower, 
but methodologically sound. Full before/after reasoning is 
documented inside the notebook.

## Top Predictive Features

1. Glucose (0.273)
2. BMI (0.164)
3. DiabetesPedigreeFunction (0.125)

Glucose emerging as the top feature (rather than Insulin, as in 
the leaky version) aligns with its role as the direct clinical 
diagnostic marker for diabetes.

## Honest Limitations

- 768 patients is small for clinical deployment — real medical AI 
  needs far larger, more diverse cohorts
- 39% of diabetic patients are missed by the current model — not 
  clinically deployable as-is
- Class-imbalance techniques (class-weighting, XGBoost weighting) 
  were tested and did not meaningfully improve results

## How to Run

This notebook was built on Kaggle with the Pima Indians Diabetes 
Database attached as input. To run elsewhere, download the dataset 
from [Kaggle/UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) 
and update the `pd.read_csv()` path accordingly.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

---
**Author:** Mohammad Danish | BCA, Amity University | Healthcare AI Researcher in Progress
