# ❤️ Heart Disease Prediction

Exploratory data analysis and (upcoming) machine learning models to predict the presence of heart disease from patient clinical data. Part of the [`ml-healthcare-journey`](.) portfolio — a series of healthcare-focused ML projects built toward research in medical imaging / healthcare AI.

## Status: EDA Complete → Preprocessing & Modeling Next

## Overview

Heart disease is one of the leading causes of death worldwide, and early detection from routine clinical measurements can help flag high-risk patients before more invasive testing is needed. This project uses the classic **UCI Cleveland Heart Disease dataset** to explore which patient features are most associated with heart disease, with the goal of eventually training and evaluating classification models on top of these insights.

## Dataset

- **Source:** [UCI Heart Disease dataset](https://www.kaggle.com/datasets/mragpavank/heart-diseaseuci) (Cleveland subset, via Kaggle)
- **Size:** 303 rows × 14 columns (302 rows after removing 1 duplicate)
- **Target:** `target` — presence (1) or absence (0) of heart disease
- **Class balance:** 165 positive / 138 negative — no severe imbalance

| Column | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dL) |
| `fbs` | Fasting blood sugar > 120 mg/dL (1 = true) |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia result (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| `target` | 1 = heart disease present, 0 = absent |

## Data Quality Findings

- **No missing values** in the raw data — but the dataset has *disguised* invalid values that a plain `isnull()` check won't catch:
  - `ca` should only take values 0–3, but a value of **4** appears (invalid).
  - `thal` should only take values 1–3, but a value of **0** appears (invalid).
  - Together these affect **6 rows**, flagged for handling during preprocessing (likely imputation or removal — not yet decided).
- **1 duplicate row** found and removed (303 → 302 rows).

## EDA Highlights

- **Target distribution** is close to balanced, which is favorable for classification metrics like accuracy.
- **Chest pain type (`cp`), max heart rate (`thalach`), exercise-induced angina (`exang`), `oldpeak`, and number of major vessels (`ca`)** show the strongest relationships with the target in the correlation analysis — these are expected to be the most predictive features.
- **Age** trends higher in patients with heart disease, though distributions overlap substantially between the two groups — age alone isn't a strong classifier.
- **Oldpeak (ST depression)** is noticeably higher and more variable in the heart disease group — clinically consistent with reduced blood flow to the heart under exercise stress.
- **Cholesterol, resting blood pressure, and max heart rate** all show mild right/left skew with a handful of outliers; given this is clinical data, outliers were treated as likely genuine (not data errors) rather than removed outright.
- Full visual EDA (distributions, boxplots, correlation heatmap, gender/chest-pain breakdowns) is in the notebook.

## Tech Stack

- Python, pandas, NumPy
- matplotlib, seaborn (visualization)
- Jupyter Notebook

## Project Structure

```
heart-disease-prediction/
├── heart-disease-prediction.ipynb   # Data loading, cleaning, and EDA
└── README.md
```

## Next Steps

- [ ] Resolve invalid `ca` (=4) and `thal` (=0) values (6 rows)
- [ ] Feature encoding / scaling as needed
- [ ] Train/test split (**done before** any imputation or leakage-prone preprocessing, to avoid the leakage issue caught in the Diabetes Prediction project)
- [ ] Train and compare baseline classification models
- [ ] Evaluate with accuracy, precision/recall, ROC-AUC (accuracy alone is misleading even on a near-balanced target)
- [ ] Document final results honestly, with any caveats

## Related Projects in `ml-healthcare-journey`

- Titanic Survival Analysis (81% accuracy)
- Diabetes Prediction (~77.27% accuracy, corrected after fixing a data leakage bug)
