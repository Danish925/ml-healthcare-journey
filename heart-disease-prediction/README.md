# ❤️ Heart Disease Prediction using Machine Learning

Exploratory Data Analysis and preprocessing pipeline on the UCI Cleveland Heart Disease dataset, built as part of a Healthcare AI portfolio. The goal is to understand which clinical and demographic features are associated with heart disease and prepare a leakage-free pipeline for model training.

---

## 📋 Project Overview

Heart disease is one of the leading causes of death worldwide, and early prediction can help clinicians identify high-risk patients sooner. This project works through the full early-stage ML workflow on real patient data:

1. Data quality assessment and cleaning
2. Exploratory Data Analysis (EDA) with visual and statistical validation
3. Handling disguised missing values
4. Leakage-safe preprocessing (stratified split → imputation → scaling)

Model training and evaluation is the next phase of this project (see [Next Steps](#-next-steps)).

---

## 📊 Dataset

**Source:** UCI Heart Disease Dataset (Cleveland subset)
**Size:** 303 rows × 14 columns (302 rows after duplicate removal)

| Column | Description | Values |
|---|---|---|
| `age` | Patient's age | Numeric (years) |
| `sex` | Biological sex | 1 = Male, 0 = Female |
| `cp` | Chest pain type | 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic |
| `trestbps` | Resting blood pressure at admission | Numeric (mm Hg) |
| `chol` | Serum cholesterol | Numeric (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 = True, 0 = False |
| `restecg` | Resting ECG results | 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy |
| `thalach` | Max heart rate achieved | Numeric (bpm) |
| `exang` | Exercise-induced angina | 1 = Yes, 0 = No |
| `oldpeak` | ST depression induced by exercise vs. rest | Numeric |
| `slope` | Slope of peak exercise ST segment | 0 = Upsloping, 1 = Flat, 2 = Downsloping |
| `ca` | Number of major vessels colored by fluoroscopy | 0–3 valid; **4 is a disguised missing value** |
| `thal` | Thalassemia test result | 1 = Normal, 2 = Fixed defect, 3 = Reversible defect; **0 is a disguised missing value** |
| `target` | Heart disease diagnosis (label) | 0 = No disease, 1 = Disease present |

---

## 🧹 Data Cleaning

- **Missing values:** None detected via `.isnull()` — but see disguised missing values below.
- **Duplicates:** 1 duplicate row found and removed (303 → 302 rows).
- **Disguised missing values:** `ca` contained 4 rows with an invalid code of `4` (valid range is 0–3), and `thal` contained 2 rows with an invalid code of `0` (valid categories are 1–3). Per the original UCI documentation, these are not real categories — they're missing data encoded as numbers. Since the dataset is small (302 rows), these rows were preserved and the invalid codes converted to `NaN` rather than dropped.

---

## 🔍 Key EDA Insights

- **Class balance:** Target distribution is 165 (disease) vs. 138 (no disease) — reasonably balanced, which is favorable for classification without needing resampling.
- **Chest pain type (`cp`)** shows one of the strongest relationships with the target, making it a likely high-importance feature.
- **Maximum heart rate (`thalach`)** correlates positively with disease presence; **`ca`, `exang`, and `oldpeak`** correlate negatively — these stood out clearly in the correlation heatmap and the feature-correlation bar chart.
- **Gender:** Male patients dominate the dataset, so gender-based patterns should be read with that imbalance in mind rather than treated as a standalone predictor.
- **Counter-intuitive finding:** Patients *without* heart disease had a higher average age (56.6 vs. 52.6) and higher average `oldpeak` (1.59 vs. 0.59) than patients *with* heart disease — the opposite of the "textbook" expectation. This was verified against actual group means and correlation values rather than assumed, and is a known quirk of how `target` is encoded in this dataset version.
- **Outliers:** `chol` and `trestbps` have a handful of high-value outliers; since this is clinical data, these were treated as potentially genuine medical extremes rather than errors, and left untouched pending further modeling decisions.

---

## ⚙️ Preprocessing Pipeline

Every step below was ordered specifically to avoid data leakage:

1. **Invalid value handling** — `ca == 4` and `thal == 0` replaced with `NaN`.
2. **Feature/target split** — `X` (13 features) and `y` (`target`) separated.
3. **Stratified train-test split** — 80/20 split (241 train / 61 test rows), stratified on `target` to preserve class balance in both sets.
4. **Missing value imputation** — mode computed **only on the training set**, then applied to both train and test sets, simulating how the model would see genuinely unseen data in production.
5. **Feature scaling** — `StandardScaler` fit only on training data, then applied to both sets. Scaled copies are kept separate from the raw versions so distance-based/gradient-based models (Logistic Regression, KNN, SVM) can use them without affecting tree-based models that don't need scaling.
6. **Verification** — confirmed scaled training features have mean ≈ 0 and std ≈ 1.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn (`train_test_split`, `StandardScaler`)

---

## 📁 Repository Structure

```
heart-disease-prediction/
│
├── heart-disease-prediction.ipynb   # Full EDA + preprocessing notebook
└── README.md                        # Project documentation
```

---

## 🚀 Next Steps

- Train and compare baseline models (Logistic Regression, KNN, Random Forest, SVM) on the scaled/unscaled splits as appropriate.
- Evaluate using accuracy, precision, recall, F1, and ROC-AUC — not accuracy alone, given the clinical context where false negatives carry real cost.
- Perform feature importance analysis to cross-check against the EDA correlation findings.
- Consider hyperparameter tuning and cross-validation before finalizing a model.

---

## 👤 Author

**Danish**
Part of an ongoing Healthcare AI portfolio — see [`ml-healthcare-journey`](https://github.com/) for related projects (Titanic Survival Analysis, Diabetes Prediction).
