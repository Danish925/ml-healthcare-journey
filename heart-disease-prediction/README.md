# ❤️ Heart Disease Prediction using Machine Learning

An end-to-end machine learning project on the UCI Cleveland Heart Disease dataset, built as part of my Healthcare AI portfolio. This project demonstrates the complete machine learning workflow—from data cleaning and exploratory data analysis to model training, hyperparameter tuning, and performance evaluation—to predict the presence of heart disease using clinical patient data.

---

## ✨ Highlights

- Complete end-to-end machine learning workflow
- Leakage-free preprocessing pipeline
- Exploratory Data Analysis (EDA)
- Comparison of four classification algorithms
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- Error analysis on misclassified samples
- Healthcare-focused evaluation using Recall and ROC-AUC

---

## 📋 Project Overview

Heart disease is one of the leading causes of death worldwide, and early prediction can help clinicians identify high-risk patients sooner.

This project follows a complete machine learning workflow on real patient data:

1. Data quality assessment and cleaning
2. Exploratory Data Analysis (EDA)
3. Handling disguised missing values
4. Leakage-safe preprocessing
5. Training multiple machine learning models
6. Hyperparameter tuning using GridSearchCV
7. Model evaluation and comparison
8. Feature importance analysis
9. Error analysis and interpretation

The primary objective is not only to build an accurate predictive model but also to develop a reproducible and leakage-free machine learning pipeline suitable for healthcare data.

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

- **Missing values:** None detected via `.isnull()` — but disguised missing values were identified.
- **Duplicates:** 1 duplicate row found and removed (303 → 302 rows).
- **Disguised missing values:** `ca` contained invalid values (`4`) and `thal` contained invalid values (`0`). According to the original UCI documentation, these represent missing values rather than valid categories. They were converted to `NaN` instead of removing the affected rows to preserve valuable clinical information.

---

## 🔍 Key EDA Insights

- **Class balance:** Target distribution is reasonably balanced, making the dataset suitable for classification without resampling.
- **Chest pain type (`cp`)** shows one of the strongest relationships with heart disease.
- **Maximum heart rate (`thalach`)** correlates positively with disease presence.
- **`ca`, `exang`, and `oldpeak`** show strong negative correlations with the target.
- **Male patients** dominate the dataset; therefore, gender-based conclusions should be interpreted carefully.
- **Unexpected finding:** Patients without heart disease had higher average age and `oldpeak` values than patients diagnosed with heart disease in this dataset version. This observation was validated statistically instead of relying on assumptions.
- **Outliers:** High values in `chol` and `trestbps` were retained because they likely represent genuine clinical observations rather than data entry errors.

---

## ⚙️ Preprocessing Pipeline

Every preprocessing step was carefully ordered to prevent data leakage.

1. Invalid values (`ca == 4`, `thal == 0`) converted to `NaN`.
2. Feature matrix (`X`) and target variable (`y`) separated.
3. Stratified train-test split (80/20) to preserve class distribution.
4. Missing values imputed using statistics calculated **only from the training data**.
5. StandardScaler fitted exclusively on training data and applied to both training and test datasets.
6. Separate scaled and unscaled datasets maintained because tree-based algorithms do not require feature scaling.
7. Verification performed to ensure correctly standardized features.

---

## 🤖 Machine Learning Models

Four supervised learning algorithms were trained and compared.

| Model | Scaling Required | Purpose |
|--------|------------------|----------|
| Logistic Regression | ✅ | Linear baseline classifier |
| K-Nearest Neighbors (KNN) | ✅ | Distance-based classifier |
| Support Vector Machine (SVM) | ✅ | Margin-based classifier |
| Random Forest | ❌ | Ensemble tree-based classifier |

The Random Forest model was further optimized using **GridSearchCV** with **5-fold cross-validation**.

---

## 📈 Model Performance

| Model | Accuracy | ROC-AUC |
|--------|---------:|---------:|
| Logistic Regression | 80.33% | 0.895 |
| KNN | 77.05% | 0.875 |
| SVM | 80.33% | **0.908** |
| Random Forest (Tuned) | **81.97%** | 0.899 |

### 🏆 Best Model

The tuned **Random Forest** model delivered the best overall balance between predictive performance and generalization.

**Best Hyperparameters**

- `n_estimators = 100`
- `max_depth = 5`
- `min_samples_split = 5`
- `min_samples_leaf = 2`

**Cross Validation Accuracy**

**85.09%**

**Final Test Performance**

- Accuracy: **81.97%**
- Precision: **77.50%**
- Recall: **93.94%**
- F1 Score: **84.93%**
- ROC-AUC: **0.899**

---

## 📌 Feature Importance & Error Analysis

The tuned Random Forest model was used to identify the most influential clinical features contributing to heart disease prediction.

An additional error analysis was performed by examining False Positives and False Negatives to better understand model limitations.

The high Recall score (**93.94%**) indicates that the model successfully identified the majority of patients with heart disease, which is particularly important in medical screening applications.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Selection:** GridSearchCV
- **Notebook Environment:** Jupyter Notebook

---

## 📁 Repository Structure

```text
heart-disease-prediction/
│
├── heart-disease-prediction.ipynb
├── README.md
└── requirements.txt
```

---

## 🔮 Future Improvements

- Evaluate additional ensemble algorithms such as XGBoost and LightGBM.
- Validate the model using external clinical datasets.
- Improve model interpretability using SHAP values.
- Deploy the trained model as a web application.
- Investigate probability calibration for clinical decision support.

---

## 🙏 Acknowledgements

- UCI Machine Learning Repository
- Scikit-learn Documentation
- Matplotlib & Seaborn Documentation

---

## 👤 Author

**Mohd Danish**

This project is part of my **Healthcare AI Portfolio**, where I apply machine learning techniques to real-world clinical datasets while emphasizing reproducible workflows, proper evaluation, and leakage-free model development.
