# ❤️ Heart Disease Prediction using Machine Learning

An end-to-end machine learning project using the **UCI Cleveland Heart Disease dataset**, developed as part of my Healthcare AI portfolio.

This project applies a complete machine learning workflow to structured clinical data, including data quality assessment, exploratory data analysis, leakage-safe preprocessing, model comparison, cross-validation, hyperparameter tuning, feature importance, and error analysis.

The goal is not to build a clinical diagnostic system, but to demonstrate how machine learning can be applied to healthcare data using careful preprocessing, transparent evaluation, and responsible interpretation.

---

## ✨ Highlights

* End-to-end machine learning workflow
* Leakage-safe preprocessing
* Exploratory Data Analysis (EDA)
* Comparison of four classification algorithms
* 5-fold cross-validation
* ROC and Precision–Recall curve analysis
* Random Forest hyperparameter tuning using GridSearchCV
* Feature importance analysis
* False Positive and False Negative error analysis
* Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC

---

## 📋 Project Overview

Heart disease is a major global health concern. This project explores clinical patient data to identify patterns associated with the presence of heart disease and evaluate multiple machine learning approaches for binary classification.

The project follows the workflow:

1. Data quality assessment and cleaning
2. Exploratory Data Analysis (EDA)
3. Identification of invalid feature values
4. Stratified train-test split
5. Leakage-safe missing-value imputation
6. One-hot encoding of categorical features
7. Feature scaling where required
8. Training and comparison of multiple classifiers
9. Cross-validation
10. ROC and Precision–Recall analysis
11. Random Forest hyperparameter tuning
12. Feature importance analysis
13. Error analysis

The emphasis throughout the project is on building a reproducible workflow while preventing information from the test set from leaking into model training.

---

## 📊 Dataset

**Source:** UCI Heart Disease Dataset — Cleveland subset

**Original Size:** 303 rows × 14 columns
**After Duplicate Removal:** 302 rows

| Column     | Description                                        |
| ---------- | -------------------------------------------------- |
| `age`      | Patient age                                        |
| `sex`      | Biological sex                                     |
| `cp`       | Chest pain type                                    |
| `trestbps` | Resting blood pressure                             |
| `chol`     | Serum cholesterol                                  |
| `fbs`      | Fasting blood sugar indicator                      |
| `restecg`  | Resting ECG results                                |
| `thalach`  | Maximum heart rate achieved                        |
| `exang`    | Exercise-induced angina                            |
| `oldpeak`  | ST depression induced by exercise relative to rest |
| `slope`    | Slope of the peak exercise ST segment              |
| `ca`       | Number of major vessels observed by fluoroscopy    |
| `thal`     | Thalassemia-related categorical feature            |
| `target`   | Heart disease classification label                 |

### Target Variable

* `0` → No heart disease
* `1` → Heart disease present

---

## 🧹 Data Quality & Cleaning

Initial data-quality checks were performed for missing values, duplicate records, class distribution, and unusual feature values.

### Key Findings

* No standard `NaN` values were initially detected.
* One duplicate record was identified and removed.
* The target classes were reasonably balanced.
* Unusual category values were identified in the `ca` and `thal` features.

Rather than treating these unusual values as normal categories, they were converted to missing values and handled later during preprocessing.

Importantly, imputation was performed **after the train-test split**, ensuring that replacement values were learned only from the training data.

---

## 🔍 Exploratory Data Analysis

EDA was performed to understand feature distributions, relationships with the target, potential outliers, and patterns within the dataset.

### Key Observations

* The target classes are reasonably balanced.
* Chest pain categories show noticeable differences in heart disease occurrence.
* Maximum heart rate (`thalach`) shows variation between target groups.
* Features such as `oldpeak`, `ca`, and exercise-induced angina show relationships with the target.
* Male patients represent a larger proportion of the dataset, so gender-related patterns require careful interpretation.
* Age distributions overlap considerably between the two target classes.
* High values in cholesterol and resting blood pressure were retained rather than automatically removed because they may represent genuine patient measurements.

EDA observations were treated as associations within this dataset rather than evidence of clinical causation.

---

## ⚙️ Preprocessing Pipeline

The preprocessing workflow was designed to prevent data leakage.

1. Invalid or unusual `ca` and `thal` values were converted to missing values.
2. Features (`X`) and target (`y`) were separated.
3. A stratified 80/20 train-test split was performed.
4. Missing values were imputed using the mode calculated **only from the training data**.
5. Categorical variables were converted using one-hot encoding.
6. Test-set columns were aligned with the encoded training features.
7. `StandardScaler` was fitted only on the encoded training data and then applied to the test data.
8. Scaled encoded features were used for Logistic Regression, KNN, and SVM.
9. Encoded but unscaled features were used for Random Forest.

This ensures that the test set remains independent throughout preprocessing and model development.

---

## 🤖 Machine Learning Models

Four supervised classification algorithms were trained and evaluated.

| Model                        | Feature Representation | Role                           |
| ---------------------------- | ---------------------- | ------------------------------ |
| Logistic Regression          | Encoded + Scaled       | Linear baseline classifier     |
| K-Nearest Neighbors (KNN)    | Encoded + Scaled       | Distance-based classifier      |
| Support Vector Machine (SVM) | Encoded + Scaled       | Margin-based classifier        |
| Random Forest                | Encoded + Unscaled     | Ensemble tree-based classifier |

Multiple evaluation metrics were considered rather than selecting a model based on accuracy alone.

---

## 🔄 Cross-Validation

Five-fold cross-validation was performed on the training data to assess model stability beyond a single train-test split.

Mean validation accuracy and standard deviation were examined to understand how consistently each model performed across different subsets of the training data.

---

## 📈 Model Evaluation

Models were evaluated using:

* **Accuracy** — overall proportion of correct predictions
* **Precision** — proportion of predicted positive cases that were actually positive
* **Recall** — proportion of actual positive cases correctly identified
* **F1-score** — balance between Precision and Recall
* **ROC-AUC** — ability to distinguish between the two target classes across classification thresholds

ROC curves and Precision–Recall curves were also used to compare model behavior across different decision thresholds.

### Baseline Model Performance

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |   ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | --------: |
| Logistic Regression |     80.33% | **78.38%** |     87.88% |     82.86% |     0.895 |
| KNN                 |     77.05% |     75.68% |     84.85% |     80.00% |     0.875 |
| **SVM**             | **80.33%** |     76.92% | **90.91%** | **83.33%** | **0.908** |
| Random Forest       |     75.41% |     73.68% |     84.85% |     78.87% |     0.879 |

---

## 🏆 Model Selection

Among the baseline models, **Support Vector Machine (SVM)** achieved the strongest overall predictive performance.

SVM achieved:

* **Accuracy:** 80.33%
* **Precision:** 76.92%
* **Recall:** 90.91%
* **F1 Score:** 83.33%
* **ROC-AUC:** 0.908

It achieved the highest Recall, F1-score, and ROC-AUC among the evaluated baseline models.

Logistic Regression also performed competitively, achieving the same test accuracy and slightly higher Precision.

Therefore, **SVM is considered the strongest predictive model in this experiment**.

---

## 🔧 Random Forest Hyperparameter Tuning

Random Forest was additionally optimized using **GridSearchCV with 5-fold cross-validation**.

The search explored different values for:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

**ROC-AUC** was used as the optimization metric during hyperparameter tuning.

### Best Hyperparameters

```text
max_depth = 5
min_samples_leaf = 2
min_samples_split = 2
n_estimators = 200
```

### Best Cross-Validation ROC-AUC

**0.9015**

### Tuned Random Forest Test Performance

* **Accuracy:** 75.41%
* **Precision:** 73.68%
* **Recall:** 84.85%
* **F1 Score:** 78.87%
* **ROC-AUC:** 0.8972

The tuned Random Forest demonstrated strong discrimination, although SVM remained the strongest predictive model in the overall test-set comparison.

Random Forest was also useful for further model interpretation through feature importance and error analysis.

---

## 📌 Feature Importance

Feature importance from the tuned Random Forest was examined to understand which encoded features contributed most strongly to its predictions.

Higher feature importance indicates that a feature had greater influence on the Random Forest's prediction process.

These importance scores describe **model behavior** and should not be interpreted as evidence that a feature directly causes heart disease.

---

## 🔎 Error Analysis

Predictions from the tuned Random Forest were examined to identify individual misclassified test samples.

Errors were separated into:

* **False Positives (FP):** Cases predicted as heart disease when the actual label was negative.
* **False Negatives (FN):** Positive heart disease cases that the model failed to identify.

Examining individual errors provides additional insight into model behavior and limitations that cannot be captured by aggregate performance metrics alone.

---

## ⚠️ Limitations

Several limitations should be considered when interpreting the results:

* The dataset is relatively small, which may limit generalization.
* The available features capture only part of the complexity of cardiovascular health.
* Model performance was evaluated on this dataset and does not establish real-world clinical effectiveness.
* Feature importance reflects model behavior rather than causal relationships.
* The models were not externally validated on independent patient populations.

Therefore, this project should be considered an **educational machine learning analysis rather than a clinical diagnostic system**.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Model Selection:** GridSearchCV
* **Environment:** Jupyter Notebook

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

## 🙏 Acknowledgements

* UCI Machine Learning Repository
* Scikit-learn Documentation
* Matplotlib Documentation
* Seaborn Documentation

---

## 👤 Author

**Mohd Danish**

This project is part of my **Healthcare AI Portfolio**, where I apply machine learning techniques to healthcare datasets while focusing on reproducible workflows, careful evaluation, leakage prevention, and transparent interpretation.
