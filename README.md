# ml-healthcare-journey

Documenting my journey toward Healthcare AI research — combining a
self-taught machine learning foundation with prior intensive study of
biology and human physiology during NEET preparation. This repository
applies machine learning to healthcare datasets with an emphasis on
methodological honesty, clinical interpretation, and reliable evaluation.

## Why Healthcare AI

My interest in Healthcare AI grew from intensive NEET preparation,
where I developed a strong foundation in biology and human physiology.
Although I ultimately pursued Computer Science, my earlier exposure to
medical science shaped my interest in applying machine learning to
healthcare problems.

Through each project, I aim to go beyond model accuracy by understanding
the clinical meaning of features, identifying limitations in the data and
methodology, and evaluating model results in the context of healthcare.

## Projects

| # | Project | Status | Key Technique | Result |
|---|---------|--------|---------------|--------|
| 1 | [Diabetes Prediction](./diabetes-prediction) | ✅ Complete | Data leakage detection & fix, Random Forest | 77.27% accuracy (leakage-free) |
| 2 | Heart Disease Prediction | 🔄 In Progress | Clinical feature comparison, model benchmarking | — |
| 3 | COVID-19 Clinical Outcomes | 📋 Planned | XGBoost, SMOTE for class imbalance | — |
| 4 | Skin Lesion Detection | 📋 Planned | CNN, Transfer Learning (EfficientNet) | — |
| 5 | Brain Tumor MRI Detection | 📋 Planned | Grad-CAM, Explainable AI | — |
| 6 | Chest X-Ray Detection (Flagship) | 📋 Planned | Full research pipeline, clinical evaluation | — |
| 7 | Federated Learning Simulation | 📋 Planned | Privacy-preserving AI, FedAvg | — |

## A Recurring Principle: Honest Evaluation

The Diabetes Prediction project (#1) includes a documented case of
identifying and correcting data leakage in my own machine learning
pipeline.

The initial model achieved 88.31% accuracy, but further investigation
revealed that the result was inflated by target leakage during
missing-value imputation. After redesigning the preprocessing workflow
to maintain strict separation between training and test data, the
corrected model achieved 77.27% accuracy.

I would rather report a lower, defensible result than an inflated one.
That principle of methodological honesty is something I intend to carry
into every project in this repository.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · PyTorch · matplotlib · seaborn

## About Me

I'm Mohammad Danish, an online BCA student at Amity University,
graduating in 2027. I am building this portfolio as part of my preparation
for graduate research in Healthcare AI, with the goal of pursuing research
in Japan through the MEXT Scholarship.

- GitHub: [@Danish925](https://github.com/Danish925)
- LinkedIn: [mohd-danish-a89015313](https://www.linkedin.com/in/mohd-danish-a89015313)
