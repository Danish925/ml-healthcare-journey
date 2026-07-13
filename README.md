# ml-healthcare-journey

Documenting my path toward Healthcare AI research — combining a 
self-taught machine learning foundation with a genuine medical 
domain background (NEET preparation), applied to real clinical 
datasets with an emphasis on methodological honesty.

## Why Healthcare AI

My interest in this space comes from a background in serious NEET 
(medical entrance) preparation — which gives me a working 
understanding of clinical data, physiology, and diagnostic 
reasoning that most CS-background ML practitioners don't have. 
Every project here tries to bring that domain knowledge into the 
preprocessing and interpretation, not just the modeling.

## Projects

| # | Project | Status | Key Technique | Result |
|---|---------|--------|----------------|--------|
| 1 | [Diabetes Prediction](./diabetes-prediction) | ✅ Complete | Data leakage detection & fix, Random Forest | 77.27% accuracy (leakage-free) |
| 2 | Heart Disease Prediction | 🔄 In Progress | Clinical feature comparison, model benchmarking | — |
| 3 | COVID-19 Clinical Outcomes | 📋 Planned | XGBoost, SMOTE for class imbalance | — |
| 4 | Skin Lesion Detection | 📋 Planned | CNN, Transfer Learning (EfficientNet) | — |
| 5 | Brain Tumor MRI Detection | 📋 Planned | Grad-CAM, Explainable AI | — |
| 6 | Chest X-Ray Detection (Flagship) | 📋 Planned | Full research pipeline, clinical evaluation | — |
| 7 | Federated Learning Simulation | 📋 Planned | Privacy-preserving AI, FedAvg | — |

## A Recurring Principle: Honest Evaluation

The Diabetes Prediction project (#1) includes a documented case of 
catching and fixing a data-leakage bug in my own pipeline — the 
initial result (88.31% accuracy) turned out to be inflated by a 
target-leakage issue in missing-value imputation. The corrected, 
methodologically sound result is 77.27%. I'd rather report a lower, 
defensible number than an inflated one, and I'm carrying that habit 
into every project in this repo.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · PyTorch (from Project 3 
onward) · matplotlib · seaborn

## About Me

Mohammad Danish — BCA (Online), Amity University, graduating 2027. 
Building this portfolio toward graduate research in Healthcare AI, 
targeting the MEXT Scholarship (Japan).

- GitHub: [@Danish925](https://github.com/Danish925)
- LinkedIn: [mohd-danish-a89015313](https://www.linkedin.com/in/mohd-danish-a89015313)
