# Heart Disease Prediction and Prevention with Machine Learning

A supervised machine learning pipeline evaluating multiple classification algorithms for early detection and risk stratification of cardiovascular heart disease using clinical and demographic patient datasets.

## Overview

Cardiovascular disease remains one of the leading global causes of mortality. Early identification of high-risk patients enables targeted clinical interventions and preventive care regimens. This project benchmarks various machine learning classifiers across the Framingham Heart Study dataset and clinical cardiac telemetry to determine optimal recall and precision tradeoffs.

## Machine Learning Pipeline

1. **Data Preprocessing & Imputation**: Handles missing biometric observations, outlier detection, and feature standard scaling across systolic blood pressure, cholesterol, BMI, glucose, and lifestyle markers.
2. **Exploratory Data Analysis (EDA)**: Correlation heatmaps, feature distribution skewness, and class imbalance mitigation using SMOTE.
3. **Model Evaluation & Benchmarking**:
   - Logistic Regression
   - Support Vector Machines (SVM)
   - Random Forest Classifier
   - Decision Trees
   - Gradient Boosting (XGBoost)
4. **Performance Metrics**: Compares models based on ROC-AUC curves, precision-recall trade-offs, and confusion matrices to minimize false negatives in cardiac risk detection.

## Project Structure

```
MachineLearningProject/
├── data/
│   ├── framingham.csv          # Framingham Heart Study demographic and clinical dataset
│   └── heart.csv               # Clinical cardiac observations dataset
├── docs/
│   ├── FinalProject_Machine_Learning_Report.pdf            # Comprehensive technical report
│   ├── Monitoring_Trust_in_Human-Machine_Interactions_for.pdf # Human-AI trust analysis
│   ├── PPT2_Final.pptx                                    # Final project presentation deck
│   └── TEAM7_ML_PROJECT_PROPOSAL.pptx                     # Project proposal and milestones
├── notebooks/
│   ├── Project_Final_1.0.ipynb # Initial data exploration, cleaning, and baseline models
│   └── Project_Final_3.0.ipynb # Tuned model ensembles, SMOTE analysis, and ROC evaluations
├── src/
│   └── Group_6_Code_Prediction_Prevention_Heart_Disease.py # End-to-end Python script
├── .gitignore                  # Checkpoints, pycache, and OS ignore rules
├── LICENSE                     # MIT License
└── README.md                   # Case-study documentation and pipeline overview
```

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Siddhesh-Bhande/MachineLearningProject.git
   cd MachineLearningProject
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter imbalanced-learn
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook notebooks/Project_Final_3.0.ipynb
   ```

## Documentation

Full experimental methodologies, comparative tables, and confusion matrices are documented in [docs/FinalProject_Machine_Learning_Report.pdf](docs/FinalProject_Machine_Learning_Report.pdf).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
