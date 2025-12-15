# Stroke Risk Prediction (Healthcare ML)

## Problem Statement
Early identification of individuals at high risk of stroke is critical for preventive healthcare.
This project builds a recall-focused machine learning model to identify potential stroke patients
using demographic, lifestyle, and clinical features.

---

## Dataset
- Source: Healthcare Stroke Dataset
- Target variable: `stroke`
- Highly imbalanced (~5% positive cases)

---

## Modeling Approach
- Feature engineering for medical risk factors
- Clean preprocessing pipeline using `ColumnTransformer`
- RandomForestClassifier as final model
- RandomOverSampler applied **inside the pipeline** to handle imbalance
- Recall-focused evaluation due to high cost of false negatives

---

## Threshold Tuning
- Default threshold (0.5) was not assumed optimal
- Probability threshold tuned to minimize false negatives
- **Final decision threshold: 0.436**
- At this threshold:
  - Recall (stroke): ~97%
  - False negatives: 2 out of 62 cases

---

## Deployment
- Deployed using **Gradio directly from the notebook**
- No model serialization (`.pkl`) used to avoid versioning instability
- Risk is communicated as:
  - Low
  - Moderate
  - High
  - Extreme High

---

## Disclaimer
This model is intended for **screening and educational purposes only** and should not be used
as a substitute for professional medical diagnosis.

---

## Tech Stack
- Python
- Scikit-learn
- Imbalanced-learn
- Optuna
- Gradio
