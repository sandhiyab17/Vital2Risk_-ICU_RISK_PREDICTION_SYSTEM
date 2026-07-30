# Vitals2Risk — ICU Risk Prediction System

Vitals2Risk is a machine learning pipeline that predicts near-term patient instability (within a 2-hour window) by combining continuous vital sign trends with static clinical features. Built during an AI/ML internship at the Tamil Nadu e-Governance Agency (TNeGA).

## Overview

Traditional threshold-based ICU monitoring generates high false-alarm rates and misses subtle deterioration patterns. Vitals2Risk uses a Random Forest classifier trained on rolling-window vital sign statistics plus patient metadata to flag rising instability risk before it becomes critical, giving clinicians an earlier warning window.

- **Dataset:** [VitalDB](https://vitaldb.net/) — 628 patients (Early gastric cancer, Colon cancer sigmoid, Rectal cancer), ~3.79M second-by-second samples after cleaning
- **Model:** Random Forest Classifier (`n_estimators=200`, `max_depth=10`, `class_weight='balanced'`)
- **Result:** ROC-AUC of 0.97; F1-score of 0.52 for the unstable class after threshold optimization

## Features

- Rolling-window statistical features (mean, std, min, max) over 5-minute intervals for HR, SpO₂, RESP, NBP (sys/dia), and TEMP
- Rule-based instability labeling using a 2-hour forward-shifted clinical threshold (HR > 120 bpm, SpO₂ < 90%, RESP > 30/min, NBP sys < 90 mmHg, TEMP > 38.5°C)
- Static clinical feature integration: age, sex, BMI, ASA score, diagnosis (one-hot encoded)
- Threshold optimization via precision-recall curve to maximize F1-score on the rare "unstable" class
- Risk timeline, feature importance, and confusion matrix visualizations

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Random Forest, SVR, Decision Tree)
- XGBoost
- Matplotlib, Seaborn
- [vitaldb](https://pypi.org/project/vitaldb/) (data acquisition from VitalDB)

## Project Structure

```
vitals2risk/
├── data_acquisition.ipynb    # Pulls case-level vitals from VitalDB by diagnosis
├── preprocessing.ipynb       # Cleans, merges, and imputes vitals + clinical data
├── feature_engineering.ipynb # Rolling-window feature generation + labeling
├── model_training.ipynb      # Random Forest training and threshold tuning
├── evaluation.ipynb          # ROC-AUC, confusion matrix, feature importance
├── requirements.txt
└── README.md
```

## Results

| Threshold | Precision | Recall | F1-score |
|---|---|---|---|
| Default (0.5) | 13% | 85% | 22% |
| Optimized (~0.97) | 51% | 54% | 52% |

- **ROC-AUC:** 0.97
- **Overall accuracy:** ~99.55%
- Confusion Matrix (optimal threshold): TP 1,849 · FP 1,778 · FN 1,603 · TN 752,807

## Getting Started

```bash
git clone https://github.com/sandhiyab17/vitals2risk.git
cd vitals2risk
pip install -r requirements.txt
```

### requirements.txt
```
vitaldb
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## Limitations & Future Work

- Rule-based labels may introduce noise since not all threshold breaches reflect true deterioration
- Single-institution dataset limits generalizability without external validation
- Static model — future work could use LSTM/Transformer architectures to capture temporal dependencies
- Prospective clinical validation needed before real-world deployment

## Authors

- Sandhiya B



## References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
2. Lee, H. C., Jung, C. W. (2018). VitalDB: A high-fidelity multi-parameter vital signs database. *Scientific Data*, 5, 180010.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
