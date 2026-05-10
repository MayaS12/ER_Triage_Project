# TriageNet — AI-Powered Emergency Room Triage

An XGBoost-based triage system that predicts patient acuity levels (1–5) from clinical vitals, symptoms, and comorbidities. Built on 425,000 ED encounters from the MIMIC-IV dataset.

**96% recall on high-risk patients (acuity 4–5). Only 3% of critical cases missed entirely.**

---

## Motivation

ER overcrowding affects up to 90% of large hospitals globally. Manual triage is subject to fatigue and bias, with mis-triage rates reaching 20–30%. Every minute of delay in critical cases increases mortality risk by 7–10%.

TriageNet is designed to assist, not replace, clinical judgment, flagging high-risk patients faster and more consistently.

---

## Model Overview

- **Algorithm**: XGBoost with custom class weighting and probability thresholding
- **Target**: ESI acuity score (1 = non-urgent → 5 = critical)
- **Data**: MIMIC-IV ED module (PhysioNet, credentialed access required)
- **Key design decision**: Asymmetric loss — missing a critical patient is worse than a false alarm. The model deliberately accepts lower precision to maximize recall on acuity 4–5 cases.

### Feature Engineering

Features are derived from six raw MIMIC tables merged on `stay_id`:

- **Vitals aggregation**: initial, mean, min, max, standard deviation, and trend for each vital sign
- **Derived clinical scores**: MAP, pulse pressure, shock index, MEWS components, abnormal vitals count
- **Chief complaint features**: binary indicators for chest pain, shortness of breath, trauma, infection, neurological symptoms
- **Diagnosis features**: total diagnoses per stay, flags for high-risk conditions (MI, stroke, sepsis, hemorrhage)
- **Demographics**: age at visit, gender, transport method

> **Bias note**: Race-related features were explicitly removed from the feature set to prevent the model from encoding historical disparities in triage decisions.

### Class Imbalance

The dataset is heavily skewed (acuity 3 = 53.8%, acuity 5 = 0.26%). Two strategies were tested:

- SMOTE oversampling → increased training time and introduced synthetic noise
- **Custom class weighting** (selected): acuity 5 weighted 15×, acuity 4 weighted 8× base. This emphasizes critical cases in the loss function without generating fake data.

### Thresholding

Standard argmax predictions perform poorly on rare critical cases. Custom thresholds:
- If P(acuity 5) > 0.05 → predict 5
- If P(acuity 4) > 0.30 → predict 4

Without thresholds: critical recall = 16%. With thresholds: critical recall = 96%.

---

## Results

| Metric | Value |
|---|---|
| High-risk recall (acuity 4–5) | 96% |
| Critical cases missed (level 3 or below) | 3% |
| Nurse vs. model exact match | 42% |
| Nurse vs. model within 1 level | 87% (42% + 45%) |
| AUC (critical vs. non-critical) | 0.75 |

Most errors occur between adjacent acuity levels (2↔3, 3↔4), which are clinically less consequential. Level 5 patients are predominantly flagged as level 4 — still routed to immediate physician attention.

**Top predictive features**: MEWS Total, Shock Index, O2 Saturation, Heart Rate, Systolic BP

---

## Repository Structure

```
├── step1.py                      # Data loading, merging, preprocessing
├── modelling.py                  # Model training, class weighting, thresholding
├── graphs.py                     # Evaluation visualizations
├── create_presentation_graphs.py # Presentation-ready charts
├── triage_dashboard_final.py     # Streamlit dashboard (main)
├── triage_dashboard.py           # Earlier dashboard version
├── test_prediction.py            # Single-patient prediction demo
├── requirements.txt
├── feature_columns.txt           # Final feature list
├── feature_columns_smote.txt     # Feature list for SMOTE variant
├── feature_importance.csv        # SHAP/XGBoost feature importances
├── feature_importance.png        # Feature importance bar chart
├── data_summary.json             # Aggregate dataset statistics (no PHI)
├── model_metrics.json            # Full evaluation metrics
├── model_metrics_smote.json      # Metrics for SMOTE variant
├── triage_model.pkl              # Trained XGBoost model
├── scaler.pkl / scaler_smote.pkl
├── imputer.pkl / imputer_smote.pkl
└── presentation_*.html           # Interactive evaluation visualizations
```

---

## Data Access

This project uses **MIMIC-IV ED** (Medical Information Mart for Intensive Care), a restricted-access dataset. Raw CSV files are not included in this repository.

To reproduce results:
1. Complete the CITI human subjects research training at [PhysioNet](https://physionet.org)
2. Request credentialed access to MIMIC-IV at physionet.org/content/mimic-iv-ed
3. Place the downloaded CSVs in the project root
4. Run `step1.py` → `modelling.py`

---

## Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run triage_dashboard_final.py
```

The dashboard accepts patient vitals as input and returns a predicted acuity level with probability breakdown across all five levels.

---

## Limitations

- Trained on data from a single U.S. academic medical center (Beth Israel Deaconess, Boston). Generalizability to other hospital systems and patient populations is not established.
- The model is intended as a decision-support tool. Clinical judgment must remain primary.
- 14% precision on critical predictions means roughly 6 in 7 patients flagged as critical will not be, acceptable for safety but would require careful workflow integration to avoid alert fatigue.

---

## Built at Phillips Exeter Academy
Senior capstone project, 2024–2025.  
Dataset access via PhysioNet credentialing. Human subjects training completed prior to data use.
