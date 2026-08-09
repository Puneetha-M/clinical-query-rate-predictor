This is a work-in-progress portfolio repo — dissertation completing October 2026, results will be updated on completion

# Clinical Query Rate Predictor

**Predicting data query rates in clinical trials using machine learning** — identifying high-risk sites and visits before queries accumulate, enabling proactive data quality management.

This project is the analytical core of my MSc dissertation: *"Predictive Query Rate Modeling in Clinical Trials: A Machine Learning Framework for Data Quality Risk Scoring"*

---

## The Problem

In clinical data management, queries (data clarification requests) are raised when site-entered data is missing, inconsistent, or implausible. High query rates delay database lock, increase CDM costs, and signal data quality problems that could affect regulatory submissions.

Traditionally, query management is **reactive** — teams respond after queries appear. This project builds a model that **predicts which sites/visits are likely to generate high query volumes**, so CDM teams can intervene early.

---

## Approach

1. **Feature Engineering** from ClinicalTrials.gov public data + synthetic EDC metadata
   - Site-level features: country, experience, enrolment pace
   - Visit-level features: visit type, form count, mandatory fields
   - Study-level features: phase, therapeutic area, complexity score
   - CDISC SDTM-aligned domain structure

2. **Model Training & Comparison**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost (best performer)
   - Isolation Forest (anomaly detection for outlier sites)

3. **Explainability with SHAP**
   - Which features drive high query risk?
   - Site-level and visit-level SHAP waterfall plots
   - Clinically interpretable outputs (not black-box)

4. **ICH E6(R3) & 21 CFR Part 11 framing**
   - Risk-based monitoring alignment
   - Audit trail considerations for model outputs

---

## Tech Stack

- Python 3.10
- Scikit-Learn, XGBoost, SHAP
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Project Structure

```
clinical-query-rate-predictor/
│
├── data/
│   ├── raw/                        # ClinicalTrials.gov extracts + synthetic EDC data
│   ├── processed/                  # Feature-engineered datasets
│   └── README.md                   # Data sourcing & generation notes
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb     # All models comparison
│   ├── 04_shap_explainability.ipynb
│   └── 05_risk_scoring_output.ipynb
│
├── src/
│   ├── features.py                 # Feature engineering pipeline
│   ├── models.py                   # Model training & evaluation
│   └── explainer.py                # SHAP wrapper functions
│
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/Puneetha-M/clinical-query-rate-predictor
cd clinical-query-rate-predictor
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

---

## Clinical Relevance

This work connects directly to **Risk-Based Quality Management (RBQM)** under ICH E6(R3) — the evolving GCP guideline that requires sponsors to proactively identify and mitigate data quality risks. A model like this supports CDM teams, data review committees, and CRO oversight functions in making smarter, earlier decisions.

---

## Author

**Puneetha** — Medidata Rave Certified Study Builder | MSc Data Analytics @ BSBI Berlin  
[LinkedIn](https://www.linkedin.com/in/puneetham/) | [GitHub](https://github.com/Puneetha-M)
