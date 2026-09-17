# Clinical Query Rate Predictor: A Machine Learning Framework for Data Quality Risk Scoring

Predicting data query rates and assessing data quality risk in clinical trials using a hybrid machine learning pipeline — identifying high-risk protocols, sites, and patient groups before queries accumulate, enabling proactive Risk-Based Quality Management (RBQM).

This project is the complete analytical core of my MSc Data Analytics dissertation at the Berlin School of Business and Innovation (BSBI): *"Predictive Query Rate Modelling in Clinical Trials: A Machine Learning Framework for Data Quality Risk Scoring"* (Completed October 2026).

---

## Executive Summary & Key Results

In clinical data management, queries (data clarification requests) are raised when site-entered data is missing, inconsistent, or implausible. High query rates delay database lock, increase operational costs, and signal data quality problems that jeopardize regulatory submissions. Traditionally, query management is entirely reactive—teams respond after queries appear. 

This research establishes a validated proof-of-concept demonstrating that clinical trial data quality risk can be predicted prospectively from baseline protocol metadata prior to study initiation:
* **Dataset:** 50,036 cleaned clinical trial records derived from ClinicalTrials.gov with 74 CDISC SDTM-aligned engineered features and a multi-tier proxy target variable (`LOW` / `MEDIUM` / `HIGH` risk).
* **Supervised Machine Learning:** Evaluated Logistic Regression, Random Forest, and XGBoost. **XGBoost achieved the best performance** on held-out test data ($n = 10,008$), achieving an **$F_1$ Macro of 0.5630** and an **ROC-AUC of 0.7525** (with HIGH-risk studies discriminated at an AUC of 0.840). Stratified 5-Fold Cross-Validation confirmed model generalisation with a stable CV-to-test gap of 0.0049.
* **Unsupervised Anomaly Detection:** An Isolation Forest model identified structural protocol outliers, successfully concentrating **43.65% of all anomaly flags in the HIGH-risk class** (a 1.346x concentration ratio above baseline).
* **Explainability (TreeSHAP):** Deployed SHAP TreeExplainer to calculate exact global and local feature attributions, proving that trial scale features (**Number of Sites**, **Enrolment**, and **CRF Page Count**) are the dominant predictors of query risk.
* **Primary Expert Validation:** Triangulated computational findings against a structured primary survey of **$n = 20$ experienced clinical data management professionals**, highlighting key convergences (CRF complexity/edit check density) and divergences (expert focus on therapeutic area complexity vs. model focus on operational scale).
* **Interactive Stakeholder Dashboards:** Developed publication-ready Tableau dashboards communicating model risk scores, risk heatmaps, and SHAP feature importances for non-technical CDM stakeholders.

---

## Technical Approach & Architecture

1. **Secondary Data Processing & SDTM Feature Engineering:** Extracted raw records from ClinicalTrials.gov, cleaned via Jupyter Notebooks, and engineered 74 structured features across four CDISC SDTM domain structures (Demographics/Study, Adverse Events/Safety, Vital Signs/Procedures, and Laboratory metrics).
2. **Proxy Target Variable Construction:** Classified studies into three operational risk tiers (`LOW`, `MEDIUM`, `HIGH`) using expert-derived rule layers combining CRF page count, protocol amendments, screen failure rate, enrolment size, and therapeutic area.
3. **Supervised Modeling Pipeline:** Applied ColumnTransformer preprocessing (StandardScaler + OneHotEncoder), 80/20 stratified splitting, and Stratified 5-Fold Cross-Validation across Logistic Regression, Random Forest, and XGBoost.
4. **Explainability & Compliance:** Mapped SHAP feature attributions directly to satisfy the transparency requirements of **ICH E6(R3)** and the auditability standards of **FDA 21 CFR Part 11**.

---

## Tech Stack

* **Language:** Python 3.10
* **Machine Learning & Analytics:** Scikit-Learn, XGBoost, SHAP (TreeExplainer)
* **Data Manipulation & Processing:** Pandas, NumPy
* **Visualisation:** Matplotlib, Seaborn, Tableau (Dashboards)
* **Environment:** Jupyter Notebook / Google Colab

---

## Project Structure

```text
clinical-query-rate-predictor/
│
├── data/
│   ├── raw/                        # ClinicalTrials.gov API extracts & raw survey exports
│   ├── processed/                  # Feature-engineered 74-feature SDTM dataset (50,036 records)
│   └── README.md                   # Data sourcing & dictionary notes
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis & target variable distributions
│   ├── 02_feature_engineering.ipynb# SDTM-aligned feature extraction & proxy target creation
│   ├── 03_model_training.ipynb     # Baseline (LR), Random Forest, and XGBoost training & CV
│   ├── 04_shap_explainability.ipynb# TreeSHAP global/local attributions & beeswarm plots
│   ├── 05_risk_scoring_output.ipynb# Isolation Forest anomaly detection & final output generation
│   └── 06_survey_analysis.ipynb    # Primary survey (n=20) descriptive stats, Likert, & Cronbach's Alpha
│
├── dashboards/
│   ├── Thesis_Dashboard_1.twbx     # Clinical Trial Data Quality Risk Scoring Dashboard
│   └── Thesis_Dashboard_2.twbx     # Model Validity and Fairness Check Dashboard
│
├── requirements.txt
└── README.md
