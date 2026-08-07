"""
Clinical Query Rate Predictor - Model Training & Evaluation
Author: Puneetha (github.com/Puneetha-M)

Trains and compares Logistic Regression, Random Forest, and XGBoost
classifiers to predict high query rate risk at site/visit level.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def load_and_split(df: pd.DataFrame, target_col: str = 'high_query_risk',
                   test_size: float = 0.2, random_state: int = 42):
    """
    Split feature matrix and target into train/test sets.
    Stratified to preserve class balance given typical CDM imbalance
    (most sites are low-risk; high-risk sites are a minority).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


def train_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=500, class_weight='balanced',
                               random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_random_forest(X_train, y_train, n_estimators: int = 200):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    XGBoost with scale_pos_weight to handle class imbalance —
    typical in clinical trial data where most sites behave normally.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos  # compensates for imbalance

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    return model


def train_isolation_forest(X_train, contamination: float = 0.1):
    """
    Isolation Forest for anomaly detection — flags genuinely unusual
    sites regardless of labelled query rate thresholds.
    Useful as a secondary signal alongside the supervised models.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )
    model.fit(X_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str,
                   scaler=None) -> dict:
    """Compute AUC-ROC, precision, recall and print classification report."""
    X_eval = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(
        model, 'predict_proba') else None

    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"AUC-ROC: {auc:.4f}" if auc else "AUC not available")
    print(classification_report(y_test, y_pred))

    return {
        'model': model_name,
        'auc': auc,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score']
    }


def run_cv(model, X, y, cv: int = 5) -> float:
    """Stratified k-fold cross-validation — gives more reliable estimate
    than a single train/test split on small clinical datasets."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    print(f"CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean()
