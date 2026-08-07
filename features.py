"""
Clinical Query Rate Predictor - Feature Engineering
Author: Puneetha (github.com/Puneetha-M)

Builds predictive features from clinical trial metadata,
aligned with CDISC SDTM domain structure and RBQM principles
(ICH E6(R3)).
"""

import pandas as pd
import numpy as np


# ── Site-level features ────────────────────────────────────────────────────────

def compute_site_experience_score(df: pd.DataFrame) -> pd.Series:
    """
    Proxy for site experience: number of prior studies at the site.
    Inexperienced sites tend to generate more queries.
    """
    return df.groupby('site_id')['study_id'].transform('nunique')


def compute_enrolment_velocity(df: pd.DataFrame) -> pd.Series:
    """
    Subjects enrolled per week. Very fast or very slow enrolment
    both correlate with higher query rates.
    """
    df = df.copy()
    df['enrolment_date'] = pd.to_datetime(df['enrolment_date'])
    df['study_start'] = pd.to_datetime(df['study_start_date'])
    df['weeks_active'] = (
        (df['enrolment_date'] - df['study_start'])
        .dt.days / 7
    ).clip(lower=1)
    return df['subject_count'] / df['weeks_active']


# ── Visit-level features ───────────────────────────────────────────────────────

def compute_visit_complexity(df: pd.DataFrame) -> pd.Series:
    """
    Complexity score = number of forms × mandatory field ratio.
    More complex visits → more opportunities for errors → more queries.
    """
    return df['form_count'] * df['mandatory_field_ratio']


def flag_unscheduled_visit(df: pd.DataFrame) -> pd.Series:
    """
    Unscheduled visits are associated with protocol deviations
    and tend to have higher query rates.
    """
    return (df['visit_type'] == 'UNSCHEDULED').astype(int)


# ── Study-level features ───────────────────────────────────────────────────────

def encode_therapeutic_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode therapeutic area.
    Oncology studies tend to have higher query rates due to
    complex endpoints and adverse event reporting requirements.
    """
    return pd.get_dummies(df, columns=['therapeutic_area'], prefix='ta')


def compute_study_complexity_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite study complexity: phase weight + endpoint count + arm count.
    Phase I = 1, Phase II = 2, Phase III = 3.
    """
    phase_map = {'Phase I': 1, 'Phase II': 2, 'Phase III': 3, 'Phase IV': 2}
    phase_score = df['study_phase'].map(phase_map).fillna(2)
    return phase_score + df['primary_endpoint_count'] + df['arm_count']


# ── Target variable ────────────────────────────────────────────────────────────

def create_query_risk_label(df: pd.DataFrame,
                            query_rate_col: str = 'query_rate_per_subject',
                            threshold: float = None) -> pd.Series:
    """
    Binarise query rate into high-risk (1) vs low-risk (0).
    Threshold defaults to the 75th percentile of observed query rates —
    flags the top quartile of sites as high risk.
    This aligns with RBQM tier definitions commonly used in CDM.
    """
    if threshold is None:
        threshold = df[query_rate_col].quantile(0.75)
        print(f"Query risk threshold (75th pct): {threshold:.2f} queries/subject")
    return (df[query_rate_col] >= threshold).astype(int)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.
    Inputs: raw merged clinical trial metadata DataFrame
    Output: model-ready feature matrix with target column
    """
    df = df.copy()

    # Site features
    df['site_experience'] = compute_site_experience_score(df)
    df['enrolment_velocity'] = compute_enrolment_velocity(df)

    # Visit features
    df['visit_complexity'] = compute_visit_complexity(df)
    df['is_unscheduled'] = flag_unscheduled_visit(df)

    # Study features
    df['study_complexity'] = compute_study_complexity_score(df)
    df = encode_therapeutic_area(df)

    # Target
    df['high_query_risk'] = create_query_risk_label(df)

    # Drop raw columns used only for feature construction
    drop_cols = [
        'query_rate_per_subject', 'enrolment_date', 'study_start_date',
        'visit_type', 'study_phase', 'therapeutic_area',
        'site_id', 'study_id'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df
