"""Evaluation metrics for the Brain CT Triage Challenge.

The competition documentation states QWK is the primary metric, but the
actual platform displays f1_macro as "Score" (confirmed empirically: our
internal f1_macro matched the platform's reported value almost exactly).
This module computes all three (QWK, accuracy, f1_macro) so we can track
what we're actually being scored on, not just the documented metric.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.triage import triage_from_intermediates

TARGET_COLUMNS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "fracture_prob", "MLS_mm"]


def _to_triage_classes(df: pd.DataFrame) -> list:
    return [
        triage_from_intermediates(row.to_dict())
        for _, row in df[TARGET_COLUMNS].iterrows()
    ]


def compute_qwk(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> float:
    """Compute QWK between predicted and true intermediates (via the official triage rule)."""
    assert len(pred_df) == len(true_df), (
        f"pred_df has {len(pred_df)} rows but true_df has {len(true_df)} rows."
    )
    pred_triage = _to_triage_classes(pred_df)
    true_triage = _to_triage_classes(true_df)
    return cohen_kappa_score(true_triage, pred_triage, weights="quadratic")


def compute_metrics(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> dict:
    """Compute QWK, accuracy, and f1_macro together, from the same triage-class
    derivation, so all three metrics are always computed consistently.

    Returns:
        Dict with keys "qwk", "accuracy", "f1_macro".
    """
    assert len(pred_df) == len(true_df), (
        f"pred_df has {len(pred_df)} rows but true_df has {len(true_df)} rows."
    )
    pred_triage = _to_triage_classes(pred_df)
    true_triage = _to_triage_classes(true_df)

    return {
        "qwk": cohen_kappa_score(true_triage, pred_triage, weights="quadratic"),
        "accuracy": accuracy_score(true_triage, pred_triage),
        "f1_macro": f1_score(true_triage, pred_triage, average="macro"),
    }
