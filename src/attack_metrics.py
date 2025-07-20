"""Evaluation metrics for membership inference attacks."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from membership_attack import calibrate_threshold


def attack_success_probability(scores: Iterable[float], labels: Iterable[int], threshold: float) -> float:
    """Compute the attack success probability (ASP)."""
    scores = np.array(list(scores))
    labels = np.array(list(labels))
    preds = scores > threshold
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    total = tp + tn + fp + fn
    return float(tp + tn) / total if total else float("nan")


def tpr_at_fpr(scores: Iterable[float], labels: Iterable[int], target_fpr: float = 0.01) -> float:
    """Return the TPR when FPR first exceeds ``target_fpr``."""
    scores = np.array(list(scores))
    labels = np.array(list(labels))
    desc_idx = np.argsort(-scores)
    sorted_labels = labels[desc_idx]
    cum_pos = np.cumsum(sorted_labels == 1)
    cum_neg = np.cumsum(sorted_labels == 0)
    total_pos = np.sum(labels == 1)
    total_neg = np.sum(labels == 0)
    fprs = cum_neg / max(total_neg, 1)
    tprs = cum_pos / max(total_pos, 1)
    idx = np.searchsorted(fprs, target_fpr, side="right")
    idx = min(idx, len(tprs) - 1)
    return float(tprs[idx])


def roc_auc(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Compute area under the ROC curve."""
    scores = np.array(list(scores))
    labels = np.array(list(labels))
    return float(roc_auc_score(labels, scores))


def evaluate_attack(scores: Iterable[float], labels: Iterable[int]) -> Tuple[float, float, float, float]:
    """Evaluate attack metrics and return ``(ASP, TPR@1%FPR, AUC, threshold)``."""
    scores = list(scores)
    labels = list(labels)
    tau = calibrate_threshold(scores, labels)
    asp = attack_success_probability(scores, labels, tau)
    tpr = tpr_at_fpr(scores, labels, 0.01)
    auc = roc_auc(scores, labels)
    return asp, tpr, auc, tau