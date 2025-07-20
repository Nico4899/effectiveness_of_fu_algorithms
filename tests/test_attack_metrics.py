import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from attack_metrics import attack_success_probability, tpr_at_fpr, roc_auc, evaluate_attack


def test_metrics_simple():
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    labels = np.array([1, 1, 1, 0, 0, 0])
    asp = attack_success_probability(scores, labels, 0.5)
    assert asp == 1.0
    tpr = tpr_at_fpr(scores, labels, 0.01)
    assert 0.0 <= tpr <= 1.0
    auc = roc_auc(scores, labels)
    assert 0.0 <= auc <= 1.0
    asp2, tpr2, auc2, tau = evaluate_attack(scores, labels)
    assert 0.0 <= asp2 <= 1.0
    assert 0.0 <= tpr2 <= 1.0
    assert 0.0 <= auc2 <= 1.0
    assert isinstance(tau, float)