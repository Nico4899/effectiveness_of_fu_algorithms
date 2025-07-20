"""Membership inference attack utilities."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf

from utils import flatten_weights


def compute_gradient_sum(model: tf.keras.Model, x: np.ndarray, num_classes: int = 100) -> List[np.ndarray]:
    r"""Compute ``\sum_a \nabla_W L(a, x)`` across all labels.

    Parameters
    ----------
    model:
        ``tf.keras.Model`` used for gradient computation.
    x:
        Single input sample with shape matching the model input.
    num_classes:
        Number of possible labels in the task.

    Returns
    -------
    list[np.ndarray]
        List of gradient arrays corresponding to ``model.trainable_variables``.
    """
    grads_sum = [np.zeros_like(w.numpy()) for w in model.trainable_variables]
    x_inp = tf.convert_to_tensor(x[None, ...])
    for a in range(num_classes):
        with tf.GradientTape() as tape:
            logits = model(x_inp, training=False)
            y_onehot = tf.one_hot([a], depth=num_classes)
            loss = tf.keras.losses.categorical_crossentropy(y_onehot, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        for i, g in enumerate(grads):
            grads_sum[i] += g.numpy()
    return grads_sum


def norm_difference(update: List[np.ndarray], candidate_grad: List[np.ndarray]) -> float:
    """Compute the norm difference metric for membership inference."""
    upd_flat, _ = flatten_weights(update)
    grad_flat, _ = flatten_weights(candidate_grad)
    orig_norm = np.linalg.norm(upd_flat)
    diff_norm = np.linalg.norm(upd_flat - grad_flat)
    return orig_norm ** 2 - diff_norm ** 2


def calibrate_threshold(deltas: Iterable[float], labels: Iterable[int]) -> float:
    """Choose the threshold that maximizes accuracy on a validation set."""
    deltas = list(deltas)
    labels = list(labels)
    best_acc = -1.0
    best_tau = 0.0
    for tau in sorted(deltas):
        preds = [1 if d > tau else 0 for d in deltas]
        acc = np.mean([p == l for p, l in zip(preds, labels)])
        if acc > best_acc:
            best_acc = acc
            best_tau = tau
    return best_tau