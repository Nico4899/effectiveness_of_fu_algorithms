"""
This module implements Subspace Federated Unlearning (SFU) as well as a
baseline retraining routine.  Both functions operate on ``tf.keras`` models
and plain NumPy client datasets so they can be used outside of Flower if
needed.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf
import flwr as fl

from model import create_model


def _flatten_weights(weights: List[np.ndarray]) -> Tuple[np.ndarray, List[tuple]]:
    """Flatten a list of weight arrays into a single vector."""
    shapes = [w.shape for w in weights]
    flat = np.concatenate([w.ravel() for w in weights])
    return flat, shapes


def _unflatten_weights(flat: np.ndarray, shapes: List[tuple]) -> List[np.ndarray]:
    """Inverse of :func:`_flatten_weights`."""
    weights = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        weights.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return weights


def _compute_gradient(model: tf.keras.Model, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    """Compute gradients of the loss w.r.t. model weights."""
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    return [g.numpy() for g in grads]


def apply_SFU(
    global_model: tf.keras.Model,
    target_data: Tuple[np.ndarray, np.ndarray],
    remaining_data: Iterable[Tuple[np.ndarray, np.ndarray]],
    eta: float = 1e-3,
    epochs: int = 1,
) -> List[np.ndarray]:
    """Apply the Subspace Federated Unlearning procedure.

    Parameters
    ----------
    global_model:
        Trained global model to be unlearned. The model is updated in place.
    target_data:
        ``(X_t, y_t)`` dataset of the client requesting deletion.
    remaining_data:
        Iterable of datasets ``[(X_i, y_i), ...]`` for the other clients.
    eta:
        Gradient ascent step size for the target client.
    epochs:
        Number of ascent epochs on the target data.

    Returns
    -------
    list[np.ndarray]
        The updated weights after unlearning.
    """
    # Save original weights
    original_weights = [w.numpy() for w in global_model.trainable_variables]
    original_flat, shapes = _flatten_weights(original_weights)

    # ----- Target gradient ascent -----
    target_model = create_model()
    target_model.set_weights(original_weights)
    x_t, y_t = target_data

    dataset = tf.data.Dataset.from_tensor_slices((x_t, y_t)).batch(32)
    for _ in range(epochs):
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                logits = target_model(batch_x, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, logits)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, target_model.trainable_variables)
            for var, grad in zip(target_model.trainable_variables, grads):
                var.assign_add(eta * grad)

    target_weights = [w.numpy() for w in target_model.trainable_variables]
    target_update = [nw - ow for nw, ow in zip(target_weights, original_weights)]

    # Flatten target update for projection
    g_t, shapes = _flatten_weights(target_update)

    # ----- Gather representation from remaining clients -----
    subspace_vectors = []
    for x_i, y_i in remaining_data:
        rep_model = create_model()
        rep_model.set_weights(original_weights)
        grads = _compute_gradient(rep_model, x_i, y_i)
        g_vec, _ = _flatten_weights(grads)
        subspace_vectors.append(g_vec)

    if subspace_vectors:
        M = np.stack(subspace_vectors, axis=1)
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        projection = U @ (U.T @ g_t)
        g_t_perp = g_t - projection
    else:
        g_t_perp = g_t

    new_weights = _unflatten_weights(g_t_perp + original_flat, shapes)
    global_model.set_weights(new_weights)
    return new_weights


def retrain_without_client(
    partitions: List[Tuple[np.ndarray, np.ndarray]],
    target_id: int,
    rounds: int = 100,
    address: str = "0.0.0.0:8080",
) -> None:
    """Retrain a model from scratch excluding ``target_id``."""
    remaining = [(x, y) for i, (x, y) in enumerate(partitions) if i != target_id]

    # Setup Flower server
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(remaining),
        min_evaluate_clients=len(remaining),
        min_available_clients=len(remaining),
    )
    fl.server.start_server(server_address=address, config={"num_rounds": rounds}, strategy=strategy)
