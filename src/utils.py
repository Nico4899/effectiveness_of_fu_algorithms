from __future__ import annotations

from typing import List, Tuple

import numpy as np


def flatten_weights(weights: List[np.ndarray]) -> Tuple[np.ndarray, List[tuple]]:
    """Flatten a list of weight arrays into a single vector.

    Parameters
    ----------
    weights : List[np.ndarray]
        List of weight arrays to be flattened.

    Returns
    -------
    flat : np.ndarray
        Flattened array of weights.
    shapes : List[tuple]
        Original shapes of the weight arrays.
    """
    shapes = [w.shape for w in weights]
    # Flatten the weight arrays into a single vector
    flat = np.concatenate([w.ravel() for w in weights])
    return flat, shapes


def unflatten_weights(flat: np.ndarray, shapes: List[tuple]) -> List[np.ndarray]:
    """Inverse of :func:`flatten_weights`.

    Given a flattened array of weights and the original shapes, recreate the
    original list of weight arrays.

    Parameters
    ----------
    flat : np.ndarray
        Flattened array of weights.
    shapes : List[tuple]
        Original shapes of the weight arrays.

    Returns
    -------
    List[np.ndarray]
        List of weight arrays with the same shapes as the original.
    """
    weights = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        weights.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return weights
