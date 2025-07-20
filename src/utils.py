from __future__ import annotations

from typing import List, Tuple

import numpy as np


def flatten_weights(weights: List[np.ndarray]) -> Tuple[np.ndarray, List[tuple]]:
    """Flatten a list of weight arrays into a single vector."""
    shapes = [w.shape for w in weights]
    flat = np.concatenate([w.ravel() for w in weights])
    return flat, shapes


def unflatten_weights(flat: np.ndarray, shapes: List[tuple]) -> List[np.ndarray]:
    """Inverse of :func:`flatten_weights`."""
    weights = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        weights.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return weights