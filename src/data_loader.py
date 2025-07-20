"""Data loader for the Texas-100 hospital discharge dataset.

This module can download the preprocessed dataset provided by the
`privacytrustlab/datasets` GitHub repository and load it into NumPy arrays.
Each sample consists of 6,169 binary features and belongs to one of 100
classes representing hospitals.  The raw labels are in the range 1-100 and
are converted to 0-99.

The dataset tarball (``dataset_texas.tgz``) is around 15MB.  Downloading
requires network access.  If your environment does not allow outbound
connections you can manually obtain the archive from
``https://github.com/privacytrustlab/datasets`` and place it in the
``root_dir`` directory provided to :func:`load_texas100`.
"""

from __future__ import annotations

import os
import tarfile
from urllib.request import urlretrieve
from typing import Tuple

import numpy as np

TEXAS_URL = (
    "https://raw.githubusercontent.com/privacytrustlab/datasets/"
    "master/dataset_texas.tgz"
)


def download_texas100(root_dir: str = "data") -> Tuple[str, str]:
    """Download and extract the Texas-100 dataset.

    Parameters
    ----------
    root_dir:
        Directory where the dataset should be stored.

    Returns
    -------
    Tuple[str, str]
        Paths to the extracted feature and label files.
    """
    os.makedirs(root_dir, exist_ok=True)
    tar_path = os.path.join(root_dir, "dataset_texas.tgz")
    if not os.path.exists(tar_path):
        print(f"Downloading Texas-100 dataset from {TEXAS_URL} ...")
        urlretrieve(TEXAS_URL, tar_path)

    # Extract only the 100-class version
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("texas/100/")]
        tar.extractall(path=root_dir, members=members, filter=lambda tarinfo, _: tarinfo)

    feats_path = os.path.join(root_dir, "texas", "100", "feats")
    labels_path = os.path.join(root_dir, "texas", "100", "labels")
    return feats_path, labels_path


def load_texas100(root_dir: str = "data", download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load the Texas-100 dataset.

    Parameters
    ----------
    root_dir:
        Location where the dataset is (or will be) stored.
    download:
        If ``True`` and the dataset is not found, it will be downloaded.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(features, labels)`` where ``features`` is a 2-D NumPy array of
        dtype ``int8`` and ``labels`` is a 1-D array of integers in the
        range ``0-99``.
    """
    feats_path = os.path.join(root_dir, "texas", "100", "feats")
    labels_path = os.path.join(root_dir, "texas", "100", "labels")

    if not (os.path.exists(feats_path) and os.path.exists(labels_path)):
        if download:
            feats_path, labels_path = download_texas100(root_dir)
        else:
            raise FileNotFoundError(
                "Texas-100 dataset not found; set download=True to fetch it."
            )

    X = np.loadtxt(feats_path, delimiter=",", dtype=np.int8)
    y = np.loadtxt(labels_path, dtype=int) - 1
    return X, y


def sample_records(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 30_000,
    seed: int | None = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly sample a subset of the dataset.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(N, d)``.
    y:
        Array of labels of shape ``(N,)``.
    n_samples:
        Number of samples to draw from the dataset.
    seed:
        Seed for the random generator to ensure reproducibility. ``None``
        uses NumPy's global generator state.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(X_subset, y_subset)`` containing ``n_samples`` randomly selected
        entries from ``X`` and ``y``.
    """

    if n_samples > len(X):
        raise ValueError(
            "Requested more samples than are available in the dataset"
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n_samples, replace=False)
    return X[idx], y[idx]


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int | None = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a dataset into train and test portions.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Labels corresponding to ``X``.
    test_size:
        Fraction of the dataset to allocate to the test set.
    seed:
        Seed for the random generator to ensure reproducible shuffling.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(X_train, X_test, y_train, y_test)`` split according to
        ``test_size``.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in the interval (0, 1)")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    n_test = int(round(len(X) * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    dirichlet_alpha: float,
    num_clients: int = 10,
    seed: int | None = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition data among ``num_clients`` using a Dirichlet distribution.

    Each class label's samples are distributed to the clients according to
    probabilities drawn from :class:`numpy.random.Generator.dirichlet` with
    parameter ``dirichlet_alpha``.  The returned list contains ``num_clients``
    tuples ``(X_i, y_i)`` representing the dataset for client ``i``.

    Parameters
    ----------
    X:
        Feature matrix of the *training* portion of the dataset.
    y:
        Labels corresponding to ``X``.
    dirichlet_alpha:
        Concentration parameter for the Dirichlet distribution.  Smaller
        values yield more uneven label distributions across clients.
    num_clients:
        Number of client datasets to create.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        ``[(X_0, y_0), (X_1, y_1), ...]`` containing the partitioned data.
    """

    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be > 0")
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")

    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    classes = np.unique(y)
    for c in classes:
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
        n_per_client = (proportions * len(class_idx)).astype(int)

        # Distribute any rounding remainder
        remainder = len(class_idx) - n_per_client.sum()
        if remainder > 0:
            extra_clients = rng.choice(num_clients, size=remainder, p=proportions)
            for client_id in extra_clients:
                n_per_client[client_id] += 1

        start = 0
        for client_id, n in enumerate(n_per_client):
            if n > 0:
                client_indices[client_id].extend(class_idx[start : start + n])
            start += n

    partitions = []
    for indices in client_indices:
        idx_array = np.array(indices, dtype=int)
        partitions.append((X[idx_array], y[idx_array]))
    return partitions


def save_partitions(
    partitions: list[tuple[np.ndarray, np.ndarray]],
    root_dir: str = "data/clients",
) -> None:
    """Persist client datasets to ``root_dir`` in ``.npz`` format."""

    os.makedirs(root_dir, exist_ok=True)
    for i, (X_i, y_i) in enumerate(partitions):
        path = os.path.join(root_dir, f"client_{i}.npz")
        np.savez(path, X=X_i, y=y_i)


if __name__ == "__main__":
    X, y = load_texas100()
    print("Loaded Texas-100:", X.shape, y.shape)

    partitions = partition_data(X, y, dirichlet_alpha=1.0, num_clients=10)
    save_partitions(partitions)
    print("Saved example client partitions to data/clients")