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
        tar.extractall(path=root_dir, members=members, filter=lambda tarinfo: tarinfo)

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


if __name__ == "__main__":
    X, y = load_texas100()
    print("Loaded Texas-100:", X.shape, y.shape)