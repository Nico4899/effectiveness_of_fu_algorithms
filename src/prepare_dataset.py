"""Utility script to create a fixed 30k subset of Texas-100.

This script loads the full Texas-100 dataset, randomly samples 30,000
records using a fixed random seed and then creates an 80/20 train-test
split.  The resulting arrays are stored in ``.npz`` format so they can be
reused across experiments.
"""

from __future__ import annotations

import argparse
import os
import numpy as np

from data_loader import load_texas100, sample_records, split_train_test


DEFAULT_OUTPUT = "data/texas100_subset.npz"


def main(args: argparse.Namespace) -> None:
    """Load the full Texas-100 dataset, sample 30,000 records, split into
    train-test and store the result in an .npz file.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments parsed by argparse.  The following arguments are
        required:

        - ``root_dir``: Where the Texas-100 dataset is stored/downloaded.
        - ``output``: Destination .npz file for the subset split.
        - ``seed``: Random seed for sampling and splitting.
    """
    X, y = load_texas100(root_dir=args.root_dir, download=True)
    X_sub, y_sub = sample_records(X, y, n_samples=30_000, seed=args.seed)
    X_train, X_test, y_train, y_test = split_train_test(
        X_sub, y_sub, test_size=0.2, seed=args.seed
    )

    # Create the output directory if it doesn't exist yet
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Store the split arrays in an .npz file
    np.savez(
        args.output,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print(f"Saved split dataset to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sampled Texas-100 split")
    parser.add_argument(
        "--root-dir",
        default="data",
        help="Where the Texas-100 dataset is stored/downloaded",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Destination .npz file for the subset split",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for sampling and splitting"
    )

    main(parser.parse_args())