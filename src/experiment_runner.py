import argparse
import csv
import itertools
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from data_loader import partition_data
from model import create_model
from unlearning import apply_SFU
from membership_attack import compute_gradient_sum, norm_difference
from attack_metrics import evaluate_attack


def federated_train(
    partitions: List[Tuple[np.ndarray, np.ndarray]],
    rounds: int,
    C: float,
    seed: int = 0,
) -> tf.keras.Model:
    """Simple FedAvg implementation used for experiments."""
    rng = np.random.default_rng(seed)
    num_clients = len(partitions)
    model = create_model()
    weights = model.get_weights()

    for _ in range(rounds):
        m = max(1, int(C * num_clients))
        clients = rng.choice(num_clients, size=m, replace=False)
        updates = []
        num_samples = []
        for cid in clients:
            x, y = partitions[cid]
            client_model = create_model()
            client_model.set_weights(weights)
            client_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.9),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            )
            client_model.fit(x, y, batch_size=64, epochs=1, verbose=0)
            updates.append(client_model.get_weights())
            num_samples.append(len(x))

        # FedAvg aggregation
        new_weights = [np.zeros_like(w) for w in weights]
        total = sum(num_samples)
        for w_update, n in zip(updates, num_samples):
            for i, w in enumerate(w_update):
                new_weights[i] += w * (n / total)
        weights = new_weights

    model.set_weights(weights)
    return model


def membership_attack_scores(
    model: tf.keras.Model,
    model_update: List[np.ndarray],
    positive_data: Tuple[np.ndarray, np.ndarray],
    negative_data: Tuple[np.ndarray, np.ndarray],
) -> Tuple[List[float], List[int]]:
    """Compute membership inference scores for a small sample.

    Parameters
    ----------
    model:
        The trained model prior to applying the update. Gradients are
        computed with respect to this model.
    model_update:
        Weights update applied to ``model`` during unlearning.
    positive_data, negative_data:
        Tuples of input arrays and labels representing members and
        non-members.
    """

    x_pos, _ = positive_data
    x_neg, _ = negative_data

    scores = []
    labels = []

    for x in x_pos[:10]:
        grad = compute_gradient_sum(model, x)
        score = norm_difference(model_update, grad)
        scores.append(score)
        labels.append(1)

    for x in x_neg[:10]:
        grad = compute_gradient_sum(model, x)
        score = norm_difference(model_update, grad)
        scores.append(score)
        labels.append(0)

    return scores, labels


def run_trial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float,
    C: float,
    eta: float,
    epochs: int,
    method: str = "SFU",
    seed: int = 0,
) -> Dict[str, float]:
    """Run training, unlearning and attack for one configuration."""

    partitions = partition_data(
        X_train, y_train, dirichlet_alpha=alpha, num_clients=10, seed=seed
    )
    model = federated_train(partitions, rounds=100, C=C, seed=seed)

    acc_before = float(model.evaluate(X_test, y_test, verbose=0)[1])

    before_weights = [w.copy() for w in model.get_weights()]

    if method.lower() == "retrain":
        retrain_model = federated_train(partitions[1:], rounds=100, C=C, seed=seed)
        model.set_weights(retrain_model.get_weights())
    else:
        apply_SFU(model, partitions[0], partitions[1:], eta=eta, epochs=epochs)

    after_weights = model.get_weights()
    acc_after = float(model.evaluate(X_test, y_test, verbose=0)[1])

    update = [nw - bw for nw, bw in zip(after_weights, before_weights)]
    # Build a model with the pre-unlearning weights for gradient computation
    attack_model = create_model()
    attack_model.set_weights(before_weights)
    scores, labels = membership_attack_scores(
        attack_model, update, partitions[0], partitions[1]
    )

    asp, tpr, auc, _ = evaluate_attack(scores, labels)

    return {
        "alpha": alpha,
        "C": C,
        "eta": eta,
        "E": epochs,
        "method": method,
        "ASP": asp,
        "TPR": tpr,
        "AUC": auc,
        "Accuracy_before": acc_before,
        "Accuracy_after": acc_after,
        "Accuracy_drop": acc_before - acc_after,
    }


def _append_row(path: str, row: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "C",
                "eta",
                "E",
                "method",
                "ASP",
                "TPR",
                "AUC",
                "Accuracy_before",
                "Accuracy_after",
                "Accuracy_drop",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def experiment1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output: str,
) -> None:
    alphas = [0.1, 5.0, 10.0]
    Cs = [0.1, 0.5, 1.0]
    for alpha, C in itertools.product(alphas, Cs):
        metrics = run_trial(
            X_train,
            y_train,
            X_test,
            y_test,
            alpha=alpha,
            C=C,
            eta=0.01,
            epochs=1,
        )
        _append_row(output, metrics)


def experiment2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output: str,
) -> None:
    etas = [1e-3, 1e-2, 1e-1]
    epochs_list = [1, 5, 10]
    for eta, E in itertools.product(etas, epochs_list):
        metrics = run_trial(
            X_train,
            y_train,
            X_test,
            y_test,
            alpha=5.0,
            C=0.5,
            eta=eta,
            epochs=E,
        )
        _append_row(output, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated unlearning experiments")
    parser.add_argument("--data", default="data/texas100_subset.npz", help="Path to texas100_subset.npz")
    parser.add_argument("--exp", choices=["system", "algorithm"], default="system")
    parser.add_argument("--output", default="logs/results.csv", help="CSV file to append metrics")
    args = parser.parse_args()

    data = np.load(args.data)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data.get("X_test")
    y_test = data.get("y_test")

    if X_test is None or y_test is None:
        raise ValueError("Dataset missing test split")

    if args.exp == "system":
        experiment1(X_train, y_train, X_test, y_test, args.output)
    else:
        experiment2(X_train, y_train, X_test, y_test, args.output)
