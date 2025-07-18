import os
from typing import Dict, List

import flwr as fl
import numpy as np
import tensorflow as tf

from data_loader import split_train_test

# Training hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 64
MOMENTUM = 0.9
BASE_LR = 5e-3


def lr_for_round(round_num: int) -> float:
    """Piecewise constant learning rate schedule."""
    if round_num >= 75:
        return BASE_LR * 0.01
    if round_num >= 50:
        return BASE_LR * 0.1
    return BASE_LR


def create_model() -> tf.keras.Model:
    """Create the baseline softmax classifier."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(6169,)),
            tf.keras.layers.Dense(
                100,
                activation="softmax",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
        ]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=BASE_LR, momentum=MOMENTUM)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


class TexasClient(fl.client.NumPyClient):
    """Flower client handling local training for a single data partition."""

    def __init__(self, client_id: int, data_dir: str = "data/clients"):
        self.client_id = client_id
        path = os.path.join(data_dir, f"client_{client_id}.npz")
        data = np.load(path)
        X, y = data["X"], data["y"]
        self.X_train, self.X_test, self.y_train, self.y_test = split_train_test(
            X, y, test_size=0.2, seed=client_id
        )
        self.model = create_model()
        self.last_update: List[np.ndarray] | None = None

    def get_parameters(self, config: Dict[str, str] | None = None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        round_num = int(config.get("server_round", 1))
        lr = lr_for_round(round_num)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

        initial_weights = [w.copy() for w in self.model.get_weights()]
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        new_weights = self.model.get_weights()
        self.last_update = [iw - nw for iw, nw in zip(initial_weights, new_weights)]
        return new_weights, len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(
            self.X_test,
            self.y_test,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        return loss, len(self.X_test), {"accuracy": float(acc)}