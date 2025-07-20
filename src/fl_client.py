import os
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import flwr as fl

from model import create_model


def load_client_data(client_id: int, data_dir: str = "data/clients") -> Tuple[np.ndarray, np.ndarray]:
    """Load a client's local dataset from ``data_dir``."""
    path = os.path.join(data_dir, f"client_{client_id}.npz")
    data = np.load(path)
    return data["X"], data["y"]


class TexasClient(fl.client.NumPyClient):
    """Flower client performing local training on Texas-100."""

    def __init__(self, client_id: int, data_dir: str = "data/clients") -> None:
        self.client_id = client_id
        self.x_train, self.y_train = load_client_data(client_id, data_dir)
        self.model = create_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Load provided parameters
        self.model.set_weights(parameters)

        round_num = int(config.get("server_round", 0))
        lr = 5e-3
        if round_num >= 75:
            lr *= 0.01
        elif round_num >= 50:
            lr *= 0.1

        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

        self.model.fit(self.x_train, self.y_train, batch_size=64, epochs=10, verbose=0)
        return self.model.get_weights(), len(self.x_train), {"cid": self.client_id}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"accuracy": float(acc)}