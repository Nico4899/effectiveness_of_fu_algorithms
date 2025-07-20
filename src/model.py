"""Model definitions for Texas-100 experiments.

This module provides utilities to construct the fully-connected neural
network used in the evaluation.  The architecture matches the one
reported in the project documentation: three hidden layers with ReLU
activations and dropout, followed by a softmax classification layer.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, regularizers, models


def create_model() -> tf.keras.Model:
    """Build the multilayer perceptron for hospital classification.

    The network consists of three hidden ``Dense`` layers of sizes 2048,
    1024 and 512 with ReLU activations and 50% dropout after each layer.
    All ``Dense`` layers use He normal initialization and ``L2`` weight
    decay of ``1e-4``.  The output layer has 100 units with a softmax
    activation.  The model is compiled with sparse categorical cross-
    entropy loss and accuracy metrics.  No optimizer is specified as
    this is handled by the federated learning clients.
    """

    # Regularizer
    l2 = regularizers.l2(1e-4)

    # Define inputs
    inputs = layers.Input(shape=(6170,), name="input_features")

    # Define layers
    x = layers.Dense(
        2048,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=l2,
    )(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        1024,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=l2,
    )(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        512,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=l2,
    )(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(
        100,
        activation="softmax",
        kernel_initializer="he_normal",
        kernel_regularizer=l2,
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="texas_mlp")
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = create_model()
    model.summary()