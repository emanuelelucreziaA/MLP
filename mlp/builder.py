"""Shared model builders for MLP training and evaluation scripts."""

from __future__ import annotations

from typing import Sequence

from .activations import relu, relu_derivative
from .layer import DenseLayer
from .network import MLP


def build_classifier(
    input_size: int = 784,
    output_size: int = 10,
    hidden_sizes: Sequence[int] = (256, 128, 64),
) -> MLP:
    """Build a classifier MLP with ReLU hidden layers and linear logits output."""
    model = MLP()
    previous_size = input_size

    for hidden_size in hidden_sizes:
        model.add_layer(
            DenseLayer(
                input_size=previous_size,
                output_size=hidden_size,
                activation_fn=relu,
                activation_derivative=relu_derivative,
            )
        )
        previous_size = hidden_size

    model.add_layer(
        DenseLayer(
            input_size=previous_size,
            output_size=output_size,
            activation_fn=None,
            activation_derivative=None,
        )
    )
    return model
