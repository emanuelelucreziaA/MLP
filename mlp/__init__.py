"""
MLP (Multi-Layer Perceptron) Neural Network Package

A from-scratch implementation of feed-forward neural networks.
Includes dense layers, activations, losses, and optimizers.
"""

# Core layer imports - lightweight, no data dependency
from .layer import DenseLayer
from .network import MLP

# Activation functions
from .activations import (
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    softmax,
)

# Loss and optimizer imports
from .losses import CrossEntropy, MSE
from .optimizers import SGD, Adam
from .builder import build_classifier
from .metrics import (
    prediction_labels,
    compute_accuracy,
    confusion_matrix,
    classification_metrics,
)


def __getattr__(name):
    """Lazy import data utilities on demand."""
    if name == 'MNISTLoader':
        from .data import MNISTLoader
        return MNISTLoader
    if name == 'one_hot_encode':
        from .data import one_hot_encode
        return one_hot_encode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'DenseLayer', 'MLP',
    'sigmoid', 'sigmoid_derivative',
    'relu', 'relu_derivative',
    'tanh', 'tanh_derivative',
    'softmax',
    'CrossEntropy', 'MSE',
    'SGD', 'Adam',
    'build_classifier',
    'prediction_labels', 'compute_accuracy', 'confusion_matrix', 'classification_metrics',
    'MNISTLoader', 'one_hot_encode',
]
