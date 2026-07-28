"""
Activation functions and their derivatives.

Key concept (chain rule):
  - Forward: y = activation(x)
  - Backward: dy/dx = derivative(x)
  - Used in: dL/dz = dL/dy * dy/dx (propagating gradient backwards)
"""

import numpy as np


# ============= SIGMOID =============
def sigmoid(x):
    """
    Sigmoid function: σ(x) = 1 / (1 + e^(-x))
    Range: (0, 1)
    Use case: Binary classification, or older networks (rarely used now)
    """
    # Clip to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def sigmoid_derivative(x):
    """
    Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
    Note: x here is the PRE-ACTIVATION (z), not the output
    """
    sig = sigmoid(x)
    return sig * (1 - sig)


# ============= RELU =============
def relu(x):
    """
    ReLU (Rectified Linear Unit): max(0, x)
    Use case: Hidden layers (most common activation)
    Advantages: Simple, non-saturating, sparse activation
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    ReLU derivative: 
      - 1 if x > 0
      - 0 if x <= 0
    """
    return (x > 0).astype(float)


# ============= TANH =============
def tanh(x):
    """
    Hyperbolic tangent: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Range: (-1, 1)
    Use case: Hidden layers (alternative to ReLU, centered output)
    Advantage over sigmoid: Centered at 0, converges faster
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Tanh derivative: tanh'(x) = 1 - tanh^2(x)
    """
    t = np.tanh(x)
    return 1 - t ** 2


# ============= SOFTMAX =============
def softmax(x):
    """
    Softmax: converts logits to probability distribution
    s_i(x) = e^(x_i) / Σ(e^(x_j))  for all j
    
    Uses: Output layer for multi-class classification
    Returns: Probability distribution (sum = 1)
    
    Numerical stability trick: subtract max(x) before exp
    """
    # Numerical stability: subtract max from each row
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(x):
    """
    Softmax derivative (Jacobian): 
      J_ij = s_i(x) * (δ_ij - s_j(x))
    
    Where δ_ij is Kronecker delta (1 if i==j, 0 otherwise)
    
    In practice, for cross-entropy loss + softmax, we don't use this directly.
    Instead, we use: dL/dz = (y_pred - y_true) directly.
    """
    s = softmax(x)
    # For single sample: Jacobian is J = diag(s) - s @ s^T
    # For batches: we typically skip this and use dL/dz = (y_pred - y_true)
    return s * (1 - s)  # Simplified version (element-wise)


# ============= LINEAR (Identity) =============
def linear(x):
    """
    Linear activation (identity): f(x) = x
    Use case: Regression tasks
    """
    return x


def linear_derivative(x):
    """Linear derivative: f'(x) = 1"""
    return np.ones_like(x)


# Alias for backward compatibility
identity = linear
identity_derivative = linear_derivative
