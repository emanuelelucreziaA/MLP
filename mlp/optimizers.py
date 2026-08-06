"""
Optimizers for weight updates.

Key concept:
  - Optimizer receives gradient (dL/dW, dL/db)
  - Returns update step: W_new = W_old - update
  - Different strategies: SGD (simple), Adam (adaptive learning rates)
"""

from __future__ import annotations

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent with optional momentum.
    
    Update rule (no momentum):
      W_new = W - learning_rate * dW
    
    With momentum (classic momentum):
      v = momentum * v + learning_rate * dW
      W_new = W - v
    
    Args:
        learning_rate: Step size for updates (typically 0.001 - 0.1)
        momentum: Acceleration parameter (typically 0.9, use 0 to disable)
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_W = None
        self.velocity_b = None
    
    def update(self, dW: np.ndarray, db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute update step for weights and biases.
        
        Args:
            dW: Gradient w.r.t. weights
            db: Gradient w.r.t. biases
        
        Returns:
            update_W, update_b: Update steps to subtract from W and b
        """
        # Initialize velocity on first call
        if self.velocity_W is None:
            self.velocity_W = np.zeros_like(dW)
            self.velocity_b = np.zeros_like(db)
        
        # Update velocity (momentum)
        self.velocity_W = self.momentum * self.velocity_W + self.learning_rate * dW
        self.velocity_b = self.momentum * self.velocity_b + self.learning_rate * db
        
        return self.velocity_W, self.velocity_b
    
    def reset(self) -> None:
        """Reset momentum buffers"""
        self.velocity_W = None
        self.velocity_b = None


class Adam:
    """
    Adaptive Moment Estimation (Adam).
    
    Combines ideas from momentum and RMSprop.
    
    Update rule:
      m = beta1 * m + (1 - beta1) * dW              # 1st moment (momentum)
      v = beta2 * v + (1 - beta2) * dW^2            # 2nd moment (RMSprop)
      m_hat = m / (1 - beta1^t)                      # Bias correction
      v_hat = v / (1 - beta2^t)                      # Bias correction
      W_new = W - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    
    Args:
        learning_rate: Step size (typically 0.001)
        beta1: Exponential decay rate for 1st moment (typically 0.9)
        beta2: Exponential decay rate for 2nd moment (typically 0.999)
        epsilon: Small constant for numerical stability (typically 1e-8)
    
    Advantage: Adaptive learning rates per parameter, converges faster
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # First moment (mean) estimates
        self.m_W = None
        self.m_b = None
        
        # Second moment (variance) estimates
        self.v_W = None
        self.v_b = None
        
        # Time step counter
        self.t = 0
    
    def update(self, dW: np.ndarray, db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Adam update step.
        
        Args:
            dW: Gradient w.r.t. weights
            db: Gradient w.r.t. biases
        
        Returns:
            update_W, update_b: Update steps
        """
        # Check if dimensions changed (different layer)
        if self.m_W is None or self.m_W.shape != dW.shape:
            # Reset optimizer state for new layer dimensions
            self.m_W = np.zeros_like(dW)
            self.m_b = np.zeros_like(db)
            self.v_W = np.zeros_like(dW)
            self.v_b = np.zeros_like(db)
            self.t = 0
        
        # Increment time step
        self.t += 1
        
        # Update biased first moment estimate (momentum)
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        
        # Update biased second raw moment estimate (variance)
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (dW ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
        
        # Bias correction
        m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        
        # Compute updates
        update_W = self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        update_b = self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        return update_W, update_b
    
    def reset(self) -> None:
        """Reset optimizer state"""
        self.m_W = None
        self.m_b = None
        self.v_W = None
        self.v_b = None
        self.t = 0
