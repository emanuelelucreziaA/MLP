"""
Loss functions for training.

Key concept:
  - Loss = L(y_true, y_pred): measures prediction error
  - Derivative dL/dy_pred: used in backward pass to start gradient flow
"""

from __future__ import annotations

import numpy as np


class CrossEntropy:
    """
    Cross-Entropy Loss for multi-class classification.
    
    Formula:
      L = -Σ(y_true * log(y_pred))
    
    Where:
      - y_true: one-hot encoded targets (batch_size, num_classes)
      - y_pred: softmax probabilities (batch_size, num_classes)
    
    Gradient w.r.t. softmax output: dL/dy_pred = -y_true / y_pred
    But typically combined with softmax: dL/dz = (y_pred - y_true)
    """
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute cross-entropy loss
        
        Args:
            y_true: One-hot encoded targets (batch_size, num_classes)
            y_pred: Softmax probabilities (batch_size, num_classes)
        
        Returns:
            loss: Scalar average loss
        """
        batch_size = y_true.shape[0]
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-7
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy: -Σ(y_true * log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / batch_size
        return loss
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. output.
        
        For softmax + cross-entropy combination:
        dL/dz_softmax = (y_pred - y_true)
        
        This is the gradient that gets passed to backward pass.
        """
        batch_size = y_true.shape[0]
        # For softmax + cross-entropy with mean reduction over batch:
        return (y_pred - y_true) / batch_size


class MSE:
    """
    Mean Squared Error Loss for regression.
    
    Formula:
      L = (1/batch_size) * Σ(y_true - y_pred)^2
    
    Gradient: dL/dy_pred = -2 * (y_true - y_pred) / n
    """
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute MSE loss
        
        Args:
            y_true: Target values (batch_size, num_outputs)
            y_pred: Predicted values (batch_size, num_outputs)
        
        Returns:
            loss: Scalar average loss
        """
        batch_size = y_true.shape[0]
        diff = y_true - y_pred
        loss = np.sum(diff ** 2) / batch_size
        return loss
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE w.r.t. output.
        
        dL/dy_pred = -2 * (y_true - y_pred) / batch_size
        """
        batch_size = y_true.shape[0]
        return -2 * (y_true - y_pred) / batch_size


class BinaryCrossentropy:
    """
    Binary Cross-Entropy Loss for binary classification.
    
    Formula:
      L = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    
    Where:
      - y_true: target (0 or 1) (batch_size, 1)
      - y_pred: sigmoid output (batch_size, 1)
    """
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss
        
        Args:
            y_true: Binary targets (batch_size, 1)
            y_pred: Sigmoid probabilities (batch_size, 1)
        
        Returns:
            loss: Scalar average loss
        """
        batch_size = y_true.shape[0]
        
        epsilon = 1e-7
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy
        loss = -np.mean(y_true * np.log(y_pred_clipped) + 
                        (1 - y_true) * np.log(1 - y_pred_clipped))
        return loss
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Gradient of binary cross-entropy w.r.t. output.
        
        dL/dy_pred = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / batch_size
        """
        epsilon = 1e-7
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        batch_size = y_true.shape[0]
        
        return (-(y_true / y_pred_clipped) + (1 - y_true) / (1 - y_pred_clipped)) / batch_size
