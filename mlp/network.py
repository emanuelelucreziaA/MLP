"""
Multi-Layer Perceptron: Composition of Dense layers
"""

from __future__ import annotations

import copy
import numpy as np

from .layer import DenseLayer


class MLP:
    """
    Multi-Layer Perceptron: stack of fully connected layers.
    
    Handles:
    - Forward propagation through all layers
    - Backward propagation through all layers
    - Weight/bias updates via optimizer
    """
    
    def __init__(self) -> None:
        self.layers: list[DenseLayer] = []
        self.optimizer = None
        self.loss_fn = None
    
    def add_layer(self, layer: DenseLayer) -> MLP:
        """Add a dense layer to the network"""
        self.layers.append(layer)
        return self
    
    def set_optimizer(self, optimizer) -> MLP:
        """Set optimizer for weight updates"""
        self.optimizer = optimizer
        return self
    
    def set_loss(self, loss_fn) -> MLP:
        """Set loss function"""
        self.loss_fn = loss_fn
        return self
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: propagate input through all layers
        
        Args:
            x: Input of shape (batch_size, input_features)
        
        Returns:
            output: Network output after all layers
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dL_doutput: np.ndarray) -> None:
        """
        Backward pass: propagate gradient through all layers (reverse order)
        
        Args:
            dL_doutput: Gradient of loss w.r.t. network output
        
        Returns:
            None (gradients stored in each layer)
        """
        # Propagate gradient backwards through layers in reverse order
        for layer in reversed(self.layers):
            dL_doutput = layer.backward(dL_doutput)
    
    def update_weights(self) -> None:
        """
        Update weights and biases using optimizer.

        Each layer keeps its own optimizer state so adaptive methods like Adam
        can maintain separate moment estimates per parameter tensor.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Use model.set_optimizer()")

        for layer in self.layers:
            if not hasattr(layer, 'optimizer') or layer.optimizer is None:
                layer.optimizer = copy.deepcopy(self.optimizer)

            dW, db = layer.get_gradients()

            # Get update from optimizer for this layer
            dW_update, db_update = layer.optimizer.update(dW, db)

            # Update parameters
            layer.update_parameters(dW_update, db_update)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict: forward pass + argmax (for classification)
        
        Args:
            x: Input of shape (batch_size, input_features)
        
        Returns:
            predictions: Class indices of shape (batch_size,)
        """
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get probability predictions (before argmax)"""
        return self.forward(x)
    
    def get_layers(self) -> list[DenseLayer]:
        """Return list of layers"""
        return self.layers
    
    def get_weights(self) -> list[dict[str, np.ndarray]]:
        """Get all weights and biases"""
        weights = []
        for layer in self.layers:
            weights.append({'W': layer.W.copy(), 'b': layer.b.copy()})
        return weights
    
    def set_weights(self, weights: list[dict[str, np.ndarray]]) -> None:
        """Set weights and biases"""
        for i, layer in enumerate(self.layers):
            layer.W = weights[i]['W'].copy()
            layer.b = weights[i]['b'].copy()
    
    def summary(self) -> None:
        """Print network architecture summary"""
        total_params = 0
        print("=" * 70)
        print("Model Summary")
        print("=" * 70)
        
        for i, layer in enumerate(self.layers):
            input_size = layer.input_size
            output_size = layer.output_size
            activation_name = getattr(layer.activation_fn, '__name__', 'None')
            
            weight_params = input_size * output_size
            bias_params = output_size
            layer_params = weight_params + bias_params
            total_params += layer_params
            
            print(f"Layer {i+1}: Dense({input_size} → {output_size})")
            print(f"  Activation: {activation_name}")
            print(f"  Parameters: {layer_params:,} (W: {weight_params:,}, b: {bias_params:,})")
            print()
        
        print("=" * 70)
        print(f"Total Parameters: {total_params:,}")
        print("=" * 70)
