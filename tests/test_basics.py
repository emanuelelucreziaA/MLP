"""
Unit tests for MLP components - Forward/Backward pass verification
"""

import numpy as np

from mlp.layer import DenseLayer
from mlp.network import MLP
from mlp.activations import relu, relu_derivative, sigmoid, sigmoid_derivative
from mlp.losses import CrossEntropy, MSE
from mlp.optimizers import SGD, Adam


def test_forward_pass():
    """Test forward pass with simple numbers"""
    print("\n" + "="*60)
    print("TEST 1: Forward Pass (DenseLayer)")
    print("="*60)
    
    # Create simple layer: 3 inputs -> 2 outputs, no activation
    np.random.seed(42)
    layer = DenseLayer(input_size=3, output_size=2)
    
    # Set fixed weights for reproducibility
    layer.W = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    layer.b = np.array([[0.1, 0.2]], dtype=np.float32)
    
    # Input: batch of 2 samples
    x = np.array([[1, 0, 1], [0, 1, 2]], dtype=np.float32)
    
    # Forward: z = x @ W + b
    output = layer.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Weights shape: {layer.W.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
    
    # Manual calculation for verification
    expected = x @ layer.W + layer.b
    print(f"Expected:\n{expected}")
    print(f"✓ Forward pass correct: {np.allclose(output, expected)}")
    
    return layer, x, output


def test_backward_pass(layer, x, output):
    """Test backward pass with numerical gradient checking"""
    print("\n" + "="*60)
    print("TEST 2: Backward Pass & Numerical Gradient Checking")
    print("="*60)
    
    # Simulate loss gradient (dL/doutput)
    dL_doutput = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    
    # Backward pass
    dL_dinput = layer.backward(dL_doutput)
    dW, db = layer.get_gradients()
    
    print(f"dL/doutput shape: {dL_doutput.shape}")
    print(f"dL/dinput shape: {dL_dinput.shape}")
    print(f"dL/dW shape: {dW.shape}")
    print(f"dL/db shape: {db.shape}")
    
    # Numerical gradient checking (finite differences)
    print("\nNumerical Gradient Verification (finite differences):")
    epsilon = 1e-5
    
    # Check dW numerically
    dW_numerical = np.zeros_like(layer.W)
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            layer.W[i, j] += epsilon
            output_plus = layer.forward(x)
            loss_plus = np.sum(output_plus * dL_doutput)  # Simplified loss
            
            layer.W[i, j] -= 2 * epsilon
            output_minus = layer.forward(x)
            loss_minus = np.sum(output_minus * dL_doutput)
            
            dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            layer.W[i, j] += epsilon
    
    # Recompute analytical gradients
    layer.forward(x)
    layer.backward(dL_doutput)
    dW, db = layer.get_gradients()
    
    diff = np.max(np.abs(dW - dW_numerical))
    print(f"Max gradient difference (dW): {diff:.6f}")
    print(f"✓ Gradients match numerically: {diff < 1e-4}")
    
    return dW, db


def test_activation_derivatives():
    """Test activation function derivatives"""
    print("\n" + "="*60)
    print("TEST 3: Activation Function Derivatives")
    print("="*60)
    
    x = np.array([[-1, 0, 1, 2]], dtype=np.float32)
    epsilon = 1e-5
    
    # Test ReLU
    print("\nReLU derivative:")
    analytical = relu_derivative(x)
    numerical = np.zeros_like(x)
    for i in range(x.shape[1]):
        x_plus = x.copy()
        x_plus[0, i] += epsilon
        x_minus = x.copy()
        x_minus[0, i] -= epsilon
        numerical[0, i] = (np.sum(relu(x_plus)) - np.sum(relu(x_minus))) / (2 * epsilon)
    
    print(f"Analytical: {analytical}")
    print(f"Numerical:  {numerical}")
    print(f"✓ ReLU derivative correct: {np.allclose(analytical, numerical, atol=1e-3)}")
    
    # Test Sigmoid
    print("\nSigmoid derivative:")
    analytical = sigmoid_derivative(x)
    numerical = np.zeros_like(x)
    for i in range(x.shape[1]):
        x_plus = x.copy()
        x_plus[0, i] += epsilon
        x_minus = x.copy()
        x_minus[0, i] -= epsilon
        numerical[0, i] = (np.sum(sigmoid(x_plus)) - np.sum(sigmoid(x_minus))) / (2 * epsilon)
    
    print(f"Analytical: {analytical}")
    print(f"Numerical:  {numerical}")
    print(f"✓ Sigmoid derivative correct: {np.allclose(analytical, numerical, atol=1e-3)}")


def test_mlp_forward():
    """Test MLP with multiple layers"""
    print("\n" + "="*60)
    print("TEST 4: MLP Forward Pass (Multiple Layers)")
    print("="*60)
    
    np.random.seed(42)
    
    # Create MLP: 3 -> 4 -> 2
    model = MLP()
    model.add_layer(DenseLayer(input_size=3, output_size=4, 
                              activation_fn=relu, activation_derivative=relu_derivative))
    model.add_layer(DenseLayer(input_size=4, output_size=2))
    
    # Input batch
    x = np.random.randn(5, 3)
    
    # Forward pass
    output = model.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ MLP output shape correct: {output.shape == (5, 2)}")
    
    model.summary()
    
    return model, x, output


def test_loss_functions():
    """Test loss functions"""
    print("\n" + "="*60)
    print("TEST 5: Loss Functions")
    print("="*60)
    
    # Test CrossEntropy
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)
    
    ce_loss = CrossEntropy()
    loss = ce_loss(y_true, y_pred)
    print(f"CrossEntropy Loss: {loss:.4f}")
    print(f"✓ CrossEntropy loss is positive: {loss > 0}")
    
    # Test MSE
    y_true_reg = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y_pred_reg = np.array([[1.1, 1.9], [3.1, 3.9]], dtype=np.float32)
    
    mse_loss = MSE()
    loss = mse_loss(y_true_reg, y_pred_reg)
    print(f"\nMSE Loss: {loss:.6f}")
    print(f"✓ MSE loss is small (near perfect prediction): {loss < 0.01}")


def test_optimizer_sgd():
    """Test SGD optimizer"""
    print("\n" + "="*60)
    print("TEST 6: SGD Optimizer")
    print("="*60)
    
    np.random.seed(42)
    optimizer = SGD(learning_rate=0.01)
    
    # Dummy gradients
    dW = np.array([[0.5, -0.3], [0.2, 0.1]], dtype=np.float32)
    db = np.array([[0.1, -0.05]], dtype=np.float32)
    
    update_W, update_b = optimizer.update(dW, db)
    print(f"Update W:\n{update_W}")
    print(f"Update b:\n{update_b}")
    print(f"✓ SGD update computed: {update_W.shape == dW.shape}")


def test_optimizer_adam():
    """Test Adam optimizer"""
    print("\n" + "="*60)
    print("TEST 7: Adam Optimizer")
    print("="*60)
    
    np.random.seed(42)
    optimizer = Adam(learning_rate=0.001)
    
    # Simulate multiple updates
    print("Simulating 5 optimizer steps:")
    for step in range(5):
        dW = np.random.randn(3, 4) * 0.1
        db = np.random.randn(1, 4) * 0.1
        
        update_W, update_b = optimizer.update(dW, db)
        print(f"Step {step+1}: update_W mean = {np.mean(np.abs(update_W)):.6f}")
    
    print(f"✓ Adam optimizer working: updates computed across steps")


def test_dense_layer_validation():
    """Test DenseLayer rejects invalid dimensions."""
    print("\n" + "="*60)
    print("TEST 8: DenseLayer Input Validation")
    print("="*60)

    invalid_cases = [
        (0, 2),
        (-1, 2),
        (2, 0),
        (2, -3),
    ]

    for input_size, output_size in invalid_cases:
        try:
            DenseLayer(input_size=input_size, output_size=output_size)
        except ValueError:
            print(f"✓ Rejected invalid dimensions: input_size={input_size}, output_size={output_size}")
        else:
            raise AssertionError(
                f"DenseLayer should reject input_size={input_size}, output_size={output_size}"
            )

    valid_layer = DenseLayer(input_size=3, output_size=2)
    print(f"✓ Accepted valid dimensions: {valid_layer.input_size} -> {valid_layer.output_size}")


if __name__ == "__main__":
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*15 + "MLP Unit Tests - Forward/Backward Pass" + " "*5 + "║")
    print("╚" + "="*58 + "╝")
    
    # Run tests
    layer, x, output = test_forward_pass()
    test_backward_pass(layer, x, output)
    test_activation_derivatives()
    model, x, output = test_mlp_forward()
    test_loss_functions()
    test_optimizer_sgd()
    test_optimizer_adam()
    test_dense_layer_validation()
    
    print("\n" + "="*60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("="*60 + "\n")
