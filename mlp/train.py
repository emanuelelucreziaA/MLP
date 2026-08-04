"""
Training script: Train MLP on MNIST classification task

Usage:
    python -m mlp.train
    mlp-train
"""

import os
import pickle
from pathlib import Path

import numpy as np

from .activations import relu, relu_derivative, softmax
from .data import MNISTLoader, one_hot_encode
from .layer import DenseLayer
from .losses import CrossEntropy
from .network import MLP
from .optimizers import Adam
from .runtime import initialize_environment

PROJECT_ROOT, DATA_DIR = initialize_environment(ensure_data_dir=True)


def build_model(input_size=784, output_size=10, hidden_sizes=(256, 128, 64)):
    """Build MLP architecture for MNIST classification."""
    print("\n" + "=" * 70)
    arch = [input_size] + list(hidden_sizes) + [output_size]
    print("Building MLP Model for MNIST")
    print(f"Architecture: {' -> '.join(str(v) for v in arch)}")
    print("=" * 70)

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

    model.set_optimizer(Adam(learning_rate=0.001))
    model.set_loss(CrossEntropy())

    model.summary()
    return model


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy"""
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_true)
    return accuracy


def train_epoch(model, x_train, y_train_onehot, batch_size=32):
    """
    Train model for one epoch with mini-batch gradient descent.

    Returns:
        average_loss: Average loss across batches
    """
    n_samples = x_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    total_loss = 0.0
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        x_batch = x_train[batch_indices]
        y_batch = y_train_onehot[batch_indices]

        y_pred = model.forward(x_batch)
        y_pred_softmax = softmax(y_pred)

        loss = model.loss_fn(y_batch, y_pred_softmax)
        total_loss += loss
        dL_doutput = model.loss_fn.gradient(y_batch, y_pred_softmax)

        model.backward(dL_doutput)
        model.update_weights()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{n_batches}, Loss: {loss:.6f}", end="\r")

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate(model, x_test, y_test):
    """Evaluate model on test set"""
    y_pred_logits = model.forward(x_test)
    y_pred_softmax = softmax(y_pred_logits)

    accuracy = compute_accuracy(y_test, y_pred_softmax)
    loss = model.loss_fn(one_hot_encode(y_test, 10), y_pred_softmax)

    return accuracy, loss


def main():
    print("\n" + "=" * 70)
    print("MLP Training Pipeline - dataset mode: mnist")
    print("=" * 70)

    n_epochs = 50
    batch_size = 36
    hidden_sizes = (256, 128, 64)

    print("Loading MNIST dataset...")
    loader = MNISTLoader(use_keras=True)
    loader.info()

    x_train, y_train = loader.get_train_data()
    x_test, y_test = loader.get_test_data()

    y_train_onehot = one_hot_encode(y_train, num_classes=10)

    model = build_model(
        input_size=x_train.shape[1],
        output_size=10,
        hidden_sizes=hidden_sizes,
    )

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    train_losses = []
    test_accuracies = []
    test_losses = []

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, x_train, y_train_onehot, batch_size)
        train_losses.append(train_loss)

        test_acc, test_loss = evaluate(model, x_test, y_test)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Test Accuracy: {test_acc:.4f} ({int(test_acc * 100)}%)")
        print(f"  Test Loss: {test_loss:.6f}")

    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Final Test Accuracy: {test_accuracies[-1]:.4f} ({int(test_accuracies[-1] * 100)}%)")
    print(f"Final Test Loss: {test_losses[-1]:.6f}")
    print(f"Best Test Accuracy: {max(test_accuracies):.4f} ({int(max(test_accuracies) * 100)}%)")

    np.save(DATA_DIR / "train_losses.npy", np.array(train_losses))
    np.save(DATA_DIR / "test_accuracies.npy", np.array(test_accuracies))
    np.save(DATA_DIR / "test_losses.npy", np.array(test_losses))

    model_save_path = PROJECT_ROOT / "mlp_model.pkl"
    model_data = {
        "dataset_mode": "mnist",
        "input_size": x_train.shape[1],
        "hidden_sizes": list(hidden_sizes),
        "output_size": 10,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_metrics": test_accuracies,
        "weights": model.get_weights(),
    }
    with model_save_path.open("wb") as file_handle:
        pickle.dump(model_data, file_handle)

    print("\n✓ Results saved to data/")
    print("  - train_losses.npy")
    print("  - test_accuracies.npy")
    print("  - test_losses.npy")
    print("✓ Model checkpoint saved to mlp_model.pkl")

    return model, train_losses, test_accuracies, test_losses


if __name__ == "__main__":
    main()
