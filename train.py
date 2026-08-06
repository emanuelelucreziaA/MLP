"""
Training script: Train MLP on MNIST classification task.

Usage:
    python train.py
    mlp-train
"""

import pickle

import numpy as np

from mlp.builder import build_classifier
from mlp.data import MNISTLoader, create_batches, one_hot_encode
from mlp.losses import CrossEntropy
from mlp.metrics import compute_accuracy
from mlp.optimizers import Adam
from mlp.runtime import initialize_environment

PROJECT_ROOT, DATA_DIR = initialize_environment(ensure_data_dir=True)


def build_model(input_size=784, output_size=10, hidden_sizes=(256, 128, 64)):
    """Build MLP architecture for MNIST classification."""
    print("\n" + "=" * 70)
    arch = [input_size] + list(hidden_sizes) + [output_size]
    print("Building MLP Model for MNIST")
    print(f"Architecture: {' -> '.join(str(v) for v in arch)}")
    print("=" * 70)

    model = build_classifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
    )

    model.set_optimizer(Adam(learning_rate=0.001))
    model.set_loss(CrossEntropy())

    model.summary()
    return model


def train_epoch(model, x_train, y_train_onehot, batch_size=32):
    """
    Train model for one epoch with mini-batch gradient descent.

    Returns:
        average_loss: Average loss across batches
    """
    n_samples = x_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    total_loss = 0.0

    for batch_idx, (x_batch, y_batch) in enumerate(
        create_batches(x_train, y_train_onehot, batch_size=batch_size, shuffle=True)
    ):

        y_pred_softmax = model.predict_proba(x_batch)

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
    """Evaluate model on test set."""
    y_pred_softmax = model.predict_proba(x_test)

    accuracy = compute_accuracy(y_test, y_pred_softmax)
    loss = model.loss_fn(one_hot_encode(y_test, y_pred_softmax.shape[1]), y_pred_softmax)

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
    n_classes = int(np.max(y_train)) + 1

    y_train_onehot = one_hot_encode(y_train, num_classes=n_classes)

    model = build_model(
        input_size=x_train.shape[1],
        output_size=n_classes,
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
        "output_size": n_classes,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_metrics": test_accuracies,
        "weights": model.get_weights(),
    }
    with model_save_path.open("wb") as file_handle:
        pickle.dump(model_data, file_handle)

    print(f"\nResults saved to {DATA_DIR}/")
    print(f"  - {DATA_DIR / 'train_losses.npy'}")
    print(f"  - {DATA_DIR / 'test_accuracies.npy'}")
    print(f"  - {DATA_DIR / 'test_losses.npy'}")
    print(f"Model checkpoint saved to {model_save_path}")

    return model, train_losses, test_accuracies, test_losses


if __name__ == "__main__":
    main()
