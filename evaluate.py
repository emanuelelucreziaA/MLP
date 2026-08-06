"""
Evaluation script: Load trained model metadata and evaluate on test set.

Usage:
    python evaluate.py
    mlp-evaluate
"""

import pickle
from pathlib import Path

import numpy as np

from mlp.activations import relu, relu_derivative
from mlp.data import MNISTLoader
from mlp.layer import DenseLayer
from mlp.network import MLP
from mlp.runtime import initialize_environment

PROJECT_ROOT, _ = initialize_environment()


def load_model(model_path):
    """Load trained model checkpoint with metadata."""
    try:
        with Path(model_path).open("rb") as file_handle:
            model_data = pickle.load(file_handle)

        print("Model metadata loaded successfully")
        return model_data
    except FileNotFoundError:
        print(f"Model checkpoint not found at {model_path}")
        print("  First train the model using: python train.py")
        return None


def build_model(input_size=784, output_size=10, hidden_sizes=(256, 128, 64)):
    """Build MLP model from checkpoint metadata."""
    model = MLP()
    previous_size = input_size

    for hidden_size in hidden_sizes:
        model.add_layer(
            DenseLayer(
                previous_size,
                hidden_size,
                activation_fn=relu,
                activation_derivative=relu_derivative,
            )
        )
        previous_size = hidden_size

    model.add_layer(DenseLayer(previous_size, output_size, activation_fn=None, activation_derivative=None))
    return model


def confusion_matrix(y_true, y_pred, num_classes=10):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes))

    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1

    return cm


def print_metrics(y_true, y_pred_proba, model_name="Model"):
    """Print detailed evaluation metrics."""
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = np.mean(y_pred == y_true)
    cm = confusion_matrix(y_true, y_pred)

    precision = np.zeros(10)
    recall = np.zeros(10)
    f1 = np.zeros(10)

    for c in range(10):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'=' * 70}")
    print(f"Overall Accuracy: {accuracy:.4f} ({int(accuracy * 100)}%)")
    print("\nPer-Class Metrics:")
    print(f"{'Class':<8} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 53)
    for c in range(10):
        print(f"{c:<8} {precision[c]:<15.4f} {recall[c]:<15.4f} {f1[c]:<15.4f}")

    print("\nMacro Average:")
    print(f"  Precision: {np.mean(precision):.4f}")
    print(f"  Recall: {np.mean(recall):.4f}")
    print(f"  F1-Score: {np.mean(f1):.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def main():
    print("\n" + "=" * 70)
    print("MLP Evaluation: MNIST Classification")
    print("=" * 70)

    print("\nLoading MNIST dataset...")
    loader = MNISTLoader(use_keras=True)
    x_test, y_test = loader.get_test_data()

    model_path = PROJECT_ROOT / "mlp_model.pkl"
    model_data = load_model(model_path)
    if model_data is None:
        return

    if "weights" not in model_data or model_data["weights"] is None:
        print("No trained weights found in checkpoint.")
        print("  Train the model first using: python train.py")
        return

    hidden_sizes = tuple(model_data.get("hidden_sizes", [256, 128, 64]))
    model = build_model(
        input_size=model_data.get("input_size", 784),
        output_size=model_data.get("output_size", 10),
        hidden_sizes=hidden_sizes,
    )
    model.set_weights(model_data["weights"])

    print("\nRunning inference on test set...")
    y_pred_proba = model.predict_proba(x_test)

    metrics = print_metrics(y_test, y_pred_proba, "Trained MLP")

    if "test_metrics" in model_data and model_data["test_metrics"]:
        print("\n" + "=" * 70)
        print("Training History")
        print("=" * 70)
        print("\nTest Accuracy by Epoch:")
        for epoch, metric in enumerate(model_data["test_metrics"], start=1):
            print(f"  Epoch {epoch}: {metric:.4f}")

    if "test_losses" in model_data and model_data["test_losses"]:
        print("\nTest Loss by Epoch:")
        for epoch, loss in enumerate(model_data["test_losses"], start=1):
            print(f"  Epoch {epoch}: {loss:.4f}")

    print(f"\n{'=' * 70}")
    print("Sample Predictions (first 10 test samples):")
    print(f"{'=' * 70}")
    print(f"{'True Class':<15} {'Predicted':<15} {'Confidence':<15} {'Correct':<10}")
    print("-" * 55)

    y_pred = np.argmax(y_pred_proba, axis=1)
    for i in range(min(10, x_test.shape[0])):
        true_class = y_test[i]
        pred_class = y_pred[i]
        confidence = y_pred_proba[i, pred_class]
        correct = "yes" if true_class == pred_class else "no"

        print(f"{true_class:<15} {pred_class:<15} {confidence:<15.4f} {correct:<10}")

    return model, metrics


if __name__ == "__main__":
    main()
