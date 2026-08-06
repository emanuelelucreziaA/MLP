"""
Evaluation script: Load trained model metadata and evaluate on test set.

Usage:
    python evaluate.py
    mlp-evaluate
"""

import pickle
from pathlib import Path

from mlp.builder import build_classifier
from mlp.data import MNISTLoader
from mlp.metrics import classification_metrics
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


def print_metrics(y_true, y_pred_proba, model_name="Model"):
    """Print detailed evaluation metrics."""
    metrics = classification_metrics(y_true, y_pred_proba)
    accuracy = metrics["accuracy"]
    cm = metrics["confusion_matrix"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    num_classes = cm.shape[0]

    print(f"\n{'=' * 70}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'=' * 70}")
    print(f"Overall Accuracy: {accuracy:.4f} ({int(accuracy * 100)}%)")
    print("\nPer-Class Metrics:")
    print(f"{'Class':<8} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 53)
    for c in range(num_classes):
        print(f"{c:<8} {precision[c]:<15.4f} {recall[c]:<15.4f} {f1[c]:<15.4f}")

    print("\nMacro Average:")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall: {metrics['macro_recall']:.4f}")
    print(f"  F1-Score: {metrics['macro_f1']:.4f}")

    return metrics


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
    model = build_classifier(
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

    y_pred = metrics["y_pred"]
    for i in range(min(10, x_test.shape[0])):
        true_class = y_test[i]
        pred_class = y_pred[i]
        confidence = y_pred_proba[i, pred_class]
        correct = "yes" if true_class == pred_class else "no"

        print(f"{true_class:<15} {pred_class:<15} {confidence:<15.4f} {correct:<10}")

    return model, metrics


if __name__ == "__main__":
    main()
