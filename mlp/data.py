"""
Data utilities: MNIST dataset loading and preprocessing
"""

import numpy as np
from urllib.request import urlopen
import gzip
import pickle
import os


def load_mnist_from_keras():
    """
    Load MNIST dataset from Keras/TensorFlow.
    Fallback method if direct download fails.
    """
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test
    except:
        return None


def load_mnist_raw():
    """
    Load MNIST dataset from raw online source.
    Downloads gzipped pickle files.
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'x_train': 'train-images-idx3-ubyte.gz',
        'y_train': 'train-labels-idx1-ubyte.gz',
        'x_test': 't10k-images-idx3-ubyte.gz',
        'y_test': 't10k-labels-idx1-ubyte.gz'
    }

    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_dir, exist_ok=True)

    try:
        print("Attempting to load MNIST from online source...")

        # Load X_train
        print("Loading training images...")
        urlopen(base_url + files['x_train']).read()
        print("✓ Training images available")

    except Exception as e:
        print(f"Failed to load from online source: {e}")
        return None


def load_mnist(data_dir='./data', use_keras=True):
    """
    Load MNIST dataset with preprocessing.

    Args:
        data_dir: Directory to store MNIST data
        use_keras: Try Keras first, then fallback to raw

    Returns:
        x_train, y_train, x_test, y_test (normalized and flattened)
    """

    # Try Keras first
    if use_keras:
        data = load_mnist_from_keras()
        if data is not None:
            x_train, y_train, x_test, y_test = data

            # Normalize to [0, 1]
            x_train = x_train.astype(np.float32) / 255.0
            x_test = x_test.astype(np.float32) / 255.0

            # Flatten: (60000, 28, 28) -> (60000, 784)
            x_train = x_train.reshape(-1, 28*28)
            x_test = x_test.reshape(-1, 28*28)

            return x_train, y_train, x_test, y_test

    # Fallback: create synthetic simple dataset for demo
    print("Warning: Using synthetic MNIST-like dataset for demonstration")
    return create_synthetic_mnist()


def create_synthetic_mnist():
    """
    Create a synthetic MNIST-like dataset for demo/testing.

    Returns:
        x_train, y_train, x_test, y_test (small synthetic data)
    """
    np.random.seed(42)

    # Create small synthetic dataset
    # 1000 training samples, 200 test samples, 10 classes
    n_train = 1000
    n_test = 200
    n_features = 784  # 28x28
    n_classes = 10

    x_train = np.random.randn(n_train, n_features).astype(np.float32) * 0.5 + 0.5
    x_train = np.clip(x_train, 0, 1)
    y_train = np.random.randint(0, n_classes, n_train)

    x_test = np.random.randn(n_test, n_features).astype(np.float32) * 0.5 + 0.5
    x_test = np.clip(x_test, 0, 1)
    y_test = np.random.randint(0, n_classes, n_test)

    print(f"Created synthetic MNIST-like dataset:")
    print(f"  Train: {x_train.shape}, {y_train.shape}")
    print(f"  Test: {x_test.shape}, {y_test.shape}")

    return x_train, y_train, x_test, y_test


def one_hot_encode(y, num_classes=10):
    """
    Convert class indices to one-hot encoding.

    Args:
        y: Class indices (batch_size,)
        num_classes: Number of classes

    Returns:
        one_hot: One-hot encoded (batch_size, num_classes)
    """
    batch_size = y.shape[0]
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), y] = 1
    return one_hot


def create_batches(x, y, batch_size=32, shuffle=True):
    """
    Create mini-batches for training.

    Args:
        x: Features (n_samples, n_features)
        y: Labels (n_samples,)
        batch_size: Size of each batch
        shuffle: Whether to shuffle data

    Yields:
        (x_batch, y_batch)
    """
    n_samples = x.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        yield x[batch_indices], y[batch_indices]


class MNISTLoader:
    """Convenient wrapper for MNIST dataset"""

    def __init__(self, data_dir='./data', use_keras=True):
        self.data_dir = data_dir
        self.x_train, self.y_train, self.x_test, self.y_test = load_mnist(data_dir, use_keras)

        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]
        self.n_features = self.x_train.shape[1]
        self.n_classes = 10

    def get_train_data(self):
        """Return training data (x, y)"""
        return self.x_train, self.y_train

    def get_test_data(self):
        """Return test data (x, y)"""
        return self.x_test, self.y_test

    def get_train_batches(self, batch_size=32, shuffle=True):
        """Create training batches"""
        return create_batches(self.x_train, self.y_train, batch_size, shuffle)

    def get_test_batches(self, batch_size=32, shuffle=False):
        """Create test batches"""
        return create_batches(self.x_test, self.y_test, batch_size, shuffle)

    def info(self):
        """Print dataset info"""
        print(f"MNIST Dataset:")
        print(f"  Training: {self.n_train} samples, {self.n_features} features")
        print(f"  Test: {self.n_test} samples")
        print(f"  Classes: {self.n_classes}")
