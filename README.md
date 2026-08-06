# MLP from Scratch

A from-scratch implementation of a Multi-Layer Perceptron (MLP) for MNIST digit classification.

## Features

- **Dense Layers**: Fully connected feed-forward layers with configurable dimensions
- **Activations**: Sigmoid, Tanh, ReLU, Softmax
- **Loss Functions**: Cross-Entropy and MSE
- **Optimizers**: SGD and Adam
- **Training Pipeline**: End-to-end training, checkpoint serialization, and metric tracking
- **Evaluation Pipeline**: Checkpoint-based model reconstruction and classification metrics

## Project Structure

```
MLP/
├── mlp/
│   ├── __init__.py           # Package init
│   ├── layer.py              # Dense layer implementation
│   ├── network.py            # MLP composition
│   ├── activations.py        # Activation functions
│   ├── losses.py             # Loss functions
│   ├── optimizers.py         # Optimizers (SGD, Adam)
│   └── data.py               # MNIST loading and preprocessing
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── tests/
│   └── test_basics.py        # Unit tests
├── notebooks/
│   └── analysis.ipynb        # Analysis notebook
├── data/                     # Saved metrics and artifacts
├── mlp_model.pkl             # Trained model checkpoint (generated)
└── requirements.txt          # Dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. Train Model

```bash
python train.py
```

This will:
- Load MNIST (via Keras if available, otherwise synthetic fallback)
- Train an MLP classifier for 10 epochs
- Save metrics to `data/`
- Save checkpoint metadata and weights to `mlp_model.pkl`

You can also use the console script installed by the package:

```bash
mlp-train
```

### 3. Evaluate Model

```bash
python evaluate.py
```

This will:
- Load `mlp_model.pkl`
- Rebuild the exact training architecture from checkpoint metadata
- Run inference on the test set
- Print overall and per-class metrics

You can also use the console script installed by the package:

```bash
mlp-evaluate
```

### 4. Run Tests

```bash
python -m pytest tests/test_basics.py
```

Tests cover:
- Dense layer forward/backward consistency
- Activation derivative checks
- MLP forward shape checks
- Loss and optimizer sanity checks

## Architecture

Default classifier architecture:

```
Input (784)
  -> Dense(256, ReLU)
  -> Dense(128, ReLU)
  -> Dense(64, ReLU)
  -> Dense(10)
  -> Softmax (for probability output)
```

## Training Configuration

- Optimizer: Adam (learning rate: 0.001)
- Loss: Cross-Entropy
- Epochs: 10
- Batch size: 32

## Implementation Notes

- The output layer uses logits; softmax is applied outside the final layer.
- Checkpoints include architecture metadata to avoid train/eval mismatch.
- Metric histories are stored as NumPy arrays in `data/` for notebook analysis.

## Known Limitations

- No regularization (dropout, weight decay) by default
- No learning-rate scheduling
- Educational implementation; production systems should use frameworks like PyTorch or TensorFlow
