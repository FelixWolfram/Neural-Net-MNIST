# MNIST Neural Network from Scratch

A complete implementation of a neural network for handwritten digit recognition, built from scratch using only NumPy. This project demonstrates the fundamentals of deep learning without relying on high-level frameworks like TensorFlow or PyTorch.

## 🚀 Features

- **From-scratch implementation**: Pure NumPy neural network with backpropagation
- **MNIST digit classification**: Recognizes handwritten digits (0-9)
- **Model persistence**: Save and load trained models
- **High accuracy**: Achieves ~95% accuracy on test data

## 📋 Requirements

- Python 3.8+
- uv (for dependency management)

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/FelixWolfram/Neural-Net-MNIST/tree/main
   cd Neuronales_Netz
   ```

2. **Install dependencies with uv:**

   ```bash
   uv sync
   ```

3. **Download MNIST dataset:**
   - Place `mnist_train.csv` and `mnist_test.csv` in `mnist_neural_net/data/`
   - You can download from [Kaggle MNIST](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

## 🎯 Usage

### Training a New Model

```python
# Open mnist_neural_net/number_recognition.ipynb
# Configure hyperparameters:
learning_rate = 0.3
epochs = 1000
neuron_layers = [784, 64, 32, 10]  # Input, Hidden1, Hidden2, Output

# Run training cells to train the model
```

### Testing the Model

```python
# Evaluate on test set
predict(test_X, test_Y, w, b)

# Test individual predictions
predict_index(42)  # Shows image and prediction for index 42
```

### Key Hyperparameters

- **Architecture**: `[784, 64, 32, 10]` (customizable)
- **Learning Rate**: `0.3` (adjust for convergence)
- **Batch Size**: `1024` (for mini-batch training)
- **Epochs**: `1000` (stop when validation loss plateaus)

## 🗂️ Project Structure

```
├── mnist_neural_net/
│   ├── number_recognition.ipynb    # Main training notebook
│   ├── data/                       # MNIST dataset (not in repo)
│   └── models/                     # Saved model weights
├── README.md
├── pyproject.toml                  # uv configuration
└── .gitignore
```

## 🧮 Mathematical Foundation

The neural network implements:

- **Forward Propagation**: `z = W·x + b`, `a = σ(z)`
- **Backpropagation**: Chain rule for gradient computation
- **Softmax + Cross-Entropy**: For multi-class classification
- **Sigmoid Activation**: For hidden layers

## 🎓 Learning Objectives

This project demonstrates:

- ✅ Neural network fundamentals
- ✅ Gradient descent optimization
- ✅ Backpropagation algorithm
- ✅ Overfitting prevention techniques
- ✅ Model evaluation and visualization
