# Neural Networks from Scratch

A lightweight neural network framework built entirely with NumPy.

The goal of this project was to build and train neural networks without relying on high-level deep learning frameworks such as PyTorch or TensorFlow.

The framework includes layers, loss functions, optimizers, data loading utilities, checkpointing, learning rate scheduling, and evaluation tools implemented from scratch using NumPy.

**Fashion-MNIST Results**

* Best Validation Accuracy: **90.22%**
* Test Accuracy: **89.98%**

### Training Curves
<img width="65%" alt="training_curves_square_fixed" src="https://github.com/user-attachments/assets/22a5ebdf-f44b-4112-b698-41146c9c620a" />


---

## Features

### Layers

* Linear
* ReLU
* LeakyReLU
* Sigmoid
* Tanh
* Flatten
* Dropout

### Loss Functions

* Binary Cross Entropy (BCE)
* Softmax Cross Entropy
* L1 Loss

### Optimizers

* SGD
* SGD with Momentum
* Adam

### Data Utilities

* Dataset abstraction
* DataLoader with batching and shuffling
* Dataset splitting utilities
* Transform pipeline

### Training Utilities

* Training and validation loops
* Checkpointing
* Learning rate scheduling
* Training history tracking
* Model parameter serialization

### Evaluation Tools

* Accuracy computation
* Confusion matrix generation and visualization
* Misclassified sample visualization
* Prediction analysis utilities

---

## Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<username>/neural-networks-from-scratch.git
cd neural-networks-from-scratch
pip install -r requirements.txt
```

## Reproducing the Results

To train the Fashion-MNIST model, go through the `main.ipynb` notebook.

Before running the notebook, download the Fashion-MNIST dataset from the official source and update the dataset paths in the configuration section of the notebook.

---

## Example

```python
model = Network(
    layers=[
        Flatten(),
        Linear(784, 512),
        ReLU(),
        Linear(512, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10)
    ]
)

optimizer = Adam(
    model.parameters(),
    learning_rate=1e-3,
    weight_decay=5e-4
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_func=SoftmaxCrossEntropy(),
    optimizer=optimizer
)

history = trainer.train(epochs=30)
```

---

## Fashion-MNIST Benchmark

The framework was evaluated on the Fashion-MNIST dataset using a fully connected neural network.

### Model Architecture

```text
Flatten
→ Linear(784, 512)
→ ReLU
→ Linear(512, 256)
→ ReLU
→ Linear(256, 128)
→ ReLU
→ Linear(128, 10)
```

### Training Configuration

* Optimizer: Adam
* Learning Rate: 1e-3
* Weight Decay: 5e-4
* Batch Size: 128
* Epochs: 30
* Learning Rate Scheduler: Step Decay
* Dataset Split: 80% Train / 20% Validation

### Performance

| Metric                   | Value  |
| ------------------------ | ------ |
| Best Validation Accuracy | 90.22% |
| Test Accuracy            | 89.98% |

### Confusion Matrix

<img width="65%" alt="confusion_matrix" src="https://github.com/user-attachments/assets/1fd1fa3a-9358-4768-9502-0575bdf22d3a" />




### Misclassified Samples

<img width="65%" alt="wrong_predictions" src="https://github.com/user-attachments/assets/fbfd947b-a3bb-48e9-bb55-b3799a909d58" />



---

## Learning Objectives

This project was built to better understand:

* Forward propagation
* Backpropagation
* Gradient computation
* Neural network optimization
* Weight initialization
* Regularization techniques
* Training and evaluation workflows
* Software design for machine learning systems

