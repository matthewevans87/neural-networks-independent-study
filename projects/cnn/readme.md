# Convolutional Neural Network

This project demonstrates a simple convolutional neural network trained to predict the digits of the MNIST dataset.

## Progression of Network Architecture

This project includes an pytorch based implementation of a multi-layer perceptron (MLP) network which was previously "manually" implemented (without pytorch), see `mlp.py`.

The pytorch MLP implementation was then copied to serve as the base of a convolutional neural network (CNN), see `cnn.py`.

## CNN Architecture

| Layer  | Type            | Dimensions | Parameters  | Notes                                       |
| ------ | --------------- | ---------- | ----------- | ------------------------------------------- |
| Input  | Image           | 28×28×1    | -           | MNIST dataset grayscale images              |
| Conv1  | Convolutional   | 14×14×8    | 3×3 kernels | 8 feature maps with 3×3 receptive fields    |
| Hidden | Fully Connected | 120        | -           | Flattened features from convolutional layer |
| Output | Fully Connected | 10         | -           | One neuron per digit (0-9)                  |

## Training and Tuning

The network uses cross-entropy loss and was optimized using stochastic gradient descent in batches (i.e., mini-batch). The fixed set of training examples was shuffled on each epoch. Input pixel values were normalized according to the following.

```
normalized_pixel = (pixel - mean) / std
```

Weights are initialized using the `xavier_uniform_` which samples from a uniform distribution `U(-a, a)` where

```
a = sqrt(6 / (fan_in + fan_out)).
```

Where `fan_in` and `fan_out` are the number of connections into and out of a given layer.

Biases are initialized to `0`.

The following hyper-parameters were used.

| Hyperparameter | Value |
| -------------- | ----- |
| Epochs         | 10    |
| Batch Size     | 32    |
| Learning Rate  | 0.01  |

## Results

The model achieved an accuracy of 97.99% on the test set.
