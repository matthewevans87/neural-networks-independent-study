from sklearn.datasets import fetch_openml
import torch
import json

# Load MNIST dataset
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff', cache=True)
       
    # Extract features and target
    X, y = mnist[0], mnist[1]

    # Convert Pandas DataFrame & Series directly to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32) / 255.0 # Normalize pixel values to [0, 1]
    y_tensor = torch.tensor(y.astype(int), dtype=torch.long)  # Convert labels to integers

    # Split into train and test sets
    X_train, X_test = X_tensor[:60000], X_tensor[60000:]
    y_train, y_test = y_tensor[:60000], y_tensor[60000:]
    return X_train, y_train, X_test, y_test

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Assuming input x is already sigmoid-activated

def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdims=True).values)  # Stability trick
    return exp_x / torch.sum(exp_x, dim=1, keepdims=True)

# One-hot encoding
def one_hot(y, num_classes):
    return torch.eye(num_classes)[y]

# Initialize weights and biases
def initialize_network(input_size, hidden_size, output_size):
    torch.random.manual_seed(42)
    scaling_factor = 0.01
    W1 = torch.randn(input_size, hidden_size) * scaling_factor
    b1 = torch.zeros((1, hidden_size))
    W2 = torch.randn(hidden_size, output_size) * scaling_factor
    b2 = torch.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward pass
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Backward pass
def backward(X, Y, Z1, A1, Z2, A2, W1, b1, W2, b2, learning_rate):
    m = X.shape[0]
    
    # Compute gradients
    dZ2 = A2 - Y
    dW2 = (A1.T @ dZ2) # / m
    db2 = torch.sum(dZ2, dim=0, keepdims=True) / m
    dZ1 = (dZ2 @ W2.T) * sigmoid_derivative(A1)
    dW1 = (X.T @ dZ1) # / m
    db1 = torch.sum(dZ1, dim=0, keepdims=True) / m

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2

# Compute accuracy
def compute_accuracy(X, Y, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    predictions = torch.argmax(A2, dim=1)
    labels = torch.argmax(Y, dim=1)
    return (predictions == labels).float().mean()

# Training loop
def train_mlp(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, epochs=10, batch_size=64, learning_rate=0.1):
    W1, b1, W2, b2 = initialize_network(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        losses = []
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            Z1, A1, Z2, A2 = forward(X_batch, W1, b1, W2, b2)
            W1, b1, W2, b2 = backward(X_batch, Y_batch, Z1, A1, Z2, A2, W1, b1, W2, b2, learning_rate)
            loss = torch.mean(-torch.sum(Y_batch * torch.log(A2 + 1e-10), dim=1))  # Avoid log(0) by adding a small constant
            losses.append(loss)
            
        loss = torch.mean(torch.tensor(losses))
        
        print(f"Loss: {loss:.4f}")
        train_acc = compute_accuracy(X_train, Y_train, W1, b1, W2, b2)
        test_acc = compute_accuracy(X_test, Y_test, W1, b1, W2, b2)
        print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    return W1, b1, W2, b2

# Load data
X_train, Y_train, X_test, Y_test = load_mnist()

# Convert labels to one-hot encoding
Y_train = one_hot(Y_train, 10)
Y_test = one_hot(Y_test, 10)

# Train the model
W1, b1, W2, b2 = train_mlp(X_train, Y_train, X_test, Y_test, input_size=784, hidden_size=20, output_size=10, epochs=10, batch_size=64, learning_rate=0.1)

# Save model weights and biases as JSON
weights = {
    "W_hidden": W1.tolist(),
    "W_output": W2.tolist(),
    "b_hidden": b1.tolist(),
    "b_output": b2.tolist()
}

with open("weights.json", "w") as f:
    json.dump(weights, f)
