import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

class MLP(nn.Module):
    def __init__(self, hidden_units=20):
        super(MLP, self).__init__()
        # Input layer to hidden layer: 28x28 = 784 input features -> hidden_units
        self.fc1 = nn.Linear(28 * 28, hidden_units)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Hidden layer to output layer: hidden_units -> 10 classes (digits 0-9)
        self.fc2 = nn.Linear(hidden_units, 10)
        # Softmax is included in CrossEntropyLoss, so we don't need it here

    def forward(self, x):
        # Reshape input images to [batch_size, 784]
        x = x.view(-1, 28 * 28)
        # First fully connected layer with ReLU
        x = self.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x


def train(model, epochs=10, batch_size=32, hidden_units=20, learning_rate=0.01):
    # Create data loaders
    train_dataset, _ = get_mnist_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    # return model

def predict(model, test_dataset):
    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation for evaluation
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)
            
            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.tolist())
            
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    return predictions

# Main code
train_dataset, test_dataset = get_mnist_dataset()
model = MLP(hidden_units=20)
train(model)
predictions = predict(model, test_dataset)
