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

class CNN(nn.Module):
    def __init__(self, hidden_units=20, num_feature_maps=8, kernel_size=3, stride=1):
        super(CNN, self).__init__()
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_feature_maps, 
                               kernel_size=kernel_size, stride=stride, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Calculate the size of the output from conv+pool layers
        # For MNIST (28x28 images), after conv with padding=1 and kernel=3, 
        # the size remains 28x28, after pooling with kernel=2, it becomes 14x14
        conv_output_size = ((28 - kernel_size + 2*1) // stride) + 1  # With padding=1
        pool_output_size = conv_output_size // 2  # After max pooling with kernel=2
        fc1_input_size = num_feature_maps * pool_output_size * pool_output_size
        
        # First fully connected layer
        self.fc1 = nn.Linear(fc1_input_size, hidden_units)
        # Output layer
        self.fc2 = nn.Linear(hidden_units, 10)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]
        x = self.relu(self.conv1(x))  # Apply convolution and ReLU
        x = self.pool(x)             # Apply pooling
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)


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
model = CNN(hidden_units=20)
train(model)
predictions = predict(model, test_dataset)
