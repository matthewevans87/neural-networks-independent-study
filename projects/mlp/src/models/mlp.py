import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class MultiLayerPerceptron:
    def __init__(self, train_loader, val_loader=None, learning_rate=0.01):
        """
        Initialize MLP with a single hidden layer
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            learning_rate (float): Learning rate for gradient descent
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        
        
        self.input_size = train_loader.dataset.data.shape[1] * train_loader.dataset.data.shape[2]  # 28 * 28 = 784
        self.hidden_size = 196  # Single hidden layer size
        self.output_size = len(train_loader.dataset.classes)  # Number of classes determined from dataset
        
        # Initialize single hidden layer architecture
        scaling_factor = 0.01
        self.W1 = nn.Parameter(torch.randn(self.input_size, self.hidden_size) * scaling_factor)
        self.W2 = nn.Parameter(torch.randn(self.hidden_size, self.output_size) * scaling_factor)
        self.b1 = nn.Parameter(torch.zeros(self.hidden_size))
        self.b2 = nn.Parameter(torch.zeros(self.output_size))
        
        # Initialize parameter list for optimization
        self.parameters = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        """
        Forward propagation through single hidden layer
        """
        # Reshape input from [batch_size, 1, 28, 28] to [batch_size, 784]
        x = x.view(x.size(0), -1)
        
        # Hidden layer with ReLU activation
        hidden = F.relu(x @ self.W1 + self.b1)
        # Output layer (no activation yet)
        output = hidden @ self.W2 + self.b2
        return output, [x, hidden]

    def backward(self, x, y_true, activations):
        """
        Backward propagation through single hidden layer
        """
        y_pred = F.log_softmax(activations[-1], dim=1)
        loss = F.nll_loss(y_pred, y_true)
        
        # Zero all gradients before backward pass
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
        
        # Compute gradients
        loss.backward()
        
        # Update weights and biases
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param -= self.learning_rate * param.grad
        
        return loss.item()

    def train(self, epochs=10):
        """
        Train the model using the DataLoader provided in constructor
        Args:
            epochs (int): Number of training epochs
        """
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_x, batch_y in self.train_loader:
                output, activations = self.forward(batch_x)
                loss = self.backward(batch_x, batch_y, activations)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validate if validation loader is available
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

    def validate(self):
        """
        Validate the model using the validation DataLoader
        Returns:
            float: Average validation loss
        """
        if self.val_loader is None:
            return None
            
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                output, activations = self.forward(batch_x)
                y_pred = F.log_softmax(activations[-1], dim=1)
                loss = F.nll_loss(y_pred, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

    def predict(self, test_loader):
        """
        Make predictions using a test DataLoader
        Args:
            test_loader (DataLoader): Test data loader
        Returns:
            torch.Tensor: Predictions
        """
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                output, _ = self.forward(batch_x)
                pred = torch.argmax(F.softmax(output, dim=1), dim=1)
                predictions.append(pred)
        
        return torch.cat(predictions)