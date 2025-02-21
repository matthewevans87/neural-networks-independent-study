from src.data.data_loader import get_data_loader
from src.models.mlp import MultiLayerPerceptron
import torch
import numpy as np

def evaluate_accuracy(model: MultiLayerPerceptron, data_loader):
    """Calculate accuracy on given data loader"""
    # Get all predictions in one tensor
    predictions = model.predict(data_loader)
    
    # Collect all labels
    all_labels = []
    for _, labels in data_loader:
        all_labels.append(labels)
    labels = torch.cat(all_labels)
    
    # Calculate accuracy
    correct = (predictions == labels).sum().item()
    total = len(labels)
    
    return 100 * correct / total

def main():
    # Initialize data loaders
    train_loader = get_data_loader(train=True)
    val_loader = get_data_loader(train=False)  # Using test set as validation set
    
    # Define network parameters
    
    # Initialize the MLP with single hidden layer
    model = MultiLayerPerceptron(
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.01
    )

    # Train the model
    print("Starting training...")
    model.train(epochs=20)
    
    # Evaluate final model
    accuracy = evaluate_accuracy(model, val_loader)
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()