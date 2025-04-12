import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from datetime import timedelta
import os

class TextDataset(Dataset):
    def __init__(self, text: str, sequence_length=50):
        chars = sorted(set(text))
        self.charToIdx = {ch: i for i, ch in enumerate(chars)}
        self.idxToChar = {i: ch for ch, i in self.charToIdx.items()}
        self.vocab_size = len(chars)
        self.data = [self.charToIdx[c] for c in text]
        self.seq_length = sequence_length
            
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x), torch.tensor(y)
    
    def get_text_from_path(path):
        with open(path, 'r') as f:
            text = f.read()
        return text

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch, sequence)
        embedded = self.embed(x)
        # embedded shape: (batch, sequence, embed_size)
        output, hidden = self.rnn(embedded, hidden)
        # output shape: (batch, sequence, hidden_size)
        output = self.fc(output)
        # output shape: (batch, sequence, vocab_size)
        return output, hidden
    

def save_model(model, path='model.pth'):
    """Save model weights and vocabulary information."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers
    }, path)
    print(f'Model saved to {path}')

def load_model(path='model.pth'):
    """Load model weights and vocabulary information."""
    if not os.path.exists(path):
        return None
    
    checkpoint = torch.load(path)
    vocab_size = checkpoint['vocab_size']
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint['num_layers']
    
    model = RNN(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded from {path}')
    return model

def generate_sample(model, dataset: TextDataset, start_text="The", length=100, device='cpu'):
    """Generate a sample text sequence using the trained model."""
    model.eval()
    input_sequence = torch.tensor([dataset.charToIdx[c] for c in start_text], device=device).unsqueeze(0)
    generated_text = start_text

    hidden = None
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_sequence, hidden)
            output = output[:, -1, :]  # Get the last character's output
            predicted_idx = output.argmax(dim=-1).item()
            generated_text += dataset.idx2char[predicted_idx]
            input_sequence = torch.tensor([[predicted_idx]], device=device)

    return generated_text

def train_model(model: RNN, dataset: TextDataset, epochs=10, batch_size=64, learning_rate=0.001, device='cpu'):
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate total batches and epochs for progress tracking
    total_batches = len(train_loader)
    total_epochs = epochs
    total_steps = total_batches * epochs
    start_time = time.time()
    
    # Progress tracking variables
    processed_steps = 0
    
    # Define loss function and optimizer
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_chars = 0
        total_chars = 0
        epoch_start_time = time.time()
        
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate top-1 accuracy
            _, predicted = outputs.max(1)
            correct_chars += (predicted == targets).sum().item()
            total_chars += targets.size(0)
            
            processed_steps += 1
            
            # Enhanced progress logging
            if i % 10 == 9:
                current_batch = i + 1
                epoch_progress = current_batch / total_batches * 100
                total_progress = (epoch * total_batches + current_batch) / (total_epochs * total_batches) * 100
                
                # Calculate time estimates
                elapsed_time = time.time() - start_time
                steps_per_sec = processed_steps / elapsed_time
                remaining_steps = total_steps - processed_steps
                estimated_remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                eta_str = str(timedelta(seconds=int(estimated_remaining_time)))
                
                perplexity = torch.exp(torch.tensor(running_loss / 10))
                accuracy = 100 * correct_chars / total_chars
                print(f'[{elapsed_str}] Epoch: {epoch+1}/{epochs} ({(epoch+1)/epochs*100:.1f}%) | '
                      f'Batch: {current_batch}/{total_batches} ({epoch_progress:.1f}%) | '
                      f'Total Progress: {total_progress:.1f}% | '
                      f'Loss: {running_loss/10:.3f} | Perplexity: {perplexity:.3f} | '
                      f'Accuracy: {accuracy:.2f}% | '
                      f'Rate: {steps_per_sec:.1f} batches/sec | '
                      f'ETA: {eta_str}')
                running_loss = 0.0
        
        # End of epoch statistics
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1} completed in {epoch_time:.2f}s\n')
        
        # Generate sample output after each epoch
        sample_text = generate_sample(model, dataset, start_text="The", length=100, device=device)
        print(f"Sample generated text after epoch {epoch+1}:\n{sample_text}\n")
    
    total_time = time.time() - start_time
    avg_batches_per_sec = processed_steps / total_time
    print(f'Finished Training in {str(timedelta(seconds=int(total_time)))}')
    print(f'Average processing speed: {avg_batches_per_sec:.1f} batches/sec')
    
    # Save the model after training
    save_model(model)
    
    return model

def evaluate_model(model, test_dataset):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, _ = model(inputs)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            
            total_loss += criterion(outputs_flat, targets_flat).item()
            
            # Calculate top-1 accuracy
            _, predicted = outputs_flat.max(1)
            correct_chars += (predicted == targets_flat).sum().item()
            total_chars += targets_flat.size(0)
    
    avg_loss = total_loss / len(test_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    accuracy = 100 * correct_chars / total_chars
    print(f'Test Loss: {avg_loss:.4f} | Perplexity: {perplexity:.3f} | Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy

# Main code
MODEL_PATH = 'model.pth'

# Try to load existing model first
model = load_model(MODEL_PATH)

if model is None:
    # No saved model found, train a new one
    train_dataset = TextDataset(TextDataset.get_text_from_path('./data/train.txt'))
    model = RNN(train_dataset.vocab_size)
    train_model(model, train_dataset, epochs=10, batch_size=64, learning_rate=0.002, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = TextDataset(TextDataset.get_text_from_path('./data/test.txt'))
    evaluate_model(model, test_dataset)
else:
    # Use loaded model and vocabulary
    test_dataset = TextDataset(TextDataset.get_text_from_path('./data/test.txt'))
    evaluate_model(model, test_dataset)
