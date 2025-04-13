import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from datetime import timedelta
import os
import argparse

class TextDataset(Dataset):
    def __init__(self, text: str, sequence_length=50, charToIdx: dict[str, int]=None, idxToChar: dict[int, str]=None):
        chars = sorted(set(text))
        self.charToIdx = charToIdx if charToIdx is not None else {ch: i for i, ch in enumerate(chars)}
        self.idxToChar = idxToChar if idxToChar is not None else {i: ch for ch, i in self.charToIdx.items()}
        self.vocab_size = len(self.charToIdx)
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
    def __init__(self, vocab_size: int, embedding_dim=128, hidden_dim=256, num_layers=2):
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
    

def save_model(model: RNN, path='model.pth'):
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

def generate_sample(model: RNN, dataset: TextDataset, start_text="The", length=100, device='cpu'):
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
            generated_text += dataset.idxToChar[predicted_idx]
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

def evaluate_model(model, test_dataset, device='cpu'):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0
    correct_chars = 0
    correct_top5_chars = 0  # Counter for top-5 accuracy
    correct_top10_chars = 0  # Counter for top-10 accuracy
    total_chars = 0
    
    # Progress tracking variables
    total_batches = len(test_loader)
    start_time = time.time()
    
    print(f"Evaluating model on {len(test_dataset)} examples ({total_batches} batches)...")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            
            total_loss += criterion(outputs_flat, targets_flat).item()
            
            # Calculate top-1 accuracy
            _, predicted = outputs_flat.max(1)
            correct_chars += (predicted == targets_flat).sum().item()
            
            # Calculate top-5 accuracy
            _, top5_predicted = outputs_flat.topk(5, dim=1)
            correct_top5 = torch.zeros_like(targets_flat).bool()
            for k in range(5):
                correct_top5 = correct_top5 | (top5_predicted[:, k] == targets_flat)
            correct_top5_chars += correct_top5.sum().item()
            
            # Calculate top-10 accuracy
            _, top10_predicted = outputs_flat.topk(10, dim=1)
            correct_top10 = torch.zeros_like(targets_flat).bool()
            for k in range(10):
                correct_top10 = correct_top10 | (top10_predicted[:, k] == targets_flat)
            correct_top10_chars += correct_top10.sum().item()
            
            total_chars += targets_flat.size(0)
            
            # Report progress every 10 batches
            if i % 10 == 9 or i == total_batches - 1:
                current_batch = i + 1
                progress = current_batch / total_batches * 100
                
                # Calculate time estimates
                elapsed_time = time.time() - start_time
                batches_per_sec = current_batch / elapsed_time if elapsed_time > 0 else 0
                remaining_batches = total_batches - current_batch
                eta = remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
                
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                eta_str = str(timedelta(seconds=int(eta)))
                
                # Current metrics
                current_loss = total_loss / total_chars
                current_perplexity = torch.exp(torch.tensor(current_loss))
                current_accuracy = 100 * correct_chars / total_chars
                current_top5_accuracy = 100 * correct_top5_chars / total_chars
                current_top10_accuracy = 100 * correct_top10_chars / total_chars
                
                print(f'[{elapsed_str}] Batch: {current_batch}/{total_batches} ({progress:.1f}%) | '
                      f'Loss: {current_loss:.4f} | Perplexity: {current_perplexity:.3f} | '
                      f'Top-1: {current_accuracy:.2f}% | Top-5: {current_top5_accuracy:.2f}% | '
                      f'Top-10: {current_top10_accuracy:.2f}% | ETA: {eta_str}')
    
    # Calculate final per-character loss and perplexity
    avg_loss = total_loss / total_chars  # Per-character loss
    perplexity = torch.exp(torch.tensor(avg_loss))
    accuracy = 100 * correct_chars / total_chars
    top5_accuracy = 100 * correct_top5_chars / total_chars
    top10_accuracy = 100 * correct_top10_chars / total_chars
    
    total_time = time.time() - start_time
    print(f'\nEvaluation completed in {str(timedelta(seconds=int(total_time)))}')
    print(f'Final Test Loss: {avg_loss:.4f} | Perplexity: {perplexity:.3f}')
    print(f'Top-1 Accuracy: {accuracy:.2f}% | Top-5: {top5_accuracy:.2f}% | Top-10: {top10_accuracy:.2f}%')
    
    return avg_loss, accuracy, top5_accuracy, top10_accuracy

def main():
    parser = argparse.ArgumentParser(description="RNN Text Generation")
    parser.add_argument("mode", choices=["train", "evaluate", "generate"], nargs="?", default="evaluate", help="Mode of operation: train, evaluate, or generate (default: generate)")
    args = parser.parse_args()

    MODEL_PATH = 'model.pth'

    if args.mode == "train":
        train_dataset = TextDataset(TextDataset.get_text_from_path('./data/train.txt'))
        model = load_model(MODEL_PATH)
        if model is None:
            model = RNN(train_dataset.vocab_size)
        train_model(model, train_dataset, epochs=10, batch_size=64, learning_rate=0.002, device='cuda' if torch.cuda.is_available() else 'cpu')
        test_dataset = TextDataset(TextDataset.get_text_from_path('./data/test.txt'))
        evaluate_model(model, test_dataset)

    elif args.mode == "evaluate":
        model = load_model(MODEL_PATH)
        if model is None:
            print("No saved model found. Please train the model first.")
            return
        train_dataset = TextDataset(TextDataset.get_text_from_path('./data/train.txt'))
        test_dataset = TextDataset(TextDataset.get_text_from_path('./data/test.txt'), train_dataset.seq_length, train_dataset.charToIdx, train_dataset.idxToChar)
        evaluate_model(model, test_dataset)

    elif args.mode == "generate":
        model = load_model(MODEL_PATH)
        if model is None:
            print("No saved model found. Please train the model first.")
            return
        train_dataset = TextDataset(TextDataset.get_text_from_path('./data/train.txt'))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        print("Entering text generation mode (REPL). Type '-r' to choose a random seed, '-e' to quit.")
        while True:
            seed_phrase = input("Enter a seed phrase: ")
            if seed_phrase.lower() == "-e":
                break
            if seed_phrase.lower() == "-r":
                seed_phrase = train_dataset.idxToChar[torch.randint(0, train_dataset.vocab_size, (1,)).item()]
                print(f"Random seed phrase: {seed_phrase}")
            try:
                sequence_length = int(input("Enter sequence length: "))
                generated_text = generate_sample(model, train_dataset, start_text=seed_phrase, length=sequence_length, device=device)
                print(f"Generated text:\n{generated_text}\n")
            except ValueError:
                print("Invalid sequence length. Please enter an integer.")

if __name__ == "__main__":
    main()
