import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
Z_DIM = 100
BATCH_SIZE = 100
LEARNING_RATE = 0.0002
EPOCHS = 200
IMAGE_SIZE = 28*28

# Data loader for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    
# Generator network
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
        
        # self.model = nn.Sequential(
        #     nn.Linear(latent_dim, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.Linear(512, 1024),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.Linear(1024, 28*28),
        #     nn.Tanh()
        # )
    def forward(self, x):
        # img = self.model(z)
        # return img.view(z.size(0), 1, 28, 28)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, d_input_dim, dropout_prob=0.3):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        # self.model = nn.Sequential(
        #     nn.Flatten(),
            
        #     nn.Linear(28*28, 1024),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(dropout_prob),
            
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(dropout_prob),
            
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(dropout_prob),
            
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # return self.model(img)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

# Initialize networks
G = Generator(g_input_dim=Z_DIM, g_output_dim=IMAGE_SIZE).to(device)
D = Discriminator(d_input_dim=IMAGE_SIZE).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE) #betas=(0.5, 0.999)
optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE) #betas=(0.5, 0.999)

def D_train(x):
    D.zero_grad()
    
    x_real, y_real = x.view(-1, IMAGE_SIZE).to(device), torch.ones(x.size(0), 1, device=device)
    d_real_output = D(x_real)
    d_real_loss = criterion(d_real_output, y_real)
    
    z = torch.randn(x.size(0), Z_DIM, device=device)
    
    x_fake, y_fake = G(z), torch.zeros(x.size(0), 1, device=device)
    d_fake_output = D(x_fake)
    d_fake_loss = criterion(d_fake_output, y_fake)
    
    d_loss = (d_real_loss + d_fake_loss)
    d_loss.backward()
    optimizer_D.step()
    
    return d_loss

def G_train(x):
    G.zero_grad()
    # optimizer_G.zero_grad()
    # Unclear if I should be calling G.zero_grad() instead
    
    z = torch.randn(x.size(0), Z_DIM).to(device)
    g_output = G(z)
    d_output = D(g_output)
    
    
    # G's loss is how well it fooled D
    # i.e., how many fakes did D classify as 1:
    y = torch.ones(x.size(0), 1, device=device)
    g_loss = criterion(d_output, y)
    
    # Only move G forward
    g_loss.backward()
    optimizer_G.step()
    
    return g_loss

# Training loop
for epoch in range(EPOCHS):
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)

        g_loss = G_train(x)
        d_loss = D_train(x)


        if batch_idx % 200 == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {batch_idx}/{len(train_loader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    if (epoch + 1) % 1 == 0 or epoch == EPOCHS - 1:
        # Generate sample images and save to disk
        with torch.no_grad():
            z = torch.randn(16, Z_DIM, device=device)
            sample_imgs = G(z).cpu()
        grid = utils.make_grid(sample_imgs.view(-1,1,28,28), nrow=4, normalize=True)

        os.makedirs("images", exist_ok=True)

        plt.figure(figsize=(5,5))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), interpolation='nearest')
        plt.axis('off')
        plt.savefig(f"images/epoch_{epoch+1}.png", bbox_inches='tight')
        plt.close()
    

