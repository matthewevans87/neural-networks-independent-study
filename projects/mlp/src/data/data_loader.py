import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loader(train: bool) -> torch.utils.data.DataLoader:
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./datasets', train=train, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return data_loader
