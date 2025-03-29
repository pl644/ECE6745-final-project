import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import config

def get_mnist_loaders():
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST(
        root=config.Train_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=config.Test_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.Batch_size,
        shuffle=True,
        num_workers=config.Num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.Batch_size,
        shuffle=False,
        num_workers=config.Num_workers
    )
    
    return train_loader, test_loader