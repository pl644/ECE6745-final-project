import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import os
import config
from model import CNNModel, initialize_weights
from dataset import get_mnist_loaders
from utils import evaluate_model, visualize_predictions

def train_fn(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    loop = tqdm.tqdm(train_loader)
    for index, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f"Training Loss: {total_loss/len(train_loader)}")
    print(f"Training Accuracy: {100 * correct / total:.2f}%")
    return total_loss/len(train_loader)

def main():
    # Create model directories
    os.makedirs("model", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize model
    model = CNNModel(num_classes=config.Num_classes).to(config.Device)
    initialize_weights(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Learning_rate, betas=(0.9, 0.999))
    
    # Get data loaders
    train_loader, test_loader = get_mnist_loaders()
    
    # Load model if specified
    if config.Load_model:
        print("Loading model")
        model.load_state_dict(torch.load('model/mnist_cnn.pth'))
        optimizer.load_state_dict(torch.load('model/mnist_optimizer.pth'))
    
    # Lists to store metrics
    train_losses = []
    test_accuracies = []
    
    for epoch in range(config.Num_epochs):
        print(f"Epoch: {epoch}/{config.Num_epochs}")
        
        # Train for one epoch
        train_loss = train_fn(model, train_loader, criterion, optimizer, config.Device)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, config.Device)
        test_accuracies.append(test_acc)
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Save model if specified
        if config.Save_model:
            print("Saving model")
            torch.save(model.state_dict(), 'model/mnist_cnn.pth')
            torch.save(optimizer.state_dict(), 'model/mnist_optimizer.pth')
        
        # Visualize predictions every few epochs
        if epoch % 2 == 0:
            with torch.no_grad():
                fig = visualize_predictions(model, test_loader, config.Device)
                plt.savefig(f"results/mnist_predictions_{epoch}.png")
                plt.close(fig)
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('results/training_metrics.png')

if __name__ == "__main__":
    main()