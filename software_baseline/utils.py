import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def visualize_predictions(model, test_loader, device, num_samples=8):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # Plot images with true and predicted labels
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        image = images[i].cpu().squeeze().numpy()
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {labels[i].item()}\nPred: {predictions[i].item()}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig