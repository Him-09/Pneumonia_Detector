import torch
import torch.nn as nn
import pandas as pd
from torchvision import models
from data.dataset import get_test_dataloaders
import os





def evaluate(model, test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:

            outputs = model(images)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()

    # Calculate accuracy
    acc = 100. * correct / len(test_loader.dataset)
    return acc


def main():
    # Set paths
    csv_path = "src/test_labeled_data.csv"
    image_folder = "src/img_data"
    
    
    # Get test data loader
    test_loader = get_test_dataloaders(csv_path, image_folder, batch_size=64)
    
    # Initialize model
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer to match the training architecture
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Adjust for 2 classes
    
    # Load best model
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint)
        print("Loaded best model successfully")
    else:
        print("No saved model found. Please train the model first.")
        return
    
    # Evaluate on test set
    acc = evaluate(model, test_loader)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    main()