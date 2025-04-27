import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from torchvision import models

from data.dataset import get_tr_dataloaders



def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0


    for images, labels in train_loader:

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)


        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / len(train_loader.dataset)

    return running_loss / len(train_loader), acc









def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()

    # Calculate accuracy
    acc = 100. * correct / len(val_loader.dataset)
    return running_loss / len(val_loader), acc


def main():

    initial_lr = 0.001
    num_epochs = 40

    csv_path = "src/labeled_data.csv"
    csv_path2 = "src/labeled_data2.csv"
    images_path = "src/img_data"

    train_loader, val_loader = get_tr_dataloaders(
        csv_path,
        csv_path2,
        images_path,
        batch_size= 64
    )

    # Replace MyModel with ResNet18
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Adjust the final layer for 2 classes

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=initial_lr/10,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-5)

    history = []

    for epoch in range(num_epochs):
        if epoch < 5:
            lr = initial_lr/10 + (initial_lr - initial_lr/10)*epoch/5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()

        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)


        # Save metrics
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Print progress
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)

        # Save history to CSV
        pd.DataFrame(history).to_csv('training_history.csv', index=False)

        # Save model if it's the best validation accuracy so far
        if epoch == 0 or val_acc > max(h['val_accuracy'] for h in history[:-1] if 'val_accuracy' in h):
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    main()









