import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import tensor
import torch


class MyDataset(Dataset):
    def __init__(self, csv_file, image_folder, is_train=True):
        self.image_folder = image_folder
        self.data=pd.read_csv(csv_file)

        if is_train:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
                transforms.Resize((64, 64)),
                transforms.RandomApply([
                    transforms.RandomRotation(3),
                    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03), shear=2)
                ], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization for 3 channels
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization for 3 channels
            ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['img_name']
        label = self.data.iloc[idx]['label']
        img_path = os.path.join(self.image_folder, img_name)

        label_mapping = {'normal': 0, 'pneumonia': 1}  # Map string labels to integers
        label = label_mapping[label]  # Convert label to numeric value

        try:
            img = Image.open(img_path)
            img = self.transform(img)
            label = tensor(int(label), dtype=torch.long)  # Convert label to integer before tensor
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise

def get_tr_dataloaders(tr_csv_file, val_csv_file, image_folder, batch_size):
    train_dataset = MyDataset(tr_csv_file, image_folder, is_train=True)
    val_dataset = MyDataset(val_csv_file, image_folder, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_test_dataloaders(test_csv_file, image_folder, batch_size):
    test_dataset = MyDataset(test_csv_file, image_folder, is_train=False)
    test_loader =DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader