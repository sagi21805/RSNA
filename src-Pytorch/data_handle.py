import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from torch import nn
from config import TRAIN_IMG_PATH, SHUFFLE, TARGET_CLASSES, TRAIN_LABEL_PATH, BATCH_SIZE

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



train_data = pd.read_csv(TRAIN_LABEL_PATH)
id_label_dict = {label[0]: label[1:-1] for label in train_data[TARGET_CLASSES].values}
train_img_paths = []
train_labels = []
for patient_id in os.listdir(TRAIN_IMG_PATH):
    for img_path in os.listdir(TRAIN_IMG_PATH + "/" + patient_id):
        train_img_paths.append(TRAIN_IMG_PATH + "/" + patient_id + "/" + img_path)
        train_labels.append(id_label_dict[int(train_img_paths[-1].split("/")[-2])]) 




# Iterate through the groups and split them, handling single-sample groups

class CustomDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = torch.tensor(cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2GRAY), dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define any image transformations you want to apply, here we also add augmentation. 
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256),   # Random crop and resize
    transforms.RandomHorizontalFlip(),    # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Create the datasets
def get_dataset():
    dataset_train = CustomDataset(train_img_paths, train_labels)
    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=10)
    return train_dataloader




# Define your dataset size and other configuration parameters
