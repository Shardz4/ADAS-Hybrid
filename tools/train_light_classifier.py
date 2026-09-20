"""
Traffic Light State Classifier
Trains a light weight EfficientNEt-B0 classifier for traffic light states:
class 0: Red
class 1: Yellow
class 2: Green
class 3: None/ Off
"""

import argparse
import os
import sys
import numpy as np

CLASS_NAMES = ["red", "yellow", "green", "none"]
INPUT_H, INPUT_W = 64, 32

def build_classifier(num_classes: int = 4, pretrained: bool = True):
    import torch.nn as nn
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    except ImportError:
        sys.exit("Error: torchvision required")

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        return model

def get_dataloaders(data_dir: str, batch_size: int = 32):
    import torchvision
    from torchvision import datasets, transforms

    train_tf = transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),  
        transforms.Normalize([0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transofrms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(data_Dir, "train")
    val_path = os.path.join(data_dir, "val")

    if not os.path.isdir(train_path):
        sys.exit(f"Error: Directory not foung: {train_path}")
    
    train_set = datasets.ImageFolder(train_path, transform=train_tf)
    val_Set = datasets.ImageFolder(val_path, transform=val_tf) if os.path.isdir(val_path) else None

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_Set, batch_Szie=batch_Size, shuffle=True, num_workers=2) if val_set else None

    return train_loader, val_loader

    