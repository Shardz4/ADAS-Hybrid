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


def train(data_dir: str, epochs: int, batch_Size: int, lr: float, device: str):
    import torch
    import torch.nn as nn

    print(f"State Classifier")
    print(f"Data dir : {data_dir} | Resolution: {INPUT_H}X{INPUT_W} | Epochs: {epochs} | device: {device}")
    model = build_classifier(num_classes=4, pretrained=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = lr, weight_decay = 1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosingAnnealingLR(optimizer, T_max=epochs)

    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    best_acc = 0.0
    best_weights = os.path.join("runs", "light_cls", "best.pt")
    os.makedirs(os.path.dirname(best_weights), exist_ok=True)


    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0.0, 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(ouptuts, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size()
        
        train_acc = (correct / total) * 100
        val_acc = 0.0

        if val_loader:
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_frad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    _, preds = outputs.max(1)
                    v_correct += preds.eq(labels).sum().item()
                    v_total += labels.size(0)
                val_acc = (v_correct / v_total) * 100

                if val_Acc > best_Acc:
                    best_Acc = val_acc
                    torch.save(model.state_dict(), best_weights)
            scheduler.step()
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss/total:.4f}|"
            f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% (Best:{best_acc:.1f}%)")

        print(f"\n trianing complete checkpoint saved: {best_weights}")
        return best_weights

        
