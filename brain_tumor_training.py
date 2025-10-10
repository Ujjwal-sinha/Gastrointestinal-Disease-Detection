#!/usr/bin/env python3
"""
Brain Tumor Classification Model Training
Using CNN for Medical Image Analysis

This script provides a complete pipeline for training a brain tumor classification model
with comprehensive analysis and metrics visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class BrainTumorDataset(Dataset):
    """Custom dataset class for brain tumor images"""

    def __init__(self, data_dir, split='Training', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        # Load all images and labels
        for label_idx, class_name in enumerate(config['classes']):
            class_dir = os.path.join(data_dir, split, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class Results:
    """Results class for compatibility with existing visualization functions"""

    def __init__(self, history):
        self.results_dict = {
            'train/box_loss': history['train_loss'],
            'val/box_loss': history['val_loss'],
            'metrics/precision': [acc/100 for acc in history['train_acc']],  # Convert to 0-1 scale
            'metrics/recall': [acc/100 for acc in history['val_acc']],       # Convert to 0-1 scale
            'metrics/mAP50': [acc/100 for acc in history['val_acc']],        # Use val acc as proxy for mAP
            'metrics/accuracy': history['val_acc']
        }


def setup_config(args):
    """Setup configuration parameters"""
    config = {
        'model_name': args.model_name,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'classes': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'data_dir': args.data_dir,
        'num_classes': 4
    }
    return config


def setup_data_transforms(config):
    """Setup data augmentation transforms"""
    train_transform = A.Compose([
        A.Resize(config['img_size'], config['img_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(config['img_size'], config['img_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


def create_datasets_and_loaders(config, train_transform, val_transform):
    """Create datasets and data loaders"""
    train_dataset = BrainTumorDataset(config['data_dir'], 'Training', train_transform)
    val_dataset = BrainTumorDataset(config['data_dir'], 'Testing', val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    return train_dataset, val_dataset, train_loader, val_loader


def setup_model(config):
    """Initialize and setup the model"""
    # Initialize CNN model (ResNet50)
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config['num_classes'])

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)

    print(f"Using device: {device}")
    print(f"Model: {config['model_name']}")
    print(f"Number of classes: {config['num_classes']}")

    return model, device


def train_model(model, train_loader, val_loader, config, device):
    """Train the model"""
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Training loop
    best_val_acc = 0.0
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Step scheduler
        scheduler.step()

        print(f'Epoch {epoch+1}/{config["epochs"]}:')
        print(f'  Train Loss: {train_loss".4f"}, Train Acc: {train_acc".2f"}%')
        print(f'  Val Loss: {val_loss".4f"}, Val Acc: {val_acc".2f"}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_brain_tumor_model.pth')
            print(f'  Best model saved with validation accuracy: {best_val_acc".2f"}%')

    # Save final model
    torch.save(model.state_dict(), 'brain_tumor_model_final.pth')

    return history


def plot_training_metrics(results):
    """Plot training metrics"""
    metrics = ['train/box_loss', 'val/box_loss', 'metrics/precision', 'metrics/recall', 'metrics/mAP50']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        if idx < len(axes):
            axes[idx].plot(results.results_dict[metric])
            axes[idx].set_title(metric)
            axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, val_loader, device):
    """Evaluate model and return predictions"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_metrics(y_true, y_pred, class_names):
    """Plot class-wise performance metrics"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()

    # Plot precision, recall, and F1-score for each class
    plt.figure(figsize=(15, 6))
    x = np.arange(len(class_names))
    width = 0.25

    plt.bar(x - width, df.loc[class_names, 'precision'], width, label='Precision')
    plt.bar(x, df.loc[class_names, 'recall'], width, label='Recall')
    plt.bar(x + width, df.loc[class_names, 'f1-score'], width, label='F1-score')

    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Class')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('class_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save classification report
    df.to_csv('classification_report.csv')
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Brain Tumor Classification Training')
    parser.add_argument('--data_dir', type=str, default='./Dataset', help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model name')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=78, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    # Setup configuration
    global config
    config = setup_config(args)

    # Check if dataset exists
    if not os.path.exists(config['data_dir']):
        print(f"Warning: Dataset directory {config['data_dir']} not found!")
        print("Please ensure your Dataset folder is in the correct location.")
        return

    # List dataset structure
    for split in ['Training', 'Testing']:
        split_path = os.path.join(config['data_dir'], split)
        if os.path.exists(split_path):
            print(f"\n{split} data:")
            for class_name in config['classes']:
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    num_images = len(os.listdir(class_path))
                    print(f"  {class_name}: {num_images} images")
                else:
                    print(f"  {class_name}: Directory not found")
        else:
            print(f"\n{split} directory not found: {split_path}")

    # Setup data transforms
    train_transform, val_transform = setup_data_transforms(config)

    # Create datasets and loaders
    train_dataset, val_dataset, train_loader, val_loader = create_datasets_and_loaders(
        config, train_transform, val_transform
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Setup model
    model, device = setup_model(config)

    # Train model
    print("\nStarting training...")
    history = train_model(model, train_loader, val_loader, config, device)

    # Create results object
    results = Results(history)

    # Plot training metrics
    print("\nPlotting training metrics...")
    plot_training_metrics(results)

    # Evaluate model
    print("\nEvaluating model...")
    val_predictions, val_targets = evaluate_model(model, val_loader, device)

    # Plot results
    print("\nGenerating evaluation plots...")
    plot_confusion_matrix(val_targets, val_predictions, config['classes'])
    plot_class_metrics(val_targets, val_predictions, config['classes'])

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history.csv', index=False)

    print("
Final Validation Accuracy:", history['val_acc'][-1])
    print("Best Validation Accuracy:", max(history['val_acc']))
    print("\nTraining completed! Model and results saved.")


if __name__ == "__main__":
    main()
