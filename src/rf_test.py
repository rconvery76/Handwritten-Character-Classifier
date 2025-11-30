"""
Testing script for Random Forest classifier.
Mirrors the structure of the CNN test.py for easy comparison.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import string
from pathlib import Path
from .rf_model import RFModel
from .train import PTShardDataset
from .data import OUTPUT_DIR


def evaluate_rf(model, test_loader, device='cpu'):
    """
    Evaluate Random Forest model.
    
    Args:
        model: Trained RFModel instance
        test_loader: DataLoader for test set
        device: Ignored (RF doesn't use GPU)
        
    Returns:
        overall_acc: Overall accuracy
        correct_by_class: Dict with per-class accuracy stats
    """
    total = 0
    correct = 0
    correct_by_class = {}
    
    print("Evaluating Random Forest...")
    batch_count = 0
    
    for x, y in test_loader:
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches...")
        
        # Convert to numpy
        x_np = x.numpy()
        y_np = y.numpy()
        
        # Make predictions
        preds = model.predict(x_np)
        
        # Compute accuracy
        correct += (preds == y_np).sum()
        total += len(y_np)
        
        # Per-class accuracy
        for true_label, pred_label in zip(y_np, preds):
            true_label = int(true_label)
            if true_label not in correct_by_class:
                correct_by_class[true_label] = {'correct': 0, 'total': 0}
            
            if true_label == int(pred_label):
                correct_by_class[true_label]['correct'] += 1
            correct_by_class[true_label]['total'] += 1
    
    overall_acc = correct / total
    print(f"Test Accuracy: {overall_acc * 100:.2f}%")
    
    return overall_acc, correct_by_class


def plot_per_class_accuracy(correct_by_class, num_classes=26):
    """Create bar plot of per-class accuracies."""
    letters = list(string.ascii_uppercase)
    acc = []
    
    for i in range(num_classes):
        if i in correct_by_class:
            stats = correct_by_class[i]
            class_acc = stats['correct'] / stats['total']
        else:
            class_acc = 0.0
        acc.append(class_acc * 100)
    
    plt.figure(figsize=(14, 5))
    plt.bar(range(num_classes), acc, color='steelblue')
    plt.xticks(range(num_classes), letters)
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Random Forest - Test Accuracy by Class')
    plt.ylim([0, 100])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return acc


if __name__ == '__main__':
    print("=" * 60)
    print("Random Forest Evaluation")
    print("=" * 60)
    
    # Load test data
    print("\n1. Loading test data...")
    _, _, test_loader = PTShardDataset.create_dataloader(
        OUTPUT_DIR, 
        batch_train=256, 
        batch_eval=512
    )
    
    # Load trained model
    print("\n2. Loading trained model...")
    model = RFModel()
    model.load('best_rf_model.joblib')
    
    # Evaluate
    print("\n3. Evaluating on test set...")
    overall_acc, correct_by_class = evaluate_rf(model, test_loader)
    
    # Print per-class accuracy
    print("\n4. Per-class accuracy:")
    print("-" * 50)
    for cls in range(26):
        if cls in correct_by_class:
            stats = correct_by_class[cls]
            class_acc = stats['correct'] / stats['total']
            letter = chr(ord('A') + cls)
            print(f"   Class {cls:2d} ({letter}): {class_acc * 100:6.2f}% " + 
                  f"({stats['correct']:4d}/{stats['total']:4d})")
    
    # Plot results
    print("\n5. Creating visualization...")
    acc = plot_per_class_accuracy(correct_by_class)
    plt.show()
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)