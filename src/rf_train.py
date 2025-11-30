"""
Training script for Random Forest classifier.
Uses the same data loading pipeline as the CNN (PTShardDataset).
"""

import torch
import numpy as np
from pathlib import Path
from .rf_model import RFModel
from .train import PTShardDataset
from .data import OUTPUT_DIR


def load_all_data_to_memory(loader, num_batches=None):
    """
    Load all data from DataLoader into memory as numpy arrays.
    
    Args:
        loader: PyTorch DataLoader
        num_batches: Max number of batches to load (for testing)
        
    Returns:
        X: All images, shape (N, 1, 28, 28) -> will be flattened by RF
        y: All labels, shape (N,)
    """
    X_list = []
    y_list = []
    
    for batch_idx, (x, y) in enumerate(loader):
        X_list.append(x.numpy())
        y_list.append(y.numpy())
        
        if num_batches and batch_idx >= num_batches - 1:
            break
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Loaded {batch_idx + 1} batches...")
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return X, y


if __name__ == '__main__':
    print("=" * 60)
    print("Random Forest Training Pipeline")
    print("=" * 60)
    
    # Load data using the same pipeline as CNN
    print("\n1. Loading data from shards...")
    train_loader, val_loader, test_loader = PTShardDataset.create_dataloader(
        OUTPUT_DIR, 
        batch_train=256,  # Larger batches for faster loading
        batch_eval=512
    )
    
    print("\n2. Converting PyTorch DataLoaders to numpy arrays...")
    X_train, y_train = load_all_data_to_memory(train_loader)
    X_val, y_val = load_all_data_to_memory(val_loader)
    X_test, y_test = load_all_data_to_memory(test_loader)
    
    print(f"   Training set: {X_train.shape} images, {y_train.shape} labels")
    print(f"   Validation set: {X_val.shape} images, {y_val.shape} labels")
    print(f"   Test set: {X_test.shape} images, {y_test.shape} labels")
    
    # Combine training and validation for final model
    # (Random Forest doesn't need a separate validation set during training)
    print("\n3. Combining training and validation data...")
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    print(f"   Combined training set: {X_combined.shape} images")
    
    # Train Random Forest
    print("\n4. Training Random Forest...")
    model = RFModel(
        n_estimators=150,           # Reduced from 200 (fewer trees = less memorization)
        max_depth=15,               # Reduced from 25 (shallower trees = less memorization)
        min_samples_split=10,       # NEW: Don't split on tiny groups (prevents overfitting)
        min_samples_leaf=5,         # NEW: Ensure leaves have minimum samples (smoothing)
        max_features='sqrt',        # NEW: Use ~27 out of 784 features per split (diversity)
        n_jobs=-1,                  # Use all CPU cores
        random_state=42
    )
    model.fit(X_combined, y_combined)
    
    # Evaluate on validation set
    print("\n5. Evaluating on validation set...")
    val_preds = model.predict(X_val)
    val_acc = (val_preds == y_val).mean()
    print(f"   Validation Accuracy: {val_acc * 100:.2f}%")
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_preds = model.predict(X_test)
    test_acc = (test_preds == y_test).mean()
    print(f"   Test Accuracy: {test_acc * 100:.2f}%")
    
    # Per-class accuracy
    print("\n7. Per-class accuracy on test set:")
    print("-" * 40)
    for cls in range(26):
        mask = y_test == cls
        if mask.sum() > 0:
            class_acc = (test_preds[mask] == y_test[mask]).mean()
            letter = chr(ord('A') + cls)
            print(f"   Class {cls:2d} ({letter}): {class_acc * 100:6.2f}% ({mask.sum():5d} samples)")
    
    # Save model
    print("\n8. Saving model...")
    model.save('best_rf_model.joblib')
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)