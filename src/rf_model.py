from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from pathlib import Path


class RFModel:
    """Random Forest wrapper for letter classification."""
    
    def __init__(self, n_estimators=150, max_depth=15, min_samples_split=10, 
                 min_samples_leaf=5, max_features='sqrt', n_jobs=-1, random_state=42):
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1
        )
        self.is_trained = False
    
    def fit(self, X_train, y_train):
       
        # Flatten images if needed
        if X_train.ndim == 4:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        print(f"Training Random Forest on {X_train_flat.shape[0]} samples...")
        self.model.fit(X_train_flat, y_train)
        self.is_trained = True
        print("Training complete!")
    
    def predict(self, X_test):
       
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Flatten images if needed
        if X_test.ndim == 4:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_flat = X_test
        
        return self.model.predict(X_test_flat)
    
    def predict_proba(self, X_test):
       
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Flatten images if needed
        if X_test.ndim == 4:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_flat = X_test
        
        return self.model.predict_proba(X_test_flat)
    
    def save(self, path):
        """Save model to disk."""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")
    
    def feature_importance(self):
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        return self.model.feature_importances_


if __name__ == "__main__":
    # Quick test
    import torch
    
    X_test = torch.randn(100, 1, 28, 28).numpy()
    y_test = np.random.randint(0, 26, 100)
    
    model = RFModel(n_estimators=10, max_depth=10)
    model.fit(X_test, y_test)
    preds = model.predict(X_test)
    print(f"Predictions shape: {preds.shape}")
    print(f"Accuracy: {(preds == y_test).mean():.4f}")