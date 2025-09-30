"""
Machine Learning models for return forecasting.
Implements XGBoost and MLP (Multi-Layer Perceptron) with standardization.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """XGBoost regressor for return forecasting."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42):
        """
        Initialize XGBoost forecaster.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Step size shrinkage
        subsample : float
            Subsample ratio of training instances
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Fit the model."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            objective='reg:squarederror',
            verbosity=0
        )
        
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Generate predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.model is None:
            return None
        return self.model.feature_importances_


class MLPNet(nn.Module):
    """Multi-layer perceptron network."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        """
        Initialize MLP network.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dims : list of int
            Hidden layer dimensions
        """
        super(MLPNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass."""
        return self.network(x).squeeze()


class MLPForecaster(BaseEstimator, RegressorMixin):
    """MLP regressor for return forecasting."""
    
    def __init__(self,
                 hidden_dims: list = [64, 32, 16],
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 n_epochs: int = 100,
                 early_stopping: int = 10,
                 device: str = None,
                 random_state: int = 42):
        """
        Initialize MLP forecaster.
        
        Parameters:
        -----------
        hidden_dims : list of int
            Hidden layer dimensions
        learning_rate : float
            Learning rate for Adam optimizer
        batch_size : int
            Batch size for training
        n_epochs : int
            Maximum number of epochs
        early_stopping : int
            Patience for early stopping
        device : str
            'cuda', 'mps', 'cpu', or None (auto-detect)
        random_state : int
            Random seed
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.random_state = random_state
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """Fit the model."""
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y).to(self.device)
        
        # Initialize model
        input_dim = X_scaled.shape[1]
        self.model = MLPNet(input_dim, self.hidden_dims).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            self.model.train()
            
            # Mini-batch training
            indices = np.random.permutation(len(X_scaled))
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping:
                break
        
        return self
    
    def predict(self, X):
        """Generate predictions."""
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions


class EnsembleForecaster(BaseEstimator, RegressorMixin):
    """Ensemble of multiple forecasters."""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        """
        Initialize ensemble forecaster.
        
        Parameters:
        -----------
        models : list
            List of forecaster instances
        weights : list, optional
            Weights for each model (defaults to equal weights)
        """
        self.models = models
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            self.weights = weights
    
    def fit(self, X, y):
        """Fit all models."""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        """Generate weighted average predictions."""
        predictions = np.zeros(len(X))
        
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)
        
        return predictions


def create_default_models():
    """
    Create default set of forecasting models.
    
    Returns:
    --------
    dict : Dictionary of model name -> model instance
    """
    models = {
        'xgboost': XGBoostForecaster(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        'mlp': MLPForecaster(
            hidden_dims=[64, 32, 16],
            learning_rate=0.001,
            batch_size=256,
            n_epochs=100,
            early_stopping=10
        ),
        'ensemble': None  # Will be created after training individual models
    }
    
    return models


if __name__ == '__main__':
    # Test models
    print("Testing forecasting models...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = X_train[:, 0] * 0.1 + X_train[:, 1] * 0.05 + np.random.randn(n_samples) * 0.02
    
    X_test = np.random.randn(200, n_features)
    y_test = X_test[:, 0] * 0.1 + X_test[:, 1] * 0.05 + np.random.randn(200) * 0.02
    
    # Test XGBoost
    print("\nTesting XGBoost...")
    xgb_model = XGBoostForecaster()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(np.mean((xgb_pred - y_test) ** 2))
    print(f"XGBoost RMSE: {xgb_rmse:.6f}")
    
    # Test MLP
    print("\nTesting MLP...")
    mlp_model = MLPForecaster(n_epochs=50)
    mlp_model.fit(X_train, y_train)
    mlp_pred = mlp_model.predict(X_test)
    mlp_rmse = np.sqrt(np.mean((mlp_pred - y_test) ** 2))
    print(f"MLP RMSE: {mlp_rmse:.6f}")
    
    # Test Ensemble
    print("\nTesting Ensemble...")
    ensemble = EnsembleForecaster([xgb_model, mlp_model])
    ensemble_pred = ensemble.predict(X_test)
    ensemble_rmse = np.sqrt(np.mean((ensemble_pred - y_test) ** 2))
    print(f"Ensemble RMSE: {ensemble_rmse:.6f}")
    
    print("\nAll models tested successfully!")
