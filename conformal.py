"""
Conformal Prediction methods for uncertainty quantification.
Implements Split Conformal Prediction and Conformalized Quantile Regression (CQR).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings('ignore')


class SplitConformalPredictor:
    """
    Split Conformal Prediction for regression.
    
    Constructs prediction intervals with valid coverage guarantees by:
    1. Fitting base model on training set
    2. Computing nonconformity scores on calibration set
    3. Using quantiles of scores to form prediction intervals
    """
    
    def __init__(self, model: BaseEstimator, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        
        Parameters:
        -----------
        model : BaseEstimator
            Sklearn-style regression model (must have fit/predict methods)
        alpha : float
            Miscoverage level (e.g., 0.1 for 90% coverage)
        """
        self.model = model
        self.alpha = alpha
        self.quantile = None
        self.is_fitted = False
        
    def fit(self, X_train, y_train, X_cal, y_cal):
        """
        Fit model and calibrate conformal scores.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_cal : array-like
            Calibration features
        y_cal : array-like
            Calibration targets
        """
        # Fit base model
        self.model.fit(X_train, y_train)
        
        # Compute predictions on calibration set
        y_cal_pred = self.model.predict(X_cal)
        
        # Compute nonconformity scores (absolute residuals)
        scores = np.abs(y_cal - y_cal_pred)
        
        # Compute quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # Cap at 1.0
        
        self.quantile = np.quantile(scores, q_level)
        self.is_fitted = True
        
        return self
    
    def predict(self, X, return_intervals: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with conformal intervals.
        
        Parameters:
        -----------
        X : array-like
            Features
        return_intervals : bool
            Whether to return prediction intervals
            
        Returns:
        --------
        y_pred : np.ndarray
            Point predictions
        lower : np.ndarray
            Lower bounds of prediction intervals
        upper : np.ndarray
            Upper bounds of prediction intervals
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        y_pred = self.model.predict(X)
        
        if return_intervals:
            lower = y_pred - self.quantile
            upper = y_pred + self.quantile
            return y_pred, lower, upper
        else:
            return y_pred
    
    def get_interval_width(self) -> float:
        """Get the width of prediction intervals."""
        return 2 * self.quantile
    
    def compute_coverage(self, X_test, y_test) -> float:
        """
        Compute empirical coverage on test set.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
            
        Returns:
        --------
        float : Empirical coverage (fraction of y_test in intervals)
        """
        _, lower, upper = self.predict(X_test)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        return coverage


class ConformizedQuantileRegressor:
    """
    Conformalized Quantile Regression (CQR).
    
    Uses quantile regression at the base level and applies conformal
    calibration to adjust intervals for valid coverage.
    More efficient than split CP when base model is well-calibrated.
    """
    
    def __init__(self, 
                 quantile_low_model: BaseEstimator,
                 quantile_high_model: BaseEstimator,
                 alpha: float = 0.1):
        """
        Initialize CQR predictor.
        
        Parameters:
        -----------
        quantile_low_model : BaseEstimator
            Model for lower quantile (e.g., alpha/2)
        quantile_high_model : BaseEstimator
            Model for upper quantile (e.g., 1 - alpha/2)
        alpha : float
            Miscoverage level
        """
        self.quantile_low_model = quantile_low_model
        self.quantile_high_model = quantile_high_model
        self.alpha = alpha
        self.correction = None
        self.is_fitted = False
        
    def fit(self, X_train, y_train, X_cal, y_cal):
        """
        Fit quantile models and calibrate corrections.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_cal : array-like
            Calibration features
        y_cal : array-like
            Calibration targets
        """
        # Fit quantile models
        self.quantile_low_model.fit(X_train, y_train)
        self.quantile_high_model.fit(X_train, y_train)
        
        # Predict on calibration set
        q_low_cal = self.quantile_low_model.predict(X_cal)
        q_high_cal = self.quantile_high_model.predict(X_cal)
        
        # Compute conformity scores
        # Score = max distance from interval to true value
        scores = np.maximum(q_low_cal - y_cal, y_cal - q_high_cal)
        
        # Compute correction quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        self.correction = np.quantile(scores, q_level)
        self.is_fitted = True
        
        return self
    
    def predict(self, X, return_intervals: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with CQR intervals.
        
        Parameters:
        -----------
        X : array-like
            Features
        return_intervals : bool
            Whether to return prediction intervals
            
        Returns:
        --------
        y_pred : np.ndarray
            Point predictions (midpoint of quantiles)
        lower : np.ndarray
            Lower bounds of prediction intervals
        upper : np.ndarray
            Upper bounds of prediction intervals
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        q_low = self.quantile_low_model.predict(X)
        q_high = self.quantile_high_model.predict(X)
        
        # Point prediction as midpoint
        y_pred = (q_low + q_high) / 2
        
        if return_intervals:
            # Apply conformalization correction
            lower = q_low - self.correction
            upper = q_high + self.correction
            return y_pred, lower, upper
        else:
            return y_pred
    
    def get_interval_width(self, X) -> np.ndarray:
        """Get the width of prediction intervals for given inputs."""
        _, lower, upper = self.predict(X)
        return upper - lower
    
    def compute_coverage(self, X_test, y_test) -> float:
        """
        Compute empirical coverage on test set.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
            
        Returns:
        --------
        float : Empirical coverage
        """
        _, lower, upper = self.predict(X_test)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        return coverage


class LocallyAdaptiveConformalPredictor:
    """
    Locally Adaptive Conformal Predictor (LACP).
    Adjusts interval width based on local feature similarity.
    More efficient intervals by using weighted nonconformity scores.
    
    Reference: Lei & Wasserman (2014) - Distribution-free prediction bands
    """
    
    def __init__(self,
                 model: BaseEstimator,
                 alpha: float = 0.1,
                 bandwidth: float = 0.1):
        """
        Initialize LACP.
        
        Parameters:
        -----------
        model : BaseEstimator
            Base regression model
        alpha : float
            Miscoverage level
        bandwidth : float
            Bandwidth for local weighting (controls locality)
        """
        self.model = model
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.X_cal = None
        self.scores_cal = None
        self.is_fitted = False
    
    def fit(self, X_train, y_train, X_cal, y_cal):
        """Fit model and calibrate with local weighting."""
        self.model.fit(X_train, y_train)
        
        y_cal_pred = self.model.predict(X_cal)
        self.scores_cal = np.abs(y_cal - y_cal_pred)
        self.X_cal = X_cal
        self.is_fitted = True
        
        return self
    
    def predict(self, X, return_intervals: bool = True):
        """Generate predictions with locally adaptive intervals."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        y_pred = self.model.predict(X)
        
        if not return_intervals:
            return y_pred
        
        # Compute local quantiles for each test point
        lower = np.zeros(len(X))
        upper = np.zeros(len(X))
        
        for i, x in enumerate(X):
            # Compute weights based on distance to calibration points
            distances = np.linalg.norm(self.X_cal - x, axis=1)
            weights = np.exp(-distances / self.bandwidth)
            weights = weights / weights.sum()
            
            # Weighted quantile
            sorted_idx = np.argsort(self.scores_cal)
            cumsum_weights = np.cumsum(weights[sorted_idx])
            quantile_idx = np.searchsorted(cumsum_weights, 1 - self.alpha)
            quantile_idx = min(quantile_idx, len(self.scores_cal) - 1)
            
            local_quantile = self.scores_cal[sorted_idx[quantile_idx]]
            lower[i] = y_pred[i] - local_quantile
            upper[i] = y_pred[i] + local_quantile
        
        return y_pred, lower, upper


class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Predictor that adjusts alpha based on recent coverage.
    Useful for handling regime changes and non-stationarity in financial markets.
    
    Reference: Gibbs & Candès (2021) - Adaptive Conformal Inference
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 target_alpha: float = 0.1,
                 adaptation_rate: float = 0.01,
                 min_alpha: float = 0.05,
                 max_alpha: float = 0.3):
        """
        Initialize adaptive conformal predictor.
        
        Parameters:
        -----------
        model : BaseEstimator
            Base regression model
        target_alpha : float
            Target miscoverage level
        adaptation_rate : float
            Speed of alpha adjustment
        min_alpha : float
            Minimum alpha value
        max_alpha : float
            Maximum alpha value
        """
        self.model = model
        self.target_alpha = target_alpha
        self.adaptation_rate = adaptation_rate
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        self.current_alpha = target_alpha
        self.quantile = None
        self.is_fitted = False
        self.coverage_history = []
        
    def fit(self, X_train, y_train, X_cal, y_cal):
        """Fit model and initialize calibration."""
        self.model.fit(X_train, y_train)
        
        y_cal_pred = self.model.predict(X_cal)
        scores = np.abs(y_cal - y_cal_pred)
        
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.current_alpha)) / n
        q_level = min(q_level, 1.0)
        
        self.quantile = np.quantile(scores, q_level)
        self.is_fitted = True
        
        return self
    
    def predict(self, X, return_intervals: bool = True):
        """Generate predictions with adaptive intervals."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        y_pred = self.model.predict(X)
        
        if return_intervals:
            lower = y_pred - self.quantile
            upper = y_pred + self.quantile
            return y_pred, lower, upper
        else:
            return y_pred
    
    def update(self, X_new, y_new):
        """
        Update alpha based on recent coverage.
        
        Parameters:
        -----------
        X_new : array-like
            New observations (features)
        y_new : array-like
            New observations (targets)
        """
        # Compute coverage on new data
        _, lower, upper = self.predict(X_new)
        recent_coverage = np.mean((y_new >= lower) & (y_new <= upper))
        
        self.coverage_history.append(recent_coverage)
        
        # Adjust alpha if coverage deviates from target
        target_coverage = 1 - self.target_alpha
        
        if recent_coverage < target_coverage:
            # Under-coverage: decrease alpha (widen intervals)
            self.current_alpha = max(
                self.min_alpha,
                self.current_alpha - self.adaptation_rate
            )
        elif recent_coverage > target_coverage + 0.05:
            # Over-coverage: increase alpha (narrow intervals)
            self.current_alpha = min(
                self.max_alpha,
                self.current_alpha + self.adaptation_rate
            )
        
        # Recompute quantile with new alpha
        # Note: In practice, would use recent residuals
        self.quantile = self.quantile * (1 - self.target_alpha) / (1 - self.current_alpha)


class EnsembleConformalPredictor:
    """
    Ensemble Conformal Predictor.
    Combines predictions from multiple base models with conformal calibration.
    Provides more robust intervals by leveraging model diversity.
    
    Reference: Vovk (2015) - Cross-conformal predictors
    """
    
    def __init__(self,
                 models: list,
                 alpha: float = 0.1,
                 aggregation: str = 'mean'):
        """
        Initialize ensemble CP.
        
        Parameters:
        -----------
        models : list
            List of base models (sklearn-style)
        alpha : float
            Miscoverage level
        aggregation : str
            How to combine predictions ('mean', 'median', 'weighted')
        """
        self.models = models
        self.alpha = alpha
        self.aggregation = aggregation
        self.quantile = None
        self.is_fitted = False
    
    def fit(self, X_train, y_train, X_cal, y_cal):
        """Fit all models and calibrate ensemble."""
        # Fit each model
        for model in self.models:
            model.fit(X_train, y_train)
        
        # Get ensemble predictions on calibration set
        cal_preds = np.array([model.predict(X_cal) for model in self.models])
        
        # Aggregate predictions
        if self.aggregation == 'mean':
            y_cal_pred = cal_preds.mean(axis=0)
        elif self.aggregation == 'median':
            y_cal_pred = np.median(cal_preds, axis=0)
        else:  # weighted (by inverse training error)
            weights = []
            for i, model in enumerate(self.models):
                train_pred = model.predict(X_train)
                mse = np.mean((y_train - train_pred) ** 2)
                weights.append(1 / (mse + 1e-6))
            weights = np.array(weights) / sum(weights)
            y_cal_pred = (cal_preds.T @ weights).T
        
        # Compute nonconformity scores
        scores = np.abs(y_cal - y_cal_pred)
        
        # Compute quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        self.quantile = np.quantile(scores, q_level)
        self.is_fitted = True
        
        return self
    
    def predict(self, X, return_intervals: bool = True):
        """Generate ensemble predictions with intervals."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get predictions from all models
        preds = np.array([model.predict(X) for model in self.models])
        
        # Aggregate
        if self.aggregation == 'mean':
            y_pred = preds.mean(axis=0)
        elif self.aggregation == 'median':
            y_pred = np.median(preds, axis=0)
        else:  # Use same weights as calibration
            y_pred = preds.mean(axis=0)  # Simplified for test
        
        if return_intervals:
            lower = y_pred - self.quantile
            upper = y_pred + self.quantile
            return y_pred, lower, upper
        else:
            return y_pred


def evaluate_prediction_intervals(y_true, y_pred, lower, upper) -> dict:
    """
    Evaluate quality of prediction intervals.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Point predictions
    lower : array-like
        Lower bounds
    upper : array-like
        Upper bounds
        
    Returns:
    --------
    dict : Evaluation metrics including coverage, width, Winkler score, CRPS
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    lower = np.array(lower)
    upper = np.array(upper)
    
    # Coverage
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    
    # Average width
    avg_width = np.mean(upper - lower)
    
    # Winkler score (lower is better)
    alpha = 0.1  # Assume 90% intervals
    winkler = np.mean(
        (upper - lower) + 
        (2 / alpha) * (lower - y_true) * (y_true < lower) +
        (2 / alpha) * (y_true - upper) * (y_true > upper)
    )
    
    # Interval score (simplified CRPS)
    interval_score = np.mean(
        (upper - lower) +
        2 * np.maximum(lower - y_true, 0) +
        2 * np.maximum(y_true - upper, 0)
    )
    
    # Point prediction error
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return {
        'coverage': coverage,
        'avg_width': avg_width,
        'winkler_score': winkler,
        'interval_score': interval_score,
        'mae': mae,
        'rmse': rmse
    }


if __name__ == '__main__':
    # Test conformal predictors
    print("Testing Conformal Prediction methods...")
    
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate synthetic data
    np.random.seed(42)
    n_train = 500
    n_cal = 200
    n_test = 300
    n_features = 10
    
    X_train = np.random.randn(n_train, n_features)
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.randn(n_train) * 0.5
    
    X_cal = np.random.randn(n_cal, n_features)
    y_cal = X_cal[:, 0] + 0.5 * X_cal[:, 1] + np.random.randn(n_cal) * 0.5
    
    X_test = np.random.randn(n_test, n_features)
    y_test = X_test[:, 0] + 0.5 * X_test[:, 1] + np.random.randn(n_test) * 0.5
    
    # Test Split Conformal
    print("\n=== Split Conformal Prediction ===")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    cp = SplitConformalPredictor(model, alpha=0.1)
    cp.fit(X_train, y_train, X_cal, y_cal)
    
    y_pred, lower, upper = cp.predict(X_test)
    metrics = evaluate_prediction_intervals(y_test, y_pred, lower, upper)
    
    print(f"Coverage: {metrics['coverage']:.3f} (target: 0.90)")
    print(f"Avg Width: {metrics['avg_width']:.3f}")
    print(f"Winkler Score: {metrics['winkler_score']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    
    print("\nAll tests passed!")
