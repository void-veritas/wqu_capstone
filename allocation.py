"""
Portfolio allocation strategies.
Implements classical methods (Mean-Variance, Risk Parity, ML-only) and
conformal prediction-aware methods (CP-Gate, CP-Size, CP-Lower-Bound).
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class BaseAllocator:
    """Base class for portfolio allocation strategies."""
    
    def __init__(self, 
                 vol_target: float = 0.10,
                 max_turnover: float = 0.20,
                 max_weight: float = 0.30,
                 min_weight: float = 0.0):
        """
        Initialize allocator.
        
        Parameters:
        -----------
        vol_target : float
            Target annualized portfolio volatility
        max_turnover : float
            Maximum portfolio turnover per rebalance
        max_weight : float
            Maximum weight per asset
        min_weight : float
            Minimum weight per asset
        """
        self.vol_target = vol_target
        self.max_turnover = max_turnover
        self.max_weight = max_weight
        self.min_weight = min_weight
        
    def allocate(self, **kwargs) -> np.ndarray:
        """
        Compute portfolio weights.
        
        Returns:
        --------
        np.ndarray : Portfolio weights (sum to 1)
        """
        raise NotImplementedError
    
    def apply_constraints(self, 
                         weights: np.ndarray,
                         prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply portfolio constraints (turnover, position limits).
        
        Parameters:
        -----------
        weights : np.ndarray
            Proposed weights
        prev_weights : np.ndarray, optional
            Previous weights for turnover constraint
            
        Returns:
        --------
        np.ndarray : Constrained weights
        """
        n_assets = len(weights)
        
        # CVXPY optimization
        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.sum_squares(w - weights))
        
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight
        ]
        
        # Add turnover constraint if previous weights provided
        if prev_weights is not None:
            constraints.append(
                cp.norm1(w - prev_weights) <= self.max_turnover
            )
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status == 'optimal':
            return w.value
        else:
            # Fallback: just normalize and clip
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / weights.sum()
            return weights


class MeanVarianceAllocator(BaseAllocator):
    """
    Mean-Variance (Markowitz) portfolio optimization.
    Maximizes expected return for given risk or minimizes risk for given return.
    """
    
    def __init__(self, 
                 risk_aversion: float = 1.0,
                 **kwargs):
        """
        Initialize Mean-Variance allocator.
        
        Parameters:
        -----------
        risk_aversion : float
            Risk aversion parameter (higher = more risk-averse)
        """
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion
        
    def allocate(self, 
                 expected_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute mean-variance optimal weights.
        
        Parameters:
        -----------
        expected_returns : np.ndarray
            Expected returns for each asset
        cov_matrix : np.ndarray
            Covariance matrix
        prev_weights : np.ndarray, optional
            Previous weights
            
        Returns:
        --------
        np.ndarray : Optimal weights
        """
        n_assets = len(expected_returns)
        
        # CVXPY optimization
        w = cp.Variable(n_assets)
        
        # Objective: maximize return - risk_aversion * variance
        portfolio_return = expected_returns @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_variance)
        
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight
        ]
        
        if prev_weights is not None:
            constraints.append(cp.norm1(w - prev_weights) <= self.max_turnover)
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status == 'optimal':
            weights = w.value
        else:
            # Fallback: equal weight
            weights = np.ones(n_assets) / n_assets
        
        # Apply volatility target
        weights = self._apply_vol_target(weights, cov_matrix)
        
        return weights
    
    def _apply_vol_target(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Scale weights to achieve volatility target."""
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        if portfolio_vol > 1e-6:
            scale = self.vol_target / portfolio_vol
            scale = min(scale, 1.5)  # Don't over-leverage
            weights = weights * scale
            
            # Renormalize if needed
            if weights.sum() > 0:
                weights = weights / weights.sum()
        
        return weights


class RiskParityAllocator(BaseAllocator):
    """
    Risk Parity allocation.
    Allocates capital so each asset contributes equally to portfolio risk.
    """
    
    def allocate(self,
                 cov_matrix: np.ndarray,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute risk parity weights.
        
        Parameters:
        -----------
        cov_matrix : np.ndarray
            Covariance matrix
        prev_weights : np.ndarray, optional
            Previous weights
            
        Returns:
        --------
        np.ndarray : Risk parity weights
        """
        n_assets = cov_matrix.shape[0]
        
        # Objective: minimize sum of squared differences in risk contribution
        def risk_parity_objective(w):
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            marginal_contrib = cov_matrix @ w
            risk_contrib = w * marginal_contrib / (portfolio_vol + 1e-8)
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        bounds = [(self.min_weight, self.max_weight)] * n_assets
        
        # Initial guess: inverse volatility
        vols = np.sqrt(np.diag(cov_matrix))
        w0 = 1 / (vols + 1e-8)
        w0 = w0 / w0.sum()
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
        else:
            # Fallback: inverse volatility
            weights = w0
        
        # Apply turnover constraint if needed
        if prev_weights is not None:
            weights = self.apply_constraints(weights, prev_weights)
        
        # Apply volatility target
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        if portfolio_vol > 1e-6:
            scale = self.vol_target / portfolio_vol
            scale = min(scale, 1.5)
            weights = weights * scale
            weights = weights / weights.sum()
        
        return weights


class MLOnlyAllocator(BaseAllocator):
    """
    ML-Only allocation.
    Weights proportional to ML forecasted returns (with constraints).
    """
    
    def allocate(self,
                 forecasts: np.ndarray,
                 cov_matrix: Optional[np.ndarray] = None,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute ML-based weights.
        
        Parameters:
        -----------
        forecasts : np.ndarray
            ML return forecasts
        cov_matrix : np.ndarray, optional
            Covariance matrix for vol targeting
        prev_weights : np.ndarray, optional
            Previous weights
            
        Returns:
        --------
        np.ndarray : ML-based weights
        """
        # Weights proportional to positive forecasts
        weights = np.maximum(forecasts, 0)
        
        if weights.sum() < 1e-6:
            # All negative forecasts: equal weight or cash
            weights = np.ones(len(forecasts)) / len(forecasts)
        else:
            weights = weights / weights.sum()
        
        # Apply constraints
        weights = self.apply_constraints(weights, prev_weights)
        
        # Apply volatility target if covariance provided
        if cov_matrix is not None:
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_vol > 1e-6:
                scale = self.vol_target / portfolio_vol
                scale = min(scale, 1.5)
                weights = weights * scale
                weights = weights / weights.sum()
        
        return weights


class CPGateAllocator(BaseAllocator):
    """
    CP-Gate allocation.
    Set weight to zero if prediction interval straddles zero (uncertain direction).
    Otherwise, weight proportional to forecast.
    """
    
    def allocate(self,
                 forecasts: np.ndarray,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 cov_matrix: Optional[np.ndarray] = None,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute CP-Gate weights.
        
        Parameters:
        -----------
        forecasts : np.ndarray
            Point forecasts
        lower_bounds : np.ndarray
            Lower bounds of prediction intervals
        upper_bounds : np.ndarray
            Upper bounds of prediction intervals
        cov_matrix : np.ndarray, optional
            Covariance matrix
        prev_weights : np.ndarray, optional
            Previous weights
            
        Returns:
        --------
        np.ndarray : CP-Gate weights
        """
        # Gate: zero weight if interval contains zero
        gate = ~((lower_bounds <= 0) & (upper_bounds >= 0))
        
        # Weights proportional to forecast (gated)
        weights = forecasts * gate
        weights = np.maximum(weights, 0)
        
        if weights.sum() < 1e-6:
            # All gated out: equal weight
            weights = np.ones(len(forecasts)) / len(forecasts)
        else:
            weights = weights / weights.sum()
        
        # Apply constraints
        weights = self.apply_constraints(weights, prev_weights)
        
        # Volatility targeting
        if cov_matrix is not None:
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_vol > 1e-6:
                scale = self.vol_target / portfolio_vol
                scale = min(scale, 1.5)
                weights = weights * scale
                weights = weights / weights.sum()
        
        return weights


class CPSizeAllocator(BaseAllocator):
    """
    CP-Size allocation.
    Weight inversely proportional to prediction interval width.
    Allocates more to assets with higher forecast precision.
    
    Note: Uses cross-sectional standardization to amplify differences
    when interval widths are similar across assets.
    """
    
    def __init__(self, use_standardization: bool = True, **kwargs):
        """
        Initialize CP-Size allocator.
        
        Parameters:
        -----------
        use_standardization : bool
            Whether to standardize widths cross-sectionally
        """
        super().__init__(**kwargs)
        self.use_standardization = use_standardization
    
    def allocate(self,
                 forecasts: np.ndarray,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 cov_matrix: Optional[np.ndarray] = None,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute CP-Size weights.
        
        Parameters:
        -----------
        forecasts : np.ndarray
            Point forecasts
        lower_bounds : np.ndarray
            Lower bounds
        upper_bounds : np.ndarray
            Upper bounds
        cov_matrix : np.ndarray, optional
            Covariance matrix
        prev_weights : np.ndarray, optional
            Previous weights
            
        Returns:
        --------
        np.ndarray : CP-Size weights
        """
        # Interval widths
        widths = upper_bounds - lower_bounds
        
        # Check width dispersion
        width_std = widths.std()
        width_mean = widths.mean()
        dispersion = width_std / (width_mean + 1e-8)
        
        # If widths are very similar and standardization enabled, standardize them
        if self.use_standardization and dispersion < 0.3:
            # Z-score the widths to amplify differences
            widths_std = (widths - width_mean) / (width_std + 1e-8)
            # Map to precision: high width → low precision
            # Use exponential to maintain positivity and amplify differences
            precision = np.exp(-widths_std)
        else:
            # Original method: direct inverse
            epsilon = np.median(widths) * 0.1
            precision = 1.0 / (widths + epsilon)
        
        # Weight = forecast * precision (only positive forecasts)
        weights = np.maximum(forecasts, 0) * precision
        
        if weights.sum() < 1e-6:
            # Fallback: inverse width
            weights = precision
        
        weights = weights / weights.sum()
        
        # Apply constraints
        weights = self.apply_constraints(weights, prev_weights)
        
        # Volatility targeting
        if cov_matrix is not None:
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_vol > 1e-6:
                scale = self.vol_target / portfolio_vol
                scale = min(scale, 1.5)
                weights = weights * scale
                weights = weights / weights.sum()
        
        return weights


class CPLowerBoundAllocator(BaseAllocator):
    """
    CP-Lower-Bound (Safety First) allocation.
    Maximize portfolio's lower bound return subject to risk constraints.
    Conservative approach focusing on worst-case scenario.
    """
    
    def __init__(self, risk_aversion: float = 0.5, **kwargs):
        """
        Initialize CP-LB allocator.
        
        Parameters:
        -----------
        risk_aversion : float
            Trade-off between lower bound and variance
        """
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion
        
    def allocate(self,
                 lower_bounds: np.ndarray,
                 cov_matrix: np.ndarray,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute CP-Lower-Bound weights.
        
        Parameters:
        -----------
        lower_bounds : np.ndarray
            Lower bounds of prediction intervals (conservative returns)
        cov_matrix : np.ndarray
            Covariance matrix
        prev_weights : np.ndarray, optional
            Previous weights
            
        Returns:
        --------
        np.ndarray : CP-LB weights
        """
        n_assets = len(lower_bounds)
        
        # CVXPY optimization
        w = cp.Variable(n_assets)
        
        # Objective: maximize lower bound return - risk penalty
        portfolio_lower_bound = lower_bounds @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(
            portfolio_lower_bound - self.risk_aversion * portfolio_variance
        )
        
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight
        ]
        
        if prev_weights is not None:
            constraints.append(cp.norm1(w - prev_weights) <= self.max_turnover)
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status == 'optimal':
            weights = w.value
        else:
            # Fallback: proportional to positive lower bounds
            weights = np.maximum(lower_bounds, 0)
            if weights.sum() < 1e-6:
                weights = np.ones(n_assets) / n_assets
            else:
                weights = weights / weights.sum()
        
        # Volatility targeting
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        if portfolio_vol > 1e-6:
            scale = self.vol_target / portfolio_vol
            scale = min(scale, 1.5)
            weights = weights * scale
            weights = weights / weights.sum()
        
        return weights


def create_all_allocators(vol_target: float = 0.10,
                          max_turnover: float = 0.20) -> Dict[str, BaseAllocator]:
    """
    Create all allocation strategies.
    
    Parameters:
    -----------
    vol_target : float
        Target portfolio volatility
    max_turnover : float
        Maximum turnover per rebalance
        
    Returns:
    --------
    dict : Strategy name -> allocator instance
    """
    allocators = {
        'mean_variance': MeanVarianceAllocator(
            risk_aversion=1.0,
            vol_target=vol_target,
            max_turnover=max_turnover
        ),
        'risk_parity': RiskParityAllocator(
            vol_target=vol_target,
            max_turnover=max_turnover
        ),
        'ml_only': MLOnlyAllocator(
            vol_target=vol_target,
            max_turnover=max_turnover
        ),
        'cp_gate': CPGateAllocator(
            vol_target=vol_target,
            max_turnover=max_turnover
        ),
        'cp_size': CPSizeAllocator(
            vol_target=vol_target,
            max_turnover=max_turnover
        ),
        'cp_lower_bound': CPLowerBoundAllocator(
            risk_aversion=0.5,
            vol_target=vol_target,
            max_turnover=max_turnover
        ),
    }
    
    return allocators


if __name__ == '__main__':
    # Test allocation strategies
    print("Testing allocation strategies...")
    
    np.random.seed(42)
    n_assets = 5
    
    # Generate test data
    forecasts = np.array([0.02, 0.01, -0.005, 0.015, 0.008])
    lower_bounds = forecasts - 0.03
    upper_bounds = forecasts + 0.03
    cov_matrix = np.random.rand(n_assets, n_assets) * 0.01
    cov_matrix = cov_matrix @ cov_matrix.T + np.eye(n_assets) * 0.001
    
    # Test each strategy
    allocators = create_all_allocators()
    
    print("\nForecasts:", forecasts)
    print("\nWeights by strategy:")
    
    for name, allocator in allocators.items():
        if name == 'mean_variance':
            weights = allocator.allocate(forecasts, cov_matrix)
        elif name == 'risk_parity':
            weights = allocator.allocate(cov_matrix)
        elif name == 'ml_only':
            weights = allocator.allocate(forecasts, cov_matrix)
        elif name == 'cp_gate':
            weights = allocator.allocate(forecasts, lower_bounds, upper_bounds, cov_matrix)
        elif name == 'cp_size':
            weights = allocator.allocate(forecasts, lower_bounds, upper_bounds, cov_matrix)
        elif name == 'cp_lower_bound':
            weights = allocator.allocate(lower_bounds, cov_matrix)
        
        print(f"{name:20s}: {weights.round(3)}")
    
    print("\nAll strategies tested successfully!")
