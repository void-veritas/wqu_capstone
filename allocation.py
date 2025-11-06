"""
Portfolio allocation strategies.
Implements classical methods (Mean-Variance, Risk Parity, ML-only) and
conformal prediction-aware methods (CP-Gate, CP-Size, CP-Lower-Bound).

Updated with signal-aware optimization to prevent equal-weight collapse.
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
                 min_weight: float = 0.0,
                 # New parameters
                 long_only: bool = True,
                 sum_to_one: bool = True,
                 eps: float = 1e-4,
                 alpha: float = 0.1,
                 gamma: float = 1.0,
                 turnover_penalty_lambda: float = 0.0,
                 use_turnover_constraint: bool = True,
                 inverse_vol_tilt: bool = False):
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
        long_only : bool
            If True, only long positions (no shorts)
        sum_to_one : bool
            If True, fully invested (sum=1). If False, allows cash (sum<=1)
        eps : float
            Margin for gating (strict gate threshold)
        alpha : float
            Signal strength parameter (reward for signal alignment)
        gamma : float
            Risk aversion parameter (penalty for variance)
        turnover_penalty_lambda : float
            Turnover penalty coefficient (0 = no penalty, use constraint)
        use_turnover_constraint : bool
            If True, use hard turnover constraint. If False, use penalty only.
        inverse_vol_tilt : bool
            If True, apply inverse-volatility tilt to raw weights
        """
        self.vol_target = vol_target
        self.max_turnover = max_turnover
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.long_only = long_only
        self.sum_to_one = sum_to_one
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.turnover_penalty_lambda = turnover_penalty_lambda
        self.use_turnover_constraint = use_turnover_constraint
        self.inverse_vol_tilt = inverse_vol_tilt
        
    def allocate(self, **kwargs) -> np.ndarray:
        """
        Compute portfolio weights.
        
        Returns:
        --------
        np.ndarray : Portfolio weights
        """
        raise NotImplementedError
    
    def apply_constraints(self, 
                         raw: np.ndarray,
                         forecasts: Optional[np.ndarray] = None,
                         cov_matrix: Optional[np.ndarray] = None,
                         prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply portfolio constraints with signal-aware optimization.
        
        Parameters:
        -----------
        raw : np.ndarray
            Raw/unconstrained weights (anchor)
        forecasts : np.ndarray, optional
            Return forecasts (for signal reward term)
        cov_matrix : np.ndarray, optional
            Covariance matrix (for risk penalty)
        prev_weights : np.ndarray, optional
            Previous weights for turnover constraint/penalty
            
        Returns:
        --------
        np.ndarray : Constrained weights
        """
        n = len(raw)
        
        # Ensure forecasts available for signal-aware optimization
        if forecasts is None:
            forecasts = raw.copy()
        
        # Ensure covariance is PSD
        if cov_matrix is not None:
            if cov_matrix.shape != (n, n):
                cov_matrix = None
            else:
                # Make PSD by adding small diagonal
                eigenvals = np.linalg.eigvals(cov_matrix)
                if np.min(eigenvals) < 0:
                    cov_matrix = cov_matrix + np.eye(n) * 1e-6
        
        w = cp.Variable(n)
        
        # Signal-aware objective: keep raw as anchor, add signal reward, risk penalty
        obj = 0.5 * cp.sum_squares(w - raw)  # Anchor to raw
        
        # Risk penalty
        if cov_matrix is not None and self.gamma > 0:
            Σ = cp.psd_wrap(cov_matrix)
            obj += self.gamma * cp.quad_form(w, Σ)
        
        # Signal reward (negative because we minimize)
        if self.alpha > 0:
            obj -= self.alpha * forecasts @ w
        
        # Turnover penalty (if using penalty instead of constraint)
        if prev_weights is not None and self.turnover_penalty_lambda > 0:
            obj += self.turnover_penalty_lambda * cp.norm1(w - prev_weights)
        
        # Constraints
        if self.sum_to_one:
            constraints = [cp.sum(w) == 1]
        else:
            constraints = [cp.sum(w) <= 1]
        
        constraints.extend([
            w >= self.min_weight,
            w <= self.max_weight
        ])
        
        # Turnover constraint (if using constraint instead of penalty)
        if prev_weights is not None and self.max_turnover is not None:
            if self.use_turnover_constraint:
                constraints.append(cp.norm1(w - prev_weights) <= self.max_turnover)
        
        problem = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except:
            # Fallback solver
            try:
                problem.solve(solver=cp.SCS, verbose=False)
            except:
                pass
        
        if problem.status in ("optimal", "optimal_inaccurate"):
            return w.value
        else:
            # Fallback: clip and normalize safely
            w_fb = np.clip(raw, self.min_weight, self.max_weight)
            if self.sum_to_one:
                s = w_fb.sum()
                if s > 1e-8:
                    return w_fb / s
                else:
                    # Equal weight fallback
                    w_eq = np.ones(n) / n
                    w_eq = np.clip(w_eq, self.min_weight, self.max_weight)
                    return w_eq / w_eq.sum()
            else:
                # sum <= 1 mode allows cash
                return w_fb
    
    def _apply_constraints_weak_anchor(self,
                                       raw: np.ndarray,
                                       forecasts: Optional[np.ndarray] = None,
                                       cov_matrix: Optional[np.ndarray] = None,
                                       prev_weights: Optional[np.ndarray] = None,
                                       anchor_weight: float = 0.1) -> np.ndarray:
        """
        Apply constraints with weak anchor - allows signal to dominate.
        
        This is used by CP strategies to preserve CP adjustments.
        Weak anchor (0.1) + strong signal allows CP differences to show through.
        """
        n = len(raw)
        
        # Ensure forecasts available
        if forecasts is None:
            forecasts = raw.copy()
        
        # Ensure covariance is PSD
        if cov_matrix is not None:
            if cov_matrix.shape != (n, n):
                cov_matrix = None
            else:
                eigenvals = np.linalg.eigvals(cov_matrix)
                if np.min(eigenvals) < 0:
                    cov_matrix = cov_matrix + np.eye(n) * 1e-6
        
        w = cp.Variable(n)
        
        # Weak anchor + strong signal - allows CP adjustments to dominate
        obj = anchor_weight * cp.sum_squares(w - raw)  # Weak anchor
        
        # Risk penalty
        if cov_matrix is not None and self.gamma > 0:
            Σ = cp.psd_wrap(cov_matrix)
            obj += self.gamma * cp.quad_form(w, Σ)
        
        # Strong signal reward (negative because we minimize)
        # Increase signal strength significantly to preserve CP adjustments
        signal_strength = max(self.alpha * 3.0, 1.0)  # Much stronger signal
        if signal_strength > 0:
            obj -= signal_strength * forecasts @ w
        
        # Turnover penalty
        if prev_weights is not None and self.turnover_penalty_lambda > 0:
            obj += self.turnover_penalty_lambda * cp.norm1(w - prev_weights)
        
        # Constraints
        if self.sum_to_one:
            constraints = [cp.sum(w) == 1]
        else:
            constraints = [cp.sum(w) <= 1]
        
        constraints.extend([
            w >= self.min_weight,
            w <= self.max_weight
        ])
        
        # Turnover constraint
        if prev_weights is not None and self.max_turnover is not None:
            if self.use_turnover_constraint:
                constraints.append(cp.norm1(w - prev_weights) <= self.max_turnover)
        
        problem = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except:
            try:
                problem.solve(solver=cp.SCS, verbose=False)
            except:
                pass
        
        if problem.status in ("optimal", "optimal_inaccurate"):
            weights = w.value
            if weights is None or np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                # Fallback: normalize raw weights
                weights = self._safe_normalize(raw)
            return weights
        else:
            # Fallback: normalize raw weights
            return self._safe_normalize(raw)
    
    def apply_constraints_with_gate(self,
                                    raw: np.ndarray,
                                    forecasts: Optional[np.ndarray],
                                    gate: np.ndarray,
                                    cov_matrix: Optional[np.ndarray] = None,
                                    prev_weights: Optional[np.ndarray] = None,
                                    anchor_weight: float = 0.5) -> np.ndarray:
        """
        Apply constraints with gate enforcement (for CP-Gate).
        Ensures gated-out assets remain at zero weight.
        
        Parameters:
        -----------
        raw : np.ndarray
            Raw weights (gated assets already zero)
        forecasts : np.ndarray
            Signal for optimization
        gate : np.ndarray
            Boolean array indicating which assets pass the gate
        cov_matrix : np.ndarray, optional
            Covariance matrix
        prev_weights : np.ndarray, optional
            Previous weights
            
        Returns:
        --------
        np.ndarray : Constrained weights (gated-out assets = 0)
        """
        n = len(raw)
        
        # Ensure forecasts available
        if forecasts is None:
            forecasts = raw.copy()
        
        # Ensure covariance is PSD
        if cov_matrix is not None:
            if cov_matrix.shape != (n, n):
                cov_matrix = None
            else:
                eigenvals = np.linalg.eigvals(cov_matrix)
                if np.min(eigenvals) < 0:
                    cov_matrix = cov_matrix + np.eye(n) * 1e-6
        
        w = cp.Variable(n)
        
        # Signal-aware objective with adjustable anchor strength
        obj = anchor_weight * cp.sum_squares(w - raw)
        
        if cov_matrix is not None and self.gamma > 0:
            Σ = cp.psd_wrap(cov_matrix)
            obj += self.gamma * cp.quad_form(w, Σ)
        
        if self.alpha > 0:
            obj -= self.alpha * forecasts @ w
        
        if prev_weights is not None and self.turnover_penalty_lambda > 0:
            obj += self.turnover_penalty_lambda * cp.norm1(w - prev_weights)
        
        # Constraints
        if self.sum_to_one:
            constraints = [cp.sum(w) == 1]
        else:
            constraints = [cp.sum(w) <= 1]
        
        constraints.extend([
            w >= self.min_weight,
            w <= self.max_weight
        ])
        
        # ENFORCE GATE: gated-out assets must be zero
        for i in range(n):
            if not gate[i]:
                constraints.append(w[i] == 0)
        
        if prev_weights is not None and self.max_turnover is not None:
            if self.use_turnover_constraint:
                constraints.append(cp.norm1(w - prev_weights) <= self.max_turnover)
        
        problem = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except:
            try:
                problem.solve(solver=cp.SCS, verbose=False)
            except:
                pass
        
        if problem.status in ("optimal", "optimal_inaccurate"):
            return w.value
        else:
            # Fallback: clip and normalize, but enforce gate
            w_fb = np.clip(raw, self.min_weight, self.max_weight)
            w_fb[~gate] = 0  # Force gated-out to zero
            if self.sum_to_one:
                s = w_fb.sum()
                if s > 1e-8:
                    return w_fb / s
                else:
                    w_eq = np.ones(n) / n
                    w_eq[~gate] = 0  # Force gated-out to zero
                    w_eq = np.clip(w_eq, self.min_weight, self.max_weight)
                    s = w_eq.sum()
                    return w_eq / s if s > 1e-8 else w_eq
            return w_fb
    
    def _apply_vol_target(self, weights: np.ndarray, cov_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply volatility targeting.
        
        If sum_to_one=True: vol targeting done via risk penalty in optimization.
        Post-scaling is only used as safety check if way off target.
        If sum_to_one=False: post-scale without renormalizing (allows cash).
        """
        if cov_matrix is None:
            return weights
        
        # Safety check: ensure dimensions match
        if cov_matrix.shape[0] != len(weights) or cov_matrix.shape[1] != len(weights):
            # Dimensions don't match - skip vol targeting
            return weights
        
        # Ensure covariance is PSD
        eigenvals = np.linalg.eigvals(cov_matrix)
        if np.min(eigenvals) < -1e-6:
            # Not PSD - skip vol targeting
            return weights
        
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        if portfolio_vol > 1e-6:
            scale = self.vol_target / portfolio_vol
            scale = min(scale, 1.5)  # Don't over-leverage
            scale = max(scale, 0.5)  # Don't under-leverage too much
            
            if self.sum_to_one:
                # For sum=1, vol targeting should ideally be done in optimization
                # via gamma tuning. Post-scaling and renormalizing changes weights.
                # Only scale if way off target (safety check)
                if scale < 0.8 or scale > 1.2:
                    weights = weights * scale
                    # Renormalize to maintain sum=1 constraint
                    s = weights.sum()
                    if s > 1e-8:
                        weights = weights / s
                    else:
                        # If weights sum to zero, return equal weights
                        weights = np.ones(len(weights)) / len(weights)
            else:
                # For sum<=1, scale without renormalizing (allows cash)
                weights = weights * scale
        
        return weights
    
    def _safe_normalize(self, raw: np.ndarray) -> np.ndarray:
        """
        Safe normalization that handles edge cases.
        
        Parameters:
        -----------
        raw : np.ndarray
            Raw weights to normalize
            
        Returns:
        --------
        np.ndarray : Normalized weights
        """
        if self.long_only:
            s = raw.sum()
            if s > 1e-8:
                return raw / s
            else:
                # Equal weight fallback across all assets
                n = len(raw)
                return np.ones(n) / n
        else:
            # Long/short: normalize by L1 norm
            s = np.sum(np.abs(raw))
            if s > 1e-8:
                return raw / s
            else:
                # Allow cash (zero weights)
                return np.zeros_like(raw)


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
            if self.use_turnover_constraint:
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
            weights = self.apply_constraints(weights, None, cov_matrix, prev_weights)
        
        # Apply volatility target
        weights = self._apply_vol_target(weights, cov_matrix)
        
        return weights


class MLOnlyAllocator(BaseAllocator):
    """
    ML-Only allocation: Baseline strategy without CP information.
    
    This is the simplest strategy - uses point forecasts directly without any uncertainty adjustment.
    All CP strategies should be compared against this baseline.
    
    Pipeline:
    1. Take positive forecasts (long-only)
    2. Normalize to sum to 1
    3. Apply constraints with signal-aware optimization
    4. Volatility targeting
    """
    
    def allocate(self,
                 forecasts: np.ndarray,
                 cov_matrix: Optional[np.ndarray] = None,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute ML-based weights - SIMPLEST APPROACH.
        
        No CP adjustments - uses forecasts directly.
        """
        # Step 1: Raw weights from forecasts (long-only: only positive forecasts)
        raw = np.maximum(forecasts, 0)
        
        # Step 2: Optional inverse-volatility tilt (before normalization)
        if self.inverse_vol_tilt and cov_matrix is not None:
            iv = np.sqrt(np.clip(np.diag(cov_matrix), 1e-12, None))
            raw = raw / iv
        
        # Step 3: Apply constraints with signal-aware optimization
        # Use ORIGINAL forecasts as signal (no CP adjustment)
        # ML-Only uses standard anchor weight (0.5) - this is baseline
        weights = self.apply_constraints(raw, forecasts, cov_matrix, prev_weights)
        
        # Step 4: Volatility targeting
        weights = self._apply_vol_target(weights, cov_matrix)
        
        return weights


class CPGateAllocator(BaseAllocator):
    """
    CP-Gate allocation: CP Layer on Top of ML-Only.
    
    Core Idea: Start with ML-Only weights, then use CP intervals to refine them.
    This layers CP uncertainty information on top of a working strategy.
    
    Methodology:
    1. Compute ML-Only weights (baseline)
    2. Compute uncertainty scores from CP intervals
    3. Reduce weights for uncertain assets (soft filtering)
    4. Increase risk penalty for uncertain assets
    5. Re-optimize with CP-adjusted inputs
    
    Rationale:
    - ML-Only works well → use it as foundation
    - CP adds uncertainty information → refine the allocation
    - Layered approach preserves ML-Only strengths while adding CP benefits
    """
    
    def __init__(self, 
                 uncertainty_penalty: float = 0.4,
                 risk_adjustment: float = 0.6,
                 ml_weight: float = 0.7,
                 **kwargs):
        """
        Initialize CP-Gate allocator.
        
        Parameters:
        -----------
        uncertainty_penalty : float
            How much to reduce weights for uncertain assets (0 = no penalty, 1 = max)
        risk_adjustment : float
            How much to increase risk for uncertain assets (0 = no adjustment, 1 = max)
        ml_weight : float
            Weight given to ML-Only baseline (0.7 = 70% ML, 30% CP adjustment)
        """
        super().__init__(**kwargs)
        self.uncertainty_penalty = uncertainty_penalty
        self.risk_adjustment = risk_adjustment
        self.ml_weight = ml_weight
        self.ml_allocator = MLOnlyAllocator(**kwargs)  # ML-Only baseline
    
    def allocate(self,
                 forecasts: np.ndarray,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 cov_matrix: Optional[np.ndarray] = None,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute CP-Gate weights: ML-Only + CP refinement.
        
        DISTINCT FROM ML-ONLY: Adds CP uncertainty layer on top.
        DISTINCT FROM CP-SIZE: Penalizes uncertainty (not rewards precision).
        """
        n = len(forecasts)
        
        # Input validation
        if len(lower_bounds) != n or len(upper_bounds) != n:
            raise ValueError(f"Dimension mismatch: forecasts={n}, lower_bounds={len(lower_bounds)}, upper_bounds={len(upper_bounds)}")
        
        if np.any(lower_bounds > upper_bounds + 1e-8):
            raise ValueError("Lower bounds must be <= upper bounds")
        
        # ========================================================================
        # STEP 1: Get ML-Only baseline weights
        # ========================================================================
        ml_weights = self.ml_allocator.allocate(forecasts, cov_matrix, prev_weights)
        
        # ========================================================================
        # STEP 2: Compute CP uncertainty scores
        # ========================================================================
        widths = upper_bounds - lower_bounds
        
        # Normalize widths to [0, 1]
        if widths.max() > widths.min() + 1e-8:
            normalized_widths = (widths - widths.min()) / (widths.max() - widths.min())
        else:
            normalized_widths = np.zeros(n)
        
        # Check if interval spans zero (uncertain direction)
        spans_zero = (lower_bounds <= 0) & (upper_bounds >= 0)
        uncertainty_scores = normalized_widths * 0.5 + spans_zero.astype(float) * 0.5
        
        # ========================================================================
        # STEP 3: Adjust ML-Only weights based on uncertainty
        # ========================================================================
        # Reduce weights for uncertain assets
        # Blend ML weights with CP-adjusted weights
        cp_adjusted_weights = ml_weights * (1 - self.uncertainty_penalty * uncertainty_scores)
        cp_adjusted_weights = np.maximum(cp_adjusted_weights, 0)  # Ensure non-negative
        
        # Combine ML and CP: ml_weight * ML + (1 - ml_weight) * CP
        blended_weights = self.ml_weight * ml_weights + (1 - self.ml_weight) * cp_adjusted_weights
        
        # Normalize
        if blended_weights.sum() > 1e-8:
            blended_weights = blended_weights / blended_weights.sum()
        else:
            blended_weights = ml_weights  # Fallback to ML-Only
        
        # ========================================================================
        # STEP 4: Adjust covariance matrix for uncertain assets
        # ========================================================================
        adjusted_cov_matrix = cov_matrix.copy() if cov_matrix is not None else None
        if adjusted_cov_matrix is not None and self.risk_adjustment > 0:
            uncertainty_multiplier = 1 + self.risk_adjustment * uncertainty_scores * 2.0
            uncertainty_multiplier = np.where(spans_zero,
                                             uncertainty_multiplier * 1.5,  # Extra penalty for zero-spanning
                                             uncertainty_multiplier)
            uncertainty_multiplier = np.clip(uncertainty_multiplier, 1.0, 3.0)
            
            adjusted_cov_matrix = adjusted_cov_matrix.copy()
            np.fill_diagonal(adjusted_cov_matrix,
                           np.diag(adjusted_cov_matrix) * uncertainty_multiplier)
            
            # Ensure PSD
            eigenvals = np.linalg.eigvals(adjusted_cov_matrix)
            if np.min(eigenvals) < -1e-6:
                adjusted_cov_matrix = cov_matrix.copy()
        
        # ========================================================================
        # STEP 5: Re-optimize with CP-adjusted inputs
        # ========================================================================
        # Use blended weights as anchor, CP-adjusted forecasts as signal
        adjusted_forecasts = forecasts * (1 - self.uncertainty_penalty * uncertainty_scores * 0.5)
        signal = adjusted_forecasts if not self.long_only else np.maximum(adjusted_forecasts, 0)
        
        weights = self.apply_constraints(blended_weights, signal, adjusted_cov_matrix, prev_weights)
        
        # Volatility targeting
        weights = self._apply_vol_target(weights, cov_matrix)  # Use original covariance
        
        # Final safety checks
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            weights = ml_weights  # Fallback to ML-Only
        
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        if self.sum_to_one:
            s = weights.sum()
            if s > 1e-8:
                weights = weights / s
            else:
                weights = ml_weights
        
        return weights


class CPSizeAllocator(BaseAllocator):
    """
    CP-Size allocation: CP Layer on Top of ML-Only.
    
    Core Idea: Start with ML-Only weights, then use CP precision to refine them.
    This layers CP precision information on top of a working strategy.
    
    Methodology:
    1. Compute ML-Only weights (baseline)
    2. Compute precision scores from CP intervals
    3. Increase weights for precise predictions
    4. Re-optimize with precision-weighted inputs
    
    Rationale:
    - ML-Only works well → use it as foundation
    - CP precision indicates confidence → refine allocation toward confident predictions
    - Layered approach preserves ML-Only strengths while adding CP benefits
    """
    
    def __init__(self, 
                 precision_strength: float = 0.5,
                 ml_weight: float = 0.7,
                 **kwargs):
        """
        Initialize CP-Size allocator.
        
        Parameters:
        -----------
        precision_strength : float
            How much to boost weights for precise predictions (0 = no boost, 1 = max)
        ml_weight : float
            Weight given to ML-Only baseline (0.7 = 70% ML, 30% CP adjustment)
        """
        super().__init__(**kwargs)
        self.precision_strength = precision_strength
        self.ml_weight = ml_weight
        self.ml_allocator = MLOnlyAllocator(**kwargs)  # ML-Only baseline
    
    def allocate(self,
                 forecasts: np.ndarray,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 cov_matrix: Optional[np.ndarray] = None,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute CP-Size weights: ML-Only + CP precision refinement.
        
        DISTINCT FROM ML-ONLY: Adds CP precision layer on top.
        DISTINCT FROM CP-GATE: Rewards precision (not penalizes uncertainty).
        """
        n = len(forecasts)
        
        # Input validation
        if len(lower_bounds) != n or len(upper_bounds) != n:
            raise ValueError(f"Dimension mismatch: forecasts={n}, lower_bounds={len(lower_bounds)}, upper_bounds={len(upper_bounds)}")
        
        # ========================================================================
        # STEP 1: Get ML-Only baseline weights
        # ========================================================================
        ml_weights = self.ml_allocator.allocate(forecasts, cov_matrix, prev_weights)
        
        # ========================================================================
        # STEP 2: Compute CP precision scores
        # ========================================================================
        widths = upper_bounds - lower_bounds
        
        # Safety check
        if np.any(widths < 0):
            widths = np.maximum(widths, 1e-8)
        
        # Compute precision (inverse width, normalized)
        epsilon = np.median(widths) * 0.01
        precision = 1.0 / (widths + epsilon)
        
        # Normalize precision to [0, 1]
        if precision.max() > precision.min() + 1e-8:
            precision_normalized = (precision - precision.min()) / (precision.max() - precision.min())
        else:
            precision_normalized = np.ones(n)
        
        # ========================================================================
        # STEP 3: Adjust ML-Only weights based on precision
        # ========================================================================
        # Boost weights for precise predictions
        # Use exponential weighting for stronger effect
        precision_shifted = precision_normalized - 0.5  # Center at 0
        precision_multiplier = np.exp(self.precision_strength * precision_shifted * 2.0)
        precision_multiplier = np.clip(precision_multiplier, 0.5, 2.0)
        
        # Apply precision boost to ML weights
        cp_adjusted_weights = ml_weights * precision_multiplier
        cp_adjusted_weights = np.maximum(cp_adjusted_weights, 0)
        
        # Combine ML and CP: ml_weight * ML + (1 - ml_weight) * CP
        blended_weights = self.ml_weight * ml_weights + (1 - self.ml_weight) * cp_adjusted_weights
        
        # Normalize
        if blended_weights.sum() > 1e-8:
            blended_weights = blended_weights / blended_weights.sum()
        else:
            blended_weights = ml_weights  # Fallback to ML-Only
        
        # ========================================================================
        # STEP 4: Re-optimize with precision-weighted inputs
        # ========================================================================
        # Use precision-weighted forecasts as signal
        precision_weighted_forecasts = forecasts * precision_multiplier
        signal = precision_weighted_forecasts if not self.long_only else np.maximum(precision_weighted_forecasts, 0)
        
        weights = self.apply_constraints(blended_weights, signal, cov_matrix, prev_weights)
        
        # Volatility targeting
        weights = self._apply_vol_target(weights, cov_matrix)
        
        # Final safety checks
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            weights = ml_weights  # Fallback to ML-Only
        
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        if self.sum_to_one:
            s = weights.sum()
            if s > 1e-8:
                weights = weights / s
            else:
                weights = ml_weights
        
        return weights


class CPKatoHRLRAllocator(BaseAllocator):
    """
    Kato's High-Return-from-Low-Risk (HR-LR) CPPS allocation.
    Implements a two-stage filtering approach:
    1. Filter assets with highest lower bounds (reduce downside risk)
    2. From filtered set, weight by upper bounds (maximize upside potential)
    
    Reference: Kato (2024) - Conformal Predictive Portfolio Selection
    """
    
    def __init__(self, 
                 lower_percentile: float = 0.5,
                 **kwargs):
        """
        Initialize Kato HR-LR allocator.
        
        Parameters:
        -----------
        lower_percentile : float
            Percentile threshold for first filter (0.5 = top 50% by lower bound)
        """
        super().__init__(**kwargs)
        self.lower_percentile = lower_percentile
    
    def allocate(self,
                 forecasts: np.ndarray,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 cov_matrix: Optional[np.ndarray] = None,
                 prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Kato HR-LR weights.
        
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
        np.ndarray : Kato HR-LR weights
        """
        n_assets = len(forecasts)
        
        # Stage 1: Filter by lower bounds (reduce downside risk)
        lower_threshold = np.quantile(lower_bounds, self.lower_percentile)
        safe_assets = lower_bounds >= lower_threshold
        
        if not np.any(safe_assets):
            # Fallback: use all assets
            safe_assets = np.ones(n_assets, dtype=bool)
        
        # Stage 2: Among safe assets, weight by upper bounds (maximize upside)
        # Use confidence-weighted approach for consistency
        upper_filtered = upper_bounds * safe_assets
        if self.long_only:
            raw = np.maximum(upper_filtered, 0.0)
        else:
            raw = upper_filtered
        
        # Safe normalization
        raw = self._safe_normalize(raw)
        
        # Optional inverse-volatility tilt
        if self.inverse_vol_tilt and cov_matrix is not None:
            iv = np.sqrt(np.clip(np.diag(cov_matrix), 1e-12, None))
            raw = raw / iv
            raw = self._safe_normalize(raw)
        
        # Apply constraints with signal-aware optimization
        # Use upper bounds as signal (optimistic but filtered by safety)
        signal = upper_bounds * safe_assets if self.long_only else upper_bounds
        weights = self.apply_constraints(raw, signal, cov_matrix, prev_weights)
        
        # Volatility targeting
        weights = self._apply_vol_target(weights, cov_matrix)
        
        return weights


class CPLowerBoundAllocator(BaseAllocator):
    """
    CP-Lower-Bound allocation: Conservative Optimization Approach.
    
    Core Idea: Optimize portfolio using worst-case scenario (lower bounds) rather than point forecasts.
    This is distinct from CP-Gate (uncertainty-penalty) and CP-Size (precision-reward).
    
    Methodology:
    1. Use lower bounds as conservative return estimates (worst-case within interval)
    2. Optimize: max_w (w^T L - λ w^T Σ w) where L = lower bounds
    3. This is a proper quadratic optimization problem (not normalization)
    
    Justification: Lower bounds represent the worst-case scenario within the prediction interval.
    By optimizing on lower bounds, we create a conservative portfolio that protects against
    downside risk while still considering covariance structure. This is fundamentally different
    from adjusting forecasts - it changes the optimization objective itself.
    
    Formula (from report): max_w (w^T L - λ w^T Σ w)
    where L is vector of interval lower bounds, λ is risk aversion parameter.
    """
    
    def __init__(self, risk_aversion: float = 0.5, **kwargs):
        """
        Initialize CP-LB allocator.
        
        Parameters:
        -----------
        risk_aversion : float
            Risk aversion parameter (λ in the optimization objective)
        """
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion
        
    def allocate(self,
                 lower_bounds: np.ndarray,
                 cov_matrix: np.ndarray,
                 prev_weights: Optional[np.ndarray] = None,
                 upper_bounds: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute CP-Lower-Bound weights using optimization.
        
        Parameters:
        -----------
        lower_bounds : np.ndarray
            Lower bounds of prediction intervals (conservative returns)
        cov_matrix : np.ndarray
            Covariance matrix
        prev_weights : np.ndarray, optional
            Previous weights
        upper_bounds : np.ndarray, optional
            Upper bounds (used for adaptive risk aversion)
            
        Returns:
        --------
        np.ndarray : CP-LB weights
        """
        n_assets = len(lower_bounds)
        
        # Input validation
        if cov_matrix.shape[0] != n_assets or cov_matrix.shape[1] != n_assets:
            raise ValueError(f"Dimension mismatch: lower_bounds={n_assets}, cov_matrix={cov_matrix.shape}")
        
        if prev_weights is not None and len(prev_weights) != n_assets:
            raise ValueError(f"Dimension mismatch: lower_bounds={n_assets}, prev_weights={len(prev_weights)}")
        
        # Adaptive risk aversion based on interval widths (if available)
        # Wide intervals → higher uncertainty → more conservative (higher risk_aversion)
        adaptive_risk_aversion = self.risk_aversion
        if upper_bounds is not None:
            widths = upper_bounds - lower_bounds
            avg_width = np.mean(widths)
            median_width = np.median(widths)
            # Use both average and median for more robust adaptation
            # Wider intervals → higher risk_aversion (more conservative)
            # Normalize to [0.3, 3.0] multiplier for stronger effect
            width_metric = (avg_width + median_width) / 2
            width_factor = 0.3 + 2.7 * np.tanh(width_metric / (width_metric + 0.01))  # Smooth normalization
            adaptive_risk_aversion = self.risk_aversion * width_factor
            
            # Also adjust based on how many intervals span zero (uncertainty indicator)
            spans_zero = (lower_bounds <= 0) & (upper_bounds >= 0)
            zero_span_ratio = np.mean(spans_zero)
            # More zero-spanning → more conservative
            adaptive_risk_aversion *= (1 + zero_span_ratio * 0.5)  # Up to 50% increase
        
        # Check covariance is PSD
        eigenvals = np.linalg.eigvals(cov_matrix)
        if np.min(eigenvals) < -1e-6:
            # Make PSD by adding small diagonal
            cov_matrix = cov_matrix + np.eye(n_assets) * (abs(np.min(eigenvals)) + 1e-6)
        
        # Only consider positive lower bounds for long-only
        if self.long_only:
            lower_bounds_used = np.maximum(lower_bounds, 0.0)
            # Safety check: if all lower bounds are zero or negative, use forecasts as fallback
            if np.sum(lower_bounds_used) < 1e-8:
                # This shouldn't happen if CP is working correctly, but handle gracefully
                lower_bounds_used = np.ones(n_assets) / n_assets  # Equal weight fallback
        else:
            lower_bounds_used = lower_bounds.copy()
        
        # CVXPY optimization: maximize lower bound return - risk penalty
        w = cp.Variable(n_assets)
        
        # Objective: maximize w^T L - λ w^T Σ w
        # Use adaptive risk_aversion based on interval widths
        portfolio_lower_bound = lower_bounds_used @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(
            portfolio_lower_bound - adaptive_risk_aversion * portfolio_variance
        )
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight
        ]
        
        # Turnover constraint
        if prev_weights is not None:
            if self.use_turnover_constraint:
                constraints.append(cp.norm1(w - prev_weights) <= self.max_turnover)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status in ('optimal', 'optimal_inaccurate'):
            weights = w.value
            # Safety check: ensure no NaN/Inf
            if weights is None or np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                weights = lower_bounds_used.copy()
                weights = self._safe_normalize(weights)
        else:
            # Fallback: proportional to positive lower bounds
            weights = lower_bounds_used.copy()
            weights = self._safe_normalize(weights)
        
        # Apply volatility targeting
        weights = self._apply_vol_target(weights, cov_matrix)
        
        # Final safety check: ensure weights sum to 1 and no NaN/Inf
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            weights = np.ones(n_assets) / n_assets
        weights = np.clip(weights, self.min_weight, self.max_weight)
        if self.sum_to_one:
            s = weights.sum()
            if s > 1e-8:
                weights = weights / s
            else:
                weights = np.ones(n_assets) / n_assets
        
        return weights


def create_all_allocators(vol_target: float = 0.10,
                          max_turnover: float = 0.20,
                          include_kato: bool = False,
                          # New parameters for signal-aware optimization
                          alpha: float = 0.1,
                          gamma: float = 1.0,
                          eps: float = 1e-4,
                          turnover_penalty_lambda: float = 0.0,
                          use_turnover_constraint: bool = True,
                          # Load optimized parameters if available
                          use_optimized_params: bool = True,
                          optimized_params_file: str = 'optimal_parameters.json') -> Dict[str, BaseAllocator]:
    """
    Create all allocation strategies.
    
    Parameters:
    -----------
    use_optimized_params : bool
        If True, load optimized parameters from JSON file if it exists
    optimized_params_file : str
        Path to JSON file with optimized parameters
    """
    import json
    import os
    
    # Try to load optimized parameters
    optimized_params = {}
    if use_optimized_params and os.path.exists(optimized_params_file):
        try:
            with open(optimized_params_file, 'r') as f:
                optimized_params = json.load(f)
            print(f"✓ Loaded optimized parameters from {optimized_params_file}")
        except Exception as e:
            print(f"⚠ Could not load optimized parameters: {e}")
            optimized_params = {}
    
    # Extract optimized parameters (with defaults)
    cp_gate_params = optimized_params.get('cp_gate', {})
    cp_size_params = optimized_params.get('cp_size', {})
    cp_lower_bound_params = optimized_params.get('cp_lower_bound', {})
    
    # Common parameters
    common_kwargs = {
        'vol_target': vol_target,
        'max_turnover': max_turnover,
        'long_only': True,
        'sum_to_one': True,
        'eps': eps,
        'alpha': alpha,
        'gamma': gamma,
        'turnover_penalty_lambda': turnover_penalty_lambda,
        'use_turnover_constraint': use_turnover_constraint,
        'inverse_vol_tilt': False
    }
    
    allocators = {
        'mean_variance': MeanVarianceAllocator(
            risk_aversion=1.0,
            **common_kwargs
        ),
        'risk_parity': RiskParityAllocator(
            **common_kwargs
        ),
        'ml_only': MLOnlyAllocator(
            **common_kwargs
        ),
        'cp_gate': CPGateAllocator(
            uncertainty_penalty=cp_gate_params.get('uncertainty_penalty', 0.4),
            risk_adjustment=cp_gate_params.get('risk_adjustment', 0.6),
            ml_weight=cp_gate_params.get('ml_weight', 0.7),
            **common_kwargs
        ),
        'cp_size': CPSizeAllocator(
            precision_strength=cp_size_params.get('precision_strength', 0.5),
            ml_weight=cp_size_params.get('ml_weight', 0.7),
            **common_kwargs
        ),
        'cp_lower_bound': CPLowerBoundAllocator(
            risk_aversion=cp_lower_bound_params.get('risk_aversion', 0.8),
            **common_kwargs
        ),
    }
    
    if include_kato:
        allocators['cp_kato_hrlr'] = CPKatoHRLRAllocator(
            lower_percentile=0.5,
            **common_kwargs
        )
    
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
