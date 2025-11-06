"""
Hyperparameter optimization for CP-based allocation strategies.

Uses walk-forward validation to find optimal parameters for each strategy.
Supports parallel execution for faster optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from itertools import product
from tqdm import tqdm
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os
warnings.filterwarnings('ignore')

from allocation import create_all_allocators, BaseAllocator
from backtester import Backtester, PerformanceMetrics


class HyperparameterOptimizer:
    """
    Optimize hyperparameters for CP allocation strategies using walk-forward validation.
    """
    
    def __init__(self,
                 backtester: Backtester,
                 optimization_metric: str = 'sharpe_ratio',
                 validation_split: float = 0.3,
                 n_jobs: int = 1):
        """
        Initialize hyperparameter optimizer.
        
        Parameters:
        -----------
        backtester : Backtester
            Backtester instance with data loaded
        optimization_metric : str
            Metric to optimize ('sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'return')
        validation_split : float
            Fraction of test period to use for validation (rest for final test)
        """
        self.backtester = backtester
        self.optimization_metric = optimization_metric
        self.validation_split = validation_split
        self.n_jobs = n_jobs
        
    def optimize_strategy(self,
                         strategy_name: str,
                         model,
                         conformal_predictor,
                         param_grid: Dict[str, List[float]],
                         test_start_date: str = '2020-01-01',
                         verbose: bool = True,
                         n_iter: int = 30,
                         method: str = 'random') -> Tuple[Dict, float]:
        """
        Optimize hyperparameters for a single strategy using efficient methods.
        
        Parameters:
        -----------
        strategy_name : str
            Name of strategy to optimize
        model : BaseEstimator
            ML model
        conformal_predictor : object or None
            Conformal predictor
        param_grid : dict
            Dictionary of parameter name -> list of values to try
        test_start_date : str
            Start date for testing
        verbose : bool
            Whether to print progress
        n_iter : int
            Number of random iterations (for random search)
        method : str
            'random' for random search, 'grid' for full grid search
            
        Returns:
        --------
        best_params : dict
            Best parameter combination
        best_score : float
            Best validation score
        """
        # Get test dates
        returns = self.backtester.data_loader.returns
        test_dates = returns[returns.index >= test_start_date].index
        
        # Split into validation and test
        n_validation = int(len(test_dates) * self.validation_split)
        validation_end_date = test_dates[n_validation]
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_ranges = {name: (min(vals), max(vals)) for name, vals in param_grid.items()}
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimizing {strategy_name}")
            print(f"{'='*60}")
            print(f"Method: {method}")
            if method == 'random':
                print(f"Random iterations: {n_iter}")
            else:
                total_combos = np.prod([len(v) for v in param_grid.values()])
                print(f"Total combinations: {total_combos}")
            print(f"Validation period: {test_start_date} to {validation_end_date}")
        
        best_score = -np.inf
        best_params = None
        results = []
        
        if method == 'random':
            # Random search: sample uniformly from parameter space
            import random
            random.seed(42)
            np.random.seed(42)  # For reproducibility
            
            for i in range(n_iter):
                # Sample random parameters
                param_dict = {}
                for name, (min_val, max_val) in param_ranges.items():
                    param_dict[name] = np.random.uniform(min_val, max_val)
                
                result = self._test_combination(
                    strategy_name, model, conformal_predictor,
                    param_dict, test_start_date, validation_end_date
                )
                if result:
                    results.append(result)
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_params = result['params'].copy()
                
                if verbose and (i + 1) % 5 == 0:
                    print(f"  [{strategy_name}] Iteration {i+1}/{n_iter}: Best score = {best_score:.4f}")
        else:
            # Grid search (original method)
            param_values = list(param_grid.values())
            combinations = list(product(*param_values))
            
            if self.n_jobs > 1 and len(combinations) > 10:
                results = self._optimize_combinations_parallel(
                    strategy_name, model, conformal_predictor,
                    combinations, param_names, test_start_date, validation_end_date, verbose
                )
            else:
                for params in tqdm(combinations, desc=f"Optimizing {strategy_name}", disable=not verbose):
                    param_dict = dict(zip(param_names, params))
                    result = self._test_combination(
                        strategy_name, model, conformal_predictor,
                        param_dict, test_start_date, validation_end_date
                    )
                    if result:
                        results.append(result)
        
        # Find best
        if results:
            best_result = max(results, key=lambda x: x['score'])
            best_params = best_result['params']
            best_score = best_result['score']
            
            if verbose:
                print(f"\nBest parameters: {best_params}")
                print(f"Best {self.optimization_metric}: {best_score:.4f}")
                print(f"Valid combinations tested: {len(results)}/{n_iter if method == 'random' else len(combinations)}")
                if len(results) > 1:
                    print(f"\nTop 5 combinations:")
                    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
                    for i, r in enumerate(sorted_results, 1):
                        print(f"{i}. {r['params']} -> {r['score']:.4f}")
        else:
            if verbose:
                print(f"\nWarning: No valid results found for {strategy_name}")
                print("Using default parameters")
            # Return default parameters if no valid results
            best_params = {}
            best_score = -np.inf
        
        return best_params, best_score
    
    def _test_combination(self, strategy_name, model, conformal_predictor,
                         param_dict, test_start_date, validation_end_date):
        """Test a single parameter combination."""
        try:
            # Create allocator with these parameters
            allocator = self._create_allocator(strategy_name, param_dict)
            
            # Run backtest
            results_dict = self.backtester.backtest_strategy(
                strategy_name,
                model,
                conformal_predictor,
                allocator,
                test_start_date=test_start_date
            )
            
            # Get validation period only
            validation_results = self._get_validation_period(
                results_dict,
                validation_end_date
            )
            
            # Check if we have valid returns
            if len(validation_results['returns']) == 0:
                return None
            
            # Compute score
            score = self._compute_score(validation_results)
            
            # Check for invalid scores
            if not np.isfinite(score):
                return None
            
            return {
                'params': param_dict,
                'score': score,
                'metrics': validation_results['metrics']
            }
        except Exception as e:
            # Return None silently - errors are expected for some parameter combinations
            return None
    
    def _optimize_combinations_parallel(self, strategy_name, model, conformal_predictor,
                                       combinations, param_names, test_start_date,
                                       validation_end_date, verbose):
        """Optimize parameter combinations in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            
            for params in combinations:
                param_dict = dict(zip(param_names, params))
                future = executor.submit(
                    self._test_combination,
                    strategy_name, model, conformal_predictor,
                    param_dict, test_start_date, validation_end_date
                )
                futures.append(future)
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), 
                             total=len(futures),
                             desc=f"Optimizing {strategy_name}",
                             disable=not verbose):
                result = future.result()
                if result:
                    results.append(result)
        
        return results
    
    def _create_allocator(self, strategy_name: str, params: Dict) -> BaseAllocator:
        """Create allocator with given parameters."""
        from allocation import (
            CPGateAllocator, CPSizeAllocator, CPLowerBoundAllocator
        )
        
        # Common kwargs
        common_kwargs = {
            'vol_target': 0.10,
            'max_turnover': 0.20,
            'max_weight': 0.30,
            'min_weight': 0.0,
            'long_only': True,
            'sum_to_one': True,
            'alpha': 0.1,
            'gamma': 1.0,
        }
        
        if strategy_name == 'cp_gate':
            return CPGateAllocator(
                uncertainty_penalty=params.get('uncertainty_penalty', 0.4),
                risk_adjustment=params.get('risk_adjustment', 0.6),
                ml_weight=params.get('ml_weight', 0.7),
                **common_kwargs
            )
        elif strategy_name == 'cp_size':
            return CPSizeAllocator(
                precision_strength=params.get('precision_strength', 0.5),
                ml_weight=params.get('ml_weight', 0.7),
                **common_kwargs
            )
        elif strategy_name == 'cp_lower_bound':
            return CPLowerBoundAllocator(
                risk_aversion=params.get('risk_aversion', 0.8),
                **common_kwargs
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def _get_validation_period(self, results_dict: Dict, validation_end_date: str) -> Dict:
        """Extract validation period from results."""
        returns = results_dict['returns']
        positions = results_dict['positions']
        
        # Convert validation_end_date to Timestamp if string
        if isinstance(validation_end_date, str):
            validation_end_date = pd.Timestamp(validation_end_date)
        
        # Filter to validation period
        validation_mask = returns.index <= validation_end_date
        validation_returns = returns[validation_mask]
        
        # Handle positions
        if positions is not None and len(positions) > 0:
            validation_positions = positions[validation_mask]
        else:
            validation_positions = None
        
        # Recompute metrics for validation period
        metrics = PerformanceMetrics.calculate_metrics(
            validation_returns,
            validation_positions,
            None,
            annualize=True
        )
        
        return {
            'returns': validation_returns,
            'positions': validation_positions,
            'metrics': metrics
        }
    
    def _compute_score(self, results: Dict) -> float:
        """Compute optimization score from results."""
        metrics = results['metrics']
        
        if self.optimization_metric == 'sharpe_ratio':
            return metrics.get('sharpe', -np.inf)  # Note: key is 'sharpe' not 'sharpe_ratio'
        elif self.optimization_metric == 'sortino_ratio':
            return metrics.get('sortino', -np.inf)
        elif self.optimization_metric == 'calmar_ratio':
            return metrics.get('calmar', -np.inf)
        elif self.optimization_metric == 'return':
            return metrics.get('ann_return', -np.inf)
        elif self.optimization_metric == 'risk_adjusted':
            # Combined metric: Sharpe * (1 - turnover_penalty)
            sharpe = metrics.get('sharpe', 0)
            turnover = metrics.get('avg_turnover', 1.0)
            turnover_penalty = min(turnover / 0.5, 1.0)  # Penalize if turnover > 50%
            return sharpe * (1 - turnover_penalty * 0.2)  # 20% penalty max
        else:
            raise ValueError(f"Unknown metric: {self.optimization_metric}")


def optimize_all_cp_strategies(backtester: Backtester,
                               model,
                               conformal_predictor,
                               test_start_date: str = '2020-01-01',
                               optimization_metric: str = 'risk_adjusted',
                               verbose: bool = True,
                               n_jobs: int = -1,
                               parallel: bool = True,
                               n_iter: int = 30,
                               method: str = 'random') -> Dict[str, Dict]:
    """
    Optimize all CP strategies using efficient random search.
    
    Parameters:
    -----------
    backtester : Backtester
        Backtester instance
    model : BaseEstimator
        ML model
    conformal_predictor : object
        Conformal predictor
    test_start_date : str
        Start date for testing
    optimization_metric : str
        Metric to optimize
    verbose : bool
        Whether to print progress
    n_jobs : int
        Number of parallel jobs (-1 = use all CPUs)
    parallel : bool
        Whether to run strategies in parallel
    n_iter : int
        Number of random iterations per strategy (default: 30)
    method : str
        'random' for random search (fast), 'grid' for full grid search (slow)
        
    Returns:
    --------
    best_params : dict
        Dictionary of strategy_name -> best_parameters
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    optimizer = HyperparameterOptimizer(
        backtester,
        optimization_metric=optimization_metric,
        validation_split=0.3,
        n_jobs=n_jobs
    )
    
    # CP-Gate parameter grid (ranges for random search)
    cp_gate_grid = {
        'uncertainty_penalty': [0.2, 0.3, 0.4, 0.5, 0.6],  # Range: [0.2, 0.6]
        'risk_adjustment': [0.4, 0.5, 0.6, 0.7, 0.8],  # Range: [0.4, 0.8]
        'ml_weight': [0.6, 0.7, 0.8, 0.9]  # Range: [0.6, 0.9]
    }
    
    # CP-Size parameter grid
    cp_size_grid = {
        'precision_strength': [0.3, 0.4, 0.5, 0.6, 0.7],  # Range: [0.3, 0.7]
        'ml_weight': [0.6, 0.7, 0.8, 0.9]  # Range: [0.6, 0.9]
    }
    
    # CP-Lower-Bound parameter grid
    cp_lower_bound_grid = {
        'risk_aversion': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Range: [0.5, 1.0]
    }
    
    if verbose:
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        print(f"Method: {method}")
        if method == 'random':
            print(f"Random iterations per strategy: {n_iter}")
            print(f"Estimated time: ~{n_iter * 3 * 20 / 60:.1f} minutes")
        if parallel:
            print(f"Running strategies in parallel with {n_jobs} workers")
        print("="*60)
    
    # Optimize strategies in parallel
    if parallel and n_jobs > 1:
        best_params = _optimize_parallel(
            optimizer,
            model,
            conformal_predictor,
            cp_gate_grid,
            cp_size_grid,
            cp_lower_bound_grid,
            test_start_date,
            n_jobs,
            verbose,
            n_iter,
            method
        )
    else:
        # Sequential optimization
        best_params = {}
        
        # CP-Gate
        best_params['cp_gate'], _ = optimizer.optimize_strategy(
            'cp_gate',
            model,
            conformal_predictor,
            cp_gate_grid,
            test_start_date,
            verbose,
            n_iter=n_iter,
            method=method
        )
        
        # CP-Size
        best_params['cp_size'], _ = optimizer.optimize_strategy(
            'cp_size',
            model,
            conformal_predictor,
            cp_size_grid,
            test_start_date,
            verbose,
            n_iter=n_iter,
            method=method
        )
        
        # CP-Lower-Bound
        best_params['cp_lower_bound'], _ = optimizer.optimize_strategy(
            'cp_lower_bound',
            model,
            conformal_predictor,
            cp_lower_bound_grid,
            test_start_date,
            verbose,
            n_iter=n_iter,
            method=method
        )
    
    if verbose:
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print("\nBest parameters:")
        for strategy, params in best_params.items():
            print(f"  {strategy}: {params}")
    
    return best_params


def _optimize_parallel(optimizer, model, conformal_predictor,
                      cp_gate_grid, cp_size_grid, cp_lower_bound_grid,
                      test_start_date, n_jobs, verbose, n_iter, method):
    """Optimize strategies in parallel using multiprocessing."""
    from concurrent.futures import ThreadPoolExecutor
    
    # Prepare optimization tasks
    tasks = [
        ('cp_gate', cp_gate_grid),
        ('cp_size', cp_size_grid),
        ('cp_lower_bound', cp_lower_bound_grid)
    ]
    
    best_params = {}
    
    with ThreadPoolExecutor(max_workers=min(n_jobs, len(tasks))) as executor:
        futures = {}
        
        for strategy_name, param_grid in tasks:
            future = executor.submit(
                optimizer.optimize_strategy,
                strategy_name,
                model,
                conformal_predictor,
                param_grid,
                test_start_date,
                verbose,
                n_iter=n_iter,
                method=method
            )
            futures[future] = strategy_name
        
        # Collect results as they complete
        for future in as_completed(futures):
            strategy_name = futures[future]
            try:
                best_params[strategy_name], _ = future.result()
                if verbose:
                    print(f"✓ Completed {strategy_name}")
            except Exception as e:
                if verbose:
                    print(f"✗ Error optimizing {strategy_name}: {e}")
    
    return best_params


def update_allocation_parameters(best_params: Dict[str, Dict], 
                                 allocation_file: str = 'allocation.py'):
    """
    Update allocation.py with optimal parameters.
    
    Parameters:
    -----------
    best_params : dict
        Dictionary of strategy_name -> best_parameters
    allocation_file : str
        Path to allocation.py file
    """
    import re
    
    with open(allocation_file, 'r') as f:
        content = f.read()
    
    # Update CP-Gate parameters
    if 'cp_gate' in best_params:
        params = best_params['cp_gate']
        pattern = r"('cp_gate': CPGateAllocator\([^)]+\))"
        replacement = (
            f"'cp_gate': CPGateAllocator(\n"
            f"            uncertainty_penalty={params.get('uncertainty_penalty', 0.4)},  # Optimized\n"
            f"            risk_adjustment={params.get('risk_adjustment', 0.6)},  # Optimized\n"
            f"            ml_weight={params.get('ml_weight', 0.7)},  # Optimized\n"
            f"            **common_kwargs\n"
            f"        )"
        )
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Update CP-Size parameters
    if 'cp_size' in best_params:
        params = best_params['cp_size']
        pattern = r"('cp_size': CPSizeAllocator\([^)]+\))"
        replacement = (
            f"'cp_size': CPSizeAllocator(\n"
            f"            precision_strength={params.get('precision_strength', 0.5)},  # Optimized\n"
            f"            ml_weight={params.get('ml_weight', 0.7)},  # Optimized\n"
            f"            **common_kwargs\n"
            f"        )"
        )
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Update CP-Lower-Bound parameters
    if 'cp_lower_bound' in best_params:
        params = best_params['cp_lower_bound']
        pattern = r"('cp_lower_bound': CPLowerBoundAllocator\([^)]+\))"
        replacement = (
            f"'cp_lower_bound': CPLowerBoundAllocator(\n"
            f"            risk_aversion={params.get('risk_aversion', 0.8)},  # Optimized\n"
            f"            **common_kwargs\n"
            f"        )"
        )
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write back
    with open(allocation_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {allocation_file} with optimal parameters")


if __name__ == '__main__':
    # Example usage
    print("Hyperparameter optimization module loaded.")
    print("Use optimize_all_cp_strategies() to find optimal parameters.")
    print("Use update_allocation_parameters() to update allocation.py automatically.")

