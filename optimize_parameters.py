"""
Example script for hyperparameter optimization.

This script demonstrates how to optimize CP strategy parameters.
"""

import numpy as np
from data_loader import DataLoader
from backtester import Backtester
from conformal import SplitConformalPredictor
from sklearn.ensemble import GradientBoostingRegressor
from hyperparameter_tuning import optimize_all_cp_strategies, update_allocation_parameters

# Set random seed
np.random.seed(42)

# Load data
print("Loading data...")
loader = DataLoader(
    start_date='2015-01-01',
    end_date='2024-12-31',
    lookback=60
)
data = loader.load_data()

# Initialize backtester
print("Initializing backtester...")
backtester = Backtester(
    data_loader=loader,
    train_window=756,
    cal_window=126,
    rebalance_freq=5,  # Weekly rebalancing
    transaction_cost=0.001,
    regime_detection=True
)

# Create model and conformal predictor
print("Creating models...")
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

cp_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

conformal_predictor = SplitConformalPredictor(cp_model, alpha=0.1)

# Optimize all CP strategies
print("\n" + "="*60)
print("Starting hyperparameter optimization...")
print("="*60)

best_params = optimize_all_cp_strategies(
    backtester=backtester,
    model=model,
    conformal_predictor=conformal_predictor,
    test_start_date='2020-01-01',
    optimization_metric='risk_adjusted',  # Optimize Sharpe with turnover penalty
    verbose=True,
    n_jobs=-1,  # Use all CPUs
    parallel=True,  # Run strategies in parallel
    n_iter=20,  # Random search: 20 iterations per strategy (faster!)
    method='random'  # Use random search instead of grid search
)

# Print results
print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)
print("\nBest parameters found:")
for strategy, params in best_params.items():
    print(f"\n{strategy.upper()}:")
    if params:
        for param, value in params.items():
            print(f"  {param}: {value}")
    else:
        print("  (Using default parameters - no valid optimization found)")

# Save results (only save non-empty parameters)
best_params_to_save = {k: v for k, v in best_params.items() if v}
if best_params_to_save:
    import json
    with open('optimal_parameters.json', 'w') as f:
        json.dump(best_params_to_save, f, indent=2)
    print("\n✓ Optimal parameters saved to 'optimal_parameters.json'")
else:
    print("\n⚠ No optimal parameters found - using defaults")

