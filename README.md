# Conformal Prediction-Based Portfolio Allocation

## Overview
This project implements a backtesting framework that compares classical portfolio allocation methods against conformal-prediction-aware strategies to test whether asset-level conformal intervals improve risk-adjusted performance.

## Project Structure
```
├── data_loader.py      # ETF data fetching and feature engineering
├── models.py           # ML forecasters (XGBoost, MLP)
├── conformal.py        # Conformal prediction intervals (Split CP, CQR)
├── allocation.py       # Portfolio allocation strategies
├── backtester.py       # Rolling backtest with regime detection
├── report.ipynb        # Analysis, visualization, and results
└── requirements.txt    # Python dependencies
```

## Strategies

### Baseline Methods
- **Mean-Variance (Markowitz)**: Historical covariance optimization
- **Risk Parity**: Equal risk contribution
- **ML-Only**: Weights proportional to ML forecasts

### Conformal Prediction Methods
- **CP-Gate**: Zero weight if interval straddles zero
- **CP-Size**: Weight inversely proportional to interval width
- **CP-Lower-Bound**: Safety-first using interval lower bounds

## Data
- **Universe**: SPY + ~10 S&P 500 sector ETFs (XLF, XLK, XLE, etc.)
- **Period**: Daily data 2015-2024
- **Features**: Past returns, realized volatility, rolling Sharpe

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
# 1. Load and preprocess data
from data_loader import DataLoader
loader = DataLoader(start_date='2015-01-01', end_date='2024-12-31')
data = loader.load_data()

# 2. Run backtest
from backtester import Backtester
bt = Backtester(data)
results = bt.run_all_strategies()

# 3. Analyze results in report.ipynb
```

## Key Metrics
- Risk-adjusted returns: Sharpe, Sortino, Max Drawdown
- CP quality: Coverage, interval width, efficiency
- Trading costs: Turnover, transaction costs (10bps)
- Regime-specific performance (HMM-based volatility regimes)
