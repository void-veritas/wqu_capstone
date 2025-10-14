# Conformal Prediction for Robust Portfolio Allocation

## Overview
This project investigates whether incorporating conformal prediction intervals at the asset level improves risk-adjusted portfolio performance compared to classical allocation methods. The framework includes rigorous backtesting with realistic market frictions across S&P 500 sector ETFs (2020-2024).

## Project Structure
```
├── data_loader.py      # ETF data fetching and feature engineering
├── models.py           # ML forecasters (XGBoost, MLP)
├── conformal.py        # Conformal prediction methods (Split CP, CQR, Adaptive, Ensemble)
├── allocation.py       # Portfolio allocation strategies (6 methods)
├── backtester.py       # Rolling backtest with regime detection
├── visualization.py    # Comprehensive plotting utilities
├── report.ipynb        # Complete analysis and results
└── requirements.txt    # Python dependencies
```

## Strategies

### Classical Methods
- **Mean-Variance**: Markowitz optimization with risk aversion
- **Risk Parity**: Equal risk contribution per asset
- **ML-Only**: Weights proportional to ML forecasts

### Conformal Prediction Methods
- **CP-Gate**: Filters assets with uncertain direction (interval spans zero)
- **CP-Size**: Precision-weighted allocation (inverse to interval width)
- **CP-Lower-Bound**: Conservative optimization on interval lower bounds

## Data
- **Universe**: 10 S&P 500 sector ETFs (SPY, XLF, XLK, XLE, XLV, XLY, XLP, XLI, XLB, XLU)
- **Period**: Daily data 2015-2024 (2,515 trading days)
- **Out-of-sample**: 2020-2024
- **Features**: 24 features per asset (returns, volatility, momentum, statistical moments)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
# Load data
from data_loader import DataLoader
loader = DataLoader(start_date='2015-01-01', end_date='2024-12-31')
data = loader.load_data()

# Initialize backtester (3-year training window)
from backtester import Backtester
backtester = Backtester(
    data_loader=loader,
    train_window=756,  # ~3 years
    rebalance_freq=5   # Weekly
)

# Run complete analysis
from allocation import create_all_allocators
allocators = create_all_allocators()
results = backtester.run_all_strategies(allocators=allocators)

# See report.ipynb for full analysis and visualizations
```

## Key Results
- **Best Performer**: CP-Gate (Sharpe: 1.47, Turnover: 0.26%)
- **Transaction Cost Advantage**: CP methods show 40-60x lower turnover vs. traditional methods
- **Robustness**: Validated across stress periods (COVID-19 2020, Inflation 2022)
- **Coverage**: 90% conformal intervals with empirical validation

## Key Metrics
- **Performance**: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown
- **CP Quality**: Coverage, interval width, efficiency (Winkler score)
- **Trading**: Turnover, transaction costs (10 bps each side)
- **Regime Analysis**: HMM-based volatility regime detection

## References
- Vovk et al. (2005): Conformal prediction theory
- Lei et al. (2018): Split conformal regression
- Kato (2024): Conformal predictive portfolio selection
- Alonso (2024): Conformal prediction in finance
