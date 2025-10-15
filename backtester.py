"""
Backtesting framework for portfolio strategies.
Implements rolling walk-forward backtesting with:
- Transaction costs
- Regime detection (Hidden Markov Model)
- Performance metrics
- Stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from hmmlearn import hmm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class RegimeDetector:
    """Hidden Markov Model for market regime detection."""
    
    def __init__(self, n_regimes: int = 2, random_state: int = 42):
        """
        Initialize regime detector.
        
        Parameters:
        -----------
        n_regimes : int
            Number of hidden states (regimes)
        random_state : int
            Random seed
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.regimes = None
        
    def fit_predict(self, returns: pd.DataFrame, vol_window: int = 20) -> np.ndarray:
        """
        Fit HMM and predict regimes based on realized volatility.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        vol_window : int
            Window for computing realized volatility
            
        Returns:
        --------
        np.ndarray : Regime labels (0 = low-vol, 1 = high-vol, ...)
        """
        # Compute realized volatility
        realized_vol = returns.rolling(vol_window).std().mean(axis=1) * np.sqrt(252)
        realized_vol = realized_vol.dropna()
        
        # Reshape for HMM
        X = realized_vol.values.reshape(-1, 1)
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=self.random_state
        )
        
        self.model.fit(X)
        
        # Predict regimes
        regimes = self.model.predict(X)
        
        # Map regimes: sort by mean volatility (0 = low-vol, 1 = high-vol)
        means = [self.model.means_[i][0] for i in range(self.n_regimes)]
        sorted_regimes = np.argsort(means)
        regime_mapping = {old: new for new, old in enumerate(sorted_regimes)}
        regimes = np.array([regime_mapping[r] for r in regimes])
        
        # Create Series with dates
        self.regimes = pd.Series(regimes, index=realized_vol.index)
        
        return self.regimes
    
    def get_regime_at_date(self, date: pd.Timestamp) -> int:
        """Get regime label at specific date."""
        if self.regimes is None:
            return 0
        
        if date in self.regimes.index:
            return self.regimes[date]
        else:
            # Return last known regime
            past_regimes = self.regimes[self.regimes.index <= date]
            if len(past_regimes) > 0:
                return past_regimes.iloc[-1]
            else:
                return 0


class PerformanceMetrics:
    """Calculate portfolio performance metrics."""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, 
                         positions: pd.DataFrame = None,
                         regime_labels: pd.Series = None,
                         annualize: bool = True) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        positions : pd.DataFrame, optional
            Position weights over time
        regime_labels : pd.Series, optional
            Regime labels for regime-specific metrics
        annualize : bool
            Whether to annualize metrics (False for short stress test periods)
            
        Returns:
        --------
        dict : Performance metrics
        """
        # Basic returns
        total_return = (1 + returns).prod() - 1
        
        # Annualization - only if period > 90 days OR explicitly requested
        if annualize and len(returns) >= 90:
            ann_return = (1 + total_return) ** (252 / len(returns)) - 1
            ann_vol = returns.std() * np.sqrt(252)
        else:
            # Don't annualize short periods - use actual values
            ann_return = total_return
            ann_vol = returns.std()
        
        # Sharpe ratio
        if annualize and len(returns) >= 90:
            sharpe = ann_return / (ann_vol + 1e-8)
        else:
            # For short periods: use raw mean/std ratio
            sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(len(returns))
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = ann_return / (downside_vol + 1e-8)
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = ann_return / (abs(max_drawdown) + 1e-8)
        
        # Turnover (if positions provided)
        if positions is not None:
            weight_changes = positions.diff().abs().sum(axis=1)
            avg_turnover = weight_changes.mean()
        else:
            avg_turnover = np.nan
        
        metrics = {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'avg_turnover': avg_turnover
        }
        
        # Regime-specific metrics
        if regime_labels is not None:
            for regime in regime_labels.unique():
                regime_mask = regime_labels == regime
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) > 0:
                    regime_sharpe = (
                        regime_returns.mean() * 252 / 
                        (regime_returns.std() * np.sqrt(252) + 1e-8)
                    )
                    metrics[f'sharpe_regime_{regime}'] = regime_sharpe
        
        return metrics


class Backtester:
    """
    Rolling walk-forward backtesting framework.
    """
    
    def __init__(self,
                 data_loader,
                 train_window: int = 756,
                 cal_window: int = 126,
                 rebalance_freq: int = 1,
                 transaction_cost: float = 0.001,
                 regime_detection: bool = True):
        """
        Initialize backtester.
        
        Parameters:
        -----------
        data_loader : DataLoader
            Loaded data object
        train_window : int
            Training window size (days), default 756 (~3 years)
            Extended from 2 years to provide more robust estimation
            and reduce overfitting bias in stress period testing
        cal_window : int
            Calibration window size (days), default 126 (~6 months)
        rebalance_freq : int
            Rebalancing frequency (days)
        transaction_cost : float
            Transaction cost (10 bps = 0.001)
        regime_detection : bool
            Whether to perform regime detection
        """
        self.data_loader = data_loader
        self.train_window = train_window
        self.cal_window = cal_window
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.regime_detection = regime_detection
        
        self.regime_detector = None
        if regime_detection:
            self.regime_detector = RegimeDetector()
        
        self.results = {}
        
    def backtest_strategy(self,
                         strategy_name: str,
                         model,
                         conformal_predictor,
                         allocator,
                         test_start_date: str = '2020-01-01') -> Dict:
        """
        Run backtest for a single strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        model : BaseEstimator
            ML forecasting model
        conformal_predictor : object or None
            Conformal prediction object (None for non-CP strategies)
        allocator : BaseAllocator
            Portfolio allocation strategy
        test_start_date : str
            Start date for out-of-sample testing
            
        Returns:
        --------
        dict : Backtest results
        """
        print(f"\nBacktesting {strategy_name}...")
        
        # Get data
        returns = self.data_loader.returns
        features = self.data_loader.features
        targets = self.data_loader.targets
        
        # Fit regime detector on historical data
        if self.regime_detector:
            train_returns = returns[returns.index < test_start_date]
            self.regime_detector.fit_predict(train_returns)
        
        # Get test period dates
        test_dates = returns[returns.index >= test_start_date].index
        tickers = self.data_loader.tickers
        n_assets = len(tickers)
        
        # Initialize tracking
        portfolio_returns = []
        positions = []
        turnover = []
        forecasts_history = []
        coverage_history = []
        width_history = []
        
        prev_weights = np.ones(n_assets) / n_assets
        
        # Rolling backtest
        for i, test_date in enumerate(tqdm(test_dates[::self.rebalance_freq], desc=strategy_name)):
            
            # Get training window
            train_end = test_date
            train_start_idx = returns.index.get_loc(train_end) - self.train_window - self.cal_window
            
            if train_start_idx < 0:
                continue
            
            # Split data
            train_dates = returns.index[train_start_idx:train_start_idx + self.train_window]
            cal_dates = returns.index[train_start_idx + self.train_window:train_start_idx + self.train_window + self.cal_window]
            
            # Prepare training/calibration data
            X_train_list, y_train_list = [], []
            X_cal_list, y_cal_list = [], []
            X_test_list = []
            
            for ticker in tickers:
                # Training
                ticker_train = features[
                    (features['date'].isin(train_dates)) & 
                    (features['ticker'] == ticker)
                ]
                ticker_train_targets = targets[
                    (targets['date'].isin(train_dates)) & 
                    (targets['ticker'] == ticker)
                ]
                
                if len(ticker_train) > 0 and len(ticker_train_targets) > 0:
                    X_train_list.append(ticker_train.drop(columns=['date', 'ticker']))
                    y_train_list.append(ticker_train_targets['target'])
                
                # Calibration
                ticker_cal = features[
                    (features['date'].isin(cal_dates)) & 
                    (features['ticker'] == ticker)
                ]
                ticker_cal_targets = targets[
                    (targets['date'].isin(cal_dates)) & 
                    (targets['ticker'] == ticker)
                ]
                
                if len(ticker_cal) > 0 and len(ticker_cal_targets) > 0:
                    X_cal_list.append(ticker_cal.drop(columns=['date', 'ticker']))
                    y_cal_list.append(ticker_cal_targets['target'])
                
                # Test (current date)
                ticker_test = features[
                    (features['date'] == test_date) & 
                    (features['ticker'] == ticker)
                ]
                
                if len(ticker_test) > 0:
                    X_test_list.append(ticker_test.drop(columns=['date', 'ticker']))
            
            if len(X_train_list) == 0 or len(X_test_list) != n_assets:
                continue
            
            X_train = pd.concat(X_train_list, axis=0)
            y_train = pd.concat(y_train_list, axis=0)
            X_cal = pd.concat(X_cal_list, axis=0)
            y_cal = pd.concat(y_cal_list, axis=0)
            X_test = pd.concat(X_test_list, axis=0)
            
            # Fit model
            try:
                if conformal_predictor:
                    # Conformal prediction strategy
                    conformal_predictor.fit(X_train, y_train, X_cal, y_cal)
                    y_pred, lower, upper = conformal_predictor.predict(X_test)
                    
                    # Track coverage/width
                    forecasts_history.append(y_pred)
                    width_history.append(np.mean(upper - lower))
                    
                else:
                    # Non-CP strategy
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    lower, upper = None, None
                
            except Exception as e:
                print(f"Error at {test_date}: {e}")
                continue
            
            # Get covariance matrix
            cov_matrix = self.data_loader.get_covariance_matrix(test_date)
            cov_matrix = cov_matrix.loc[tickers, tickers].values
            
            # Compute weights
            try:
                if strategy_name == 'mean_variance':
                    weights = allocator.allocate(y_pred, cov_matrix, prev_weights)
                elif strategy_name == 'risk_parity':
                    weights = allocator.allocate(cov_matrix, prev_weights)
                elif strategy_name == 'ml_only':
                    weights = allocator.allocate(y_pred, cov_matrix, prev_weights)
                elif strategy_name in ['cp_gate', 'cp_size']:
                    weights = allocator.allocate(y_pred, lower, upper, cov_matrix, prev_weights)
                elif strategy_name == 'cp_lower_bound':
                    weights = allocator.allocate(lower, cov_matrix, prev_weights)
                else:
                    weights = prev_weights
                    
            except Exception as e:
                print(f"Allocation error at {test_date}: {e}")
                weights = prev_weights
            
            # Calculate turnover
            current_turnover = np.sum(np.abs(weights - prev_weights))
            turnover.append(current_turnover)
            
            # Calculate transaction costs
            transaction_cost = current_turnover * self.transaction_cost
            
            # Get next period return
            next_date_idx = returns.index.get_loc(test_date) + 1
            if next_date_idx >= len(returns):
                break
            
            next_date = returns.index[next_date_idx]
            asset_returns = returns.loc[next_date, tickers].values
            
            # Portfolio return (after costs)
            portfolio_ret = np.dot(weights, asset_returns) - transaction_cost
            portfolio_returns.append(portfolio_ret)
            positions.append(weights)
            
            # Update for next iteration
            prev_weights = weights
        
        # Convert to Series/DataFrame
        portfolio_returns = pd.Series(
            portfolio_returns,
            index=test_dates[:len(portfolio_returns) * self.rebalance_freq:self.rebalance_freq]
        )
        
        positions = pd.DataFrame(
            positions,
            columns=tickers,
            index=test_dates[:len(positions) * self.rebalance_freq:self.rebalance_freq]
        )
        
        # Get regime labels for test period
        regime_labels = None
        if self.regime_detector:
            regime_labels = pd.Series(
                [self.regime_detector.get_regime_at_date(d) for d in portfolio_returns.index],
                index=portfolio_returns.index
            )
        
        # Calculate metrics (annualize=True for main backtest)
        metrics = PerformanceMetrics.calculate_metrics(
            portfolio_returns,
            positions,
            regime_labels,
            annualize=True
        )
        
        # Add CP-specific metrics
        if len(width_history) > 0:
            metrics['avg_interval_width'] = np.mean(width_history)
        
        results = {
            'returns': portfolio_returns,
            'positions': positions,
            'weights_history': positions.values,  # Add weights history for visualization
            'turnover': turnover,
            'metrics': metrics,
            'regime_labels': regime_labels
        }
        
        return results
    
    def run_all_strategies(self,
                          model_class,
                          allocators: Dict,
                          test_start_date: str = '2020-01-01',
                          alpha: float = 0.1) -> Dict:
        """
        Run backtest for all strategies.
        
        Parameters:
        -----------
        model_class : class
            ML model class to instantiate
        allocators : dict
            Dictionary of strategy name -> allocator
        test_start_date : str
            Start date for testing
        alpha : float
            Conformal prediction alpha level
            
        Returns:
        --------
        dict : Results for all strategies
        """
        from conformal import SplitConformalPredictor
        
        all_results = {}
        
        for strategy_name, allocator in allocators.items():
            # Create fresh model instance
            model = model_class()
            
            # Create conformal predictor for CP strategies
            if strategy_name.startswith('cp_'):
                cp_model = model_class()
                conformal_predictor = SplitConformalPredictor(cp_model, alpha=alpha)
            else:
                conformal_predictor = None
            
            # Run backtest
            results = self.backtest_strategy(
                strategy_name,
                model,
                conformal_predictor,
                allocator,
                test_start_date
            )
            
            all_results[strategy_name] = results
        
        self.results = all_results
        return all_results
    
    def get_performance_table(self) -> pd.DataFrame:
        """
        Create performance comparison table.
        
        Returns:
        --------
        pd.DataFrame : Performance metrics for all strategies
        """
        if not self.results:
            return pd.DataFrame()
        
        metrics_df = pd.DataFrame({
            name: results['metrics']
            for name, results in self.results.items()
        }).T
        
        return metrics_df
    
    def stress_test(self, 
                   start_date: str,
                   end_date: str,
                   period_name: str = 'Stress Period') -> pd.DataFrame:
        """
        Perform stress test for a specific period.
        
        Parameters:
        -----------
        start_date : str
            Start date of stress period
        end_date : str
            End date of stress period
        period_name : str
            Name of the stress period
            
        Returns:
        --------
        pd.DataFrame : Stress test results
        """
        stress_metrics = {}
        
        for strategy_name, results in self.results.items():
            returns = results['returns']
            period_returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]
            
            if len(period_returns) > 0:
                # Don't annualize short stress test periods (misleading)
                metrics = PerformanceMetrics.calculate_metrics(period_returns, annualize=False)
                # Add period length for context
                metrics['n_days'] = len(period_returns)
                stress_metrics[strategy_name] = metrics
        
        stress_df = pd.DataFrame(stress_metrics).T
        stress_df['period'] = period_name
        
        return stress_df


if __name__ == '__main__':
    print("Backtesting framework ready!")
    print("Use with data_loader, models, conformal predictors, and allocators.")
