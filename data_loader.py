"""
Data loading and preprocessing for ETF portfolio allocation.
Fetches SPY and sector ETFs, computes features, and prepares data for modeling.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Dict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and preprocess ETF data for portfolio allocation."""
    
    # Default sector ETFs + SPY
    DEFAULT_TICKERS = [
        'SPY',   # S&P 500
        'XLF',   # Financials
        'XLK',   # Technology
        'XLE',   # Energy
        'XLV',   # Healthcare
        'XLY',   # Consumer Discretionary
        'XLP',   # Consumer Staples
        'XLI',   # Industrials
        'XLB',   # Materials
        'XLU',   # Utilities
        'XLRE',  # Real Estate
    ]
    
    def __init__(self, 
                 tickers: List[str] = None,
                 start_date: str = '2015-01-01',
                 end_date: str = '2024-12-31',
                 lookback: int = 60):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        tickers : list of str
            ETF tickers to load
        start_date : str
            Start date for data
        end_date : str
            End date for data
        lookback : int
            Rolling window size for feature computation (trading days)
        """
        self.tickers = tickers if tickers else self.DEFAULT_TICKERS
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        
        self.prices = None
        self.returns = None
        self.features = None
        self.targets = None
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and prepare all data.
        
        Returns:
        --------
        dict with keys:
            'prices': Raw price data
            'returns': Log returns
            'features': Feature DataFrame (MultiIndex: date, ticker)
            'targets': Next-day returns
        """
        print(f"Loading data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}...")
        
        # Download price data with auto_adjust to get adjusted prices directly
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True  # This returns 'Close' instead of 'Adj Close'
        )
        
        # Extract close prices (adjusted due to auto_adjust=True)
        if len(self.tickers) == 1:
            self.prices = pd.DataFrame(data['Close'], columns=self.tickers)
        else:
            self.prices = data['Close']
        
        # Drop any tickers with missing data
        missing_pct = self.prices.isna().sum() / len(self.prices)
        valid_tickers = missing_pct[missing_pct < 0.05].index.tolist()
        
        if len(valid_tickers) < len(self.tickers):
            print(f"Dropping tickers with >5% missing data: {set(self.tickers) - set(valid_tickers)}")
            self.tickers = valid_tickers
            self.prices = self.prices[self.tickers]
        
        # Forward fill small gaps
        self.prices = self.prices.fillna(method='ffill', limit=5)
        
        # Compute log returns
        self.returns = np.log(self.prices / self.prices.shift(1))
        
        # Build features
        self._build_features()
        
        print(f"Data loaded: {len(self.prices)} days, {len(self.tickers)} assets")
        print(f"Features shape: {self.features.shape}")
        
        return {
            'prices': self.prices,
            'returns': self.returns,
            'features': self.features,
            'targets': self.targets
        }
    
    def _build_features(self):
        """Build feature matrix for ML models."""
        
        feature_list = []
        
        for ticker in self.tickers:
            ticker_features = pd.DataFrame(index=self.returns.index)
            ticker_features['ticker'] = ticker
            
            ret = self.returns[ticker]
            
            # Past returns (5, 10, 20, 60 days)
            for lag in [1, 5, 10, 20, 60]:
                ticker_features[f'ret_lag_{lag}'] = ret.shift(lag)
            
            # Rolling returns
            for window in [5, 10, 20, 60]:
                ticker_features[f'ret_roll_{window}'] = ret.rolling(window).mean()
            
            # Realized volatility
            for window in [5, 10, 20, 60]:
                ticker_features[f'vol_roll_{window}'] = ret.rolling(window).std() * np.sqrt(252)
            
            # Rolling Sharpe (annualized)
            for window in [20, 60]:
                roll_mean = ret.rolling(window).mean() * 252
                roll_std = ret.rolling(window).std() * np.sqrt(252)
                ticker_features[f'sharpe_roll_{window}'] = roll_mean / (roll_std + 1e-8)
            
            # Momentum indicators
            ticker_features['momentum_10_60'] = (
                ret.rolling(10).mean() - ret.rolling(60).mean()
            )
            
            # Min/max returns in window
            for window in [20, 60]:
                ticker_features[f'min_ret_{window}'] = ret.rolling(window).min()
                ticker_features[f'max_ret_{window}'] = ret.rolling(window).max()
            
            # Skewness and kurtosis
            for window in [20, 60]:
                ticker_features[f'skew_{window}'] = ret.rolling(window).skew()
                ticker_features[f'kurt_{window}'] = ret.rolling(window).kurt()
            
            # Target: next-day return
            ticker_features['target'] = ret.shift(-1)
            
            feature_list.append(ticker_features)
        
        # Combine all tickers
        self.features = pd.concat(feature_list, axis=0)
        self.features = self.features.reset_index()
        self.features.columns.name = None
        
        # Rename the index column to 'date' - handle different possible names
        if 'index' in self.features.columns:
            self.features = self.features.rename(columns={'index': 'date'})
        elif 'Date' in self.features.columns:
            self.features = self.features.rename(columns={'Date': 'date'})
        elif self.features.columns[0] not in ['ticker', 'date']:
            # First column is likely the date
            self.features = self.features.rename(columns={self.features.columns[0]: 'date'})
        
        # Remove rows with NaN (warm-up period)
        self.features = self.features.dropna()
        
        # Separate target
        self.targets = self.features[['date', 'ticker', 'target']].copy()
        self.features = self.features.drop(columns=['target'])
        
        print(f"Built {self.features.shape[1] - 2} features per asset")
    
    def get_train_test_split(self, 
                            train_end_date: str,
                            test_start_date: str = None,
                            test_end_date: str = None) -> Tuple:
        """
        Split data into train and test sets by date.
        
        Parameters:
        -----------
        train_end_date : str
            Last date for training data
        test_start_date : str, optional
            First date for test data (defaults to day after train_end_date)
        test_end_date : str, optional
            Last date for test data (defaults to end of data)
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : DataFrames
        """
        if test_start_date is None:
            train_end_idx = self.features[self.features['date'] <= train_end_date].index.max()
            test_start_date = self.features.loc[train_end_idx + 1, 'date']
        
        train_mask = self.features['date'] <= train_end_date
        test_mask = self.features['date'] >= test_start_date
        
        if test_end_date:
            test_mask &= self.features['date'] <= test_end_date
        
        X_train = self.features[train_mask].copy()
        X_test = self.features[test_mask].copy()
        
        # Merge with targets
        y_train = X_train[['date', 'ticker']].merge(self.targets, on=['date', 'ticker'], how='left')['target']
        y_test = X_test[['date', 'ticker']].merge(self.targets, on=['date', 'ticker'], how='left')['target']
        
        # Drop date and ticker for features
        X_train = X_train.drop(columns=['date', 'ticker'])
        X_test = X_test.drop(columns=['date', 'ticker'])
        
        return X_train, X_test, y_train, y_test
    
    def get_covariance_matrix(self, date: pd.Timestamp, lookback: int = None) -> pd.DataFrame:
        """
        Compute historical covariance matrix up to a given date.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Date for which to compute covariance
        lookback : int, optional
            Number of days to look back (defaults to self.lookback)
            
        Returns:
        --------
        pd.DataFrame : Covariance matrix
        """
        if lookback is None:
            lookback = self.lookback
        
        # Get returns up to date
        mask = (self.returns.index <= date)
        historical_returns = self.returns[mask].tail(lookback)
        
        # Compute covariance (annualized)
        cov_matrix = historical_returns.cov() * 252
        
        return cov_matrix
    
    def get_returns_history(self, date: pd.Timestamp, lookback: int = None) -> pd.DataFrame:
        """
        Get historical returns up to a given date.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Date for which to get historical returns
        lookback : int, optional
            Number of days to look back
            
        Returns:
        --------
        pd.DataFrame : Historical returns
        """
        if lookback is None:
            lookback = self.lookback
        
        mask = (self.returns.index <= date)
        return self.returns[mask].tail(lookback)
