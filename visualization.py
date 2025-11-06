"""
Visualization utilities for conformal prediction and portfolio analysis.
Provides comprehensive plotting functions for:
- Conformal prediction intervals
- Portfolio allocation analysis
- Performance metrics
- Regime analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


class CPVisualizer:
    """Visualization tools for conformal prediction intervals."""
    
    @staticmethod
    def plot_prediction_intervals(forecasts: np.ndarray,
                                  lower: np.ndarray,
                                  upper: np.ndarray,
                                  actuals: Optional[np.ndarray] = None,
                                  dates: Optional[pd.DatetimeIndex] = None,
                                  asset_name: str = "Asset",
                                  figsize: Tuple[int, int] = (14, 6)):
        """
        Plot prediction intervals with forecasts and actuals.
        
        Parameters:
        -----------
        forecasts : np.ndarray
            Point predictions
        lower : np.ndarray
            Lower bounds
        upper : np.ndarray
            Upper bounds
        actuals : np.ndarray, optional
            Actual values
        dates : pd.DatetimeIndex, optional
            Time index
        asset_name : str
            Name for title
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = dates if dates is not None else np.arange(len(forecasts))
        
        # Plot intervals
        ax.fill_between(x, lower, upper, alpha=0.3, color='skyblue', 
                        label='90% Prediction Interval')
        ax.plot(x, forecasts, 'b-', linewidth=2, label='Point Forecast', alpha=0.8)
        
        if actuals is not None:
            ax.scatter(x, actuals, c='red', s=20, alpha=0.6, label='Actual Values', zorder=5)
            
            # Highlight miscoverage
            outside = (actuals < lower) | (actuals > upper)
            if np.any(outside):
                ax.scatter(x[outside], actuals[outside], c='orange', s=100, 
                          marker='x', linewidth=3, label=f'Miscoverage ({np.sum(outside)})', zorder=6)
        
        ax.set_xlabel('Time' if dates is not None else 'Sample Index', fontsize=12)
        ax.set_ylabel('Return', fontsize=12)
        ax.set_title(f'{asset_name}: Conformal Prediction Intervals', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_interval_width_distribution(widths: np.ndarray,
                                         coverage: float,
                                         target_coverage: float = 0.90,
                                         figsize: Tuple[int, int] = (12, 5)):
        """Plot distribution of interval widths."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax = axes[0]
        ax.hist(widths, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(widths.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {widths.mean():.4f}')
        ax.axvline(np.median(widths), color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {np.median(widths):.4f}')
        ax.set_xlabel('Interval Width', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Interval Widths', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Coverage gauge
        ax = axes[1]
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, alpha=0.3)
        
        # Coverage arc
        coverage_theta = coverage * np.pi
        coverage_arc = np.linspace(0, coverage_theta, 100)
        color = 'green' if abs(coverage - target_coverage) < 0.05 else 'orange'
        ax.plot(np.cos(coverage_arc), np.sin(coverage_arc), color=color, linewidth=8)
        
        # Target line
        target_theta = target_coverage * np.pi
        ax.plot([0, np.cos(target_theta)], [0, np.sin(target_theta)], 
               'r--', linewidth=2, label=f'Target: {target_coverage:.0%}')
        
        # Text
        ax.text(0, -0.3, f'{coverage:.1%}', fontsize=24, ha='center', fontweight='bold')
        ax.text(0, -0.5, 'Coverage', fontsize=14, ha='center')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.axis('off')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_calibration_analysis(forecasts: np.ndarray,
                                  actuals: np.ndarray,
                                  lower: np.ndarray,
                                  upper: np.ndarray,
                                  figsize: Tuple[int, int] = (14, 10)):
        """Comprehensive calibration analysis plots."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Residual distribution
        ax1 = fig.add_subplot(gs[0, 0])
        residuals = actuals - forecasts
        ax1.hist(residuals, bins=40, alpha=0.7, color='steelblue', edgecolor='black', density=True)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Residual', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. QQ plot
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Coverage by forecast magnitude
        ax3 = fig.add_subplot(gs[1, 0])
        forecast_bins = pd.qcut(np.abs(forecasts), q=5, duplicates='drop')
        coverage_by_bin = []
        bin_labels = []
        
        for bin_val in forecast_bins.categories:
            mask = forecast_bins == bin_val
            if mask.sum() > 0:
                coverage = np.mean((actuals[mask] >= lower[mask]) & (actuals[mask] <= upper[mask]))
                coverage_by_bin.append(coverage)
                bin_labels.append(f'{bin_val.left:.3f} to {bin_val.right:.3f}')
        
        ax3.bar(range(len(coverage_by_bin)), coverage_by_bin, alpha=0.7, color='steelblue')
        ax3.axhline(0.9, color='red', linestyle='--', linewidth=2, label='Target: 90%')
        ax3.set_xlabel('Forecast Magnitude Bin', fontsize=11)
        ax3.set_ylabel('Coverage', fontsize=11)
        ax3.set_title('Coverage by Forecast Magnitude', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(bin_labels)))
        ax3.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Interval width vs absolute forecast
        ax4 = fig.add_subplot(gs[1, 1])
        widths = upper - lower
        ax4.scatter(np.abs(forecasts), widths, alpha=0.5, s=30)
        ax4.set_xlabel('|Forecast|', fontsize=11)
        ax4.set_ylabel('Interval Width', fontsize=11)
        ax4.set_title('Interval Width vs Forecast Magnitude', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation text
        corr = np.corrcoef(np.abs(forecasts), widths)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax4.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig


class PortfolioVisualizer:
    """Visualization tools for portfolio allocation and performance."""
    
    @staticmethod
    def plot_allocation_weights(weights_dict: Dict[str, np.ndarray],
                               dates: pd.DatetimeIndex,
                               asset_names: List[str],
                               figsize: Tuple[int, int] = (16, 10)):
        """Plot allocation weights over time for multiple strategies."""
        n_strategies = len(weights_dict)
        fig, axes = plt.subplots(n_strategies, 1, figsize=figsize, sharex=True)
        
        if n_strategies == 1:
            axes = [axes]
        
        for ax, (strategy_name, weights) in zip(axes, weights_dict.items()):
            # Weights is (n_dates, n_assets)
            weights_df = pd.DataFrame(weights, index=dates, columns=asset_names)
            
            # Stacked area chart
            ax.stackplot(dates, weights_df.T, labels=asset_names, alpha=0.7)
            ax.set_ylabel('Weight', fontsize=11)
            ax.set_title(f'{strategy_name.replace("_", " ").title()}', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
        
        axes[-1].set_xlabel('Date', fontsize=12)
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_portfolio_concentration(weights_dict: Dict[str, np.ndarray],
                                    strategy_names: Optional[List[str]] = None,
                                    figsize: Tuple[int, int] = (14, 6)):
        """Plot portfolio concentration metrics (HHI, effective N)."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        if strategy_names is None:
            strategy_names = list(weights_dict.keys())
        
        # Compute metrics
        hhi_list = []
        eff_n_list = []
        
        for strategy_name in strategy_names:
            weights = weights_dict[strategy_name]
            # IMPORTANT: Calculate HHI and Effective N per period, then average
            # This correctly handles strategies that allocate to different assets over time
            # (e.g., CP-Gate which filters uncertain assets)
            hhi_per_period = np.array([(w ** 2).sum() for w in weights])
            eff_n_per_period = 1 / np.maximum(hhi_per_period, 1e-12)  # Avoid division by zero
            
            # Average across periods
            avg_hhi = hhi_per_period.mean()
            avg_eff_n = eff_n_per_period.mean()
            
            hhi_list.append(avg_hhi)
            eff_n_list.append(avg_eff_n)
        
        # HHI plot
        ax = axes[0]
        bars = ax.bar(range(len(strategy_names)), hhi_list, alpha=0.7, color='steelblue')
        ax.set_ylabel('Herfindahl-Hirschman Index', fontsize=11)
        ax.set_title('Portfolio Concentration (HHI)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(strategy_names)))
        ax.set_xticklabels([s.replace('_', ' ').title() for s in strategy_names], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, hhi_list)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Effective N plot
        ax = axes[1]
        bars = ax.bar(range(len(strategy_names)), eff_n_list, alpha=0.7, color='coral')
        ax.set_ylabel('Effective Number of Assets', fontsize=11)
        ax.set_title('Portfolio Diversification', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(strategy_names)))
        ax.set_xticklabels([s.replace('_', ' ').title() for s in strategy_names], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, eff_n_list)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_rolling_performance(returns_dict: Dict[str, pd.Series],
                                 window: int = 63,
                                 figsize: Tuple[int, int] = (16, 10)):
        """Plot rolling performance metrics."""
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        for strategy_name, returns in returns_dict.items():
            # Rolling Sharpe
            rolling_sharpe = (returns.rolling(window).mean() / 
                            returns.rolling(window).std() * np.sqrt(252))
            axes[0].plot(rolling_sharpe.index, rolling_sharpe.values, 
                        label=strategy_name.replace('_', ' ').title(), linewidth=2, alpha=0.8)
            
            # Rolling volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            axes[1].plot(rolling_vol.index, rolling_vol.values, 
                        label=strategy_name.replace('_', ' ').title(), linewidth=2, alpha=0.8)
            
            # Cumulative returns
            cum_returns = (1 + returns).cumprod()
            axes[2].plot(cum_returns.index, cum_returns.values, 
                        label=strategy_name.replace('_', ' ').title(), linewidth=2, alpha=0.8)
        
        axes[0].set_ylabel('Sharpe Ratio', fontsize=11)
        axes[0].set_title(f'Rolling Sharpe Ratio ({window}-day window)', 
                         fontsize=12, fontweight='bold')
        axes[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].legend(loc='upper left', ncol=3, fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_ylabel('Volatility (ann.)', fontsize=11)
        axes[1].set_title(f'Rolling Volatility ({window}-day window)', 
                         fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper left', ncol=3, fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_ylabel('Cumulative Return', fontsize=11)
        axes[2].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        axes[2].legend(loc='upper left', ncol=3, fontsize=9)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        return fig, axes


class RegimeVisualizer:
    """Visualization tools for market regime analysis."""
    
    @staticmethod
    def plot_regime_timeline(regimes: pd.Series,
                            returns: pd.Series,
                            regime_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (16, 8)):
        """Plot regime timeline with returns."""
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])
        
        if regime_names is None:
            regime_names = [f'Regime {i}' for i in range(int(regimes.max()) + 1)]
        
        # Cumulative returns with regime shading
        cum_returns = (1 + returns).cumprod()
        axes[0].plot(cum_returns.index, cum_returns.values, 'k-', linewidth=2)
        
        # Shade regimes
        colors = plt.cm.Set3(np.linspace(0, 1, len(regime_names)))
        for i, regime_name in enumerate(regime_names):
            regime_mask = regimes == i
            if regime_mask.sum() > 0:
                regime_dates = regimes[regime_mask].index
                for date in regime_dates:
                    axes[0].axvspan(date, date + pd.Timedelta(days=1), 
                                  alpha=0.3, color=colors[i])
        
        axes[0].set_ylabel('Cumulative Return', fontsize=12)
        axes[0].set_title('Market Regimes and Performance', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], alpha=0.3, label=regime_names[i]) 
                          for i in range(len(regime_names))]
        axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Regime indicator
        axes[1].fill_between(regimes.index, 0, regimes.values, step='post', alpha=0.7)
        axes[1].set_ylabel('Regime', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_yticks(range(len(regime_names)))
        axes[1].set_yticklabels(regime_names)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig, axes


def create_comparison_dashboard(results: Dict,
                                perf_table: pd.DataFrame,
                                figsize: Tuple[int, int] = (20, 12)):
    """Create comprehensive comparison dashboard."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Cumulative returns
    ax1 = fig.add_subplot(gs[0, :])
    for strategy_name, result in results.items():
        cum_ret = (1 + result['returns']).cumprod()
        ax1.plot(cum_ret.index, cum_ret.values, 
                label=strategy_name.replace('_', ' ').title(), linewidth=2, alpha=0.8)
    ax1.set_ylabel('Cumulative Return', fontsize=11)
    ax1.set_title('Strategy Performance Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Sharpe ratio comparison
    ax2 = fig.add_subplot(gs[1, 0])
    sharpe_ratios = perf_table['sharpe'].sort_values(ascending=True)
    colors = ['green' if x > 0 else 'red' for x in sharpe_ratios.values]
    ax2.barh(range(len(sharpe_ratios)), sharpe_ratios.values, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(sharpe_ratios)))
    ax2.set_yticklabels([s.replace('_', ' ').title() for s in sharpe_ratios.index], fontsize=9)
    ax2.set_xlabel('Sharpe Ratio', fontsize=10)
    ax2.set_title('Risk-Adjusted Performance', fontsize=11, fontweight='bold')
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Turnover comparison
    ax3 = fig.add_subplot(gs[1, 1])
    turnover = perf_table['avg_turnover'].sort_values(ascending=False) * 100
    ax3.bar(range(len(turnover)), turnover.values, alpha=0.7, color='coral')
    ax3.set_xticks(range(len(turnover)))
    ax3.set_xticklabels([s.replace('_', ' ').title() for s in turnover.index], 
                       rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Avg Turnover (%)', fontsize=10)
    ax3.set_title('Trading Activity', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Max drawdown
    ax4 = fig.add_subplot(gs[1, 2])
    mdd = perf_table['max_drawdown'].sort_values(ascending=True) * 100
    ax4.barh(range(len(mdd)), mdd.values, alpha=0.7, color='darkred')
    ax4.set_yticks(range(len(mdd)))
    ax4.set_yticklabels([s.replace('_', ' ').title() for s in mdd.index], fontsize=9)
    ax4.set_xlabel('Max Drawdown (%)', fontsize=10)
    ax4.set_title('Downside Risk', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5-6. Distribution comparisons
    ax5 = fig.add_subplot(gs[2, :2])
    for strategy_name, result in results.items():
        ax5.hist(result['returns'].dropna() * 100, bins=50, alpha=0.3, 
                label=strategy_name.replace('_', ' ').title(), density=True)
    ax5.set_xlabel('Daily Return (%)', fontsize=10)
    ax5.set_ylabel('Density', fontsize=10)
    ax5.set_title('Return Distributions', fontsize=11, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # 7. Calmar ratio
    ax6 = fig.add_subplot(gs[2, 2])
    calmar = perf_table['calmar'].sort_values(ascending=True)
    colors_calmar = ['green' if x > 1 else 'orange' for x in calmar.values]
    ax6.barh(range(len(calmar)), calmar.values, color=colors_calmar, alpha=0.7)
    ax6.set_yticks(range(len(calmar)))
    ax6.set_yticklabels([s.replace('_', ' ').title() for s in calmar.index], fontsize=9)
    ax6.set_xlabel('Calmar Ratio', fontsize=10)
    ax6.set_title('Return/Drawdown', fontsize=11, fontweight='bold')
    ax6.axvline(1, color='black', linestyle='--', linewidth=1)
    ax6.grid(True, alpha=0.3, axis='x')
    
    return fig


if __name__ == '__main__':
    print("Visualization utilities loaded successfully!")
    print("Available classes:")
    print("  - CPVisualizer: Conformal prediction plots")
    print("  - PortfolioVisualizer: Portfolio allocation plots")
    print("  - RegimeVisualizer: Market regime analysis")
    print("  - create_comparison_dashboard: Comprehensive dashboard")

