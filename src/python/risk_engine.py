"""
python/risk_engine.py - Portfolio risk engine.

RISK METRICS IMPLEMENTED:
1. VaR (Value at Risk) - three methods
2. CVaR (Conditional VaR / Expected Shortfall)
3. Sharpe and Sortino ratios
4. Maximum drawdown and drawdown statistics
5. Volatility and correlation

VaR METHODS:
- Parametric (variance-covariance): assumes normal returns
  VaR = μ - z_α × σ  where z_α = 1.645 for 95% confidence
  Fast, simple, wrong for fat tails.

- Historical simulation: uses the empirical return distribution
  VaR = percentile(returns, 1-confidence)
  No distribution assumption. Accurate but needs long history.

- Monte Carlo: simulates future paths from a fitted model
  VaR = percentile(simulated_returns, 1-confidence)
  Most flexible. Can incorporate fat tails, correlation, regime changes.

CVaR (EXPECTED SHORTFALL):
CVaR = E[loss | loss > VaR]
= average loss in the worst (1-confidence)% of scenarios
Satisfies subadditivity → measures diversification correctly.
Preferred by Basel III. VaR does NOT satisfy subadditivity.

INTERVIEW QUESTION:
"What's wrong with VaR?"
A: "Three problems:
1. Not subadditive - adding positions can increase VaR (penalises diversification)
2. Tells you nothing about the severity of losses beyond the threshold
3. Underestimates tail risk for fat-tailed distributions
CVaR/Expected Shortfall fixes all three."
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

DARK   = '#0F1117'
PANEL  = '#1A1D27'
GRID   = '#2A2D3A'
TEXT   = '#C8CCD8'
BLUE   = '#4C9BE8'
GREEN  = '#56C27D'
RED    = '#E85C5C'
AMBER  = '#F5A623'
PURPLE = '#9B7FD4'
MUTED  = '#6B7280'


class RiskEngine:
    """
    Portfolio risk engine. Computes all standard risk metrics.

    Usage:
        engine = RiskEngine(returns, initial_capital=1_000_000)
        report = engine.full_report()
        engine.plot(save_path='outputs/risk_report.png')
    """

    def __init__(self, returns: pd.Series, initial_capital: float = 1_000_000,
                  freq: str = 'daily'):
        """
        returns:         pd.Series of portfolio returns (log or simple)
        initial_capital: portfolio size in ₹ for absolute risk metrics
        freq:            'daily', 'weekly', or 'monthly' - for annualisation
        """
        self.returns  = returns.dropna()
        self.capital  = initial_capital
        self.freq     = freq
        self._ann     = {'daily': 252, 'weekly': 52, 'monthly': 12}[freq]
        self.equity   = (1 + self.returns).cumprod() * initial_capital

    # ── VaR methods ───────────────────────────────────────────────────────────

    def var_parametric(self, confidence: float = 0.95) -> float:
        """
        Parametric VaR assuming normally distributed returns.

        VaR_α = -(μ - z_α × σ)  expressed as positive loss

        z_α for common confidence levels:
        90%: 1.282   95%: 1.645   99%: 2.326   99.9%: 3.090
        """
        mu    = self.returns.mean()
        sigma = self.returns.std()
        z     = stats.norm.ppf(1 - confidence)
        var   = -(mu + z * sigma)  # positive loss
        return float(var)

    def var_historical(self, confidence: float = 0.95) -> float:
        """
        Historical simulation VaR - empirical percentile of losses.

        Loss distribution = -returns (positive = bad)
        VaR = percentile of losses at (1-confidence)% level
        """
        losses = -self.returns
        var    = float(losses.quantile(confidence))
        return var

    def var_monte_carlo(self, confidence: float = 0.95,
                         n_sim: int = 10_000) -> float:
        """
        Monte Carlo VaR - simulate from fitted normal distribution.
        For production: use fat-tailed distribution (Student-t, historical bootstrap).
        """
        np.random.seed(42)
        mu    = self.returns.mean()
        sigma = self.returns.std()
        sim   = np.random.normal(mu, sigma, n_sim)
        losses = -pd.Series(sim)
        return float(losses.quantile(confidence))

    def var_all_methods(self, confidence: float = 0.95) -> dict:
        """Compute VaR using all three methods."""
        return {
            'parametric':    self.var_parametric(confidence),
            'historical':    self.var_historical(confidence),
            'monte_carlo':   self.var_monte_carlo(confidence),
            'confidence':    confidence,
            'capital_at_risk_parametric': self.var_parametric(confidence) * self.capital,
            'capital_at_risk_historical': self.var_historical(confidence) * self.capital,
        }

    # ── CVaR ──────────────────────────────────────────────────────────────────

    def cvar(self, confidence: float = 0.95) -> float:
        """
        CVaR (Conditional Value at Risk) = Expected Shortfall.

        CVaR_α = E[loss | loss > VaR_α]
               = mean of all losses exceeding VaR

        Always ≥ VaR. Measures the average severity of tail losses.
        """
        var    = self.var_historical(confidence)
        losses = -self.returns
        tail   = losses[losses >= var]
        return float(tail.mean()) if not tail.empty else var

    # ── Return statistics ──────────────────────────────────────────────────────

    def sharpe_ratio(self, risk_free_rate: float = 0.06) -> float:
        """
        Sharpe ratio = (annualised return - risk-free rate) / annualised vol.
        Uses log returns; annualised correctly.
        """
        rf_daily  = (1 + risk_free_rate) ** (1 / self._ann) - 1
        excess    = self.returns - rf_daily
        return float((excess.mean() / excess.std()) * np.sqrt(self._ann))

    def sortino_ratio(self, risk_free_rate: float = 0.06,
                       target: float = 0.0) -> float:
        """
        Sortino ratio = excess return / downside deviation.

        Uses only negative returns for the denominator - penalises
        downside volatility only, not upside.
        """
        rf_daily    = (1 + risk_free_rate) ** (1 / self._ann) - 1
        excess      = self.returns - rf_daily
        downside    = excess[excess < target]
        down_dev    = np.sqrt((downside ** 2).mean()) * np.sqrt(self._ann)
        ann_excess  = excess.mean() * self._ann
        return float(ann_excess / down_dev) if down_dev > 0 else 0.0

    def calmar_ratio(self) -> float:
        """Calmar = annualised return / max drawdown."""
        ann_ret = self.returns.mean() * self._ann
        mdd     = self.max_drawdown()
        return float(ann_ret / abs(mdd)) if mdd != 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────

    def drawdown_series(self) -> pd.Series:
        """Running drawdown from the most recent peak."""
        cumret  = (1 + self.returns).cumprod()
        running_max = cumret.cummax()
        return (cumret / running_max - 1)

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown (negative number)."""
        return float(self.drawdown_series().min())

    def drawdown_stats(self) -> dict:
        """Full drawdown analysis: max, average, recovery time."""
        dd        = self.drawdown_series()
        in_dd     = dd < 0
        mdd       = float(dd.min())

        # Average drawdown (only during drawdown periods)
        avg_dd = float(dd[in_dd].mean()) if in_dd.any() else 0.0

        # Longest drawdown duration (consecutive bars below peak)
        max_dur = 0
        current = 0
        for x in in_dd:
            current = current + 1 if x else 0
            max_dur = max(max_dur, current)

        return {
            'max_drawdown':          round(mdd, 4),
            'avg_drawdown':          round(avg_dd, 4),
            'max_drawdown_duration': max_dur,
            'pct_time_in_drawdown':  round(in_dd.mean(), 4),
        }

    # ── Full report ───────────────────────────────────────────────────────────

    def full_report(self, confidence: float = 0.95,
                     risk_free: float = 0.06) -> dict:
        """Compute all risk metrics and print a formatted report."""
        ann        = self._ann
        ann_ret    = self.returns.mean() * ann
        ann_vol    = self.returns.std() * np.sqrt(ann)
        total_ret  = (self.equity.iloc[-1] / self.capital) - 1
        var_dict   = self.var_all_methods(confidence)
        dd_stats   = self.drawdown_stats()

        report = {
            'period':           f'{self.returns.index[0].date()} → {self.returns.index[-1].date()}',
            'n_periods':        len(self.returns),
            'total_return':     round(total_ret, 4),
            'annualised_return': round(ann_ret, 4),
            'annualised_vol':   round(ann_vol, 4),
            'sharpe_ratio':     round(self.sharpe_ratio(risk_free), 3),
            'sortino_ratio':    round(self.sortino_ratio(risk_free), 3),
            'calmar_ratio':     round(self.calmar_ratio(), 3),
            'var_95_param':     round(var_dict['parametric'], 4),
            'var_95_hist':      round(var_dict['historical'], 4),
            'var_95_mc':        round(var_dict['monte_carlo'], 4),
            'cvar_95':          round(self.cvar(confidence), 4),
            'max_drawdown':     dd_stats['max_drawdown'],
            'avg_drawdown':     dd_stats['avg_drawdown'],
            'max_dd_duration':  dd_stats['max_drawdown_duration'],
            'pct_in_drawdown':  dd_stats['pct_time_in_drawdown'],
            'capital_at_risk':  round(var_dict['capital_at_risk_historical'], 0),
        }

        print(f"\n{'═'*55}")
        print(f"  RISK REPORT")
        print(f"{'═'*55}")
        print(f"  Period              : {report['period']}")
        print(f"  Periods             : {report['n_periods']}")
        print(f"\n  RETURNS")
        print(f"  Total return        : {report['total_return']:>+10.2%}")
        print(f"  Annualised return   : {report['annualised_return']:>+10.2%}")
        print(f"  Annualised vol      : {report['annualised_vol']:>10.2%}")
        print(f"\n  RISK-ADJUSTED")
        print(f"  Sharpe ratio        : {report['sharpe_ratio']:>10.3f}")
        print(f"  Sortino ratio       : {report['sortino_ratio']:>10.3f}")
        print(f"  Calmar ratio        : {report['calmar_ratio']:>10.3f}")
        print(f"\n  VALUE AT RISK (95%)")
        print(f"  Parametric VaR      : {report['var_95_param']:>10.2%}  (₹{report['var_95_param']*self.capital:>12,.0f})")
        print(f"  Historical VaR      : {report['var_95_hist']:>10.2%}  (₹{report['var_95_hist']*self.capital:>12,.0f})")
        print(f"  Monte Carlo VaR     : {report['var_95_mc']:>10.2%}  (₹{report['var_95_mc']*self.capital:>12,.0f})")
        print(f"  CVaR (Exp Shortfall): {report['cvar_95']:>10.2%}  (₹{report['cvar_95']*self.capital:>12,.0f})")
        print(f"\n  DRAWDOWN")
        print(f"  Max drawdown        : {report['max_drawdown']:>10.2%}")
        print(f"  Avg drawdown        : {report['avg_drawdown']:>10.2%}")
        print(f"  Max DD duration     : {report['max_dd_duration']:>10d} periods")
        print(f"  % time in drawdown  : {report['pct_in_drawdown']:>10.1%}")
        print(f"{'═'*55}")

        return report

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot(self, save_path: str = None):
        """4-panel risk dashboard."""
        fig = plt.figure(figsize=(14, 11))
        fig.patch.set_facecolor(DARK)
        gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

        # Panel 1: Equity curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(PANEL)
        ax1.plot(self.equity.index, self.equity.values, color=BLUE, linewidth=1.5)
        ax1.set_title('Portfolio equity curve', color=TEXT, fontsize=10)
        ax1.set_ylabel('Portfolio value (₹)', color=TEXT)

        # Panel 2: Drawdown
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(PANEL)
        dd = self.drawdown_series() * 100
        ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.5)
        ax2.plot(dd.index, dd.values, color=RED, linewidth=0.8)
        ax2.set_title(f'Drawdown (%)  |  Max={self.max_drawdown():.1%}', color=TEXT, fontsize=10)
        ax2.set_ylabel('Drawdown (%)', color=TEXT)

        # Panel 3: Return distribution + VaR
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(PANEL)
        clean = self.returns.dropna()
        n_bins = min(60, len(clean) // 5)
        ax3.hist(clean * 100, bins=n_bins, color=BLUE, alpha=0.7,
                 edgecolor=PANEL, density=True, label='Returns (%)')
        var95 = self.var_historical(0.95)
        cvar95 = self.cvar(0.95)
        ax3.axvline(-var95 * 100, color=AMBER, linewidth=2,
                    linestyle='--', label=f'95% VaR={var95:.2%}')
        ax3.axvline(-cvar95 * 100, color=RED, linewidth=2,
                    linestyle='--', label=f'95% CVaR={cvar95:.2%}')
        x = np.linspace(clean.min() * 100, clean.max() * 100, 200)
        ax3.plot(x, stats.norm.pdf(x, clean.mean()*100, clean.std()*100),
                 color=GREEN, linewidth=1.5, linestyle=':', alpha=0.8, label='Normal fit')
        ax3.set_title('Return distribution + VaR', color=TEXT, fontsize=10)
        ax3.set_xlabel('Daily return (%)', color=TEXT)
        ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

        # Panel 4: Rolling VaR
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(PANEL)
        roll_var = (-self.returns).rolling(63).quantile(0.95) * 100
        ax4.fill_between(roll_var.index, roll_var.values, color=AMBER, alpha=0.5)
        ax4.plot(roll_var.index, roll_var.values, color=AMBER, linewidth=1, label='Rolling 95% VaR (63d)')
        ax4.axhline(var95 * 100, color=RED, linewidth=1, linestyle='--',
                    label=f'Full-period VaR={var95:.2%}')
        ax4.set_title('Rolling 63-day 95% VaR (%)', color=TEXT, fontsize=10)
        ax4.set_ylabel('VaR (%)', color=TEXT)
        ax4.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(colors=TEXT)
            ax.yaxis.label.set_color(TEXT)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4)

        plt.suptitle('Portfolio Risk Dashboard', color=TEXT, fontsize=13, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
            print(f"  Saved: {save_path}")
        else:
            plt.show()
        plt.close()
