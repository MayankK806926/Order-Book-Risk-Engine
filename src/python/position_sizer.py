"""
python/position_sizer.py — Position sizing: Kelly, vol-targeting, VaR limits.

THREE FRAMEWORKS FOR POSITION SIZING:

1. KELLY CRITERION (maximises long-run growth rate):
   f* = (p × b - q) / b   [for binary bets]
   f* = μ / σ²             [for continuous returns, Merton 1969]

   p = probability of winning
   b = odds (win/loss ratio)
   q = 1 - p = probability of losing

   f* = fraction of capital to bet on each trade.
   Full Kelly maximises log utility but has high volatility of returns.
   Fractional Kelly (f = f*/2 or f*/4) reduces volatility at cost of growth.

   INTERVIEW INSIGHT:
   "Why fractional Kelly?"
   Kelly assumes you know p and b exactly. In practice, you estimate them
   with error. Overestimating edge → overbetting → gambler's ruin.
   Fractional Kelly builds in a safety margin for estimation error.

2. VOLATILITY TARGETING:
   size_t = target_vol / realised_vol_t

   Scale position to maintain constant portfolio volatility over time.
   In high-vol regimes: reduce size. In low-vol: increase size.
   Used by risk-parity funds and CTA strategies.

   The TSMOM paper (Moskowitz et al.) showed vol-scaled momentum
   has better risk-adjusted returns than unscaled.

3. VaR-BASED LIMITS:
   Position such that 1-day 95% VaR ≤ risk_budget

   position = risk_budget / VaR_per_unit
            = risk_budget / (z_α × σ_stock)

   z_α = 1.645 for 95% confidence.
   Conservative — ensures daily losses stay within a risk budget.
   Used by banks to comply with Basel III.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# ── Kelly criterion ───────────────────────────────────────────────────────────

def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """
    Kelly fraction for a binary bet.

    f* = (p × b - q) / b
       = p - q/b
       = p - (1-p)/b

    Parameters:
    win_prob:      probability of winning (e.g. 0.55)
    win_loss_ratio: ratio of win size to loss size (e.g. 1.5 means
                    you win 1.5x for every 1x you risk)

    Returns: optimal fraction of capital to bet [0, 1].
    Negative → don't bet (negative edge).
    """
    p = win_prob
    q = 1 - p
    b = win_loss_ratio
    f = (p * b - q) / b
    return max(f, 0.0)


def kelly_continuous(returns: pd.Series,
                      lookback: int = 63) -> pd.Series:
    """
    Continuous Kelly fraction: f* = μ / σ²

    Derived from Merton's portfolio problem: maximise E[log(wealth)].

    This is the rolling Kelly fraction — recomputed every period
    using the most recent `lookback` days of returns.

    In practice, cap at 1.0 (never bet more than 100% of capital).
    Negative → short signal.
    """
    mu     = returns.rolling(lookback).mean()
    sigma2 = returns.rolling(lookback).var()
    kelly  = mu / sigma2.replace(0, np.nan)
    kelly  = kelly.clip(-1, 1)  # cap at ±100%
    kelly.name = f'kelly_{lookback}d'
    return kelly


def fractional_kelly(returns: pd.Series, fraction: float = 0.5,
                      lookback: int = 63) -> pd.Series:
    """
    Fractional Kelly: f = fraction × f*

    Common choices:
    fraction = 0.5 → half-Kelly (most common in practice)
    fraction = 0.25 → quarter-Kelly (very conservative)

    WHY FRACTIONAL KELLY:
    1. Estimation error: p and σ are estimated, not known
    2. Drawdown reduction: half-Kelly has ¾ of the growth but ½ the variance
    3. Parameter sensitivity: Kelly is very sensitive to μ estimation

    The growth-drawdown trade-off:
    Full Kelly: max growth, large drawdowns (~50% of capital typical)
    Half Kelly: 75% of growth, much smaller drawdowns
    Quarter Kelly: ~56% of growth, very small drawdowns
    """
    fk = fraction * kelly_continuous(returns, lookback)
    fk.name = f'kelly_{fraction}f_{lookback}d'
    return fk


# ── Volatility targeting ──────────────────────────────────────────────────────

def vol_targeted_position(returns: pd.Series,
                           target_vol: float = 0.10,
                           vol_window: int = 21,
                           max_leverage: float = 2.0) -> pd.Series:
    """
    Volatility-targeted position size.

    scale_t = target_vol / realised_vol_t

    Normalised so that scale = 1 when realised_vol = target_vol.
    Capped at max_leverage.

    target_vol: annualised target volatility (e.g. 0.10 = 10%)
    vol_window: lookback for realised vol (days)
    max_leverage: cap on position size

    EXAMPLE:
    target_vol = 10% annual = 10%/√252 ≈ 0.63% daily
    realised_vol = 20% annual → scale = 0.5 (half position)
    realised_vol = 5% annual  → scale = 2.0 (double position, capped)
    """
    # Daily target vol
    daily_target  = target_vol / np.sqrt(252)
    daily_realised = returns.rolling(vol_window).std()

    scale = daily_target / daily_realised.replace(0, np.nan)
    scale = scale.clip(0, max_leverage)
    scale.name = f'vol_target_{int(target_vol*100)}pct'
    return scale


def vol_targeted_returns(raw_signal: pd.Series, returns: pd.Series,
                          target_vol: float = 0.10,
                          vol_window: int = 21,
                          commission: float = 0.001) -> pd.Series:
    """
    Apply vol-targeting to a raw signal and compute strategy returns.

    raw_signal: +1 (long), -1 (short), or 0 (flat)
    """
    scale    = vol_targeted_position(returns, target_vol, vol_window)
    position = raw_signal * scale.shift(1)  # execute next day
    strat    = position * returns
    trades   = position.diff().abs()
    strat   -= trades * commission
    return strat.dropna()


# ── VaR-based position limits ─────────────────────────────────────────────────

def var_based_position(returns: pd.Series,
                        risk_budget: float = 0.02,
                        confidence: float  = 0.95,
                        vol_window: int    = 21,
                        method: str        = 'parametric') -> pd.Series:
    """
    Position size such that VaR ≤ risk_budget.

    position = risk_budget / VaR_per_unit

    VaR_per_unit (parametric) = z_α × σ
    VaR_per_unit (historical) = percentile of loss distribution

    risk_budget: maximum acceptable 1-day loss as fraction of capital
                 (e.g. 0.02 = 2% max daily loss)
    confidence:  VaR confidence level (0.95 or 0.99)

    INTERPRETATION:
    risk_budget=2%, confidence=95%: we accept a 5% chance of losing >2% in a day.
    On the days when we do lose more than 2%, the position sizing kept it bounded.
    """
    from scipy import stats as scipy_stats

    daily_vol = returns.rolling(vol_window).std()

    if method == 'parametric':
        z              = scipy_stats.norm.ppf(confidence)
        var_per_unit   = z * daily_vol
    else:  # historical
        var_per_unit   = (-returns).rolling(vol_window).quantile(confidence)

    position = risk_budget / var_per_unit.replace(0, np.nan)
    position = position.clip(0, 2.0)  # cap at 2× leverage
    position.name = f'var_position_{int(confidence*100)}pct'
    return position


# ── Comparison and visualisation ──────────────────────────────────────────────

class PositionSizerComparison:
    """
    Compare all three position sizing methods on the same strategy.
    Shows the trade-off between growth rate and drawdown.
    """

    def __init__(self, returns: pd.Series, signal: pd.Series,
                  initial_capital: float = 1_000_000):
        self.returns  = returns
        self.signal   = signal
        self.capital  = initial_capital

    def run(self, target_vol: float = 0.10,
             kelly_fraction_val: float = 0.5,
             risk_budget: float = 0.02,
             commission: float  = 0.001) -> dict:
        """
        Run all three position sizing methods and compare performance.
        """
        common = self.signal.index.intersection(self.returns.index)
        sig    = self.signal.reindex(common)
        ret    = self.returns.reindex(common)

        results = {}

        # ── Naive (fixed 100% allocation) ──
        naive_pos  = sig.shift(1).fillna(0)
        naive_ret  = naive_pos * ret - naive_pos.diff().abs() * commission
        results['naive'] = self._metrics(naive_ret, 'Naive (100%)')

        # ── Fractional Kelly ──
        kelly = fractional_kelly(ret, kelly_fraction_val, lookback=63)
        kelly_pos = (sig * kelly.shift(1)).fillna(0).clip(-2, 2)
        kelly_ret = kelly_pos * ret - kelly_pos.diff().abs() * commission
        results['kelly'] = self._metrics(kelly_ret, f'Kelly (f={kelly_fraction_val})')

        # ── Volatility targeting ──
        scale     = vol_targeted_position(ret, target_vol)
        vol_pos   = (sig * scale.shift(1)).fillna(0).clip(-2, 2)
        vol_ret   = vol_pos * ret - vol_pos.diff().abs() * commission
        results['vol_target'] = self._metrics(vol_ret, f'Vol-target ({target_vol:.0%})')

        # ── VaR-based ──
        var_scale = var_based_position(ret, risk_budget)
        var_pos   = (sig * var_scale.shift(1)).fillna(0).clip(-2, 2)
        var_ret   = var_pos * ret - var_pos.diff().abs() * commission
        results['var_based'] = self._metrics(var_ret, f'VaR-limit ({risk_budget:.0%})')

        # Print comparison
        print(f"\n{'═'*62}")
        print(f"  POSITION SIZING COMPARISON")
        print(f"  {'─'*50}")
        print(f"  {'Method':<22} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Vol':>8}")
        print(f"  {'─'*50}")
        for name, m in results.items():
            print(f"  {m['name']:<22} {m['total_ret']:>+8.2%} "
                  f"{m['sharpe']:>8.3f} {m['max_dd']:>8.2%} "
                  f"{m['ann_vol']:>8.2%}")
        print(f"{'═'*62}")

        return results

    def _metrics(self, ret: pd.Series, name: str) -> dict:
        clean  = ret.dropna()
        eq     = (1 + clean).cumprod()
        sharpe = (clean.mean() / clean.std() * np.sqrt(252)
                  if clean.std() > 0 else 0)
        mdd    = (eq / eq.cummax() - 1).min()
        total  = eq.iloc[-1] - 1
        vol    = clean.std() * np.sqrt(252)
        return {
            'name':      name,
            'returns':   clean,
            'equity':    eq * self.capital,
            'sharpe':    round(sharpe, 3),
            'max_dd':    round(mdd, 4),
            'total_ret': round(total, 4),
            'ann_vol':   round(vol, 4),
        }

    def plot(self, results: dict, save_path: str = None):
        """Plot equity curves for all sizing methods."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(DARK)
        colors = [BLUE, GREEN, AMBER, PURPLE]

        for ax in [ax1, ax2]:
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4)

        for (name, m), col in zip(results.items(), colors):
            eq = m['equity']
            ax1.plot(eq.index, eq.values, color=col, linewidth=1.5,
                     label=f"{m['name']} (SR={m['sharpe']:.2f})")
            dd = (m['equity'] / m['equity'].cummax() - 1) * 100
            ax2.plot(dd.index, dd.values, color=col, linewidth=1,
                     alpha=0.85, label=m['name'])

        ax1.set_title('Equity curves — position sizing comparison', color=TEXT, fontsize=10)
        ax1.set_ylabel('Portfolio value (₹)', color=TEXT)
        ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

        ax2.set_title('Drawdowns (%)', color=TEXT, fontsize=10)
        ax2.set_ylabel('Drawdown (%)', color=TEXT)
        ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

        plt.suptitle('Position Sizing: Kelly vs Vol-Target vs VaR vs Naive',
                     color=TEXT, fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
            print(f"  Saved: {save_path}")
        else:
            plt.show()
        plt.close()
