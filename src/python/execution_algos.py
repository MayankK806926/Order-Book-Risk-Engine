"""
python/execution_algos.py — VWAP, TWAP, implementation shortfall, TCA.

WHY EXECUTION ALGORITHMS:
A large order cannot be submitted all at once. Buying 100,000 shares of
a stock with 200,000 shares average daily volume would:
1. Move the price against you (market impact)
2. Signal your intention to other traders (information leakage)

Execution algos split the parent order into smaller child orders,
balancing market impact against timing risk (opportunity cost).

THE THREE COSTS OF EXECUTION:
1. Market impact: price moves because of your order
2. Timing risk:   price moves against you while you wait to execute
3. Spread cost:   crossing the bid-ask spread

OPTIMAL EXECUTION (Almgren-Chriss 2001):
Minimise: E[Cost] + λ × Var[Cost]
where λ controls the risk aversion.

λ=0: linear schedule (trade at constant rate) — minimises risk
λ=∞: trade immediately — minimises timing risk but maximises impact

VWAP vs TWAP:
VWAP: Volume Weighted Average Price — benchmark price weighted by volume
TWAP: Time Weighted Average Price    — simple average over the day

A VWAP algo slices orders proportional to intraday volume profile.
Executes more at open/close (high volume) and less at midday (low volume).
This minimises tracking error vs VWAP benchmark.

A TWAP algo slices orders equally over time intervals.
Simpler, less market impact, but more timing risk.
Preferred for illiquid stocks where volume profile is unreliable.

IMPLEMENTATION SHORTFALL:
IS = Decision price − Execution price (for buys)
   = E[price at trade decision] - E[price actually filled at]
Measures the total cost of going from "decision to trade" to "filled".
Includes: delay cost, impact cost, spread cost.
"""

import numpy as np
import pandas as pd
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


# ── VWAP ─────────────────────────────────────────────────────────────────────

def vwap_execution(total_shares: int, intraday_volume: pd.Series,
                    intraday_prices: pd.Series,
                    randomise: bool = True,
                    randomise_pct: float = 0.10) -> pd.DataFrame:
    """
    VWAP execution algorithm.

    STEPS:
    1. Compute the intraday volume profile (% of daily volume per interval)
    2. Allocate shares to each interval proportional to expected volume
    3. Execute allocated shares in each interval
    4. Track fill prices and compute execution quality

    randomise: if True, randomise order sizes by ±randomise_pct to
               reduce predictability and information leakage.

    Returns DataFrame with per-interval execution details and cumulative metrics.
    """
    n         = len(intraday_volume)
    total_vol = intraday_volume.sum()

    # Volume profile: fraction of daily volume in each interval
    vol_profile = intraday_volume / total_vol

    # Target shares per interval
    target_shares = (vol_profile * total_shares).round().astype(int)

    # Adjust rounding errors to hit exact total
    diff = total_shares - target_shares.sum()
    if diff != 0:
        target_shares.iloc[-1] += diff

    # Apply randomisation ±randomise_pct to hide order intentions
    if randomise:
        noise           = 1 + np.random.uniform(-randomise_pct, randomise_pct, n)
        noisy_shares    = (target_shares * noise).round().astype(int)
        # Rescale to hit total
        if noisy_shares.sum() > 0:
            scale       = total_shares / noisy_shares.sum()
            noisy_shares = (noisy_shares * scale).round().astype(int)
        noisy_shares.iloc[-1] += total_shares - noisy_shares.sum()
        executed_shares = noisy_shares
    else:
        executed_shares = target_shares

    # Simulate fills at interval VWAP with slight market impact
    # Impact model: ΔP = impact_factor × sqrt(executed / interval_volume)
    impact_factor = 0.001  # 0.1% per unit vol fraction
    fills = []

    for i in range(n):
        int_vol = intraday_volume.iloc[i]
        exe     = executed_shares.iloc[i]
        base_px = intraday_prices.iloc[i]

        if int_vol > 0 and exe > 0:
            impact   = impact_factor * np.sqrt(exe / int_vol)
            fill_px  = base_px * (1 + impact)  # buy at slightly higher price
        else:
            fill_px = base_px

        fills.append({
            'interval':         i,
            'time':             intraday_prices.index[i],
            'market_price':     base_px,
            'fill_price':       fill_px,
            'executed_shares':  exe,
            'target_shares':    target_shares.iloc[i],
            'vol_profile_pct':  vol_profile.iloc[i] * 100,
        })

    df = pd.DataFrame(fills)
    df['dollar_volume']    = df['fill_price'] * df['executed_shares']
    df['cumulative_shares'] = df['executed_shares'].cumsum()
    df['cum_avg_price']    = (df['dollar_volume'].cumsum() / df['cumulative_shares'].clip(lower=1))

    # VWAP benchmark
    vwap_benchmark = ((intraday_prices * intraday_volume).sum() / intraday_volume.sum())
    avg_fill_price = df['dollar_volume'].sum() / df['executed_shares'].sum()

    df['vwap_benchmark']    = vwap_benchmark
    df['vwap_slippage_bps'] = (avg_fill_price / vwap_benchmark - 1) * 10000

    print(f"\n  VWAP Execution Results")
    print(f"  {'─'*40}")
    print(f"  Total shares      : {total_shares:>10,}")
    print(f"  Executed shares   : {df['executed_shares'].sum():>10,}")
    print(f"  VWAP benchmark    : {vwap_benchmark:>10.2f}")
    print(f"  Avg fill price    : {avg_fill_price:>10.2f}")
    print(f"  VWAP slippage     : {(avg_fill_price/vwap_benchmark-1)*10000:>10.2f} bps")

    return df


# ── TWAP ─────────────────────────────────────────────────────────────────────

def twap_execution(total_shares: int, n_intervals: int,
                    prices: pd.Series,
                    randomise: bool = True,
                    randomise_pct: float = 0.15) -> pd.DataFrame:
    """
    TWAP execution algorithm.

    Splits total shares equally across n_intervals time intervals.
    Simpler than VWAP — does not require volume profile estimation.

    WHEN TO USE TWAP OVER VWAP:
    - Illiquid stocks: volume profile unreliable → TWAP more predictable
    - When mandate specifies TWAP benchmark
    - When you want to minimise urgency signal to market

    randomise: add noise to each interval to reduce predictability.
    """
    if n_intervals <= 0:
        raise ValueError("n_intervals must be positive")

    shares_per = total_shares // n_intervals
    remainder  = total_shares - shares_per * n_intervals

    # Build interval schedule
    step = max(len(prices) // n_intervals, 1)
    intervals = []

    for i in range(n_intervals):
        idx_start = i * step
        idx_end   = min((i + 1) * step, len(prices))
        if idx_start >= len(prices):
            break

        exe = shares_per + (remainder if i == n_intervals - 1 else 0)

        # Only randomise (and floor at 1) intervals that were actually
        # allocated shares. When total_shares < n_intervals, shares_per
        # rounds down to 0 for most intervals — max(1, ...) used to force
        # a phantom 1-share fill into EVERY such interval regardless,
        # so a 5-share order over 10 intervals could execute ~14 shares.
        if randomise and exe > 0:
            noise = 1 + np.random.uniform(-randomise_pct, randomise_pct)
            exe   = max(1, int(exe * noise))

        price = prices.iloc[idx_start]
        intervals.append({
            'interval':       i + 1,
            'time':           prices.index[idx_start],
            'market_price':   price,
            'fill_price':     price * (1 + 0.0005),  # approx 0.5bps impact
            'executed_shares': exe,
        })

    df = pd.DataFrame(intervals)
    df['dollar_volume']    = df['fill_price'] * df['executed_shares']
    df['cumulative_shares'] = df['executed_shares'].cumsum()
    df['cum_avg_price']    = df['dollar_volume'].cumsum() / df['cumulative_shares'].clip(lower=1)

    twap_benchmark = prices.mean()
    avg_fill       = df['dollar_volume'].sum() / df['executed_shares'].sum()

    print(f"\n  TWAP Execution Results")
    print(f"  {'─'*40}")
    print(f"  Total shares      : {total_shares:>10,}")
    print(f"  Intervals         : {n_intervals:>10,}")
    print(f"  TWAP benchmark    : {twap_benchmark:>10.2f}")
    print(f"  Avg fill price    : {avg_fill:>10.2f}")
    print(f"  TWAP slippage     : {(avg_fill/twap_benchmark-1)*10000:>10.2f} bps")

    return df


# ── Implementation Shortfall ──────────────────────────────────────────────────

def implementation_shortfall(decision_price: float,
                               fills: pd.DataFrame,
                               total_shares: int) -> dict:
    """
    Implementation Shortfall = Decision Price - Execution VWAP.

    Measures the full cost of acting on a trading decision:
    IS = (avg_fill_price - decision_price) / decision_price   [for buys]

    DECOMPOSITION:
    IS = Delay cost + Impact cost + Spread cost

    Delay cost = (price when first order sent - decision price) / decision_price
    Impact cost = market impact of actual trading
    Spread cost = half-spread × 2 (crossing the spread)

    BENCHMARK:
    IS = 0     → executed at exactly the decision price (ideal)
    IS > 0     → executed worse than decision price (normal)
    IS < 0     → executed better than decision price (price improved)

    For an institutional manager, IS is the primary TCA metric.
    A typical IS for NSE large-caps: 10-50 bps per trade.
    """
    avg_fill = fills['dollar_volume'].sum() / fills['executed_shares'].sum()
    is_bps   = (avg_fill / decision_price - 1) * 10000  # basis points

    # Cost in rupees
    total_cost_rs = (avg_fill - decision_price) * total_shares

    result = {
        'decision_price':    decision_price,
        'avg_fill_price':    round(avg_fill, 4),
        'is_bps':            round(is_bps, 2),
        'is_pct':            round(is_bps / 100, 4),
        'total_cost_rs':     round(total_cost_rs, 2),
        'total_shares':      total_shares,
        'execution_quality': 'GOOD' if abs(is_bps) < 10 else (
                              'FAIR' if abs(is_bps) < 30 else 'POOR'),
    }

    print(f"\n  Implementation Shortfall")
    print(f"  {'─'*40}")
    print(f"  Decision price    : {decision_price:>10.2f}")
    print(f"  Avg fill price    : {result['avg_fill_price']:>10.2f}")
    print(f"  IS                : {is_bps:>+10.2f} bps")
    print(f"  Total cost        : ₹{total_cost_rs:>12,.2f}")
    print(f"  Quality           : {result['execution_quality']}")

    return result


# ── Transaction Cost Analysis ─────────────────────────────────────────────────

def tca_report(vwap_fills: pd.DataFrame, twap_fills: pd.DataFrame,
                decision_price: float, total_shares: int,
                save_path: str = None):
    """
    Transaction Cost Analysis (TCA) report.
    Compares VWAP and TWAP execution quality.
    """
    vwap_is = implementation_shortfall(decision_price, vwap_fills, total_shares)
    twap_is = implementation_shortfall(decision_price, twap_fills, total_shares)

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(1, 2, wspace=0.3)

    # Panel 1: Cumulative execution progress
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PANEL)
    ax1.plot(vwap_fills['interval'], vwap_fills['cum_avg_price'],
             color=BLUE, linewidth=2, marker='o', ms=4, label='VWAP algo')
    ax1.plot(twap_fills['interval'], twap_fills['cum_avg_price'],
             color=GREEN, linewidth=2, marker='s', ms=4, label='TWAP algo')
    ax1.axhline(decision_price, color=RED, linewidth=1.5, linestyle='--',
                label=f'Decision price={decision_price:.2f}')
    ax1.axhline(vwap_fills['vwap_benchmark'].iloc[0] if 'vwap_benchmark' in vwap_fills.columns
                else decision_price, color=AMBER, linewidth=1,
                linestyle=':', label='VWAP benchmark')
    ax1.set_title('Cumulative average fill price', color=TEXT, fontsize=10)
    ax1.set_xlabel('Interval', color=TEXT)
    ax1.set_ylabel('Price (₹)', color=TEXT)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # Panel 2: IS comparison bar chart
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(PANEL)
    methods  = ['VWAP algo', 'TWAP algo']
    is_vals  = [vwap_is['is_bps'], twap_is['is_bps']]
    bar_cols = [GREEN if v < 20 else (AMBER if v < 50 else RED) for v in is_vals]
    bars     = ax2.bar(methods, is_vals, color=bar_cols, alpha=0.8, edgecolor=PANEL, width=0.4)
    ax2.axhline(0, color=TEXT, linewidth=0.6, linestyle=':')
    for bar, val in zip(bars, is_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f} bps', ha='center', va='bottom', color=TEXT, fontsize=11, fontweight='500')
    ax2.set_title('Implementation Shortfall (bps)\nLower = better execution quality',
                  color=TEXT, fontsize=10)
    ax2.set_ylabel('IS (basis points)', color=TEXT)

    for ax in [ax1, ax2]:
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4)

    plt.suptitle('Transaction Cost Analysis (TCA)', color=TEXT, fontsize=12, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()

    return {'vwap_is': vwap_is, 'twap_is': twap_is}
