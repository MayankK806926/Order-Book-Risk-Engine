"""
python/microstructure.py - Market microstructure analysis for NSE equities.

WHAT IS MARKET MICROSTRUCTURE:
The study of how trades are executed, how prices are set, and how
information flows through a market. Critical for HFT and execution algo design.

KEY CONCEPTS:

1. BID-ASK SPREAD DECOMPOSITION (Roll 1984, Glosten-Milgrom 1987):
   Observed spread = Adverse selection component + Order processing cost

   Adverse selection = cost of trading with informed traders
   Order processing = fixed cost of running a market maker

   Total spread = 2c  where c = half-spread
   Realised spread < quoted spread → maker captured some adverse selection

2. AMIHUD ILLIQUIDITY (Amihud 2002):
   ILLIQ_t = |r_t| / Volume_t × 10^6

   Measures price impact per unit of volume.
   High ILLIQ → stock moves a lot per ₹ traded → illiquid
   Used as: (a) a risk factor, (b) a return predictor (illiquidity premium)

3. ROLL ESTIMATOR (Roll 1984):
   Estimates the bid-ask spread from trade-to-trade price changes.
   Cov(ΔP_t, ΔP_{t-1}) = -c²  where c = half-spread

   Half-spread = sqrt(-Cov(ΔP_t, ΔP_{t-1}))
   Only valid when covariance is negative (mean-reverting prices).

4. PRICE IMPACT:
   How much does a trade of size Q move the price?
   Almgren-Chriss: impact ∝ sqrt(Q) (square root law)
   Empirically validated across all liquid markets.

5. ORDER BOOK IMBALANCE:
   OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty) ∈ [-1, 1]
   OBI > 0 → more buyers → short-term upward pressure
   Strong predictor at millisecond frequency; weaker at daily.

INTERVIEW QUESTION:
"How do you estimate the bid-ask spread from daily price data?"
A: "I use the Roll estimator: estimate Cov(ΔP_t, ΔP_{t-1}) from
autocorrelation of price changes. Under Roll's model, the half-spread
c = sqrt(-Cov(ΔP_t, ΔP_{t-1})). This requires negative autocorrelation
(bid-ask bounce) which is common in microstructure noise."
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


# ── Amihud illiquidity ────────────────────────────────────────────────────────

def amihud_illiquidity(returns: pd.Series, volume: pd.Series,
                        window: int = 21) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio.

    ILLIQ_t = |r_t| / (price_t × volume_t)  × 10^6

    For daily data without dollar volume, approximate as:
    ILLIQ_t = |r_t| / volume_t  × 10^6

    ROLLING VERSION: average of daily ILLIQ over a window.

    INTERPRETATION:
    ILLIQ = ₹-basis-points of return per ₹1M of volume.
    High ILLIQ → large price impact per unit traded.

    NSE liquidity hierarchy (typical ILLIQ):
    Large caps (TCS, Reliance): ILLIQ ~ 0.001–0.01
    Mid caps: ILLIQ ~ 0.1–1.0
    Small caps: ILLIQ > 1.0
    """
    daily_illiq = returns.abs() / volume.replace(0, np.nan) * 1e6
    rolling     = daily_illiq.rolling(window).mean()
    rolling.name = f'amihud_{window}d'
    return rolling


def amihud_from_ohlcv(close: pd.Series, volume: pd.Series,
                       window: int = 21) -> pd.Series:
    """Compute Amihud from close prices and volume."""
    ret = np.log(close / close.shift(1))
    return amihud_illiquidity(ret, volume * close, window)


# ── Roll spread estimator ─────────────────────────────────────────────────────

def roll_spread(prices: pd.Series, window: int = 63) -> pd.Series:
    """
    Roll (1984) effective spread estimator.

    Roll's model: observed price = efficient price ± half-spread c
    P_t = V_t + c·Q_t  where V_t = efficient price, Q_t ∈ {-1, +1}

    This implies:
    Cov(ΔP_t, ΔP_{t-1}) = -c²

    Therefore:
    Effective half-spread = sqrt(max(-Cov(ΔP_t, ΔP_{t-1}), 0))
    Effective full spread = 2c

    NOTE: Works only when bid-ask bounce dominates (high-frequency data).
    For daily NSE data, microstructure noise is smaller relative to
    information flow, so the covariance may be positive (no valid estimate).

    Returns NaN where covariance is positive (no valid Roll estimate).
    """
    delta_p   = prices.diff()
    roll_cov  = delta_p.rolling(window).cov(delta_p.shift(1))
    half_sprd = np.sqrt((-roll_cov).clip(lower=0))
    full_sprd = 2 * half_sprd
    full_sprd.name = f'roll_spread_{window}d'
    return full_sprd


# ── Quoted and realised spread (with bid/ask) ─────────────────────────────────

def quoted_spread(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """
    Quoted spread = ask - bid.
    Proportional spread = (ask - bid) / mid_price.
    """
    mid   = (bid + ask) / 2
    qsprd = ask - bid
    psprd = qsprd / mid
    return pd.DataFrame({'quoted_spread': qsprd,
                          'proportional_spread': psprd,
                          'mid': mid})


def realised_spread(trade_price: pd.Series, mid_price: pd.Series,
                     direction: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Realised (effective) spread = 2 × direction × (trade_price - future_mid)

    WHERE direction = +1 for buy, -1 for sell.

    Measures: how much the market maker earns AFTER the information in the
    trade is impounded into prices.

    Realised spread < Quoted spread → informed trading (adverse selection)
    Realised spread ≈ Quoted spread → mostly noise trading

    If you don't have tick data with direction, use the Lee-Ready algorithm
    to classify trades: buy if trade_price > mid, sell if below.
    """
    future_mid  = mid_price.shift(-horizon)
    real_spread = 2 * direction * (trade_price - future_mid)
    return real_spread


# ── Price impact ──────────────────────────────────────────────────────────────

def price_impact(returns: pd.Series, volume: pd.Series,
                  prices: pd.Series = None,
                  model: str = 'linear') -> dict:
    """
    Estimate price impact: how much does volume move the price?

    MODELS:
    - 'linear':    |r| = α + β × Volume/ADV  (simple OLS)
    - 'sqrt':      |r| = α + β × sqrt(Volume/ADV)  (Almgren-Chriss)
    - 'log':       |r| = α + β × log(Volume)

    The square root law (Almgren-Chriss) is best supported empirically:
    Price impact ∝ sqrt(Volume / Average Daily Volume)

    Returns regression coefficients and R².
    """
    from scipy.stats import linregress

    abs_ret = returns.abs()
    adv     = volume.rolling(63).mean()
    vol_ratio = volume / adv.replace(0, np.nan)

    if model == 'linear':
        X = vol_ratio
    elif model == 'sqrt':
        X = np.sqrt(vol_ratio)
    elif model == 'log':
        X = np.log(vol_ratio.replace(0, np.nan))

    # Align and drop NaN
    df = pd.DataFrame({'y': abs_ret, 'X': X}).dropna()

    slope, intercept, r, p, se = linregress(df['X'], df['y'])

    print(f"\n  Price Impact ({model} model)")
    print(f"  α (intercept) : {intercept:.6f}")
    print(f"  β (slope)     : {slope:.6f}")
    print(f"  R²            : {r**2:.4f}")
    print(f"  p-value       : {p:.4f}")

    return {
        'model':     model,
        'alpha':     intercept,
        'beta':      slope,
        'r_squared': r**2,
        'p_value':   p,
    }


# ── Order book imbalance signal ───────────────────────────────────────────────

def order_imbalance_from_volume(volume: pd.Series, prices: pd.Series,
                                  window: int = 5) -> pd.Series:
    """
    Approximate order book imbalance from daily volume and price change.

    Without tick data, approximate trade direction using daily returns:
    If price went up → mostly buyer-initiated → OBI proxy = +1
    If price went down → mostly seller-initiated → OBI proxy = -1

    Better (if available): use signed volume from NSE tick data.

    THIS IS A PROXY - real OBI requires L2 order book data.
    """
    ret      = np.log(prices / prices.shift(1))
    sign     = np.sign(ret)
    # Weight by volume magnitude
    signed_vol = sign * volume

    # Rolling imbalance: sum of signed volume / total volume
    buy_vol  = signed_vol.clip(lower=0).rolling(window).sum()
    sell_vol = (-signed_vol).clip(lower=0).rolling(window).sum()
    total    = (buy_vol + sell_vol).replace(0, np.nan)

    obi = (buy_vol - sell_vol) / total
    obi.name = f'obi_{window}d'
    return obi


# ── Microstructure analysis report ───────────────────────────────────────────

def microstructure_report(prices: pd.Series, volume: pd.Series = None,
                            symbol: str = '') -> pd.DataFrame:
    """
    Full microstructure report for a price series.
    Computes all available metrics given the data.
    """
    ret = np.log(prices / prices.shift(1)).dropna()

    print(f"\n{'═'*55}")
    print(f"  MICROSTRUCTURE REPORT: {symbol}")
    print(f"{'═'*55}")

    results = {}

    # Roll spread
    roll    = roll_spread(prices, window=63).dropna()
    r_mean  = roll.mean()
    results['roll_spread_mean'] = r_mean
    results['roll_spread_std']  = roll.std()
    print(f"\n  Roll spread (63d window):")
    print(f"    Mean: {r_mean:.4f}  Std: {roll.std():.4f}")
    print(f"    Valid estimates: {roll.notna().sum()} / {len(roll)} bars")

    # Return autocorrelation (microstructure noise indicator)
    ac1 = ret.autocorr(lag=1)
    results['autocorr_lag1'] = ac1
    print(f"\n  Return autocorrelation (lag=1): {ac1:.4f}")
    print(f"    {'Negative → bid-ask bounce / mean reversion' if ac1 < 0 else 'Positive → momentum / trending'}")

    # Volatility
    rv_21 = ret.rolling(21).std() * np.sqrt(252)
    results['rv_21d_mean'] = rv_21.mean()
    print(f"\n  Realised vol (21d, annualised): {rv_21.mean():.2%}")

    if volume is not None:
        # Amihud illiquidity
        amihud = amihud_from_ohlcv(prices, volume, window=21).dropna()
        results['amihud_mean'] = amihud.mean()
        results['amihud_std']  = amihud.std()
        print(f"\n  Amihud illiquidity (21d):")
        print(f"    Mean: {amihud.mean():.6f}  Std: {amihud.std():.6f}")

        # Price impact
        print(f"\n  Price impact regression:")
        impact = price_impact(ret, volume, prices, model='sqrt')
        results.update({f'impact_{k}': v for k, v in impact.items()})

        # Order imbalance
        obi = order_imbalance_from_volume(volume, prices, window=5).dropna()
        results['obi_mean'] = obi.mean()
        print(f"\n  Order book imbalance proxy (5d): mean={obi.mean():.4f}")

    return pd.Series(results)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_microstructure(prices: pd.Series, volume: pd.Series = None,
                         symbol: str = '', save_path: str = None):
    """
    4-panel microstructure dashboard:
    Panel 1: Price + volume
    Panel 2: Amihud illiquidity over time
    Panel 3: Roll spread estimate over time
    Panel 4: Return autocorrelation (rolling)
    """
    ret = np.log(prices / prices.shift(1))

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # Panel 1: Price
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    ax1.plot(prices.index, prices.values, color=BLUE, linewidth=1.2)
    ax1.set_title(f'Price - {symbol}', color=TEXT, fontsize=10)
    ax1.set_ylabel('Price (₹)', color=TEXT)

    if volume is not None:
        ax1b = ax1.twinx()
        ax1b.bar(volume.index, volume.values, color=AMBER, alpha=0.2, width=1)
        ax1b.set_ylabel('Volume', color=AMBER)
        ax1b.tick_params(colors=AMBER)

    # Panel 2: Amihud (if volume available)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    if volume is not None:
        amihud = amihud_from_ohlcv(prices, volume, 21).dropna()
        ax2.fill_between(amihud.index, amihud.values, color=RED, alpha=0.5)
        ax2.plot(amihud.index, amihud.values, color=RED, linewidth=1)
        ax2.set_title('Amihud illiquidity (21d rolling)', color=TEXT, fontsize=10)
        ax2.set_ylabel('ILLIQ × 10⁶', color=TEXT)
    else:
        ax2.text(0.5, 0.5, 'Volume data required\nfor Amihud', ha='center',
                 va='center', color=MUTED, transform=ax2.transAxes)
        ax2.set_title('Amihud illiquidity (N/A)', color=TEXT, fontsize=10)

    # Panel 3: Roll spread
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL)
    roll = roll_spread(prices, 63).dropna()
    ax3.fill_between(roll.index, roll.values, color=GREEN, alpha=0.5)
    ax3.plot(roll.index, roll.values, color=GREEN, linewidth=1)
    ax3.set_title('Roll effective spread estimate (63d)', color=TEXT, fontsize=10)
    ax3.set_ylabel('Spread (₹)', color=TEXT)

    # Panel 4: Rolling autocorrelation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL)
    rolling_ac = ret.rolling(63).apply(lambda x: x.autocorr(lag=1), raw=False)
    ax4.fill_between(rolling_ac.index, rolling_ac.values, 0,
                     where=(rolling_ac > 0), color=AMBER, alpha=0.5, label='Positive (momentum)')
    ax4.fill_between(rolling_ac.index, rolling_ac.values, 0,
                     where=(rolling_ac < 0), color=PURPLE, alpha=0.5, label='Negative (mean-rev)')
    ax4.plot(rolling_ac.index, rolling_ac.values, color=BLUE, linewidth=0.8, alpha=0.8)
    ax4.axhline(0, color=TEXT, linewidth=0.6, linestyle=':')
    ax4.set_title('Rolling lag-1 autocorrelation (63d)', color=TEXT, fontsize=10)
    ax4.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(colors=TEXT)
        ax.yaxis.label.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4)

    plt.suptitle(f'Market Microstructure Analysis - {symbol}',
                 color=TEXT, fontsize=13, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()
