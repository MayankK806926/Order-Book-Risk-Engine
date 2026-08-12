"""
tests/test_week56.py - Test suite for week 5-6 Python modules.

Run: python tests/test_week56.py
"""

import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.microstructure  import amihud_illiquidity, roll_spread, order_imbalance_from_volume
from python.risk_engine     import RiskEngine
from python.position_sizer  import (kelly_fraction, kelly_continuous,
                                     vol_targeted_position, var_based_position,
                                     fractional_kelly)
from python.execution_algos import vwap_execution, twap_execution, implementation_shortfall

passed = failed = 0

def check(name, cond, detail=''):
    global passed, failed
    if cond:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}  {detail}")
        failed += 1

def make_returns(n=500, seed=42):
    np.random.seed(seed)
    return pd.Series(np.random.normal(0.0004, 0.012, n),
                     index=pd.bdate_range('2019-01-01', periods=n))

def make_prices(n=500, seed=42):
    ret = make_returns(n, seed)
    return pd.Series(2500 * np.exp(ret.cumsum().values),
                     index=ret.index)

def make_volume(n=500, seed=42):
    np.random.seed(seed+1)
    return pd.Series(np.random.randint(100_000, 1_000_000, n).astype(float),
                     index=pd.bdate_range('2019-01-01', periods=n))

# ── Microstructure tests ──────────────────────────────────────────────────────
def test_amihud_positive():
    prices = make_prices()
    volume = make_volume()
    ret = np.log(prices / prices.shift(1)).dropna()
    amihud = amihud_illiquidity(ret, volume.iloc[1:], window=21).dropna()
    check('Amihud illiquidity is positive', (amihud > 0).all(), f"min={amihud.min():.6f}")

def test_roll_spread_nonneg():
    prices = make_prices()
    roll = roll_spread(prices, window=63).dropna()
    check('Roll spread is non-negative', (roll >= 0).all(), f"min={roll.min():.6f}")

def test_obi_bounded():
    prices = make_prices()
    volume = make_volume()
    obi = order_imbalance_from_volume(volume, prices, window=5).dropna()
    check('OBI bounded in [-1, 1]', ((obi >= -1) & (obi <= 1)).all(),
          f"range=[{obi.min():.3f}, {obi.max():.3f}]")

# ── Risk engine tests ─────────────────────────────────────────────────────────
def test_var_hierarchy():
    "CVaR >= historical VaR >= 0"
    engine = RiskEngine(make_returns())
    var    = engine.var_historical(0.95)
    cvar   = engine.cvar(0.95)
    check('CVaR >= historical VaR', cvar >= var - 1e-6,
          f"cvar={cvar:.4f}, var={var:.4f}")
    check('VaR is positive', var >= 0, f"var={var:.4f}")

def test_var_methods_similar():
    "All three VaR methods should be in the same ballpark"
    engine = RiskEngine(make_returns(n=1000))
    v = engine.var_all_methods(0.95)
    ratio = v['parametric'] / v['historical']
    check('Parametric and historical VaR within 50% of each other',
          0.5 < ratio < 2.0, f"ratio={ratio:.3f}")

def test_sharpe_positive_drift():
    "Strong positive drift should give positive Sharpe"
    np.random.seed(0)
    ret    = pd.Series(np.random.normal(0.002, 0.01, 500),
                       index=pd.bdate_range('2020-01-01', periods=500))
    engine = RiskEngine(ret, freq='daily')
    sharpe = engine.sharpe_ratio(0.0)  # zero risk-free for simplicity
    check('Positive drift → positive Sharpe', sharpe > 0, f"sharpe={sharpe:.3f}")

def test_max_drawdown_negative():
    "Max drawdown should be ≤ 0"
    engine = RiskEngine(make_returns())
    mdd    = engine.max_drawdown()
    check('Max drawdown ≤ 0', mdd <= 0, f"mdd={mdd:.4f}")

def test_drawdown_series_nonneg_magnitude():
    "Drawdown series values should be in [-1, 0]"
    engine = RiskEngine(make_returns())
    dd     = engine.drawdown_series()
    check('Drawdown series in [-1, 0]', ((dd >= -1) & (dd <= 0.001)).all(),
          f"range=[{dd.min():.4f}, {dd.max():.4f}]")

# ── Position sizer tests ──────────────────────────────────────────────────────
def test_kelly_zero_edge():
    "Zero edge → Kelly fraction = 0"
    f = kelly_fraction(win_prob=0.5, win_loss_ratio=1.0)
    check('Kelly = 0 for zero edge (p=0.5, b=1.0)', abs(f) < 1e-9, f"got {f:.6f}")

def test_kelly_positive_edge():
    "Positive edge → positive Kelly fraction"
    f = kelly_fraction(win_prob=0.6, win_loss_ratio=1.5)
    check('Kelly > 0 for positive edge', f > 0, f"got {f:.4f}")

def test_kelly_negative_edge_clipped():
    "Negative edge → Kelly = 0 (no bet)"
    f = kelly_fraction(win_prob=0.4, win_loss_ratio=1.0)
    check('Kelly = 0 for negative edge (clipped)', f == 0, f"got {f:.4f}")

def test_fractional_kelly_smaller():
    "Fractional Kelly should be smaller than full Kelly"
    returns = make_returns()
    full = kelly_continuous(returns).dropna()
    half = fractional_kelly(returns, fraction=0.5).dropna()
    common = full.index.intersection(half.index)
    check('Half-Kelly ≤ full Kelly in absolute terms',
          (half.reindex(common).abs() <= full.reindex(common).abs() + 1e-9).all())

def test_vol_target_scales_correctly():
    "In high-vol regime, vol target reduces position size"
    np.random.seed(0)
    low_vol  = pd.Series(np.random.normal(0, 0.005, 200))
    high_vol = pd.Series(np.random.normal(0, 0.025, 200))
    scale_low  = vol_targeted_position(low_vol,  target_vol=0.10, vol_window=50).dropna().mean()
    scale_high = vol_targeted_position(high_vol, target_vol=0.10, vol_window=50).dropna().mean()
    check('Higher vol → smaller position size', scale_high < scale_low,
          f"high={scale_high:.3f}, low={scale_low:.3f}")

def test_var_position_bounded():
    "VaR-based position should never exceed max leverage"
    returns = make_returns()
    pos = var_based_position(returns, risk_budget=0.02).dropna()
    check('VaR position ≤ max leverage (2.0)', (pos <= 2.0).all(),
          f"max={pos.max():.3f}")

# ── Execution algo tests ──────────────────────────────────────────────────────
def make_intraday(n=13, base=2500):
    np.random.seed(0)
    times   = pd.date_range('2024-01-01 09:15', periods=n, freq='30min')
    prices  = pd.Series(base + np.random.normal(0, 5, n).cumsum(), index=times)
    volume  = pd.Series(np.random.randint(50_000, 300_000, n).astype(float), index=times)
    return prices, volume

def test_vwap_total_shares():
    "VWAP algo should execute exactly the requested total shares"
    prices, volume = make_intraday()
    total = 10_000
    df = vwap_execution(total, volume, prices, randomise=False)
    check('VWAP executes correct total shares',
          df['executed_shares'].sum() == total,
          f"got {df['executed_shares'].sum()}")

def test_twap_intervals():
    "TWAP algo should produce the right number of intervals"
    prices, _ = make_intraday(n=20)
    n_int = 5
    df = twap_execution(10_000, n_int, prices, randomise=False)
    check('TWAP produces correct number of intervals', len(df) == n_int,
          f"got {len(df)}")

def test_implementation_shortfall_sign():
    "If fill > decision price (buy), IS should be positive"
    prices, volume = make_intraday()
    decision_price = 2490.0  # below market → expect positive IS
    df = vwap_execution(10_000, volume, prices, randomise=False)
    is_result = implementation_shortfall(decision_price, df, 10_000)
    check('IS > 0 when fill above decision price', is_result['is_bps'] > 0,
          f"IS={is_result['is_bps']:.2f} bps")

def test_twap_small_order_many_intervals_no_overexecution():
    """
    Regression test: TWAP used to force a phantom 1-share fill into every
    interval via max(1, int(exe*noise)), even for intervals that were
    legitimately allocated 0 shares (total_shares < n_intervals). A
    5-share order over 10 intervals used to execute as much as 14 shares
    (1 phantom share x 9 empty intervals + ~5 on the one real interval).
    Randomisation still applies to the one interval that has real shares,
    so the total isn't pinned to exactly 5 - but it must stay near 5,
    not balloon toward one-phantom-share-per-empty-interval.
    """
    np.random.seed(3)
    times  = pd.date_range('2024-01-01 09:15', periods=20, freq='15min')
    prices = pd.Series(2500.0, index=times)
    df = twap_execution(5, 10, prices, randomise=True)
    total = df['executed_shares'].sum()
    check('TWAP does not inflate execution with phantom fills at empty intervals',
          1 <= total <= 6,
          f"got {total} (old bug could push this to ~14)")

def test_vwap_slippage_small():
    "VWAP slippage should be less than 20 bps in normal conditions"
    prices, volume = make_intraday(n=26)
    df = vwap_execution(5_000, volume, prices, randomise=False)
    check('VWAP slippage < 20 bps',
          abs(df['vwap_slippage_bps'].iloc[-1]) < 20,
          f"slippage={df['vwap_slippage_bps'].iloc[-1]:.2f} bps")

def run_all():
    global passed, failed
    passed = failed = 0

    tests = [
        # Microstructure
        test_amihud_positive, test_roll_spread_nonneg, test_obi_bounded,
        # Risk engine
        test_var_hierarchy, test_var_methods_similar, test_sharpe_positive_drift,
        test_max_drawdown_negative, test_drawdown_series_nonneg_magnitude,
        # Position sizer
        test_kelly_zero_edge, test_kelly_positive_edge, test_kelly_negative_edge_clipped,
        test_fractional_kelly_smaller, test_vol_target_scales_correctly, test_var_position_bounded,
        # Execution
        test_vwap_total_shares, test_twap_intervals,
        test_implementation_shortfall_sign, test_twap_small_order_many_intervals_no_overexecution,
        test_vwap_slippage_small,
    ]

    sections = {
        'MICROSTRUCTURE': tests[:3],
        'RISK ENGINE':    tests[3:8],
        'POSITION SIZER': tests[8:14],
        'EXECUTION ALGOS': tests[14:],
    }

    print('\n' + '='*55)
    print('  WEEK 5-6 TEST SUITE')
    print('='*55)

    for section, section_tests in sections.items():
        print(f'\n--- {section} ---')
        for t in section_tests:
            try:
                t()
            except Exception as e:
                print(f"  FAIL: {t.__name__} [ERROR]: {e}")
                failed += 1

    print(f'\n{"="*55}')
    print(f'  Results: {passed} passed / {failed} failed / {passed+failed} total')
    print(f'  {"All tests passed ✓" if failed == 0 else f"{failed} tests failed ✗"}')
    print('='*55 + '\n')

if __name__ == '__main__':
    run_all()
