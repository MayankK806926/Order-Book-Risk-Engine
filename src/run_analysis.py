"""
run_analysis.py — Week 5-6 master runner.
Runs all analyses: order book sim, microstructure, risk engine,
position sizing comparison, and execution algo TCA.

Usage:
  cd src
  pip install -r requirements.txt
  python run_analysis.py
"""
import os, numpy as np, pandas as pd, warnings, matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# Synthetic data
np.random.seed(42)
dates   = pd.bdate_range('2019-01-01', '2024-01-01')
n       = len(dates)
ret     = np.random.normal(0.0004, 0.013, n)
prices  = pd.Series(2500*np.exp(np.cumsum(ret)), index=dates, name='RELIANCE')
volume  = pd.Series(np.random.randint(200_000,2_000_000,n).astype(float), index=dates)
returns = pd.Series(ret, index=dates)

print('\n' + '█'*55)
print('  WEEK 5-6 ANALYSIS')
print('█'*55)

# 1. Order book simulation
print('\n[1/5] Order book simulation...')
from python.order_book_simulator import run_simulation
run_simulation(n_steps=300, mid_price=2500.0, save_path='outputs/order_book_sim.png')

# 2. Microstructure
print('\n[2/5] Microstructure analysis...')
from python.microstructure import microstructure_report, plot_microstructure
microstructure_report(prices, volume, symbol='RELIANCE')
plot_microstructure(prices, volume, symbol='RELIANCE', save_path='outputs/microstructure.png')

# 3. Risk engine
print('\n[3/5] Risk engine report...')
from python.risk_engine import RiskEngine
engine = RiskEngine(returns, initial_capital=1_000_000)
report = engine.full_report()
engine.plot(save_path='outputs/risk_report.png')

# 4. Position sizing
print('\n[4/5] Position sizing comparison...')
from python.position_sizer import PositionSizerComparison
signal = pd.Series(np.sign(returns.rolling(21).mean()), index=dates)
comp   = PositionSizerComparison(returns, signal, initial_capital=1_000_000)
results = comp.run(target_vol=0.10, kelly_fraction_val=0.5, risk_budget=0.02)
comp.plot(results, save_path='outputs/position_sizing.png')

# 5. Execution algos TCA
print('\n[5/5] Execution algorithm TCA...')
from python.execution_algos import vwap_execution, twap_execution, tca_report
intraday_times  = pd.date_range('2024-01-15 09:15', periods=26, freq='15min')
intraday_prices = pd.Series(2500 + np.random.normal(0,3,26).cumsum(), index=intraday_times)
intraday_volume = pd.Series(np.random.randint(50_000,300_000,26).astype(float), index=intraday_times)
vwap_df = vwap_execution(50_000, intraday_volume, intraday_prices, randomise=True)
twap_df = twap_execution(50_000, 13, intraday_prices, randomise=True)
tca_report(vwap_df, twap_df, decision_price=2498.0, total_shares=50_000,
           save_path='outputs/tca_report.png')

print('\n\nOutputs saved to outputs/')
for f in sorted(os.listdir('outputs')):
    print(f'  {f}')
