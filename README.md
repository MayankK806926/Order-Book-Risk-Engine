# Week 5–6: C++ Systems, Risk Management & Execution

## Structure
```
cpp-systems-risk/
└── src/
    ├── cpp/
    │   ├── order_book.h/.cpp      — Full limit order book matching engine
    │   ├── risk_manager.h/.cpp    — Pre-trade risk checks, kill switch
    │   ├── test_order_book.cpp    — 57/57 tests (order book + risk manager)
    │   └── Makefile
    ├── python/
    │   ├── order_book_simulator.py — Python order book + depth chart
    │   ├── microstructure.py       — Amihud, Roll spread, price impact
    │   ├── risk_engine.py          — VaR (3 methods), CVaR, Sharpe, drawdown
    │   ├── position_sizer.py       — Kelly, vol-targeting, VaR limits
    │   └── execution_algos.py      — VWAP, TWAP, implementation shortfall
    ├── tests/
    │   └── test_week56.py          — 20/20 Python tests
    └── run_analysis.py
```

## Quickstart
```bash
cd cpp-systems-risk/src/cpp && make test       # 57/57 C++ tests
# no `make` available? build directly:
#   g++ -std=c++17 -O2 -Wall -Wextra order_book.cpp risk_manager.cpp test_order_book.cpp -o test_orderbook && ./test_orderbook
cd .. && pip install -r requirements.txt
python tests/test_week56.py  # 20/20 Python tests
python run_analysis.py       # full analysis
```

## Bugs Found & Fixed

A deep-dive review found and fixed five real bugs across the C++ engine
and the Python execution algos.

1. **The demo crashed with heap corruption.** `demo_market_simulation()`
   in `test_order_book.cpp` stored orders in a `std::vector<Order>` while
   interleaving `emplace_back()` with `ob.add_order()` — but `OrderBook`
   keeps raw `Order*` pointers into whatever container holds the orders
   (documented in `PriceLevel`). A `std::vector` reallocates its buffer
   as it grows, invalidating every pointer the book had already stored
   from earlier loop iterations. Confirmed via `gdb` backtrace: the
   segfault surfaced later, inside `~OrderBook()`, destroying an already
   heap-corrupted `deque`. Fixed by switching to `std::deque<Order>`,
   which never invalidates references to existing elements on
   `push_back`/`emplace_back`.
2. **Cancelling the last order at a price level left a phantom quote.**
   `cancel_order()` zeroed a `PriceLevel`'s quantity but never erased
   the now-empty entry from the `bids_`/`asks_` map. `best_bid()` /
   `best_ask()` only check whether the map entry exists, not whether
   it's actually empty — so they kept reporting a price with zero real
   liquidity after the only order there was cancelled. `print()`
   happened to mask this (it filters on `total_quantity > 0`), which is
   why it went unnoticed. Fixed by erasing the level when it empties.
3. **The risk manager's kill switch could trip on a normal buy.**
   `daily_pnl_` was raw realised cash flow — debited the instant you
   bought anything, credited only on a sell. A single buy of 1,000
   shares at ₹2,500 (well within the default 1,000-share order-size
   limit) produced a "loss" of ₹2.5M against a ₹50,000 kill-switch
   threshold — halting all trading on ordinary inventory acquisition,
   not an actual loss. Fixed by making `daily_pnl()` mark-to-market:
   realised cash flow + net position valued at the last traded price.
4. **Dead code.** `prune_empty_bids()`/`prune_empty_asks()` were
   declared in the header and never implemented or called (the cancel
   fix above makes an explicit pruning pass unnecessary — pruning now
   happens right where the level empties). A no-op line in
   `match_order()` (`total_quantity -= min(total_quantity,
   total_quantity)`, i.e. always zero, then immediately overwritten by
   a correct recomputation three lines later) was also removed.
5. **TWAP could execute far more than requested.** `twap_execution()`
   used `max(1, int(exe * noise))` on every interval, forcing a phantom
   1-share fill into intervals that were legitimately allocated 0 shares
   (whenever `total_shares < n_intervals`). A 5-share order sliced into
   10 intervals could execute up to ~14 shares. Fixed by only applying
   the randomise-and-floor-at-1 logic to intervals that actually got a
   nonzero allocation.

All five are covered by regression tests (`test_order_book.cpp`:
+2 tests, now 57; `tests/test_week56.py`: +1 test, now 20).

## Key Concepts

### Order Book (C++)
- Price-time priority matching
- `std::map<Price, PriceLevel, greater>` for bids (descending)
- `std::map<Price, PriceLevel>` for asks (ascending)
- `std::deque<Order*>` for FIFO time priority
- Handles: LIMIT, MARKET, IOC, FOK orders

### Risk Engine
| Metric | Formula |
|--------|---------|
| Parametric VaR | μ − z_α × σ |
| Historical VaR | percentile(-returns, α) |
| CVaR | E[loss \| loss > VaR] |
| Sharpe | (μ − r_f) / σ × √252 |
| Kelly fraction | μ / σ² |

### Interview Answers
**"Why std::map for the order book?"** — Need sorted price levels for matching. O(log n) insert/lookup is acceptable for a project engine. Production uses flat pre-allocated arrays.

**"What's wrong with VaR?"** — Not subadditive, ignores severity beyond threshold, understates fat-tail risk. CVaR fixes all three.

**"Why fractional Kelly?"** — Edge (μ) is estimated, not known. Overestimating edge → overbetting → ruin. Half-Kelly has ¾ the growth with half the variance.
