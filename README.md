# Order Book & Risk Engine

A limit order book matching engine and pre-trade risk manager written in
C++17, plus a companion Python layer for market microstructure analysis,
position sizing, and execution algorithms. The book implements
price-time priority matching — the same discipline used by NSE, BSE,
and NYSE — across LIMIT, MARKET, IOC, and FOK order types.

## Features

- **Price-time priority matching engine** — `std::map<Price, PriceLevel>`
  for sorted price levels (descending for bids, ascending for asks),
  `std::deque<Order*>` per level for FIFO time priority
- **Four order types**: LIMIT, MARKET, IOC (immediate-or-cancel), FOK
  (fill-or-kill), including multi-level sweeps and partial fills
- **Pre-trade risk manager**: position limits, order-size limits,
  order-rate limiting, and a mark-to-market daily-loss kill switch
- **57 C++ unit tests** covering the order book and risk manager
- **Python layer**: order book simulator with a live depth chart,
  microstructure metrics (Amihud illiquidity, Roll spread, price
  impact), position sizing (Kelly, vol-targeting, VaR-based limits),
  and execution algorithms (VWAP, TWAP, implementation shortfall) —
  **20 Python unit tests**

## Architecture

| Component | Responsibility |
|---|---|
| `OrderBook` | Matching engine — accepts orders, matches against the opposite side, rests unfilled limit quantity |
| `RiskManager` | Pre-trade checks on every order before it reaches the book; tracks net position and mark-to-market P&L |
| `PriceLevel` | One price, FIFO queue of orders, cached total quantity |

Time complexity: `O(log P)` to add an order (`P` = distinct price
levels), `O(1)` amortised to rest it, `O(F)` to match (`F` = fills
generated).

## Risk Metrics (Python layer)

| Metric | Formula |
|--------|---------|
| Parametric VaR | μ − z_α × σ |
| Historical VaR | percentile(-returns, α) |
| CVaR | E[loss \| loss > VaR] |
| Sharpe | (μ − r_f) / σ × √252 |
| Kelly fraction | μ / σ² |

## Project Structure

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

## Engineering Notes

Five defects found in review, each covered by a regression test:

- **Heap corruption in the demo.** A market-sim demo stored orders in a
  `std::vector` while interleaving inserts with `add_order()` calls —
  but the book keeps raw pointers into that storage, and vector
  reallocation on growth invalidates them. Confirmed via `gdb`
  backtrace (the segfault surfaced later, inside the destructor).
  Fixed by switching to `std::deque`, which never invalidates
  references to existing elements on insert.
- **Phantom quotes after cancellation.** Cancelling the last order at a
  price level zeroed its quantity but never erased the map entry, so
  `best_bid()`/`best_ask()` kept reporting a price with no real
  liquidity behind it. Fixed by erasing the level once it empties.
- **Kill switch could trip on a normal buy.** The risk manager tracked
  raw realised cash flow as "P&L" — a single buy looked like a large
  loss the instant it executed, capable of halting trading on ordinary
  inventory acquisition. Fixed with proper mark-to-market accounting
  (cash flow + position valued at last traded price).
- **TWAP could over-execute.** A rounding/randomisation step forced a
  phantom minimum fill into intervals that were legitimately allocated
  zero shares, so a small order sliced across many intervals could
  execute well above the requested size. Fixed to only apply the
  randomise-and-floor step to intervals with a real allocation.
- Removed a block of dead code (`match_order()` computed and
  immediately discarded a quantity, and two pruning functions were
  declared but never implemented or called).
