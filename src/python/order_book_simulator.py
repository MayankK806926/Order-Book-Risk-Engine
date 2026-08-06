"""
python/order_book_simulator.py — Python order book simulator and visualiser.

Simulates realistic order flow and renders a live-updating depth chart.
Companion to the C++ matching engine — same concepts, Python speed for analysis.

WHAT THIS DOES:
1. Simulates a market with random limit and market orders arriving
2. Maintains a Python order book with bid/ask sides
3. Plots a live depth chart (cumulative volume at each price level)
4. Tracks mid price, spread, and trade history over time

WHY A PYTHON VERSION:
- Rapid prototyping: test strategies against simulated microstructure
- Visualisation: C++ produces numbers, Python produces charts
- Analysis: compute microstructure metrics (spread, depth, imbalance)

USAGE:
    python order_book_simulator.py              # run simulation
    python order_book_simulator.py --static     # static snapshot only
"""

import argparse
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Colours ───────────────────────────────────────────────────────────────────
DARK   = '#0F1117'
PANEL  = '#1A1D27'
GRID   = '#2A2D3A'
TEXT   = '#C8CCD8'
BLUE   = '#4C9BE8'
GREEN  = '#56C27D'
RED    = '#E85C5C'
AMBER  = '#F5A623'
MUTED  = '#6B7280'


# ── Data structures ───────────────────────────────────────────────────────────

class Side(Enum):
    BID = 'bid'
    ASK = 'ask'

@dataclass
class Order:
    id:       int
    side:     Side
    price:    float
    quantity: int
    time:     int = 0

@dataclass
class Trade:
    price:    float
    quantity: int
    time:     int


class OrderBook:
    """
    Pure Python order book with price-time priority matching.
    Simpler than the C++ version — for analysis and visualisation.
    """

    def __init__(self, tick_size: float = 0.05):
        self.tick_size   = tick_size
        self.bids: dict[float, list[Order]] = defaultdict(list)  # price → [orders]
        self.asks: dict[float, list[Order]] = defaultdict(list)
        self.trades: list[Trade] = []
        self._order_id   = 0
        self._time       = 0

    def _next_id(self) -> int:
        self._order_id += 1
        return self._order_id

    def _round_price(self, p: float) -> float:
        return round(round(p / self.tick_size) * self.tick_size, 6)

    def add_limit_order(self, side: Side, price: float, qty: int) -> list[Trade]:
        price  = self._round_price(price)
        order  = Order(self._next_id(), side, price, qty, self._time)
        trades = self._match(order, side)
        # Rest unfilled quantity
        if order.quantity > 0:
            if side == Side.BID:
                self.bids[price].append(order)
            else:
                self.asks[price].append(order)
        return trades

    def add_market_order(self, side: Side, qty: int) -> list[Trade]:
        order  = Order(self._next_id(), side, 0, qty, self._time)
        trades = self._match(order, side)
        return trades

    def _match(self, aggressive: Order, side: Side) -> list[Trade]:
        trades = []
        if side == Side.BID:
            # Match against asks (ascending)
            for price in sorted(self.asks.keys()):
                if aggressive.quantity == 0:
                    break
                if aggressive.price > 0 and price > aggressive.price:
                    break
                queue = self.asks[price]
                while queue and aggressive.quantity > 0:
                    passive = queue[0]
                    fill = min(aggressive.quantity, passive.quantity)
                    aggressive.quantity -= fill
                    passive.quantity    -= fill
                    trade = Trade(price, fill, self._time)
                    trades.append(trade)
                    self.trades.append(trade)
                    if passive.quantity == 0:
                        queue.pop(0)
                if not queue:
                    del self.asks[price]
        else:
            for price in sorted(self.bids.keys(), reverse=True):
                if aggressive.quantity == 0:
                    break
                if aggressive.price > 0 and price < aggressive.price:
                    break
                queue = self.bids[price]
                while queue and aggressive.quantity > 0:
                    passive = queue[0]
                    fill = min(aggressive.quantity, passive.quantity)
                    aggressive.quantity -= fill
                    passive.quantity    -= fill
                    trade = Trade(price, fill, self._time)
                    trades.append(trade)
                    self.trades.append(trade)
                    if passive.quantity == 0:
                        queue.pop(0)
                if not queue:
                    del self.bids[price]
        return trades

    def cancel_random(self, side: Side):
        book = self.bids if side == Side.BID else self.asks
        if not book:
            return
        price = random.choice(list(book.keys()))
        if book[price]:
            book[price].pop(0)
        if not book[price]:
            del book[price]

    @property
    def best_bid(self) -> float | None:
        return max(self.bids.keys()) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks.keys()) if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0

    def bid_depth(self, n_levels: int = 10) -> pd.DataFrame:
        """Return top n bid levels as DataFrame."""
        rows = []
        for price in sorted(self.bids.keys(), reverse=True)[:n_levels]:
            qty = sum(o.quantity for o in self.bids[price])
            rows.append({'price': price, 'quantity': qty})
        df = pd.DataFrame(rows)
        if not df.empty:
            df['cumulative'] = df['quantity'].cumsum()
        return df

    def ask_depth(self, n_levels: int = 10) -> pd.DataFrame:
        rows = []
        for price in sorted(self.asks.keys())[:n_levels]:
            qty = sum(o.quantity for o in self.asks[price])
            rows.append({'price': price, 'quantity': qty})
        df = pd.DataFrame(rows)
        if not df.empty:
            df['cumulative'] = df['quantity'].cumsum()
        return df

    def order_imbalance(self) -> float:
        """
        Order book imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
        Range: [-1, 1]. Positive = more bids (bullish pressure).
        Used as a short-term price direction signal in HFT.
        """
        bid_qty = sum(sum(o.quantity for o in orders) for orders in self.bids.values())
        ask_qty = sum(sum(o.quantity for o in orders) for orders in self.asks.values())
        total = bid_qty + ask_qty
        if total == 0:
            return 0.0
        return (bid_qty - ask_qty) / total


# ── Market simulator ──────────────────────────────────────────────────────────

class MarketSimulator:
    """
    Generates realistic order flow around a drifting fundamental price.

    ORDER ARRIVAL PROCESS:
    - Limit orders arrive according to a Poisson process
    - 70% limit orders, 30% market orders (realistic ratio)
    - Order prices are normally distributed around the mid price
    - Order sizes follow a Pareto distribution (few large, many small)

    FUNDAMENTAL PRICE:
    The "true" value drifts as a random walk with mean reversion.
    Market makers quote around this, creating a realistic spread.
    """

    def __init__(self, mid_price: float = 2500.0, tick_size: float = 0.05,
                  volatility: float = 0.0005):
        self.book          = OrderBook(tick_size=tick_size)
        self.fundamental   = mid_price
        self.tick_size     = tick_size
        self.volatility    = volatility
        self.mid_history:  list[float] = []
        self.spread_history: list[float] = []
        self.imbalance_history: list[float] = []
        self.t             = 0
        self._seed_book(mid_price)

    def _seed_book(self, mid: float, n_levels: int = 10):
        """Place initial resting orders to create a realistic book."""
        for i in range(1, n_levels + 1):
            bid_price = self.book._round_price(mid - i * self.tick_size)
            ask_price = self.book._round_price(mid + i * self.tick_size)
            bid_qty   = random.randint(50, 500)
            ask_qty   = random.randint(50, 500)
            self.book.add_limit_order(Side.BID, bid_price, bid_qty)
            self.book.add_limit_order(Side.ASK, ask_price, ask_qty)

    def step(self):
        """Simulate one time step of market activity."""
        self.t += 1
        self.book._time = self.t

        # Update fundamental price (random walk with mean reversion)
        shock = np.random.normal(0, self.volatility) * self.fundamental
        self.fundamental += shock * 0.3  # slow drift

        mid = self.book.mid_price or self.fundamental

        # Generate orders
        n_orders = np.random.poisson(5)  # avg 5 orders per step
        for _ in range(n_orders):
            order_type = random.random()

            if order_type < 0.15:
                # Market order (15%)
                side = random.choice([Side.BID, Side.ASK])
                qty  = max(1, int(np.random.exponential(50)))
                self.book.add_market_order(side, qty)

            elif order_type < 0.25:
                # Cancel (10%)
                self.book.cancel_random(random.choice([Side.BID, Side.ASK]))

            else:
                # Limit order (75%)
                side  = random.choice([Side.BID, Side.ASK])
                # Price normally distributed around mid with half-spread
                spread_half = max(self.tick_size, mid * 0.0002)
                if side == Side.BID:
                    price = mid - abs(np.random.normal(spread_half, spread_half * 2))
                else:
                    price = mid + abs(np.random.normal(spread_half, spread_half * 2))
                qty = max(1, int(np.random.exponential(100)))
                self.book.add_limit_order(side, self.book._round_price(price), qty)

        # Record state
        if self.book.mid_price:
            self.mid_history.append(self.book.mid_price)
            self.spread_history.append(self.book.spread)
            self.imbalance_history.append(self.book.order_imbalance())


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_book_snapshot(ob: OrderBook, title: str = 'Order Book Snapshot',
                        save_path: str = None):
    """
    Plot order book depth chart:
    Left panel:  bid/ask depth at each price level (volume bars)
    Right panel: cumulative depth (market impact curve)
    """
    bids = ob.bid_depth(15)
    asks = ob.ask_depth(15)
    mid  = ob.mid_price or 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK)

    for ax in [ax1, ax2]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4)

    # Panel 1: Depth bars
    if not bids.empty:
        ax1.barh(bids['price'], bids['quantity'],
                 color=GREEN, alpha=0.7, height=ob.tick_size * 0.8, label='Bids')
    if not asks.empty:
        ax1.barh(asks['price'], asks['quantity'],
                 color=RED, alpha=0.7, height=ob.tick_size * 0.8, label='Asks')
    if mid:
        ax1.axhline(mid, color=AMBER, linewidth=1.5, linestyle='--',
                    label=f'Mid={mid:.2f}')
    ax1.set_xlabel('Quantity', color=TEXT)
    ax1.set_ylabel('Price', color=TEXT)
    ax1.set_title('Order book depth', color=TEXT, fontsize=10)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # Panel 2: Cumulative depth (market impact)
    if not bids.empty:
        ax2.fill_betweenx(bids['price'], bids['cumulative'], 0,
                          color=GREEN, alpha=0.4, label='Bid cumulative')
        ax2.plot(bids['cumulative'], bids['price'], color=GREEN, linewidth=1.5)
    if not asks.empty:
        ax2.fill_betweenx(asks['price'], asks['cumulative'], 0,
                          color=RED, alpha=0.4, label='Ask cumulative')
        ax2.plot(asks['cumulative'], asks['price'], color=RED, linewidth=1.5)
    if mid:
        ax2.axhline(mid, color=AMBER, linewidth=1.5, linestyle='--',
                    label=f'Spread={ob.spread:.3f}')
    ax2.set_xlabel('Cumulative quantity', color=TEXT)
    ax2.set_title('Cumulative depth (market impact curve)', color=TEXT, fontsize=10)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    plt.suptitle(title, color=TEXT, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_simulation_results(sim: MarketSimulator, save_path: str = None):
    """
    4-panel simulation results:
    Panel 1: Mid price over time
    Panel 2: Bid-ask spread over time
    Panel 3: Order book imbalance over time
    Panel 4: Final order book depth snapshot
    """
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    bids = sim.book.bid_depth(15)
    asks = sim.book.ask_depth(15)
    mid  = sim.book.mid_price or 0

    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4)

    # Panel 1: Mid price
    ax = axes[0]
    ax.plot(sim.mid_history, color=BLUE, linewidth=1, label='Mid price')
    ax.set_title('Mid price over time', color=TEXT, fontsize=10)
    ax.set_ylabel('Price', color=TEXT)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # Panel 2: Spread
    ax = axes[1]
    ax.fill_between(range(len(sim.spread_history)), sim.spread_history,
                    color=AMBER, alpha=0.6, label='Bid-ask spread')
    ax.plot(sim.spread_history, color=AMBER, linewidth=0.8)
    ax.set_title('Bid-ask spread over time', color=TEXT, fontsize=10)
    ax.set_ylabel('Spread (₹)', color=TEXT)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # Panel 3: Imbalance
    ax = axes[2]
    imb = sim.imbalance_history
    ax.fill_between(range(len(imb)), imb, 0,
                    where=[x > 0 for x in imb], color=GREEN, alpha=0.5, label='Bid pressure')
    ax.fill_between(range(len(imb)), imb, 0,
                    where=[x < 0 for x in imb], color=RED,   alpha=0.5, label='Ask pressure')
    ax.plot(imb, color=BLUE, linewidth=0.6, alpha=0.8)
    ax.axhline(0, color=TEXT, linewidth=0.6, linestyle=':')
    ax.set_title('Order book imbalance (bid pressure vs ask pressure)',
                 color=TEXT, fontsize=10)
    ax.set_ylabel('Imbalance [-1, 1]', color=TEXT)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # Panel 4: Final depth
    ax = axes[3]
    if not bids.empty:
        ax.barh(bids['price'], bids['quantity'],
                color=GREEN, alpha=0.7, height=sim.tick_size * 0.8, label='Bids')
    if not asks.empty:
        ax.barh(asks['price'], asks['quantity'],
                color=RED, alpha=0.7, height=sim.tick_size * 0.8, label='Asks')
    if mid:
        ax.axhline(mid, color=AMBER, linewidth=1.5, linestyle='--',
                   label=f'Mid={mid:.2f}  Spread={sim.book.spread:.3f}')
    ax.set_title('Final order book depth', color=TEXT, fontsize=10)
    ax.set_xlabel('Quantity', color=TEXT)
    ax.set_ylabel('Price', color=TEXT)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    plt.suptitle('Order Book Simulation — Market Microstructure Analysis',
                 color=TEXT, fontsize=13, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_simulation(n_steps: int = 500, mid_price: float = 2500.0,
                    save_path: str = None):
    """Run the full market simulation and plot results."""
    print(f"\n[Simulator] Running {n_steps} time steps...")
    print(f"[Simulator] Initial mid price: ₹{mid_price:.2f}")

    random.seed(42)
    np.random.seed(42)

    sim = MarketSimulator(mid_price=mid_price, tick_size=0.05, volatility=0.0003)

    for t in range(n_steps):
        sim.step()
        if t % 100 == 0:
            mid = sim.book.mid_price
            spr = sim.book.spread
            imb = sim.book.order_imbalance()
            trades = len(sim.book.trades)
            print(f"  t={t:4d}  mid={mid:.2f}  spread={spr:.3f}  "
                  f"imbalance={imb:+.3f}  trades={trades}")

    print(f"\n[Simulator] Simulation complete.")
    print(f"  Total trades   : {len(sim.book.trades)}")
    print(f"  Final mid price: ₹{sim.book.mid_price:.2f}")
    print(f"  Final spread   : ₹{sim.book.spread:.3f}")
    print(f"  Bid levels     : {len(sim.book.bids)}")
    print(f"  Ask levels     : {len(sim.book.asks)}")

    save = save_path or 'outputs/order_book_simulation.png'
    import os; os.makedirs('outputs', exist_ok=True)
    plot_simulation_results(sim, save_path=save)

    return sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--price', type=float, default=2500.0)
    parser.add_argument('--save',  type=str, default=None)
    args = parser.parse_args()
    run_simulation(n_steps=args.steps, mid_price=args.price, save_path=args.save)
