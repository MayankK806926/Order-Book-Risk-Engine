/*
 * order_book.h — Limit order book matching engine.
 *
 * ARCHITECTURE:
 *   OrderBook
 *   ├── bids_  : std::map<Price, PriceLevel, std::greater<Price>>
 *   │             Descending — best bid at top (highest price)
 *   └── asks_  : std::map<Price, PriceLevel>
 *                Ascending  — best ask at top (lowest price)
 *
 * DESIGN DECISIONS (explain these in interviews):
 *
 * 1. WHY std::map (not unordered_map):
 *    We need price levels in sorted order — best bid/ask, iteration
 *    across price levels for partial fills. std::map gives O(log n)
 *    insert/lookup with guaranteed ordering. unordered_map is O(1)
 *    average but has no ordering — wrong for an order book.
 *    In production HFT: custom sorted arrays or skip lists, but
 *    std::map is correct and readable for an interview project.
 *
 * 2. WHY std::deque<Order> for time priority:
 *    Price-time priority: at the same price, earlier orders fill first.
 *    deque supports O(1) push_back (new order) and pop_front (fill).
 *    A queue of Orders at each price level implements FIFO correctly.
 *
 * 3. WHY not a flat array:
 *    Cache-friendly flat arrays are faster but require knowing the
 *    price range in advance. For a general-purpose engine, map is safer.
 *    Production systems (e.g. LMAX Disruptor) use pre-allocated arrays.
 *
 * 4. WHAT IS MISSING FOR PRODUCTION:
 *    - Lock-free data structures (concurrent access)
 *    - Memory pool allocation (avoid heap fragmentation)
 *    - Nanosecond timestamps
 *    - Kernel bypass networking (DPDK, RDMA)
 *    - Order ID hash map for O(1) cancel lookup
 *    Acknowledge ALL of these in your README.
 *
 * INTERVIEW QUESTION:
 * "What is the time complexity of adding an order?"
 * A: "O(log P) where P = number of distinct price levels, for the
 *    map lookup. O(1) amortised for deque push. Matching is O(F)
 *    where F = number of fills generated."
 */

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace orderbook {

// ── Types ─────────────────────────────────────────────────────────────────────

using OrderId  = uint64_t;
using Price    = int64_t;   // Fixed-point: price in ticks (avoid float precision)
using Quantity = uint64_t;

/*
 * WHY FIXED-POINT PRICE:
 * Floating-point arithmetic is non-deterministic across architectures
 * and compilers. For an order book, you need exact equality checks
 * ("does this order match this price level?"). Use integer ticks.
 * E.g. if tick size = 0.05 INR, price 2500.00 = 50000 ticks.
 */

// ── Enums ─────────────────────────────────────────────────────────────────────

enum class Side : uint8_t {
    BUY  = 0,
    SELL = 1
};

enum class OrderType : uint8_t {
    LIMIT  = 0,
    MARKET = 1,
    IOC    = 2,   // Immediate-or-Cancel: fill what you can, cancel the rest
    FOK    = 3    // Fill-or-Kill: fill everything or cancel entirely
};

enum class OrderStatus : uint8_t {
    NEW          = 0,
    PARTIALLY_FILLED = 1,
    FILLED       = 2,
    CANCELLED    = 3,
    REJECTED     = 4
};

// ── Order ─────────────────────────────────────────────────────────────────────

struct Order {
    OrderId    id;
    Side       side;
    OrderType  type;
    Price      price;       // 0 for market orders
    Quantity   quantity;    // total quantity requested
    Quantity   filled;      // quantity filled so far
    OrderStatus status;
    uint64_t   timestamp;   // nanoseconds since epoch (or sequence number)

    Order(OrderId id, Side side, OrderType type,
          Price price, Quantity quantity, uint64_t timestamp = 0)
        : id(id), side(side), type(type), price(price),
          quantity(quantity), filled(0),
          status(OrderStatus::NEW), timestamp(timestamp) {}

    Quantity remaining() const { return quantity - filled; }
    bool     is_done()   const {
        return status == OrderStatus::FILLED ||
               status == OrderStatus::CANCELLED ||
               status == OrderStatus::REJECTED;
    }
};

// ── Trade (fill record) ───────────────────────────────────────────────────────

struct Trade {
    OrderId  buy_order_id;
    OrderId  sell_order_id;
    Price    price;
    Quantity quantity;
    uint64_t timestamp;

    Trade(OrderId bid, OrderId ask, Price p, Quantity q, uint64_t t = 0)
        : buy_order_id(bid), sell_order_id(ask),
          price(p), quantity(q), timestamp(t) {}
};

// ── PriceLevel ────────────────────────────────────────────────────────────────

/*
 * A PriceLevel holds all orders at a specific price, in time order.
 * FIFO: first order in → first to be filled (price-time priority).
 *
 * Using std::deque:
 * - push_back (new order):  O(1) amortised
 * - pop_front (fill):       O(1)
 * - Iteration for partial:  O(n) at this level
 */
class PriceLevel {
public:
    Price              price;
    Quantity           total_quantity;   // cached sum (avoid O(n) recompute)
    std::deque<Order*> orders;           // pointers into the order pool

    explicit PriceLevel(Price p) : price(p), total_quantity(0) {}

    void add_order(Order* o) {
        orders.push_back(o);
        total_quantity += o->remaining();
    }

    // Remove fully-filled or cancelled orders from the front
    void clean() {
        while (!orders.empty() && orders.front()->is_done()) {
            orders.pop_front();
        }
    }

    bool empty() const { return orders.empty() || total_quantity == 0; }
};

// ── OrderBook ─────────────────────────────────────────────────────────────────

class OrderBook {
public:
    explicit OrderBook(std::string symbol = "UNKNOWN");

    // ── Core operations ───────────────────────────────────────────────────────

    /*
     * add_order: submit a new order to the book.
     * Returns a vector of Trade objects generated (may be empty for resting orders).
     * Modifies order status in-place.
     *
     * PROCESS:
     * 1. For market/IOC/FOK: attempt to match immediately
     * 2. For limit orders: match first, then rest any unfilled quantity
     * 3. IOC: cancel unfilled remainder after matching
     * 4. FOK: if cannot fill entirely, cancel entire order
     */
    std::vector<Trade> add_order(Order& order);

    /*
     * cancel_order: remove an order from the book.
     * Returns true if found and cancelled, false if already filled/not found.
     *
     * NOTE: O(log P + n) where n = orders at that price level.
     * Production: use a hash map from order_id → PriceLevel* for O(1) cancel.
     */
    bool cancel_order(OrderId id);

    // ── Queries ───────────────────────────────────────────────────────────────

    std::optional<Price>    best_bid()  const;
    std::optional<Price>    best_ask()  const;
    std::optional<Price>    mid_price() const;
    Price                   spread()    const;

    Quantity bid_depth_at(Price p) const;
    Quantity ask_depth_at(Price p) const;

    // Total quantity on each side across all price levels
    Quantity total_bid_quantity() const;
    Quantity total_ask_quantity() const;

    // Number of distinct price levels on each side
    size_t   num_bid_levels() const { return bids_.size(); }
    size_t   num_ask_levels() const { return asks_.size(); }

    // All trades generated since construction
    const std::vector<Trade>& trade_history() const { return trades_; }

    // Human-readable order book snapshot
    void print(int levels = 5) const;

    const std::string& symbol() const { return symbol_; }
    uint64_t sequence_number() const  { return seq_num_; }

private:
    // ── Internal state ────────────────────────────────────────────────────────

    std::string symbol_;
    uint64_t    seq_num_ = 0;

    /*
     * bids_: descending map — best bid (highest price) comes first
     *   std::greater<Price> makes map::begin() → best bid
     *
     * asks_: ascending map — best ask (lowest price) comes first
     *   default less<Price> makes map::begin() → best ask
     */
    std::map<Price, PriceLevel, std::greater<Price>> bids_;
    std::map<Price, PriceLevel>                      asks_;

    // Trade history (for TCA, analytics)
    std::vector<Trade> trades_;

    // ── Matching engine internals ─────────────────────────────────────────────

    std::vector<Trade> match_order(Order& aggressive);
    void               rest_order(Order& passive);
    void               execute_fill(Order& aggressive, Order& passive,
                                     Price fill_price, Quantity fill_qty);
};

} // namespace orderbook
