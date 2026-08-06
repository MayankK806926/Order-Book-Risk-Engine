/*
 * test_order_book.cpp — Tests and demo for the order book matching engine.
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra order_book.cpp test_order_book.cpp -o test_ob
 *
 * Run:
 *   ./test_ob
 */

#include "order_book.h"
#include "risk_manager.h"
#include <cassert>
#include <iostream>
#include <sstream>

using namespace orderbook;

// ── Test helpers ──────────────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_EQ(a, b, msg) \
    do { \
        if ((a) == (b)) { \
            std::cout << "  PASS: " << msg << "\n"; \
            ++tests_passed; \
        } else { \
            std::cout << "  FAIL: " << msg << "\n"; \
            ++tests_failed; \
        } \
    } while(0)

#define ASSERT_TRUE(cond, msg) ASSERT_EQ((cond), true, msg)
#define ASSERT_FALSE(cond, msg) ASSERT_EQ((cond), false, msg)

static OrderId next_id = 1;
static Order make_order(Side side, OrderType type, Price price, Quantity qty) {
    return Order(next_id++, side, type, price, qty);
}

// ── Test cases ────────────────────────────────────────────────────────────────

void test_basic_resting() {
    std::cout << "\n--- test_basic_resting ---\n";
    OrderBook ob("TEST");

    // Add a resting limit bid — should NOT match (no asks yet)
    auto bid = make_order(Side::BUY, OrderType::LIMIT, 100, 10);
    auto trades = ob.add_order(bid);

    ASSERT_EQ(trades.size(), 0u, "No trades for resting bid");
    ASSERT_EQ(ob.best_bid().value(), 100, "Best bid = 100");
    ASSERT_EQ(ob.bid_depth_at(100), 10u, "Bid depth = 10");
    ASSERT_EQ(ob.best_ask().has_value(), false, "No ask yet");
    ASSERT_EQ(bid.status, OrderStatus::NEW, "Bid status = NEW");
}

void test_exact_match() {
    std::cout << "\n--- test_exact_match ---\n";
    OrderBook ob("TEST");

    // Rest a sell at 100
    auto ask = make_order(Side::SELL, OrderType::LIMIT, 100, 10);
    ob.add_order(ask);

    // Matching buy at 100
    auto bid = make_order(Side::BUY, OrderType::LIMIT, 100, 10);
    auto trades = ob.add_order(bid);

    ASSERT_EQ(trades.size(), 1u, "One trade generated");
    ASSERT_EQ(trades[0].price, 100, "Fill price = 100");
    ASSERT_EQ(trades[0].quantity, 10u, "Fill qty = 10");
    ASSERT_EQ(bid.status, OrderStatus::FILLED, "Bid fully filled");
    ASSERT_EQ(ask.status, OrderStatus::FILLED, "Ask fully filled");
    ASSERT_EQ(ob.best_bid().has_value(), false, "Book empty after match");
    ASSERT_EQ(ob.best_ask().has_value(), false, "Book empty after match");
}

void test_partial_fill() {
    std::cout << "\n--- test_partial_fill ---\n";
    OrderBook ob("TEST");

    // Rest a sell of 20 at 100
    auto ask = make_order(Side::SELL, OrderType::LIMIT, 100, 20);
    ob.add_order(ask);

    // Buy only 10
    auto bid = make_order(Side::BUY, OrderType::LIMIT, 100, 10);
    auto trades = ob.add_order(bid);

    ASSERT_EQ(trades.size(), 1u, "One trade");
    ASSERT_EQ(trades[0].quantity, 10u, "Partial fill = 10");
    ASSERT_EQ(bid.status, OrderStatus::FILLED, "Bid fully filled");
    ASSERT_EQ(ask.status, OrderStatus::PARTIALLY_FILLED, "Ask partially filled");
    ASSERT_EQ(ask.remaining(), 10u, "Ask has 10 remaining");
    ASSERT_EQ(ob.ask_depth_at(100), 10u, "10 left at ask");
}

void test_price_improvement() {
    std::cout << "\n--- test_price_improvement ---\n";
    OrderBook ob("TEST");

    // Seller resting at 99
    auto ask = make_order(Side::SELL, OrderType::LIMIT, 99, 10);
    ob.add_order(ask);

    // Buyer willing to pay 100 — should match at 99 (seller's price)
    auto bid = make_order(Side::BUY, OrderType::LIMIT, 100, 10);
    auto trades = ob.add_order(bid);

    ASSERT_EQ(trades.size(), 1u, "One trade");
    ASSERT_EQ(trades[0].price, 99, "Fill at passive (ask) price = 99");
    ASSERT_EQ(bid.status, OrderStatus::FILLED, "Bid filled");
}

void test_no_cross() {
    std::cout << "\n--- test_no_cross ---\n";
    OrderBook ob("TEST");

    // Bid at 99, ask at 101 — spread = 2, no match
    auto bid = make_order(Side::BUY,  OrderType::LIMIT, 99,  10);
    auto ask = make_order(Side::SELL, OrderType::LIMIT, 101, 10);
    ob.add_order(bid);
    auto trades = ob.add_order(ask);

    ASSERT_EQ(trades.size(), 0u, "No trade when spread exists");
    ASSERT_EQ(ob.best_bid().value(), 99,  "Best bid = 99");
    ASSERT_EQ(ob.best_ask().value(), 101, "Best ask = 101");
    ASSERT_EQ(ob.spread(), 2, "Spread = 2");
}

void test_fifo_time_priority() {
    std::cout << "\n--- test_fifo_time_priority ---\n";
    OrderBook ob("TEST");

    // Two resting bids at same price — first one should fill first
    auto bid1 = make_order(Side::BUY, OrderType::LIMIT, 100, 10);
    auto bid2 = make_order(Side::BUY, OrderType::LIMIT, 100, 10);
    ob.add_order(bid1);
    ob.add_order(bid2);

    // Sell 10 — should fill bid1 entirely, not bid2
    auto ask = make_order(Side::SELL, OrderType::LIMIT, 100, 10);
    ob.add_order(ask);

    ASSERT_EQ(bid1.status, OrderStatus::FILLED, "First bid fills first (FIFO)");
    ASSERT_EQ(bid2.status, OrderStatus::NEW,    "Second bid untouched");
    ASSERT_EQ(ob.bid_depth_at(100), 10u, "10 remaining at 100");
}

void test_cancel_order() {
    std::cout << "\n--- test_cancel_order ---\n";
    OrderBook ob("TEST");

    auto bid = make_order(Side::BUY, OrderType::LIMIT, 100, 10);
    ob.add_order(bid);

    bool cancelled = ob.cancel_order(bid.id);
    ASSERT_TRUE(cancelled, "Cancel returns true");
    ASSERT_EQ(bid.status, OrderStatus::CANCELLED, "Order is cancelled");
    ASSERT_EQ(ob.bid_depth_at(100), 0u, "Depth at 100 = 0 after cancel");

    bool cancel_again = ob.cancel_order(bid.id);
    ASSERT_FALSE(cancel_again, "Cannot cancel twice");
}

void test_market_order() {
    std::cout << "\n--- test_market_order ---\n";
    OrderBook ob("TEST");

    // Rest 10 at 100, 10 at 101
    auto ask1 = make_order(Side::SELL, OrderType::LIMIT, 100, 10);
    auto ask2 = make_order(Side::SELL, OrderType::LIMIT, 101, 10);
    ob.add_order(ask1);
    ob.add_order(ask2);

    // Market buy for 15 — sweeps level 100 (10) then partial at 101 (5)
    auto mkt = make_order(Side::BUY, OrderType::MARKET, 0, 15);
    auto trades = ob.add_order(mkt);

    ASSERT_EQ(trades.size(), 2u, "Two fills from two levels");
    ASSERT_EQ(trades[0].quantity, 10u, "First fill = 10 at 100");
    ASSERT_EQ(trades[1].quantity,  5u, "Second fill = 5 at 101");
    ASSERT_EQ(mkt.status, OrderStatus::FILLED, "Market order fully filled");
    ASSERT_EQ(ob.ask_depth_at(100), 0u, "Level 100 empty");
    ASSERT_EQ(ob.ask_depth_at(101), 5u, "Level 101 has 5 remaining");
}

void test_ioc_order() {
    std::cout << "\n--- test_ioc_order ---\n";
    OrderBook ob("TEST");

    // Only 5 available at 100
    auto ask = make_order(Side::SELL, OrderType::LIMIT, 100, 5);
    ob.add_order(ask);

    // IOC buy for 10 at 100 — fill 5, cancel remaining 5
    auto ioc = make_order(Side::BUY, OrderType::IOC, 100, 10);
    auto trades = ob.add_order(ioc);

    ASSERT_EQ(trades.size(), 1u, "One trade");
    ASSERT_EQ(trades[0].quantity, 5u, "Filled 5");
    ASSERT_EQ(ioc.status, OrderStatus::CANCELLED, "IOC remainder cancelled");
    ASSERT_EQ(ioc.filled, 5u, "5 units filled before cancel");
}

void test_fok_success() {
    std::cout << "\n--- test_fok_success ---\n";
    OrderBook ob("TEST");

    auto ask = make_order(Side::SELL, OrderType::LIMIT, 100, 20);
    ob.add_order(ask);

    // FOK buy for 10 — enough liquidity, should fill
    auto fok = make_order(Side::BUY, OrderType::FOK, 100, 10);
    auto trades = ob.add_order(fok);

    ASSERT_EQ(trades.size(), 1u, "FOK filled");
    ASSERT_EQ(fok.status, OrderStatus::FILLED, "FOK status = FILLED");
}

void test_fok_failure() {
    std::cout << "\n--- test_fok_failure ---\n";
    OrderBook ob("TEST");

    // Only 5 available — FOK for 10 should be rejected entirely
    auto ask = make_order(Side::SELL, OrderType::LIMIT, 100, 5);
    ob.add_order(ask);

    auto fok = make_order(Side::BUY, OrderType::FOK, 100, 10);
    auto trades = ob.add_order(fok);

    ASSERT_EQ(trades.size(), 0u, "FOK generates no trades when can't fill");
    ASSERT_EQ(fok.status, OrderStatus::CANCELLED, "FOK cancelled");
    ASSERT_EQ(ob.ask_depth_at(100), 5u, "Ask unchanged");
}

void test_multi_level_sweep() {
    std::cout << "\n--- test_multi_level_sweep ---\n";
    OrderBook ob("TEST");

    // Store orders in a vector so pointers remain valid
    std::vector<Order> ask_orders;
    for (Price p = 101; p <= 105; ++p) {
        ask_orders.emplace_back(make_order(Side::SELL, OrderType::LIMIT, p, 10));
    }
    for (auto& ask : ask_orders) {
        ob.add_order(ask);
    }

    // Buy 35 — sweeps 101(10), 102(10), 103(10), partial 104(5)
    auto bid = make_order(Side::BUY, OrderType::LIMIT, 104, 35);
    auto trades = ob.add_order(bid);

    ASSERT_EQ(trades.size(), 4u, "Four fills across four levels");
    ASSERT_EQ(bid.status, OrderStatus::FILLED, "Bid fully filled");
    ASSERT_EQ(ob.ask_depth_at(101), 0u, "101 empty");
    ASSERT_EQ(ob.ask_depth_at(104), 5u, "5 remaining at 104");
    ASSERT_EQ(ob.ask_depth_at(105), 10u, "105 untouched");
}

void test_cancel_removes_phantom_price_level() {
    std::cout << "\n--- test_cancel_removes_phantom_price_level ---\n";
    OrderBook ob("TEST");

    // Regression test: cancelling the only order at a price level used
    // to leave a zero-quantity PriceLevel behind in the map, so
    // best_bid()/best_ask() kept reporting that price as the best quote
    // even though there was no real liquidity there.
    auto bid = make_order(Side::BUY, OrderType::LIMIT, 100, 10);
    ob.add_order(bid);
    ob.cancel_order(bid.id);

    ASSERT_EQ(ob.bid_depth_at(100), 0u, "Depth at 100 = 0 after cancel");
    ASSERT_EQ(ob.best_bid().has_value(), false,
              "best_bid() must be empty after cancelling the only bid, not phantom 100");
}

void test_risk_manager_pnl_is_mark_to_market() {
    std::cout << "\n--- test_risk_manager_pnl_is_mark_to_market ---\n";
    using orderbook::RiskManager;
    using orderbook::Trade;

    // Regression test: daily_pnl() used to be raw realised cash flow,
    // so buying inventory (pure cash outflow, no loss) looked like an
    // enormous loss and could trip the kill switch on a normal trade.
    RiskManager rm;
    rm.set_last_price(2500.0);
    Trade t(1, 2, 2500, 1000, 1);  // buy 1000 @ 2500 = Rs 2.5M cash out
    rm.on_fill(t, Side::BUY);

    ASSERT_EQ(rm.net_position(), 1000, "Position reflects the fill");
    ASSERT_TRUE(std::abs(rm.daily_pnl()) < 1e-6,
                "Mark-to-market P&L ~0 right after buying at last_price (no real loss yet)");
    ASSERT_FALSE(rm.is_breached(),
                 "Kill switch must NOT trip from ordinary buying activity");
}

// ── Demo: simulate a mini market ──────────────────────────────────────────────

void demo_market_simulation() {
    std::cout << "\n=== DEMO: Mini Market Simulation ===\n";
    OrderBook ob("RELIANCE");

    // Build initial book.
    //
    // MUST be a deque, not a vector: OrderBook stores raw Order* pointers
    // into whatever container holds these orders (see PriceLevel::orders
    // in order_book.h). A vector reallocates its buffer as it grows,
    // which invalidates every pointer the book already stored from
    // earlier iterations of the loop below — the book ends up holding
    // dangling pointers into freed memory. std::deque never invalidates
    // references to existing elements on push_back/emplace_back, so
    // pointers taken mid-loop stay valid. (This previously crashed with
    // heap corruption inside ~OrderBook, reproducible via gdb backtrace.)
    std::deque<Order> orders;

    // Bids: 2490, 2495, 2498, 2499, 2500
    for (Price p : {2490, 2495, 2498, 2499, 2500}) {
        orders.emplace_back(make_order(Side::BUY, OrderType::LIMIT, p, 100));
        ob.add_order(orders.back());
    }
    // Asks: 2501, 2502, 2503, 2505, 2510
    for (Price p : {2501, 2502, 2503, 2505, 2510}) {
        orders.emplace_back(make_order(Side::SELL, OrderType::LIMIT, p, 100));
        ob.add_order(orders.back());
    }

    ob.print(5);

    std::cout << "Submitting market buy for 150...\n";
    auto mkt = make_order(Side::BUY, OrderType::MARKET, 0, 150);
    auto trades = ob.add_order(mkt);
    for (auto& t : trades) {
        std::cout << "  TRADE: " << t.quantity << " @ " << t.price << "\n";
    }
    ob.print(5);

    std::cout << "Cancelling best bid...\n";
    auto best_bid_id = orders[4].id; // 2500 bid
    ob.cancel_order(best_bid_id);
    ob.print(5);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Order Book Test Suite ===\n";

    test_basic_resting();
    test_exact_match();
    test_partial_fill();
    test_price_improvement();
    test_no_cross();
    test_fifo_time_priority();
    test_cancel_order();
    test_market_order();
    test_ioc_order();
    test_fok_success();
    test_fok_failure();
    test_multi_level_sweep();
    test_cancel_removes_phantom_price_level();
    test_risk_manager_pnl_is_mark_to_market();

    std::cout << "\n=== Results ===\n";
    std::cout << "  Passed: " << tests_passed << "\n";
    std::cout << "  Failed: " << tests_failed << "\n";

    if (tests_failed == 0) {
        std::cout << "  All tests passed!\n\n";
    }

    demo_market_simulation();

    return tests_failed > 0 ? 1 : 0;
}
