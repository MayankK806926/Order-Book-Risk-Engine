/*
 * risk_manager.h — Pre-trade risk checks for the order book engine.
 *
 * In a production trading system, EVERY order must pass risk checks
 * before being submitted to the matching engine. This is non-negotiable.
 *
 * THREE LAYERS OF RISK IN HFT:
 * 1. Pre-trade (this file): check before sending to exchange
 * 2. In-flight:             monitor orders in the book
 * 3. Post-trade:            reconcile fills, compute P&L, update limits
 *
 * WHAT THIS CHECKS:
 * - Position limits:    max long/short per symbol
 * - Order size limits:  max qty per single order
 * - Daily loss limits:  kill switch if P&L drops below threshold
 * - Price sanity:       order not more than N% from last trade price
 * - Order rate limits:  max orders per second (prevent runaway algos)
 *
 * INTERVIEW QUESTION:
 * "What happens if the risk manager goes down?"
 * A: "All order submission must be halted. A risk manager failure
 *    is treated the same as a breach — safe state is no trading."
 */

#pragma once
#include "order_book.h"
#include <chrono>
#include <string>
#include <unordered_map>

namespace orderbook {

struct RiskLimits {
    int64_t  max_position        = 10000;   // max net long or short
    uint64_t max_order_qty       = 1000;    // max single order size
    double   max_daily_loss      = -50000;  // daily P&L kill switch (₹)
    double   max_price_deviation = 0.05;    // 5% from last trade price
    int      max_orders_per_sec  = 100;     // order rate limit
};

enum class RiskCheckResult {
    APPROVED,
    REJECTED_POSITION_LIMIT,
    REJECTED_ORDER_SIZE,
    REJECTED_DAILY_LOSS,
    REJECTED_PRICE_SANITY,
    REJECTED_RATE_LIMIT,
    REJECTED_NO_LAST_PRICE,
};

std::string to_string(RiskCheckResult r);

class RiskManager {
public:
    explicit RiskManager(RiskLimits limits = RiskLimits{});

    // Check order before submission — returns APPROVED or rejection reason
    RiskCheckResult check(const Order& order);

    // Called after each fill to update position and P&L
    void on_fill(const Trade& trade, Side aggressor_side);

    // Called when order is cancelled — reverse any reserved position
    void on_cancel(const Order& order);

    // State queries
    int64_t  net_position()  const { return net_position_; }

    // Mark-to-market daily P&L = realised cash flow from fills (negative
    // while buying, positive while selling) + current mark-to-market
    // value of whatever position is still held, valued at last_price_.
    //
    // WHY NOT JUST THE RAW CASH FLOW:
    // A version that returned the raw cash flow directly would report
    // an ever-growing "loss" the moment you buy anything and never sell
    // it back the same day — you've only spent cash to acquire inventory
    // of equal value, not lost money. That would trip the daily-loss
    // kill switch on ordinary buying activity, not on actual losses.
    double   daily_pnl() const {
        return cash_flow_ + static_cast<double>(net_position_) * last_price_;
    }
    bool     is_breached()   const { return breached_; }
    void     reset_daily();

    // Update last traded price (from market data feed)
    void set_last_price(double price) { last_price_ = price; }

    void print_status() const;

private:
    RiskLimits limits_;
    int64_t    net_position_   = 0;
    double     cash_flow_      = 0.0;   // realised cash flow from fills only
    double     last_price_     = 0.0;
    bool       breached_       = false;

    // Rate limiting
    int                                    orders_this_second_ = 0;
    std::chrono::steady_clock::time_point  rate_window_start_;
};

} // namespace orderbook
