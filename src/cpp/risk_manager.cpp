/*
 * risk_manager.cpp — Pre-trade risk check implementation.
 */

#include "risk_manager.h"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace orderbook {

std::string to_string(RiskCheckResult r) {
    switch (r) {
        case RiskCheckResult::APPROVED:                  return "APPROVED";
        case RiskCheckResult::REJECTED_POSITION_LIMIT:  return "REJECTED: position limit";
        case RiskCheckResult::REJECTED_ORDER_SIZE:       return "REJECTED: order too large";
        case RiskCheckResult::REJECTED_DAILY_LOSS:       return "REJECTED: daily loss limit";
        case RiskCheckResult::REJECTED_PRICE_SANITY:     return "REJECTED: price sanity";
        case RiskCheckResult::REJECTED_RATE_LIMIT:       return "REJECTED: rate limit";
        case RiskCheckResult::REJECTED_NO_LAST_PRICE:    return "REJECTED: no last price";
        default:                                         return "REJECTED: unknown";
    }
}

RiskManager::RiskManager(RiskLimits limits)
    : limits_(limits),
      rate_window_start_(std::chrono::steady_clock::now()) {}

RiskCheckResult RiskManager::check(const Order& order) {
    // Hard stop: if already breached, reject everything
    if (breached_) return RiskCheckResult::REJECTED_DAILY_LOSS;

    // 1. Daily loss check (mark-to-market, not raw cash flow — see
    // daily_pnl() in risk_manager.h for why that distinction matters)
    if (daily_pnl() <= limits_.max_daily_loss) {
        breached_ = true;
        std::cerr << "[RiskManager] KILL SWITCH: daily loss limit breached. "
                  << "P&L=" << daily_pnl() << "\n";
        return RiskCheckResult::REJECTED_DAILY_LOSS;
    }

    // 2. Order size check
    if (order.quantity > limits_.max_order_qty) {
        return RiskCheckResult::REJECTED_ORDER_SIZE;
    }

    // 3. Position limit check
    // Would this order breach our position limit?
    int64_t signed_qty = (order.side == Side::BUY)
        ? static_cast<int64_t>(order.quantity)
        : -static_cast<int64_t>(order.quantity);

    int64_t projected = net_position_ + signed_qty;
    if (std::abs(projected) > limits_.max_position) {
        return RiskCheckResult::REJECTED_POSITION_LIMIT;
    }

    // 4. Price sanity check (limit orders only)
    if (order.type == OrderType::LIMIT && last_price_ > 0.0) {
        double order_price  = static_cast<double>(order.price);
        double deviation    = std::abs(order_price - last_price_) / last_price_;
        if (deviation > limits_.max_price_deviation) {
            return RiskCheckResult::REJECTED_PRICE_SANITY;
        }
    } else if (order.type == OrderType::LIMIT && last_price_ == 0.0) {
        return RiskCheckResult::REJECTED_NO_LAST_PRICE;
    }

    // 5. Order rate limit
    auto now     = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                       now - rate_window_start_).count();
    if (elapsed >= 1) {
        orders_this_second_ = 0;
        rate_window_start_  = now;
    }
    if (orders_this_second_ >= limits_.max_orders_per_sec) {
        return RiskCheckResult::REJECTED_RATE_LIMIT;
    }
    ++orders_this_second_;

    return RiskCheckResult::APPROVED;
}

void RiskManager::on_fill(const Trade& trade, Side aggressor_side) {
    // Update net position and realised cash flow. daily_pnl() combines
    // this with the mark-to-market value of net_position_ at last_price_.
    int64_t qty = static_cast<int64_t>(trade.quantity);
    if (aggressor_side == Side::BUY) {
        net_position_ += qty;
        cash_flow_    -= static_cast<double>(trade.price) * trade.quantity;
    } else {
        net_position_ -= qty;
        cash_flow_    += static_cast<double>(trade.price) * trade.quantity;
    }
}

void RiskManager::on_cancel(const Order& order) {
    // Nothing to reverse for resting orders — position not yet taken
    (void)order;
}

void RiskManager::reset_daily() {
    cash_flow_  = 0.0;
    breached_   = false;
    std::cout << "[RiskManager] Daily state reset.\n";
}

void RiskManager::print_status() const {
    std::cout << "\n=== Risk Manager Status ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Net position : " << net_position_ << "\n";
    std::cout << "  Daily P&L    : " << daily_pnl() << "\n";
    std::cout << "  Last price   : " << last_price_ << "\n";
    std::cout << "  Breached     : " << (breached_ ? "YES" : "no") << "\n";
    std::cout << "  Limits:\n";
    std::cout << "    max_position      : " << limits_.max_position << "\n";
    std::cout << "    max_order_qty     : " << limits_.max_order_qty << "\n";
    std::cout << "    max_daily_loss    : " << limits_.max_daily_loss << "\n";
    std::cout << "    max_orders/sec    : " << limits_.max_orders_per_sec << "\n";
    std::cout << "===========================\n\n";
}

} // namespace orderbook
