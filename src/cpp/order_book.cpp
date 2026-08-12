/*
 * order_book.cpp - Limit order book matching engine implementation.
 *
 * The matching engine is the heart of any exchange or HFT system.
 * This implementation uses price-time priority (FIFO at each price level)
 * which is the standard used by NSE, BSE, NYSE, and most modern exchanges.
 */

#include "order_book.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace orderbook {

// ── Constructor ───────────────────────────────────────────────────────────────

OrderBook::OrderBook(std::string symbol)
    : symbol_(std::move(symbol)) {}

// ── add_order - main entry point ──────────────────────────────────────────────

std::vector<Trade> OrderBook::add_order(Order& order) {
    order.timestamp = ++seq_num_;

    // FOK: check if we can fill entirely BEFORE matching
    if (order.type == OrderType::FOK) {
        Quantity available = 0;
        if (order.side == Side::BUY) {
            for (auto& [price, level] : asks_) {
                if (price > order.price) break;
                available += level.total_quantity;
                if (available >= order.quantity) break;
            }
        } else {
            for (auto& [price, level] : bids_) {
                if (price < order.price) break;
                available += level.total_quantity;
                if (available >= order.quantity) break;
            }
        }
        if (available < order.quantity) {
            order.status = OrderStatus::CANCELLED;
            return {};
        }
    }

    // Match against the opposite side
    std::vector<Trade> new_trades = match_order(order);

    // Add generated trades to history
    for (auto& t : new_trades) {
        trades_.push_back(t);
    }

    // Rest unfilled quantity for LIMIT orders
    if (!order.is_done()) {
        if (order.type == OrderType::LIMIT) {
            rest_order(order);
        } else if (order.type == OrderType::IOC ||
                   order.type == OrderType::MARKET) {
            // IOC/Market: cancel any unfilled remainder
            order.status = OrderStatus::CANCELLED;
        }
    }

    return new_trades;
}

// ── match_order - core matching logic ─────────────────────────────────────────

std::vector<Trade> OrderBook::match_order(Order& aggressive) {
    /*
     * MATCHING RULES:
     * BUY  order matches SELL orders at ask_price <= buy_price (price improvement)
     * SELL order matches BUY  orders at bid_price >= sell_price
     *
     * Fill price = PASSIVE order's price (maker's price)
     * This rewards the passive (resting) order with price improvement.
     *
     * WHY PASSIVE PRICE:
     * The aggressive order is "crossing the spread" - they accept the
     * resting order's price. This is standard exchange convention.
     */
    std::vector<Trade> trades;

    if (aggressive.side == Side::BUY) {
        // Match BUY against asks (ascending price - best ask first)
        auto it = asks_.begin();
        while (it != asks_.end() && aggressive.remaining() > 0) {
            Price level_price = it->first;

            // For LIMIT orders: stop if ask is above our limit price
            if (aggressive.type == OrderType::LIMIT &&
                level_price > aggressive.price) {
                break;
            }

            PriceLevel& level = it->second;
            level.clean();

            // Match against each order at this price level (FIFO)
            while (!level.orders.empty() && aggressive.remaining() > 0) {
                Order* passive = level.orders.front();
                if (passive->is_done()) {
                    level.orders.pop_front();
                    continue;
                }

                Quantity fill_qty = std::min(aggressive.remaining(),
                                              passive->remaining());
                execute_fill(aggressive, *passive, level_price, fill_qty);
                trades.emplace_back(aggressive.id, passive->id,
                                     level_price, fill_qty, seq_num_);

                if (passive->is_done()) {
                    level.orders.pop_front();
                }
            }

            // Recompute total_quantity after matching
            Quantity remaining = 0;
            for (auto* o : level.orders) {
                if (!o->is_done()) remaining += o->remaining();
            }
            level.total_quantity = remaining;

            if (level.empty()) {
                it = asks_.erase(it);
            } else {
                ++it;
            }
        }

    } else {
        // Match SELL against bids (descending price - best bid first)
        auto it = bids_.begin();
        while (it != bids_.end() && aggressive.remaining() > 0) {
            Price level_price = it->first;

            if (aggressive.type == OrderType::LIMIT &&
                level_price < aggressive.price) {
                break;
            }

            PriceLevel& level = it->second;
            level.clean();

            while (!level.orders.empty() && aggressive.remaining() > 0) {
                Order* passive = level.orders.front();
                if (passive->is_done()) {
                    level.orders.pop_front();
                    continue;
                }

                Quantity fill_qty = std::min(aggressive.remaining(),
                                              passive->remaining());
                execute_fill(*passive, aggressive, level_price, fill_qty);
                trades.emplace_back(passive->id, aggressive.id,
                                     level_price, fill_qty, seq_num_);

                if (passive->is_done()) {
                    level.orders.pop_front();
                }
            }

            // Recompute level total
            Quantity remaining = 0;
            for (auto* o : level.orders) {
                if (!o->is_done()) remaining += o->remaining();
            }
            level.total_quantity = remaining;

            if (level.empty()) {
                it = bids_.erase(it);
            } else {
                ++it;
            }
        }
    }

    return trades;
}

// ── execute_fill - update order states ───────────────────────────────────────

void OrderBook::execute_fill(Order& buy, Order& sell,
                               Price fill_price, Quantity fill_qty) {
    (void)fill_price; // used by caller for Trade record

    buy.filled  += fill_qty;
    sell.filled += fill_qty;

    auto update_status = [](Order& o) {
        if (o.filled >= o.quantity) {
            o.status = OrderStatus::FILLED;
        } else {
            o.status = OrderStatus::PARTIALLY_FILLED;
        }
    };

    update_status(buy);
    update_status(sell);
}

// ── rest_order - add unfilled LIMIT order to book ────────────────────────────

void OrderBook::rest_order(Order& passive) {
    if (passive.side == Side::BUY) {
        auto it = bids_.find(passive.price);
        if (it == bids_.end()) {
            bids_.emplace(passive.price, PriceLevel(passive.price));
            it = bids_.find(passive.price);
        }
        it->second.add_order(&passive);
    } else {
        auto it = asks_.find(passive.price);
        if (it == asks_.end()) {
            asks_.emplace(passive.price, PriceLevel(passive.price));
            it = asks_.find(passive.price);
        }
        it->second.add_order(&passive);
    }
}

// ── cancel_order ──────────────────────────────────────────────────────────────

bool OrderBook::cancel_order(OrderId id) {
    // Search bids.
    //
    // Cancelling the last live order at a price level must erase that
    // level from the map - leaving a zero-quantity PriceLevel behind
    // makes best_bid()/best_ask() report a phantom price with no real
    // liquidity (they only check whether the map entry exists, not
    // whether it's actually empty).
    for (auto it = bids_.begin(); it != bids_.end(); ++it) {
        PriceLevel& level = it->second;
        for (auto* o : level.orders) {
            if (o->id == id && !o->is_done()) {
                level.total_quantity -= o->remaining();
                o->status = OrderStatus::CANCELLED;
                if (level.empty()) {
                    bids_.erase(it);
                }
                return true;
            }
        }
    }
    // Search asks
    for (auto it = asks_.begin(); it != asks_.end(); ++it) {
        PriceLevel& level = it->second;
        for (auto* o : level.orders) {
            if (o->id == id && !o->is_done()) {
                level.total_quantity -= o->remaining();
                o->status = OrderStatus::CANCELLED;
                if (level.empty()) {
                    asks_.erase(it);
                }
                return true;
            }
        }
    }
    return false;
}

// ── Queries ───────────────────────────────────────────────────────────────────

std::optional<Price> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<Price> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<Price> OrderBook::mid_price() const {
    auto bid = best_bid();
    auto ask = best_ask();
    if (!bid || !ask) return std::nullopt;
    return (*bid + *ask) / 2;
}

Price OrderBook::spread() const {
    auto bid = best_bid();
    auto ask = best_ask();
    if (!bid || !ask) return 0;
    return *ask - *bid;
}

Quantity OrderBook::bid_depth_at(Price p) const {
    auto it = bids_.find(p);
    return (it != bids_.end()) ? it->second.total_quantity : 0;
}

Quantity OrderBook::ask_depth_at(Price p) const {
    auto it = asks_.find(p);
    return (it != asks_.end()) ? it->second.total_quantity : 0;
}

Quantity OrderBook::total_bid_quantity() const {
    Quantity total = 0;
    for (auto& [p, level] : bids_) total += level.total_quantity;
    return total;
}

Quantity OrderBook::total_ask_quantity() const {
    Quantity total = 0;
    for (auto& [p, level] : asks_) total += level.total_quantity;
    return total;
}

// ── print - human-readable snapshot ──────────────────────────────────────────

void OrderBook::print(int levels) const {
    std::cout << "\n=== Order Book: " << symbol_ << " ===\n";
    std::cout << std::setw(12) << "Price"
              << std::setw(12) << "Qty"
              << std::setw(10) << "Side" << "\n";
    std::cout << std::string(36, '-') << "\n";

    // Print asks (best ask last = closest to mid)
    std::vector<std::pair<Price, Quantity>> ask_levels;
    for (auto& [p, level] : asks_) {
        if ((int)ask_levels.size() >= levels) break;
        if (level.total_quantity > 0)
            ask_levels.emplace_back(p, level.total_quantity);
    }
    // Print asks in reverse (highest first)
    for (auto it = ask_levels.rbegin(); it != ask_levels.rend(); ++it) {
        std::cout << std::setw(12) << it->first
                  << std::setw(12) << it->second
                  << std::setw(10) << "ASK\n";
    }

    // Spread line
    if (best_bid() && best_ask()) {
        std::cout << "  --- spread: " << spread()
                  << " ticks | mid: " << *mid_price() << " ---\n";
    }

    // Print bids (best bid first = closest to mid)
    int count = 0;
    for (auto& [p, level] : bids_) {
        if (count++ >= levels) break;
        if (level.total_quantity > 0) {
            std::cout << std::setw(12) << p
                      << std::setw(12) << level.total_quantity
                      << std::setw(10) << "BID\n";
        }
    }
    std::cout << std::string(36, '=') << "\n";
    std::cout << "  Trades: " << trades_.size() << "\n\n";
}

} // namespace orderbook
