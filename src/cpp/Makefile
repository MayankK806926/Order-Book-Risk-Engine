# Makefile - Order book matching engine
#
# Usage:
#   make          → build everything
#   make test     → build and run tests
#   make clean    → remove build artifacts

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wpedantic
SRCS     = order_book.cpp risk_manager.cpp
HDRS     = order_book.h   risk_manager.h
TEST_SRC = test_order_book.cpp
TARGET   = test_orderbook

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS) $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(SRCS) $(TEST_SRC) -o $(TARGET)
	@echo "Build successful: ./$(TARGET)"

test: $(TARGET)
	@echo "\nRunning order book tests..."
	@./$(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all test clean
